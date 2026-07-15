"""Watchdog for MCMC sampling that aborts and drops stalled chains.

Enabled via ``Config.mcmc_drop_stalled_chains``; see
:func:`sample_with_watchdog` for the stall rule.
"""

import time
from typing import Any

import numpy as np
from loguru import logger

STALL_FACTOR = 5.0
"""A running chain is stalled when the time-weighted median of its draw
durations exceeds this factor times the time-weighted 90th percentile of
the slowest finished chain's draw durations."""

POLL_SECONDS = 15.0


def _weighted_quantile(observations: list[tuple[float, float]], q: float) -> float:
    """Quantile of ``(value, weight)`` pairs, weighted by the second entry."""
    if not observations:
        return float("inf")
    observations = sorted(observations)
    total = sum(w for _, w in observations)
    acc = 0.0
    for value, weight in observations:
        acc += weight
        if acc >= q * total:
            return value
    return observations[-1][0]


def _drop_unfinished_chains(trace: Any, progress: Any) -> tuple[Any, list[int]]:
    """Drop the chains that had not finished when sampling was aborted.

    ``sampler.abort()`` pads every chain to the longest chain's draw count
    (NaN for floats, zeros for ints/bools), so finished chains keep all
    their draws while unfinished chains end in padding. Completion is
    judged from the returned trace itself, because a chain may finish
    between the last progress snapshot and the abort taking effect; the
    snapshot is used only for logging. The authoritative field is the
    sampler-owned log-density ``sample_stats["logp"]``: nutpie writes one
    finite value per completed draw and pads with NaN (verified on nutpie
    0.16.10, where unfinished chains carry a contiguous NaN tail).
    Derived posterior quantities can be legitimately non-finite on a valid
    draw, so they must not be consulted. nutpie's chain coordinate is
    ``arange(n_chains)``, so the returned labels are integers coinciding
    with chain positions.
    """
    log_density = trace.sample_stats["logp"].values
    keep, dropped = [], []
    for i in range(log_density.shape[0]):
        (dropped if np.isnan(log_density[i]).any() else keep).append(i)
    if not dropped:
        return trace, []
    labels = [int(c) for c in np.asarray(trace.posterior.chain.values)[dropped]]
    logger.warning(
        f"Dropping unfinished chains {labels} and continuing with the "
        f"{len(keep)} finished chains "
        f"(finished draws/total draws/latest trajectory steps per chain: "
        f"{[(c.finished_draws, c.total_draws, c.latest_num_steps) for c in progress]})."
    )
    if len(keep) == 1:
        logger.warning(
            "Only one chain was retained: between-chain convergence "
            "diagnostics such as R-hat are unavailable for this run."
        )
    return trace.isel(chain=keep), labels


def sample_with_watchdog(
    compiled: Any, *, n_tune: int | None, **sample_kwargs: Any
) -> tuple[Any, list[int]]:
    """Sample with nutpie, aborting once only stalled chains are left running.

    Uses nutpie's non-blocking API (``blocking=False`` plus a progress
    callback) and polls ``sampler.wait``. A running chain is stalled when
    the time-weighted median of its draw durations -- counting the time
    since its last completed draw as a censored observation -- exceeds
    ``STALL_FACTOR`` times the time-weighted 90th percentile of the
    slowest finished chain's draw durations in the chain's current phase
    (warmup or sampling), since warmup draws are systematically slower.
    Durations are measured as per-poll batch means, so the quantiles are
    approximate at the poll resolution.
    Once at least one chain has finished and every unfinished chain is
    stalled, sampling is aborted and the stalled chains are dropped, so
    the estimation continues with the finished chains at their full draw
    count.

    Behavior pinned to nutpie 0.16.10: ``sampler.wait(timeout=...)``
    raises ``TimeoutError`` while sampling is still running, and on
    success consumes the results (so ``abort()`` is only called when
    ``wait`` did not succeed); ``sampler.abort()`` returns a trace padded
    to the longest chain, so finished chains keep every draw; the
    progress callback receives ``list[ChainProgress]``, one entry per
    chain, updated once per completed draw; each callback gets a freshly
    cloned snapshot, never mutated afterwards, so the poll loop can read
    it safely.

    Returns:
        ``(trace, dropped)`` where ``dropped`` lists the chain labels
        removed from the trace (empty if sampling completed normally).
    """
    import nutpie

    if n_tune is None:
        raise ValueError(
            "The stalled-chain watchdog needs an explicit warmup length to "
            "compare warmup and sampling draws separately; set "
            "config.mcmc_tune to an integer."
        )
    colliding = {"blocking", "progress_callback"} & sample_kwargs.keys()
    if colliding:
        raise ValueError(
            f"The stalled-chain watchdog controls {sorted(colliding)}; "
            f"remove them from sample_kwargs."
        )

    progress_holder: dict[str, Any] = {"chains": None}

    def _progress_callback(chains: Any) -> None:
        # Called by nutpie with list[ChainProgress], one entry per chain.
        # Runs on a nutpie background thread; exceptions are swallowed.
        progress_holder["chains"] = chains

    sampler = nutpie.sample(
        compiled,
        blocking=False,
        progress_callback=_progress_callback,
        **sample_kwargs,
    )

    prev: dict[int, tuple[int, float]] = {}  # chain -> (draws, runtime seconds)
    last_advance: dict[int, float] = {}  # chain -> time of last completed draw
    # chain -> phase (True = warmup) -> [(mean draw duration, weight)]
    durations: dict[int, dict[bool, list[tuple[float, float]]]] = {}

    while True:
        try:
            return sampler.wait(timeout=POLL_SECONDS), []
        except TimeoutError:
            pass
        chains = progress_holder["chains"]
        if chains is None:
            continue
        now = time.monotonic()
        for idx, c in enumerate(chains):
            last_advance.setdefault(idx, now)
            pf, pr = prev.get(idx, (0, 0.0))
            runtime = c.runtime_ms / 1000.0
            dd = c.finished_draws - pf
            if dd <= 0:
                continue
            mean = (runtime - pr) / dd
            phases = durations.setdefault(idx, {True: [], False: []})
            # Draws numbered up to n_tune are warmup; a batch spanning the
            # boundary contributes its mean to both phases.
            n_warm = min(max(n_tune - pf, 0), dd)
            if n_warm:
                phases[True].append((mean, mean * n_warm))
            if dd - n_warm:
                phases[False].append((mean, mean * (dd - n_warm)))
            prev[idx] = (c.finished_draws, runtime)
            last_advance[idx] = now

        finished = [
            idx for idx, c in enumerate(chains) if c.finished_draws >= c.total_draws
        ]
        unfinished = [
            idx for idx, c in enumerate(chains) if c.finished_draws < c.total_draws
        ]
        if not finished or not unfinished:
            continue
        # Anchor the threshold to the slowest finished chain (max of the
        # per-chain p90s, not a pooled quantile): a chain within
        # STALL_FACTOR of some chain that actually finished is not stuck.
        p90 = {
            phase: max(
                _weighted_quantile(durations[idx][phase], 0.90) for idx in finished
            )
            for phase in (True, False)
        }

        stalled = []
        for idx in unfinished:
            # Judge each chain only by its current phase: history from a
            # completed slow warmup must not condemn a healthy sampler.
            in_warmup = chains[idx].finished_draws < n_tune
            phases = durations.get(idx, {True: [], False: []})
            silence = now - last_advance[idx]
            observations = [*phases[in_warmup], (silence, silence)]
            median = _weighted_quantile(observations, 0.50)
            if median > STALL_FACTOR * p90[in_warmup]:
                stalled.append(idx)
        if stalled == unfinished:
            logger.warning(
                f"MCMC chains {stalled} are stalled (time-weighted median "
                f"draw duration exceeds {STALL_FACTOR}x the finished "
                f"chains' 90th percentile) and all other chains have "
                f"finished; aborting sampling."
            )
            return _drop_unfinished_chains(sampler.abort(), chains)
