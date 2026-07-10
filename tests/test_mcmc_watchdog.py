"""Tests for the stalled-chain MCMC watchdog."""

from types import SimpleNamespace

import numpy as np
import pytest

from celeri.mcmc_watchdog import (
    _drop_unfinished_chains,
    _weighted_quantile,
    sample_with_watchdog,
)


def _chain_progress(finished_per_chain, total=100):
    return [
        SimpleNamespace(finished_draws=n, total_draws=total, latest_num_steps=7)
        for n in finished_per_chain
    ]


def _make_aborted_trace(
    valid_per_chain: list[int], n_draws: int, derived_nan: bool = False
):
    """Build a synthetic aborted trace with nutpie-style padding.

    Each chain has ``valid_per_chain[i]`` valid draws followed by padding:
    NaN for floats, False for bools, mimicking what nutpie's ``abort()``
    returns. With ``derived_nan``, a derived posterior variable carries a
    legitimate NaN inside chain 0's valid draws. Returns an
    ``xarray.DataTree`` on ArviZ>=1 and an ``arviz.InferenceData`` on
    ArviZ<1.
    """
    import arviz as az

    n_chains = len(valid_per_chain)
    x = np.full((n_chains, n_draws), np.nan)
    logp = np.full((n_chains, n_draws), np.nan)
    for i, n_valid in enumerate(valid_per_chain):
        x[i, :n_valid] = 1.0
        logp[i, :n_valid] = -100.0
    posterior = {"x": x}
    if derived_nan:
        derived = x.copy()
        derived[0, min(5, n_draws - 1)] = np.nan
        posterior["derived"] = derived
    groups = {
        "posterior": posterior,
        "sample_stats": {
            "logp": logp,
            "diverging": np.zeros((n_chains, n_draws), dtype=bool),
        },
    }
    coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}

    # LEGACY-MCMC: ArviZ<1's ``from_dict`` takes groups as keyword arguments;
    # ArviZ>=1's takes a single mapping of group name -> variables.
    # (``from_datatree`` only exists on ArviZ<1, so it flags the legacy
    # stack.) On cleanup, keep only the ArviZ>=1 branch below.
    if hasattr(az, "from_datatree"):
        return az.from_dict(**groups, coords=coords)
    return az.from_dict(groups, coords=coords)


class TestDropUnfinishedChains:
    """Verify post-abort recovery drops padded chains, keeps finished ones."""

    def test_keeps_finished_chains_at_full_length(self):
        # Chains 0 and 2 finished; chains 1 and 3 are stuck stragglers.
        trace = _make_aborted_trace([100, 30, 100, 0], n_draws=100)
        progress = _chain_progress([200, 130, 200, 0], total=200)
        reduced, dropped = _drop_unfinished_chains(trace, progress)
        assert dropped == [1, 3]
        assert list(reduced.posterior.chain.values) == [0, 2]
        assert reduced.posterior.sizes["draw"] == 100
        assert np.isfinite(reduced.posterior["x"].values).all()
        assert np.isfinite(reduced.sample_stats["logp"].values).all()

    def test_keeps_chain_that_finished_during_abort(self):
        # The progress snapshot predates abort() and says chain 1 was at
        # 130 of 200 draws, but the returned trace shows it complete: the
        # trace is authoritative, so nothing is dropped.
        trace = _make_aborted_trace([100, 100], n_draws=100)
        progress = _chain_progress([200, 130], total=200)
        reduced, dropped = _drop_unfinished_chains(trace, progress)
        assert dropped == []
        assert reduced.posterior.sizes["chain"] == 2

    def test_non_finite_derived_posterior_is_not_padding(self):
        # Chain 0 finished but a derived posterior variable holds a
        # legitimate NaN within its valid draws. Completion is judged from
        # the sampler-owned log-density, so chain 0 must be retained and
        # only the genuinely unfinished chain 1 dropped.
        trace = _make_aborted_trace([100, 50], n_draws=100, derived_nan=True)
        progress = _chain_progress([200, 150], total=200)
        reduced, dropped = _drop_unfinished_chains(trace, progress)
        assert dropped == [1]
        assert list(reduced.posterior.chain.values) == [0]

    def test_all_finished_is_a_no_op(self):
        trace = _make_aborted_trace([100, 100], n_draws=100)
        progress = _chain_progress([200, 200], total=200)
        reduced, dropped = _drop_unfinished_chains(trace, progress)
        assert dropped == []
        assert reduced.posterior.sizes["chain"] == 2


class TestWeightedQuantile:
    def test_unweighted_median(self):
        assert _weighted_quantile([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)], 0.50) == 2.0

    def test_time_weighting_favors_long_draws(self):
        obs = [(1.0, 1.0)] * 9 + [(10.0, 10.0)]
        assert _weighted_quantile(obs, 0.50) == 10.0

    def test_empty_is_infinite(self):
        assert _weighted_quantile([], 0.50) == float("inf")


def _snap(finished_per_chain, runtime_s_per_chain, total=100):
    """One progress snapshot: per-chain (finished draws, cumulative seconds)."""
    return [
        SimpleNamespace(
            finished_draws=f,
            total_draws=total,
            runtime_ms=r * 1000.0,
            latest_num_steps=7,
        )
        for f, r in zip(finished_per_chain, runtime_s_per_chain, strict=True)
    ]


class _FakeSampler:
    """Stand-in for nutpie's non-blocking sampler.

    Mimics the nutpie 0.16.10 semantics the watchdog relies on: ``wait``
    raises ``TimeoutError`` until sampling completes, and ``abort``
    returns the (padded) trace. Each ``wait`` advances the fake clock by
    its timeout and plays the next snapshot of the progress script (the
    last entry repeats; ``None`` means sampling completed).
    """

    def __init__(self, progress_callback, progress_script, result, clock):
        self._progress_callback = progress_callback
        self._progress_script = progress_script
        self._result = result
        self._clock = clock
        self._calls = 0
        self.aborted = False

    def wait(self, timeout=None):
        self._clock.now += timeout
        progress = self._progress_script[
            min(self._calls, len(self._progress_script) - 1)
        ]
        self._calls += 1
        if progress is None:  # sampling complete
            return self._result
        self._progress_callback(progress)
        raise TimeoutError

    def abort(self):
        self.aborted = True
        return self._result


class TestSampleWithWatchdog:
    """Verify the watchdog poll loop against a fake non-blocking sampler.

    All scenarios use total_draws=100 with n_tune=20 and a fake clock
    advancing 1 s per poll. Healthy draws take 1 s; the stall threshold is
    therefore 5 s (``STALL_FACTOR`` times the finished chain's p90).
    """

    N_TUNE = 20

    def _run(self, monkeypatch, progress_script, result):
        import nutpie

        from celeri import mcmc_watchdog

        clock = SimpleNamespace(now=0.0)
        samplers = []

        def fake_sample(compiled, *, blocking, progress_callback, **kwargs):
            assert blocking is False
            sampler = _FakeSampler(progress_callback, progress_script, result, clock)
            samplers.append(sampler)
            return sampler

        monkeypatch.setattr(nutpie, "sample", fake_sample)
        monkeypatch.setattr(mcmc_watchdog, "POLL_SECONDS", 1.0)
        monkeypatch.setattr(
            mcmc_watchdog, "time", SimpleNamespace(monotonic=lambda: clock.now)
        )
        trace, dropped = sample_with_watchdog("compiled", n_tune=self.N_TUNE)
        return trace, dropped, samplers[0]

    def test_returns_trace_on_completion(self, monkeypatch):
        script = [_snap([50, 60], [50.0, 60.0]), None]
        trace, dropped, sampler = self._run(monkeypatch, script, "trace")
        assert trace == "trace"
        assert dropped == []
        assert not sampler.aborted

    def test_aborts_dead_chain(self, monkeypatch):
        # Chains 0 and 2 finished; chain 1 froze at 50 of 100 draws. Its
        # growing silence eventually dominates its time-weighted median.
        result = _make_aborted_trace([100, 50, 100], n_draws=100)
        script = [_snap([100, 50, 100], [100.0, 50.0, 100.0])]
        trace, dropped, sampler = self._run(monkeypatch, script, result)
        assert sampler.aborted
        assert dropped == [1]
        assert list(trace.posterior.chain.values) == [0, 2]
        assert trace.posterior.sizes["draw"] == 100
        assert np.isfinite(trace.posterior["x"].values).all()

    def test_no_abort_while_no_chain_finished(self, monkeypatch):
        # All chains frozen but none finished: the watchdog must not fire
        # (there is nothing to continue with); completion returns the trace.
        frozen = _snap([50, 50], [50.0, 50.0])
        script = [frozen] * 200 + [None]
        trace, dropped, sampler = self._run(monkeypatch, script, "trace")
        assert trace == "trace"
        assert dropped == []
        assert not sampler.aborted

    def test_keeps_slow_but_alive_chain(self, monkeypatch):
        # Chain 1 advances steadily at 3 s per draw (3x the finished
        # chain's pace, below the 5x threshold): never dropped.
        script = [_snap([100, 50 + i], [100.0, 3.0 * (50 + i)]) for i in range(40)]
        script.append(None)
        trace, dropped, sampler = self._run(monkeypatch, script, "trace")
        assert trace == "trace"
        assert dropped == []
        assert not sampler.aborted

    def test_drops_pathologically_slow_chain(self, monkeypatch):
        # Chain 1 keeps completing draws, but at 20 s each: a fixed
        # no-progress timeout would never fire on it.
        result = _make_aborted_trace([100, 50], n_draws=100)
        script = [_snap([100, 50 + i], [100.0, 20.0 * (50 + i)]) for i in range(50)]
        _trace, dropped, sampler = self._run(monkeypatch, script, result)
        assert sampler.aborted
        assert dropped == [1]

    def test_warmup_pace_is_judged_against_warmup(self, monkeypatch):
        # Warmup draws run 3x slower for everyone: the finished chain does
        # 20 warmup draws at 3 s, then 80 sampling draws at 1 s (snapshots
        # split at the boundary so each phase gets its own reference).
        # Chain 1 crawls through warmup at 8 s per draw: slow, but under
        # 5x the 3 s warmup p90, so it survives.
        script = [_snap([20, 5], [60.0, 40.0])]
        script += [_snap([100, 5 + i], [140.0, 8.0 * (5 + i)]) for i in range(1, 10)]
        script.append(None)
        trace, dropped, sampler = self._run(monkeypatch, script, "trace")
        assert trace == "trace"
        assert dropped == []
        assert not sampler.aborted

    def test_sampling_pace_is_judged_against_sampling(self, monkeypatch):
        # Same finished chain as above. Chain 1 is past warmup and draws at
        # 8 s: over 5x the 1 s sampling p90, so it is dropped -- even though
        # it would look acceptable against a pooled reference diluted by
        # the slow warmup draws (pooled p90 is 3 s, threshold 15 s).
        result = _make_aborted_trace([100, 50], n_draws=100)
        script = [_snap([20, 20], [60.0, 60.0])]
        script += [
            _snap([100, 50 + i], [140.0, 60.0 + 8.0 * (30 + i)]) for i in range(50)
        ]
        _trace, dropped, sampler = self._run(monkeypatch, script, result)
        assert sampler.aborted
        assert dropped == [1]

    def test_drops_chain_that_never_started(self, monkeypatch):
        # Chain 1 never completes a single draw: only its censored silence
        # is observable, and it alone eventually exceeds the threshold.
        result = _make_aborted_trace([100, 0], n_draws=100)
        script = [_snap([100, 0], [120.0, 0.0])]
        _trace, dropped, sampler = self._run(monkeypatch, script, result)
        assert sampler.aborted
        assert dropped == [1]

    def test_chain_finishing_during_abort_is_kept(self, monkeypatch):
        # The last snapshot shows chain 1 frozen at 50 draws, so the
        # watchdog aborts -- but the trace returned by abort() shows the
        # chain actually finished in the meantime. It must be kept.
        result = _make_aborted_trace([100, 100], n_draws=100)
        script = [_snap([100, 50], [100.0, 50.0])]
        trace, dropped, sampler = self._run(monkeypatch, script, result)
        assert sampler.aborted
        assert dropped == []
        assert trace.posterior.sizes["chain"] == 2

    def test_slow_warmup_history_does_not_condemn_healthy_sampler(self, monkeypatch):
        # Chain 1 crawled through warmup at 20 s per draw (well over 5x the
        # 3 s warmup p90) but now samples at a healthy 1 s: only its current
        # phase counts, so it must not be dropped.
        script = [_snap([20, 5], [60.0, 100.0])]
        script.append(_snap([100, 20], [140.0, 400.0]))
        script += [_snap([100, 20 + i], [140.0, 400.0 + 1.0 * i]) for i in range(1, 30)]
        script.append(None)
        trace, dropped, sampler = self._run(monkeypatch, script, "trace")
        assert trace == "trace"
        assert dropped == []
        assert not sampler.aborted

    def test_rejects_missing_n_tune(self):
        with pytest.raises(ValueError, match="warmup length"):
            sample_with_watchdog("compiled", n_tune=None)

    def test_rejects_colliding_sample_kwargs(self):
        with pytest.raises(ValueError, match="blocking"):
            sample_with_watchdog("compiled", n_tune=20, blocking=True)


class TestLiveDoubleWell:
    """End-to-end run against real nutpie, no fakes.

    The target is a double-well potential in x0 (wells at +-3, ~40-nat
    barrier, so chains never cross), and the logp function sleeps 2 ms per
    evaluation whenever x0 > 0. Chains seeded into the positive well
    therefore sample correctly but ~two orders of magnitude too slowly --
    the failure mode a fixed no-progress timeout can never catch.
    """

    def test_drops_genuinely_slow_chains(self, monkeypatch):
        import itertools
        import threading
        import time

        from nutpie.compiled_pyfunc import from_pyfunc

        from celeri import mcmc_watchdog

        monkeypatch.setattr(mcmc_watchdog, "POLL_SECONDS", 0.2)

        ndim, barrier_k, sleep_s = 4, 0.5, 0.002

        def make_logp_fn():
            def logp(x):
                x0, rest = x[0], x[1:]
                if x0 > 0:
                    time.sleep(sleep_s)
                value = -barrier_k * (x0**2 - 9.0) ** 2 - 0.5 * np.sum(rest**2)
                grad = np.empty_like(x)
                grad[0] = -4.0 * barrier_k * x0 * (x0**2 - 9.0)
                grad[1:] = -rest
                return value, grad

            return logp

        def make_expand_fn(seed1, seed2, chain):
            def expand(x):
                return {"x": x.copy()}

            return expand

        counter = itertools.count()
        lock = threading.Lock()

        def initial_point(seed):
            # Deal chains alternately into the two wells. Init calls run in
            # thread order, so which chain ids are slow varies; exactly half
            # must be dropped. The well floor has zero gradient, which
            # nutpie rejects as an initial point, hence the jitter.
            with lock:
                i = next(counter)
            rng = np.random.default_rng(seed)
            point = 0.2 * rng.normal(size=ndim)
            point[0] += 3.0 if i % 2 else -3.0
            return point

        compiled = from_pyfunc(
            ndim=ndim,
            make_logp_fn=make_logp_fn,
            make_expand_fn=make_expand_fn,
            expanded_dtypes=[np.dtype("float64")],
            expanded_shapes=[(ndim,)],
            expanded_names=["x"],
            make_initial_point_fn=initial_point,
        )

        tune, draws = 400, 800
        trace, dropped = sample_with_watchdog(
            compiled,
            n_tune=tune,
            tune=tune,
            draws=draws,
            chains=6,
            cores=6,
            seed=0,
            progress_bar=False,
        )

        post = trace.posterior
        assert len(dropped) == 3, f"expected the 3 slow chains dropped: {dropped}"
        assert post.sizes["chain"] == 3
        assert post.sizes["draw"] == draws
        assert np.isfinite(post["x"].values).all()
        x0_means = post["x"].values[..., 0].mean(axis=1)
        assert (x0_means < -2.5).all(), "kept chains must be in the fast well"
