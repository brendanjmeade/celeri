"""Filter MCMC chains by WAIC to remove those stuck in subdominant posterior modes.

When sampling a multimodal posterior with multiple chains, some chains may
converge to modes with lower predictive quality.  This module provides
:func:`select_chains`, which compares per-chain WAIC (via :func:`waic_summary`)
and excludes chains whose WAIC deficit relative to the leader exceeds a
z-score threshold calibrated by Monte Carlo noise.
"""

import warnings
from dataclasses import dataclass

import arviz as az
import numpy as np
from loguru import logger
from scipy.special import logsumexp


@dataclass
class WaicSummary:
    """Per-chain WAIC with MC noise, used to filter subdominant-mode chains.

    :func:`select_chains` compares per-chain WAIC to identify chains stuck
    in subdominant posterior modes.  ``mc_sd`` sets the scale of the
    filtering threshold: a chain is excluded when its WAIC deficit from
    the leader exceeds a z-score cutoff in units of pairwise ``mc_sd``.

    Given a log-likelihood matrix ``ll`` of shape ``(S, N)`` — S posterior
    draws by N observations — the summary is computed in two stages.

    **Stage 1 — WAIC** (aggregate over S draws at each observation):

    For each observation *i*, aggregate across draws:

    - ``lppd_i = log( (1/S) sum_s exp(ll[s,i]) )`` — how well the
      posterior predictive fits observation *i*.
    - ``p_waic_i = Var_s(ll[s,i])`` — effective complexity penalty: how
      sensitive observation *i*'s log-likelihood is to the choice of draw.

    Total ``elpd_waic = sum_i (lppd_i - p_waic_i) = lppd - p_waic``.
    ``se = sqrt(N · Var_i(elpd_waic_i))`` — uncertainty in the total due
    to having finitely many observations.

    **Stage 2 — MC noise** (sensitivity of Stage 1 to individual draws):

    The infinitesimal jackknife differentiates the total WAIC w.r.t. the
    weight of each draw *s*, yielding a per-draw influence ``h_s`` of
    shape ``(S,)``.  Then ``mc_sd = sqrt(S · Var_s(h_s))`` — uncertainty
    in the total due to having finitely many draws.

    ``lppd``, ``p_waic``, and ``se`` match ``arviz.waic()`` (``ddof=0``
    convention).  ``mc_sd`` is computed via the infinitesimal jackknife
    and is not provided by ArviZ.

    LOO-based ELPD has more powerful diagnostics (Pareto k), but the
    z-scores produced by WAIC and LOO are indistinguishable for chain
    selection (pairwise dWAIC and dLOO differ by < 0.5, negligible
    relative to ``mc_sd`` ~ 6).  We use WAIC because the IJ for
    ``mc_sd`` is simpler.

    Attributes
    ----------
    lppd : float
        Log pointwise predictive density (summed over observations).
    p_waic : float
        Effective number of parameters (summed over observations).
    se : float
        Standard error of WAIC over observations (matches ArviZ ``se``).
    mc_sd : float
        Monte Carlo noise SD of WAIC over draws (via IJ).  Sets the
        scale of the filtering threshold in :func:`select_chains`.
    """

    lppd: float
    p_waic: float
    se: float
    mc_sd: float

    @property
    def elpd_waic(self) -> float:
        """Total WAIC: ``lppd - p_waic`` (higher is better)."""
        return self.lppd - self.p_waic


def waic_summary(ll: np.ndarray) -> WaicSummary:
    """Compute WAIC and its MC noise SD from a single chain's log-likelihoods.

    See :class:`WaicSummary` for the decomposition and its role in chain
    selection.

    Parameters
    ----------
    ll : ndarray of shape (S, N)
        Pointwise log-likelihoods.  S = posterior draws, N = observations.
        ``ll[s, i] = log p(y_i | theta_s)``.

    Returns
    -------
    WaicSummary
        Quadruple ``(lppd, p_waic, se, mc_sd)``.  WAIC = ``lppd - p_waic``,
        available via ``.elpd_waic``.
    """
    S, N = ll.shape

    # ── Stage 1: WAIC (aggregate over draws at each observation) ─────────

    # lppd_i = log( (1/S) sum_s exp(ll[s,i]) )  — (N,)
    log_mean_L = logsumexp(ll, axis=0) - np.log(S)  # (N,)
    lppd = float(log_mean_L.sum())  # scalar

    # p_waic_i = Var_s(ll[s,i])  — (N,)  [ddof=0 to match ArviZ]
    var_ll = np.var(ll, axis=0)  # (N,)
    p_waic = float(var_ll.sum())  # scalar

    # elpd_waic_i = lppd_i - p_waic_i  — (N,)
    elpd_i = log_mean_L - var_ll  # (N,)
    se = float(np.sqrt(N * np.var(elpd_i)))  # scalar [ddof=0 to match ArviZ]

    # ── Stage 2: MC noise (sensitivity of WAIC to each draw) ─────────────
    #
    # The IJ differentiates the Stage 1 total w.r.t. each draw's weight.
    # h_s decomposes into LPPD and p_WAIC contributions.

    # LPPD influence of draw s:
    #   ratio[s,i] = exp(ll[s,i]) / mean_s(exp(ll[.,i]))  — (S, N)
    #   lppd_influence[s] = (1/S) * sum_i (ratio[s,i] - 1)  — (S,)
    lppd_influence = (np.exp(ll - log_mean_L) - 1).sum(axis=1) / S  # (S,)

    # p_WAIC influence of draw s:
    #   excess[s,i] = (ll[s,i] - mean(ll[.,i]))^2 - Var(ll[.,i])  — (S, N)
    #   Positive when draw s is unusually far from the per-obs mean.
    #   pwaic_influence[s] = (1/S) * sum_i excess[s,i]  — (S,)
    ll_mean = ll.mean(axis=0)  # (N,)
    excess = (ll - ll_mean) ** 2 - var_ll  # (S, N)
    pwaic_influence = excess.sum(axis=1) / S  # (S,)

    # Total per-draw influence on WAIC (= LPPD - p_WAIC)
    h = lppd_influence - pwaic_influence  # (S,)

    # IJ bootstrap SD: mc_sd = sqrt(S · Var_s(h_s))  [ddof=0 throughout]
    mc_sd = float(np.sqrt(S * np.var(h)))

    return WaicSummary(lppd=lppd, p_waic=p_waic, se=se, mc_sd=mc_sd)


def select_chains(
    trace: az.InferenceData,
    *,
    z_threshold: float = 6.0,
    ll_var: str = "station_velocity",
) -> list[int]:
    """Filter out MCMC chains stuck in subdominant posterior modes.

    When a multimodal posterior is sampled with multiple chains, some
    chains may converge to modes with lower predictive quality.  This
    function identifies such chains by comparing per-chain WAIC, using
    Monte Carlo noise (``mc_sd``) to distinguish real WAIC deficits from
    sampling noise.

    The chain with the highest WAIC is the leader.  Any chain whose WAIC
    deficit relative to the leader exceeds ``z_threshold`` times the
    pairwise MC noise SD is excluded.  The pairwise noise SD between two
    independent chains A and B is ``sqrt(mc_sd_A**2 + mc_sd_B**2)``.

    Only ``mc_sd`` (draw-axis noise) enters the z-score, not ``se``
    (observation-axis noise), because all chains share the same
    observations and that uncertainty cancels in pairwise differences.

    Diagnostics example
    -------------------
    A typical log table (4 chains)::

      Chain  div      LPPD  p_WAIC      WAIC  SE_WAIC  k>0.7  dW-dL    MC_SD   dWAIC      z
          0    0  -13710.9    59.2  -13770.1    109.0      5  +0.03    0.748  -222.9 -218.3 ✗
          1    0  -13491.1    56.2  -13547.2    105.4      9  +0.00    0.695    +0.0   +0.0 ✓
          2    0  -13492.0    55.6  -13547.6    105.4      3  +0.00    0.739    -0.4   -0.4 ✓
          3    0  -13492.1    55.9  -13547.9    105.4      4  +0.03    0.705    -0.7   -0.7 ✓

    Column guide (left to right):

    **div** — Number of divergent transitions in the chain.  Divergences
    indicate the sampler encountered regions of high curvature it could
    not follow; even a few divergences can bias posterior estimates.
    Zero is ideal.

    **LPPD** — Total log-likelihood in nats: the raw energy of the
    posterior predictive (higher = better fit).  Dominates WAIC since
    p_WAIC is comparatively small.  Chains in the same mode have nearly
    identical LPPD; a chain in a subdominant mode shows a clearly worse
    value (here chain 0 is ~220 nats below chains 1-3).

    **p_WAIC** — Effective number of parameters (complexity penalty).
    Usually a small fraction of |LPPD| (here ~57 vs ~13 500).  Stable
    across chains in the same mode.  Different modes *can* have different
    p_WAIC (here chain 0 is slightly higher at 59.2 vs ~56), reflecting
    a different effective complexity in that region of parameter space.

    **WAIC** — ``LPPD - p_WAIC`` (higher = better).  A bias-corrected
    estimate of the out-of-sample expected log pointwise predictive
    density (ELPD), where p_WAIC corrects for the optimism of the
    in-sample LPPD.  This is the basis for all comparisons, but it is
    the *difference* dWAIC (further right) that we actually evaluate.

    **SE_WAIC** — Standard error of WAIC across observations.  A
    standard diagnostic quantity included for completeness.  Since all
    chains share the same observations, SE cancels in pairwise
    differences and does *not* enter the z-score.  Useful only as a
    sanity check: a noticeably different SE (here 109.0 for chain 0 vs
    ~105 for chains 1-3) is a secondary signal that the chain sees the
    data differently.

    **k>0.7** — Number of observations with Pareto k > 0.7, a
    diagnostic from LOO cross-validation (PSIS-LOO) indicating where
    importance-sampling leave-one-out estimates are unreliable.  Zero or
    low counts are ideal.  High counts (tens to hundreds) suggest the
    model is sensitive to particular observations; compare across chains
    to see whether problematic observations are mode-specific.

    **dW-dL** — ``dWAIC - dLOO`` for each chain relative to the leader.
    WAIC and LOO are both approximations of ELPD; this column checks
    that they agree on inter-chain differences.  Values should be small
    relative to MC_SD (here ~0.03 vs MC_SD ~0.7, i.e. a few percent),
    confirming that the simpler WAIC-based z-score is trustworthy.
    Values comparable to MC_SD would indicate meaningful disagreement
    between the two estimators, warranting investigation.

    **MC_SD** — Monte Carlo noise SD of WAIC (via the infinitesimal
    jackknife over draws).  Sets the scale of dWAIC in standard
    deviations: a dWAIC of one MC_SD is indistinguishable from sampling
    noise.  MC_SD is usually similar across chains (even across modes,
    here 0.748 for subdominant chain 0 vs 0.695–0.739 for chains 1-3)
    because it depends mainly on the number of draws and observations,
    not on which mode the chain occupies.

    **dWAIC** — WAIC difference from the leader (always ≤ 0; exactly 0
    for the leader by definition).  The numerator of the z-score.
    Same-mode chains differ by at most a few MC_SD (here chains 1-3 are
    within 1 MC_SD); a subdominant-mode chain differs by many MC_SD
    (chain 0: -222.9, i.e. ~300 × MC_SD).

    **z** — z-score: distance from the leader in standard deviations,
    ``dWAIC / sqrt(mc_sd_chain² + mc_sd_leader²)``.  Always ≤ 0.
    Most same-mode chains should have |z| < 4; subdominant-mode
    outliers should have |z| > 6.

    **✓ / ✗** — Whether the chain is kept (``z ≥ -z_threshold``) or
    excluded.  In practice the gap between same-mode and subdominant-mode
    z-scores is enormous (here -218 vs -0.7), so the filtering decision
    is not sensitive to the exact threshold.

    Parameters
    ----------
    trace : arviz.InferenceData
        Must contain a ``log_likelihood`` group with variable ``ll_var``.
        The log-likelihood array has shape ``(n_chains, S, ...)``, where S is
        the number of posterior samples.  It is reshaped to ``(S, N)`` per
        chain, where ``N = product of remaining dimensions``.
    z_threshold : float
        Chains whose WAIC deficit gives ``z < -z_threshold`` vs the leader
        are excluded.  (z is always non-positive since the leader has the
        highest WAIC.)

        Default is 6.0.  The rationale: within a single posterior mode, each
        chain's WAIC can deviate from the mode's true WAIC by up to ~3 MC_SD.
        When comparing two such chains, one may be 3σ above and the other 3σ
        below, giving a pairwise difference of up to 6 noise SDs.  A
        threshold of 6 therefore keeps all chains plausibly from the same
        mode, assuming a moderate number of chains.
    ll_var : str
        Name of the log-likelihood variable in the trace.

    Returns
    -------
    kept_indices : list[int]
        Chain indices to keep (0-based, matching ``trace.posterior.chain``).
    """
    chains = trace.posterior.chain.values  # type: ignore[attr-defined]  # shape (n_chains,)
    n_chains = len(chains)
    S = trace.posterior.sizes["draw"]  # type: ignore[attr-defined]

    # ll_all has shape (n_chains, S, ...) where ... are observation dimensions
    ll_all = trace.log_likelihood[ll_var].values  # type: ignore[attr-defined]
    # Reshape to (S, N) per chain, where N = total number of observations
    ll_flat = {c: ll_all[i].reshape(S, -1) for i, c in enumerate(chains)}

    summaries = {c: waic_summary(ll_flat[c]) for c in chains}

    # LOO is logged alongside WAIC for comparison but not used in the
    # z-score.  Filter the expected ArviZ Pareto k warning (we report
    # high-k counts ourselves); let any unexpected warnings through.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Estimated shape parameter of Pareto distribution",
            category=UserWarning,
        )
        loo = {c: az.loo(trace.sel(chain=[c]), var_name=ll_var) for c in chains}

    leader = max(chains, key=lambda c: summaries[c].elpd_waic)

    leader_waic = summaries[leader].elpd_waic
    leader_loo = loo[leader].elpd_loo

    diverging = trace.sample_stats.diverging.values  # type: ignore[attr-defined]  # (n_chains, S)

    kept_indices: list[int] = []
    logger.info("Per-chain WAIC / LOO diagnostics:")
    logger.info(
        f"  {'Chain':>5} {'div':>4} {'LPPD':>10} {'p_WAIC':>7} {'WAIC':>10} "
        f"{'SE_WAIC':>8} {'k>0.7':>6} {'dW-dL':>6} "
        f"{'MC_SD':>8} {'dWAIC':>7} {'z':>6} {'':<1}"
    )
    for i, c in enumerate(chains):
        ws = summaries[c]
        lo = loo[c]
        n_div = int(diverging[i].sum())
        n_high_k = int((lo.pareto_k.values.ravel() > 0.7).sum())
        d_waic = ws.elpd_waic - leader_waic
        assert d_waic <= 0, f"Chain {c} has higher WAIC than leader {leader}"
        d_loo = lo.elpd_loo - leader_loo
        noise_sd = np.sqrt(ws.mc_sd**2 + summaries[leader].mc_sd ** 2)
        z = d_waic / noise_sd if c != leader else 0.0
        keep = z >= -z_threshold or c == leader
        if keep:
            kept_indices.append(int(c))
        mark = "\u2713" if keep else "\u2717"
        logger.info(
            f"  {c:>5} {n_div:>4} {ws.lppd:>10.1f} {ws.p_waic:>7.1f} {ws.elpd_waic:>10.1f} "
            f"{ws.se:>8.1f} {n_high_k:>6} {d_waic - d_loo:>+6.2f} "
            f"{ws.mc_sd:>8.3f} {d_waic:>+7.1f} {z:>+6.1f} {mark}"
        )

    logger.info(
        f"Keeping chains: {kept_indices} ({len(kept_indices)}/{n_chains}), "
        f"z_threshold={z_threshold}"
    )
    return kept_indices
