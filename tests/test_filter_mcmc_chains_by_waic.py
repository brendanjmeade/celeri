import arviz as az
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from celeri.filter_mcmc_chains_by_waic import select_chains, waic_summary


class TestWaicSummary:
    """Verify waic_summary matches arviz.waic on synthetic log-likelihood data."""

    @pytest.mark.parametrize(
        "S, N, loc, scale, seed",
        [
            (200, 50, -2.0, 0.5, 0),
            (500, 300, -5.0, 1.0, 1),
        ],
        ids=["small", "medium"],
    )
    def test_matches_arviz(self, S: int, N: int, loc: float, scale: float, seed: int):
        rng = np.random.default_rng(seed)
        ll = rng.normal(loc, scale, size=(S, N))

        ws = waic_summary(ll)

        idata = az.from_dict(log_likelihood={"y": ll[np.newaxis, :, :]})
        aw = az.waic(idata, var_name="y")

        assert_allclose(ws.elpd_waic, aw.elpd_waic, rtol=1e-10)
        assert_allclose(ws.p_waic, aw.p_waic, rtol=1e-10)
        assert_allclose(ws.se, aw.se, rtol=1e-10)

    def test_elpd_equals_lppd_minus_p(self):
        rng = np.random.default_rng(99)
        ll = rng.normal(-4.0, 1.0, size=(300, 80))
        ws = waic_summary(ll)
        assert_allclose(ws.elpd_waic, ws.lppd - ws.p_waic, rtol=1e-12)

    def test_mc_sd_positive(self):
        rng = np.random.default_rng(77)
        ll = rng.normal(-3.0, 0.8, size=(400, 60))
        ws = waic_summary(ll)
        assert ws.mc_sd > 0


def _make_trace(
    ll_per_chain: dict[int, np.ndarray],
    *,
    ll_var: str = "y",
) -> az.InferenceData:
    """Build a minimal InferenceData with posterior, log_likelihood, and sample_stats."""
    chains = sorted(ll_per_chain)
    S = ll_per_chain[chains[0]].shape[0]
    N = ll_per_chain[chains[0]].shape[1]
    n_chains = len(chains)

    coords = {"chain": chains, "draw": np.arange(S), "obs": np.arange(N)}

    ll_data = np.stack([ll_per_chain[c] for c in chains], axis=0)
    log_likelihood = xr.Dataset(
        {ll_var: (["chain", "draw", "obs"], ll_data)},
        coords=coords,
    )

    # Dummy posterior (select_chains only reads chain/draw dims from it)
    posterior = xr.Dataset(
        {"dummy": (["chain", "draw"], np.zeros((n_chains, S)))},
        coords={"chain": chains, "draw": np.arange(S)},
    )

    sample_stats = xr.Dataset(
        {"diverging": (["chain", "draw"], np.zeros((n_chains, S), dtype=bool))},
        coords={"chain": chains, "draw": np.arange(S)},
    )

    return az.InferenceData(
        posterior=posterior,
        log_likelihood=log_likelihood,
        sample_stats=sample_stats,
    )


class TestSelectChains:
    """Verify select_chains keeps/excludes the right chains on synthetic data."""

    def test_keeps_all_same_mode_chains(self):
        """When all chains sample from the same distribution, all are kept."""
        rng = np.random.default_rng(42)
        S, N = 500, 100
        ll = {c: rng.normal(-3.0, 0.5, size=(S, N)) for c in range(4)}
        trace = _make_trace(ll, ll_var="y")
        kept = select_chains(trace, ll_var="y")
        assert kept == [0, 1, 2, 3]

    def test_excludes_subdominant_chain(self):
        """A chain with much worse log-likelihoods is excluded."""
        rng = np.random.default_rng(42)
        S, N = 500, 100
        ll = {c: rng.normal(-3.0, 0.5, size=(S, N)) for c in range(4)}
        # Shift chain 2 far below the others
        ll[2] = rng.normal(-10.0, 0.5, size=(S, N))
        trace = _make_trace(ll, ll_var="y")
        kept = select_chains(trace, ll_var="y")
        assert 2 not in kept
        assert all(c in kept for c in [0, 1, 3])

    def test_returns_at_least_leader(self):
        """Even if all other chains are subdominant, the leader is always kept."""
        rng = np.random.default_rng(42)
        S, N = 500, 100
        ll = {
            0: rng.normal(-3.0, 0.5, size=(S, N)),
            1: rng.normal(-50.0, 0.5, size=(S, N)),
            2: rng.normal(-50.0, 0.5, size=(S, N)),
        }
        trace = _make_trace(ll, ll_var="y")
        kept = select_chains(trace, ll_var="y")
        assert kept == [0]

    def test_z_threshold_respected(self):
        """A very loose threshold keeps a borderline chain."""
        rng = np.random.default_rng(42)
        S, N = 500, 100
        ll = {c: rng.normal(-3.0, 0.5, size=(S, N)) for c in range(3)}
        # Shift chain 0 moderately below
        ll[0] = rng.normal(-5.0, 0.5, size=(S, N))
        trace = _make_trace(ll, ll_var="y")
        # Very loose threshold should keep everyone
        kept_loose = select_chains(trace, ll_var="y", z_threshold=1e6)
        assert kept_loose == [0, 1, 2]
        # Very tight threshold should exclude chain 0
        kept_tight = select_chains(trace, ll_var="y", z_threshold=0.01)
        assert 0 not in kept_tight
