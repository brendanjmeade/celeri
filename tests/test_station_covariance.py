"""Tests for the near-field station covariance (celeri.station_covariance)."""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.special import gamma, kv

from celeri import station_covariance as sc

COMPONENT = {
    "nugget": 0.05,
    "terms": [
        {"weight": 0.15, "length_km": 0.5, "smoothness": 0.5},
        {"weight": 0.80, "length_km": 13.0, "smoothness": 0.5},
    ],
}


@pytest.fixture(scope="module")
def points():
    rng = np.random.default_rng(1)
    n = 200
    xyz = np.column_stack([rng.uniform(0, 12, n), rng.uniform(0, 12, n), np.zeros(n)])
    groups = (xyz[:, 0] > 6).astype(int)
    h = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return xyz, groups, h


def _dense(h, taper_radius=None, groups=None):
    rho = sc.correlation(h, COMPONENT, taper_radius)
    if groups is not None:
        rho = rho * (groups[:, None] == groups[None])
    return rho


def _logpdf(factor, r):
    """Gaussian log density of r (n,) under the Vecchia factor."""
    e = factor.whiten(r[:, None])
    return stats.norm.logpdf(e).sum() + factor.log_det()[0]


@pytest.mark.parametrize("smoothness", [0.5, 1.5, 2.5])
def test_matern_closed_forms_match_bessel(smoothness):
    h = np.linspace(0.01, 20, 50)
    s = np.sqrt(2 * smoothness) * h / 3.0
    expected = (
        2 ** (1 - smoothness) / gamma(smoothness) * s**smoothness * kv(smoothness, s)
    )
    assert_allclose(sc.matern(h, 3.0, smoothness), expected, rtol=1e-10)


def test_wendland_taper_is_compact():
    h = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    taper = sc.wendland(h, 2.0)
    assert taper[0] == 1.0
    assert np.all(taper[2:] >= 0) and np.all(taper[3:] == 0)


@pytest.mark.parametrize("radius", [1.0, 3.0, None])
def test_tapered_correlation_is_positive_definite(points, radius):
    _, groups, h = points
    assert np.linalg.eigvalsh(_dense(h, radius)).min() > 0
    assert np.linalg.eigvalsh(_dense(h, radius, groups)).min() > 0


@pytest.mark.parametrize("radius", [None, 1.0, 3.0])
def test_full_vecchia_equals_dense_density(points, radius):
    """Conditioning on every earlier point is exact."""
    xyz, _, h = points
    neighbors, used = sc.vecchia_neighbors(xyz, len(xyz) - 1, seed=0)
    factor = sc.vecchia_kriging(xyz, neighbors, used, [COMPONENT], radius)
    r = np.random.default_rng(2).standard_normal(len(xyz))
    expected = stats.multivariate_normal(cov=_dense(h, radius)).logpdf(r)
    assert_allclose(_logpdf(factor, r), expected, rtol=1e-8)


def test_same_block_vecchia_equals_block_diagonal_density(points):
    xyz, groups, h = points
    factor = sc.vecchia_factor(xyz, [COMPONENT], len(xyz) - 1, 0, groups=groups)
    assert np.all(groups[factor.neighbors] == groups[:, None])
    r = np.random.default_rng(3).standard_normal(len(xyz))
    expected = stats.multivariate_normal(cov=_dense(h, None, groups)).logpdf(r)
    assert_allclose(_logpdf(factor, r), expected, rtol=1e-8)


def test_neighbors_within_taper_radius_approximate_dense_density(points):
    xyz, _, h = points
    factor = sc.vecchia_factor(xyz, [COMPONENT], 30, 0, taper_radius=2.0)
    distance = np.linalg.norm(xyz[factor.neighbors] - xyz[:, None], axis=-1)
    assert np.all(distance[factor.neighbors != np.arange(len(xyz))[:, None]] < 2.0)
    r = np.random.default_rng(4).standard_normal(len(xyz))
    expected = stats.multivariate_normal(cov=_dense(h, 2.0)).logpdf(r)
    assert abs(_logpdf(factor, r) - expected) < 0.01 * abs(expected)


def test_radius_below_spacing_gives_independent_stations(points):
    xyz, _, h = points
    factor = sc.vecchia_factor(
        xyz, [COMPONENT], 30, 0, taper_radius=0.9 * h[h > 0].min()
    )
    assert factor.neighbors.shape[1] == 1
    assert np.all(factor.coef == 0) and np.all(factor.inv_sd == 1)


def test_whiten_handles_extra_trailing_axis(points):
    xyz, _, _ = points
    factor = sc.vecchia_factor(xyz, [COMPONENT, COMPONENT], 10, 0, taper_radius=3.0)
    a = np.random.default_rng(5).standard_normal((len(xyz), 2, 4))
    stacked = factor.whiten(a)
    for j in range(4):
        assert_allclose(stacked[:, :, j], factor.whiten(a[:, :, j]))


def _covariance_file(tmp_path, **overrides):
    covariance = {
        "station_names": ["grid"],
        "n_neighbors": 10,
        "ordering_seed": 0,
        "components": {"east": COMPONENT, "north": COMPONENT},
    }
    covariance.update(overrides)
    path = tmp_path / "covariance.json"
    path.write_text(json.dumps(covariance))
    return path


def test_load_station_covariance_round_trips(tmp_path):
    covariance = sc.load_station_covariance(_covariance_file(tmp_path))
    assert covariance["components"]["east"] == COMPONENT


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"components": {"east": {**COMPONENT, "nugget": 0.1}}}, "sum to"),
        (
            {
                "components": {
                    "east": {
                        "nugget": 0.0,
                        "terms": [{**COMPONENT["terms"][0], "weight": 1.0}],
                    }
                }
            },
            "nugget must be positive",
        ),
        ({"components": {"sideways": COMPONENT}}, "unknown component"),
        ({"station_names": "grid"}, "station_names"),
        ({"n_neighbors": 0}, "n_neighbors"),
    ],
)
def test_load_station_covariance_rejects_bad_files(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        sc.load_station_covariance(_covariance_file(tmp_path, **overrides))


def test_load_station_covariance_requires_keys(tmp_path):
    path = tmp_path / "covariance.json"
    path.write_text(json.dumps({"station_names": None}))
    with pytest.raises(ValueError, match="missing"):
        sc.load_station_covariance(path)


def _station_table(points):
    xyz, groups, _ = points
    n = len(xyz)
    names = np.where(np.arange(n) % 10 == 0, "GNSS_STATION  ", "grid")
    return pd.DataFrame(
        {
            "name": names,
            "x": xyz[:, 0] * 1e3,
            "y": xyz[:, 1] * 1e3,
            "z": xyz[:, 2] * 1e3,
            "block_label": groups,
        }
    )


def test_station_factor_correlates_only_named_stations(points, tmp_path):
    station = _station_table(points)
    covariance = sc.load_station_covariance(_covariance_file(tmp_path))
    factor, rows = sc.station_factor(station, covariance, 2, 3.0, same_block=True)
    gnss = np.flatnonzero(station.name.str.strip() == "GNSS_STATION")
    assert set(rows) == set(np.flatnonzero(station.name == "grid"))
    assert np.all(factor.coef[gnss] == 0) and np.all(factor.inv_sd[gnss] == 1)
    # Grid stations never condition on GNSS stations or across blocks
    used = factor.neighbors[rows] != rows[:, None]
    assert not np.isin(factor.neighbors[rows][used], gnss).any()
    labels = station.block_label.to_numpy()
    assert np.all((labels[factor.neighbors[rows]] == labels[rows][:, None])[used])


def test_station_factor_null_names_selects_every_station(points, tmp_path):
    station = _station_table(points)
    covariance = sc.load_station_covariance(
        _covariance_file(tmp_path, station_names=None)
    )
    _, rows = sc.station_factor(station, covariance, 2, 3.0, same_block=False)
    assert len(rows) == len(station)


def test_station_factor_leaves_missing_components_independent(points, tmp_path):
    station = _station_table(points)
    covariance = sc.load_station_covariance(_covariance_file(tmp_path))
    factor, _ = sc.station_factor(station, covariance, 3, 3.0, same_block=False)
    assert np.all(factor.coef[:, :, 2] == 0) and np.all(factor.inv_sd[:, 2] == 1)


def test_station_factor_requires_named_stations(points, tmp_path):
    station = _station_table(points)
    covariance = sc.load_station_covariance(
        _covariance_file(tmp_path, station_names=["nobody"])
    )
    with pytest.raises(ValueError, match="No stations named"):
        sc.station_factor(station, covariance, 2, 3.0, same_block=False)


def test_mean_information_counts_independent_stations(points):
    xyz, _, _ = points
    independent = sc.vecchia_factor(xyz, [COMPONENT], 10, 0, taper_radius=1e-6)
    rows = np.arange(len(xyz))
    assert_allclose(sc.mean_information(independent, rows, 1), [len(xyz)])
    correlated = sc.vecchia_factor(xyz, [COMPONENT], 10, 0, taper_radius=3.0)
    assert sc.mean_information(correlated, rows, 1)[0] < len(xyz)


def test_whiten_tensor_matches_numpy(points):
    import pytensor
    import pytensor.tensor as pt

    xyz, _, _ = points
    factor = sc.vecchia_factor(xyz, [COMPONENT, COMPONENT], 10, 0, taper_radius=3.0)
    r = np.random.default_rng(6).standard_normal((len(xyz), 2))
    x = pt.matrix("x", dtype="float64")
    whiten = pytensor.function([x], sc.whiten_tensor(x, factor))
    assert_allclose(whiten(r), factor.whiten(r), rtol=1e-12)


def _synthetic_run(run_dir, n=500, length_km=2.0, nugget=0.1, seed=7):
    """A run directory whose residuals have a known exponential correlation."""
    rng = np.random.default_rng(seed)
    xyz = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n), np.zeros(n)])
    h = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    cov = (1 - nugget) * np.exp(-h / length_km) + nugget * np.eye(n)
    chol = np.linalg.cholesky(cov)
    residual = chol @ rng.standard_normal((n, 2))
    run_dir.mkdir()
    pd.DataFrame(
        {
            "name": "grid",
            "x": xyz[:, 0] * 1e3,
            "y": xyz[:, 1] * 1e3,
            "z": xyz[:, 2] * 1e3,
            "block_label": 0,
            "model_east_vel_residual": residual[:, 0],
            "model_north_vel_residual": residual[:, 1],
        }
    ).to_csv(run_dir / "model_station.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps({"station_file_name": "s.csv"}))
    return run_dir


def test_fit_script_recovers_correlation_and_round_trips(tmp_path):
    from celeri.scripts.celeri_fit_station_covariance import main

    run_dir = _synthetic_run(tmp_path / "run")
    output = tmp_path / "covariance.json"
    main(
        [
            str(run_dir),
            str(output),
            "--station-names",
            "grid",
            "--taper-radius",
            "10",
            "--n-terms",
            "1",
            "--anchors",
            "500",
            "--skip-convergence",
        ]
    )
    covariance = sc.load_station_covariance(output)
    assert covariance["station_names"] == ["grid"]
    assert covariance["fit"]["taper_radius_km"] == 10
    assert covariance["fit"]["same_block"] is True
    assert output.with_suffix(".png").exists()
    for name in ("east", "north"):
        (term,) = covariance["components"][name]["terms"]
        assert 0.5 < term["length_km"] < 8
        assert 0.6 < term["weight"] < 1.0


def test_fit_script_bounds_lengths_far_beyond_the_taper(tmp_path, capsys):
    from celeri.scripts.celeri_fit_station_covariance import LENGTH_BOUNDS_KM, main

    # Within a 3 km taper a 200 km correlation is flat, so the likelihood
    # cannot pin its length: the fit must stop at the bound, not drift on
    run_dir = _synthetic_run(tmp_path / "run", n=300, length_km=200.0, nugget=0.05)
    output = tmp_path / "covariance.json"
    main(
        [
            str(run_dir),
            str(output),
            "--taper-radius",
            "3",
            "--n-terms",
            "1",
            "--anchors",
            "300",
            "--skip-convergence",
            "--no-plot",
        ]
    )
    covariance = sc.load_station_covariance(output)
    for name in ("east", "north"):
        for term in covariance["components"][name]["terms"]:
            low, high = LENGTH_BOUNDS_KM
            assert low <= term["length_km"] <= high
    assert "upper bound (1000 km)" in capsys.readouterr().out


def test_fit_script_refuses_to_overwrite(tmp_path):
    from celeri.scripts.celeri_fit_station_covariance import main

    run_dir = _synthetic_run(tmp_path / "run", n=50)
    output = tmp_path / "covariance.json"
    output.write_text("{}")
    with pytest.raises(SystemExit):
        main([str(run_dir), str(output), "--no-plot"])
    assert output.read_text() == "{}"
