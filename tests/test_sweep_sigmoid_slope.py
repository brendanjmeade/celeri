"""Tests for celeri-sweep-sigmoid-slope (celeri.scripts.celeri_sweep_sigmoid_slope)."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from celeri.celeri_util import get_newest_run_folder
from celeri.scripts.celeri_sweep_sigmoid_slope import (
    main,
    parse_sweep_args,
    sweep_values,
)

CONFIG = Path("./data/config/wna_config.json")


def test_sweep_values_linear_and_log():
    assert_allclose(sweep_values(2.0, 8.0, 3, log=False), [2.0, 5.0, 8.0])
    assert_allclose(sweep_values(1.0, 16.0, 3, log=True), [1.0, 4.0, 16.0])
    assert_allclose(sweep_values(3.0, 9.0, 1, log=False), [3.0])
    assert_allclose(sweep_values(3.0, 9.0, 1, log=True), [3.0])


def test_sweep_values_rejects_bad_grid():
    with pytest.raises(ValueError):
        sweep_values(2.0, 8.0, 0, log=False)
    with pytest.raises(ValueError):
        sweep_values(0.0, 8.0, 3, log=True)
    with pytest.raises(ValueError):
        sweep_values(-1.0, 8.0, 3, log=False)


def test_parse_sweep_args_defaults_and_passthrough():
    args = parse_sweep_args([str(CONFIG), "2", "8"])
    assert (args.lower, args.upper, args.n_steps, args.log) == (2.0, 8.0, 3, False)
    assert args.mcmc_tune is None

    args = parse_sweep_args(
        [str(CONFIG), "1", "16", "4", "--log", "--mcmc-tune", "7", "--mcmc-draws", "3"]
    )
    assert (args.lower, args.upper, args.n_steps, args.log) == (1.0, 16.0, 4, True)
    assert (args.mcmc_tune, args.mcmc_draws) == (7, 3)


def test_sweep_writes_one_numbered_run_per_value():
    """Three values -> three consecutive run folders, each recording its slope."""
    values = [2.0, 5.0, 8.0]
    rc = main(
        [
            str(CONFIG),
            "2",
            "8",
            "3",
            "--mcmc-tune",
            "2",
            "--mcmc-draws",
            "2",
            "--mcmc-seed",
            "42",
            "--repl",
            "0",
            "--plot_estimation_summary",
            "0",
        ]
    )
    assert rc == 0

    base = Path("./runs")
    run_dirs = [get_newest_run_folder(base=base, rewind=r) for r in (2, 1, 0)]
    run_numbers = [int(d.name) for d in run_dirs]
    assert run_numbers == list(range(run_numbers[0], run_numbers[0] + 3))

    for run_dir, value in zip(run_dirs, values, strict=True):
        saved = json.loads((run_dir / "config.json").read_text())
        assert saved["solve_type"] == "mcmc"
        assert saved["mcmc_default_mesh_sigmoid_slope"] == value
        assert all(mp["sigmoid_slope"] == value for mp in saved["mesh_params"])
        assert f"sigmoid_slope={value:g}" in saved["description"]
        assert (run_dir / f"model_{run_dir.name}.hdf5").exists()
        assert (run_dir / "mcmc_trace.zarr").exists()

    manifest = base / f"sweep_sigmoid_slope_{run_dirs[0].name}-{run_dirs[-1].name}.json"
    entries = json.loads(manifest.read_text())
    assert [e["sigmoid_slope"] for e in entries] == values
    assert [e["run_name"] for e in entries] == [d.name for d in run_dirs]
    assert all(e["status"] == "ok" for e in entries)
    assert all(np.isfinite(e["wall_time_s"]) for e in entries)


def test_sweep_continues_after_a_failed_value(monkeypatch):
    """A failure for one value is recorded and the remaining values still run."""
    import celeri

    real_solve_mcmc = celeri.solve_mcmc
    calls = {"n": 0}

    def flaky_solve_mcmc(model, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic failure")
        return real_solve_mcmc(model, **kwargs)

    monkeypatch.setattr(celeri, "solve_mcmc", flaky_solve_mcmc)

    rc = main(
        [
            str(CONFIG),
            "3",
            "6",
            "2",
            "--mcmc-tune",
            "2",
            "--mcmc-draws",
            "2",
            "--mcmc-seed",
            "42",
            "--repl",
            "0",
            "--plot_estimation_summary",
            "0",
        ]
    )
    assert rc == 1

    base = Path("./runs")
    last = get_newest_run_folder(base=base)
    manifests = sorted(
        base.glob("sweep_sigmoid_slope_*.json"), key=lambda p: p.stat().st_mtime
    )
    entries = json.loads(manifests[-1].read_text())
    assert [e["status"] for e in entries] == ["failed", "ok"]
    assert entries[1]["run_name"] == last.name
    # The failed value still allocated its own folder, recorded in the manifest
    assert entries[0]["run_name"] == get_newest_run_folder(base=base, rewind=1).name
