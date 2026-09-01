from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from loguru import logger
from numpy.testing import assert_allclose

import celeri
from celeri.celeri_util import get_newest_run_folder
from celeri.mesh import ScalarBound

test_logger = logger.bind(name="test_output_files")


def _assert_mcmc_outputs_consistent(estimation, run_dir):
    """model_meshes.csv, the HDF5 file and model_segment.csv of an MCMC run all
    report the posterior mean (and std) of the sampled fields, and agree with
    each other.
    """
    posterior = estimation.mcmc_trace.posterior
    posterior_mean = posterior.mean(["chain", "draw"])
    meshes = pd.read_csv(run_dir / "model_meshes.csv")
    segment = pd.read_csv(run_dir / "model_segment.csv")

    with h5py.File(run_dir / f"model_{run_dir.name}.hdf5", "r") as hdf:
        for i in range(len(estimation.model.meshes)):
            rows = meshes[meshes["mesh_idx"] == i]
            grp = hdf[f"meshes/mesh_{i:05}"]
            # What the sampler used: unsmoothed, linear in the rotation, so
            # the rate at the posterior mean rotation is the posterior mean.
            kinematic = estimation.operators.kinematic_slip_rate(
                estimation.state_vector, i, smooth=False
            )
            for kind, key, component in (
                ("ss", "strike_slip", 0),
                ("ds", "dip_slip", 1),
            ):
                rate = rows[f"{key}_rate"].to_numpy()
                assert_allclose(
                    rate, posterior_mean[f"elastic_{i}_{kind}"].values, rtol=1e-6
                )
                kinematic_rate = rows[f"{key}_rate_kinematic"].to_numpy()
                assert_allclose(kinematic_rate, kinematic[component::2], rtol=1e-6)
                kinematic_var = f"kinematic_{i}_{kind}"
                if kinematic_var in posterior_mean:
                    # float32 boundary: same operand-scale round-off as the
                    # segment rates below
                    assert_allclose(
                        kinematic_rate,
                        posterior_mean[kinematic_var].values,
                        atol=5e-3,
                    )
                coupling_var = f"coupling_{i}_{kind}"
                if coupling_var in posterior_mean:
                    expected_coupling = posterior_mean[coupling_var].values
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        expected_coupling = rate / kinematic[component::2]
                assert_allclose(
                    rows[f"{key}_coupling"],
                    expected_coupling,
                    rtol=1e-6,
                    equal_nan=True,
                )
                # The HDF5 file carries the same arrays as the CSV
                for suffix, column in (
                    ("", f"{key}_rate"),
                    ("_kinematic", f"{key}_rate_kinematic"),
                    ("_coupling", f"{key}_coupling"),
                ):
                    assert_allclose(
                        grp[f"{key}{suffix}/{0:012}"][...],
                        rows[column],
                        rtol=1e-12,
                        equal_nan=True,
                    )

    # Segment rates are the posterior mean and their uncertainties the
    # posterior std (the CSV is written with 4 decimals). The sampler
    # evaluates segment_slip_rate in float32 while the CSV reports the
    # float64 rate at the posterior mean rotation. The float32 round-off
    # scales with the operand terms -- rotation cross products of up to
    # ~4000 mm/yr that cancel down to the rates -- not with the result,
    # so only an absolute tolerance sized to that ceiling works:
    # (nnz + 2) * eps_f32 * (|A| @ |x|) reaches 3.6e-3 mm/yr for the WNA
    # operator at solution-scale rotations.
    segment_mean = posterior_mean["segment_slip_rate"].values
    segment_std = posterior["segment_slip_rate"].std(["chain", "draw"]).values
    for component, name in enumerate(("strike", "dip", "tensile")):
        assert_allclose(
            segment[f"model_{name}_slip_rate"],
            segment_mean[:, component],
            atol=5e-3,
        )
        uncertainty = segment[f"model_{name}_slip_rate_uncertainty"]
        assert np.isfinite(uncertainty).all()
        assert_allclose(uncertainty, segment_std[:, component], atol=1e-4)

    # Station predictions from the state vector match the sampler's forward
    # model. float32 boundary: the sampler's mu sums rotation cross products
    # of ~1000s of mm/yr in float32, so the round-off ceiling is ~1e-3
    # regardless of the velocity magnitude (a seeded legacy run measured a
    # 1.2e-3 miss at atol=1e-3).
    mu = posterior_mean["mu"].values
    assert_allclose(estimation.station["model_east_vel"], mu[:, 0], atol=5e-3)
    assert_allclose(estimation.station["model_north_vel"], mu[:, 1], atol=5e-3)


@pytest.mark.parametrize(
    "config_file",
    [
        "data/config/wna_config.json",
    ],
)
def test_celeri_solve_creates_output_files(config_file):
    """Test that celeri_solve.py creates the HDF5 file and CSV files via write_output()."""
    config = celeri.get_config(config_file)
    config.solve_type = "dense"

    model = celeri.build_model(config)
    estimation = celeri.build_and_solve_dense(model)
    celeri.write_output(estimation)

    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")
    run_name = run_dir.name
    hdf5_file = run_dir / f"model_{run_name}.hdf5"
    assert hdf5_file.exists(), f"HDF5 file not created: {hdf5_file}"

    with h5py.File(hdf5_file, "r") as hdf:
        assert "meshes" in hdf, "HDF5 file missing 'meshes' Group"
        assert "segments" in hdf, "HDF5 file missing 'segments' Group"
        assert "segment" in hdf, "HDF5 file missing 'segment' Dataset"
        assert "station" in hdf, "HDF5 file missing 'station' Dataset"
        assert "station_names" in hdf, "HDF5 file missing 'station_names' Dataset"

    csv_files = [
        "model_station.csv",
        "model_segment.csv",
        "model_block.csv",
        "model_mogi.csv",
    ]

    for csv_file in csv_files:
        csv_path = run_dir / csv_file
        assert csv_path.exists(), f"CSV file not created: {csv_path}"


@pytest.mark.parametrize(
    "config_file",
    [
        "data/config/wna_config.json",
    ],
)
def test_celeri_solve_mcmc_creates_output_files(config_file):
    """Test that celeri_solve_mcmc.py creates the HDF5 file and CSV files required by result_manager."""
    config = celeri.get_config(config_file)
    config.solve_type = "mcmc"
    model = celeri.build_model(config)
    for mesh in model.meshes:
        if mesh.config.elastic_constraints_ss is not None:
            mesh.config.elastic_constraints_ss = ScalarBound(lower=None, upper=None)
        if mesh.config.elastic_constraints_ds is not None:
            mesh.config.elastic_constraints_ds = ScalarBound(lower=None, upper=None)
    # Seeded: with 2 tune / 2 draws the sampler is far from converged, and an
    # unseeded wild draw can inflate the float32 operand scale (and thus the
    # round-off in the consistency comparison) past any fixed tolerance.
    estimation = celeri.solve_mcmc(
        model, sample_kwargs={"tune": 2, "draws": 2, "seed": 42}
    )
    celeri.write_output(estimation)
    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")
    run_name = run_dir.name
    hdf5_file = run_dir / f"model_{run_name}.hdf5"
    assert hdf5_file.exists(), f"HDF5 file not created: {hdf5_file}"

    with h5py.File(hdf5_file, "r") as hdf:
        assert "meshes" in hdf, "HDF5 file missing 'meshes' Group"
        assert "segments" in hdf, "HDF5 file missing 'segments' Group"
        assert "segment" in hdf, "HDF5 file missing 'segment' Dataset"
        assert "station" in hdf, "HDF5 file missing 'station' Dataset"
        assert "station_names" in hdf, "HDF5 file missing 'station_names' Dataset"

    loaded = celeri.Estimation.from_disk(run_dir)
    assert loaded.mcmc_trace is not None
    # LEGACY-MCMC: group names live in ``.children`` on an xarray.DataTree
    # (ArviZ>=1) and in ``.groups()`` on an arviz.InferenceData (ArviZ<1).
    # On cleanup, assert on ``trace.children`` directly.
    trace = loaded.mcmc_trace
    groups = trace.children if hasattr(trace, "children") else trace.groups()
    assert "posterior" in groups
    assert "log_likelihood" in groups

    # Coupling mode: the coupling field itself is sampled
    assert "coupling_0_ds" in estimation.mcmc_trace.posterior
    _assert_mcmc_outputs_consistent(estimation, run_dir)


@pytest.mark.parametrize(
    "config_file",
    [
        "data/config/wna_config.json",
    ],
)
def test_celeri_solve_mcmc_elastic_mode_output_files(config_file):
    """Elastic mode with bounded rates: the bound transform makes the sampled
    elastic field nonlinear in the eigen coefficients, and the outputs must
    still report its posterior mean and match the sampler's predictions.
    """
    config = celeri.get_config(config_file)
    config.solve_type = "mcmc"
    model = celeri.build_model(config)
    for mesh in model.meshes:
        mesh.config.coupling_constraints_ss = ScalarBound(lower=None, upper=None)
        mesh.config.coupling_constraints_ds = ScalarBound(lower=None, upper=None)
        assert mesh.config.elastic_constraints_ds.upper is not None
    # Seeded: with 2 tune / 2 draws the sampler is far from converged, and an
    # unseeded wild draw can inflate the float32 operand scale (and thus the
    # round-off in the consistency comparison) past any fixed tolerance.
    estimation = celeri.solve_mcmc(
        model, sample_kwargs={"tune": 2, "draws": 2, "seed": 42}
    )
    celeri.write_output(estimation)
    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")

    posterior = estimation.mcmc_trace.posterior
    assert "coupling_0_ds" not in posterior
    assert "elastic_eigen_0_ds" in posterior
    _assert_mcmc_outputs_consistent(estimation, run_dir)
