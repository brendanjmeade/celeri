from pathlib import Path
import h5py
import pytest
from loguru import logger
import celeri
from celeri.celeri_util import get_newest_run_folder
from celeri.mesh import ScalarBound

test_logger = logger.bind(name="test_output_files")

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
    estimation = celeri.solve_mcmc(model, sample_kwargs={"tune": 2, "draws": 2})
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