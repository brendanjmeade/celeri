from pathlib import Path
import h5py
import pandas as pd
import pytest
import celeri
from celeri.celeri_util import get_newest_run_folder


@pytest.mark.parametrize(
    "config_file",
    [
        "data/config/wna_config.json",
    ],
)
def test_celeri_solve_creates_output_files(config_file):
    """Test that celeri_solve.py creates the HDF5 file and CSV files via write_output()."""
    # Load config and set solve type
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
