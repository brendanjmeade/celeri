from pathlib import Path
import subprocess
import h5py
import pytest
import json
import pandas as pd

from celeri.celeri_util import get_newest_run_folder


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
    ],
)
def test_celeri_solve_creates_output_files(config_file):
    """Test that celeri_solve.py creates the HDF5 file and CSV files via write_output()."""
    subprocess.check_call(
        [
            "python",
            "celeri/scripts/celeri_solve.py",
            config_file,
            "--solve_type",
            "dense",
        ],
    )

    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")
    run_name = run_dir.name
    hdf5_file = run_dir / f"model_{run_name}.hdf5"
    assert hdf5_file.exists(), f"HDF5 file not created: {hdf5_file}"

    with h5py.File(hdf5_file, "r") as hdf:
        assert "meshes" in hdf, "HDF5 file missing 'meshes' Group"
        assert isinstance(hdf["meshes"], h5py.Group), "'meshes' is not a Group"
        
        assert "segments" in hdf, "HDF5 file missing 'segments' Group"
        assert isinstance(hdf["segments"], h5py.Group), "'segments' is not a Group"

        assert "segment" in hdf, "HDF5 file missing 'segment' Dataset"
        segment_dataset = hdf["segment"]
        assert isinstance(segment_dataset, h5py.Dataset), "'segment' is not a Dataset"

        with open(config_file, "r") as f:
            config = json.load(f)
        segment_file_path = config["segment_file_name"]
        segment_file = Path(config_file).parent / segment_file_path

        segment_df = pd.read_csv(segment_file)
        h5_segment_shape = segment_dataset.shape
        csv_shape = segment_df.shape

        assert h5_segment_shape[0] == csv_shape[0], (
            f"Shape mismatch: HDF5 'segment' first dimension: {h5_segment_shape[0]}, "
            f"CSV segment rows: {csv_shape[0]}"
        )
        
        assert "station" in hdf, "HDF5 file missing 'station' Dataset"
        station_dataset = hdf["station"]
        assert isinstance(station_dataset, h5py.Dataset), "'station' is not a Dataset"

        with open(config_file, "r") as f:
            config = json.load(f)
        station_file_path = config["station_file_name"]
        station_file = Path(config_file).parent / station_file_path

        station_df = pd.read_csv(station_file)
        h5_station_shape = station_dataset.shape
        csv_shape = station_df.shape

        assert h5_station_shape[0] == csv_shape[0], (
            f"Shape mismatch: HDF5 'station' first dimension: {h5_station_shape[0]}, "
            f"CSV station rows: {csv_shape[0]}"
        )
        
        assert "station_names" in hdf, "HDF5 file missing 'station_names' Dataset"
        assert isinstance(hdf["station_names"], h5py.Dataset), "'station_names' is not a Dataset"

    csv_files = [
        "model_station.csv",
        "model_segment.csv",
        "model_block.csv",
        "model_mogi.csv",
    ]
    
    for csv_file in csv_files:
        csv_path = run_dir / csv_file
        assert csv_path.exists(), f"CSV file not created: {csv_path}"