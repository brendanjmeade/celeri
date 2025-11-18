import subprocess
from pathlib import Path

import pytest

from celeri.celeri_util import get_newest_run_folder


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/test_japan_config.json",
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

    csv_files = [
        "model_station.csv",
        "model_segment.csv",
        "model_block.csv",
        "model_mogi.csv",
    ]
    
    for csv_file in csv_files:
        csv_path = run_dir / csv_file
        assert csv_path.exists(), f"CSV file not created: {csv_path}"