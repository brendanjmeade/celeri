import subprocess
from pathlib import Path

import pytest

from celeri.celeri_util import get_newest_run_folder, read_run


@pytest.mark.parametrize(
    ("config_file", "solve_type"),
    [
        ("./tests/configs/test_japan_config.json", "dense_no_meshes"),
        ("./tests/configs/test_japan_config.json", "qp"),
        ("./tests/configs/test_japan_config.json", "qp2"),
        ("./tests/configs/test_japan_config.json", "mcmc"),
        ("./data/config/japan_config.json", "dense_no_meshes"),
        ("./data/config/japan_config.json", "qp2"),
        ("./data/config/japan_config.json", "mcmc"),
        ("./data/config/wna_config.json", "dense_no_meshes"),
        ("./data/config/wna_config.json", "qp2"),
        ("./data/config/wna_config.json", "mcmc"),
    ],
)
def test_celeri_solve(config_file, solve_type):
    subprocess.check_call(
        [
            "python",
            "celeri/scripts/celeri_solve.py",
            config_file,
            "--solve_type",
            solve_type,
            "--mcmc-tune",
            "5",
            "--mcmc-draws",
            "5",
        ],
    )

    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")
    read_run(run_dir)
