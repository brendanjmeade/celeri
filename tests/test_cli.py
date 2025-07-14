import subprocess
from pathlib import Path

import pytest

from celeri.celeri_util import get_newest_run_folder, read_run


@pytest.mark.parametrize(
    "solve_type",
    [
        "dense",
        "dense_no_meshes",
        "qp",
        "qp2",
    ],
)
def test_celeri_solve(solve_type):
    subprocess.check_call(
        [
            "python",
            "celeri/scripts/celeri_solve.py",
            "./tests/test_japan_config.json",
            "--solve_type",
            solve_type,
        ],
    )

    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")
    read_run(run_dir)
