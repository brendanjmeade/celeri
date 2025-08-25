import subprocess
from pathlib import Path

import pytest

from celeri.celeri_util import get_newest_run_folder, read_run


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/test_japan_config.json",
        "./data/config/japan_config.json",
        "./data/config/wna_config.json",
    ],
)
@pytest.mark.parametrize(
    "solve_type",
    [
        "dense",
        "dense_no_meshes",
        "qp",
        "qp2",
        "mcmc",
    ],
)
def test_celeri_solve(config_file, solve_type):
    if solve_type == "qp":
        if "tests" not in config_file:
            # Those are very slow
            pytest.skip()
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
