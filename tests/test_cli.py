import subprocess


def test_celeri_solve():
    subprocess.check_call(
        ["python", "celeri/scripts/celeri_solve.py", "./tests/test_japan_config.json"],
    )
