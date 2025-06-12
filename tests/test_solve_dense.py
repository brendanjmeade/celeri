import pytest

import celeri


@pytest.mark.parametrize(
    "config_file, eigen, tde",
    [
        ("./tests/test_japan_config.json", True, True),
        ("./tests/test_japan_config.json", False, True),
        ("./tests/test_japan_config.json", False, False),
        ("./tests/test_western_north_america_config.json", True, True),
        ("./tests/test_western_north_america_config.json", False, True),
        ("./tests/test_western_north_america_config.json", False, False),
    ],
)
def test_japan_dense(config_file, eigen: bool, tde: bool):
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    # Estimate block model parameters (dense)
    operators, estimation = celeri.assemble_and_solve_dense(model, eigen=eigen, tde=tde)

    estimation.tde_rates  # noqa: B018
    estimation.east_vel_residual  # noqa: B018


def test_japan_dense_error():
    config_file_name = "./tests/test_japan_config.json"
    model = celeri.build_model(config_file_name)

    with pytest.raises(ValueError):
        celeri.assemble_and_solve_dense(model, eigen=True, tde=False)
