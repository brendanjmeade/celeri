import pytest

import celeri


@pytest.mark.parametrize(
    "config_name, eigen, tde",
    [
        ("test_japan_config", True, True),
        ("test_japan_config", False, True),
        ("test_japan_config", False, False),
        ("test_wna_config", True, True),
        ("test_wna_config", False, True),
        ("test_wna_config", False, False),
    ],
)
def test_dense_sol(config_name, eigen: bool, tde: bool):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=eigen, tde=tde)

    estimation.tde_rates  # noqa: B018
    estimation.east_vel_residual  # noqa: B018


def test_japan_dense_error():
    config_file_name = "./tests/configs/test_japan_config.json"
    model = celeri.build_model(config_file_name)

    with pytest.raises(ValueError):
        celeri.assemble_and_solve_dense(model, eigen=True, tde=False)
