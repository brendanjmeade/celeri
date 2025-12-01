import pytest
import celeri


@pytest.mark.parametrize(
    "config_file, eigen, tde",
    [
        ("./tests/configs/test_japan_config.json", True, True),
        ("./tests/configs/test_japan_config.json", False, True),
        ("./tests/configs/test_japan_config.json", False, False),
        ("./tests/configs/test_wna_config.json", True, True),
        ("./tests/configs/test_wna_config.json", False, True),
        ("./tests/configs/test_wna_config.json", False, False),
    ],
)
def test_japan_dense(config_file, eigen: bool, tde: bool):
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=eigen, tde=tde)

    assert hasattr(estimation, "tde_rates")
    assert hasattr(estimation, "east_vel_residual")
    return

def test_japan_dense_error():
    config_file_name = "./tests/configs/test_japan_config.json"
    model = celeri.build_model(config_file_name)

    with pytest.raises(ValueError):
        celeri.assemble_and_solve_dense(model, eigen=True, tde=False)
    return

@pytest.mark.array_compare
def test_wna_dense_state_vector():
    config_file = "./data/config/wna_config.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)
    return estimation.state_vector

