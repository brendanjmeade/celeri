import pytest
import celeri
import numpy as np

@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
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

    assert hasattr(estimation, "tde_rates")
    assert hasattr(estimation, "east_vel_residual")

    scale = np.abs(estimation.operators.full_dense_operator).max(0)
    estimation.state_vector = estimation.state_vector * scale
    return estimation.state_vector

def test_japan_dense_error():
    config_file_name = "./tests/configs/test_japan_config.json"
    model = celeri.build_model(config_file_name)

    with pytest.raises(ValueError):
        celeri.assemble_and_solve_dense(model, eigen=True, tde=False)
    return
