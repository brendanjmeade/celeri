import pytest
import celeri
import math
import numpy as np

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

@pytest.mark.array_compare
@pytest.mark.parametrize(
    "config_name",
    ["test_japan_config", "test_wna_config"],
)
def test_operator_tde_to_velocities(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.tde is not None
    operator = estimation.operators.tde.tde_to_velocities[0]
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(operator), size=25, replace=False)
    return operator[indices, :]

@pytest.mark.array_compare
@pytest.mark.parametrize(
    "config_name",
    ["test_japan_config", "test_wna_config"],
)
def test_operator_eigen_to_tde_slip(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.eigen is not None
    operator = estimation.operators.eigen.eigenvectors_to_tde_slip[0]
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(operator), size=25, replace=False)
    return operator[indices, :]
    
@pytest.mark.array_compare
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
def test_japan_dense(config_name, eigen: bool, tde: bool):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=eigen, tde=tde)

    assert hasattr(estimation, "tde_rates")
    assert hasattr(estimation, "east_vel_residual")

    select_indices = [idx for idx in range(len(estimation.state_vector)) if is_prime(idx)]
    return estimation.state_vector[select_indices]

def test_japan_dense_error():
    config_file_name = "./tests/configs/test_japan_config.json"
    model = celeri.build_model(config_file_name)

    with pytest.raises(ValueError):
        celeri.assemble_and_solve_dense(model, eigen=True, tde=False)
    return
