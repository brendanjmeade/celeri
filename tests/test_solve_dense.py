import pytest
import celeri
import numpy as np


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_tde_to_velocities(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.tde is not None

    operator = estimation.operators.tde.tde_to_velocities[0]
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_eigen_to_velocities(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.eigen is not None

    operator = estimation.operators.eigen.eigen_to_velocities[0]
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_eigen_to_tde_slip(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.eigen is not None

    operator = estimation.operators.eigen.eigenvectors_to_tde_slip[0]
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_eigen_to_tde_bcs(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.eigen is not None

    operator = estimation.operators.eigen.eigen_to_tde_bcs[0]
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_slip_rate_to_okada_to_velocities(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.slip_rate_to_okada_to_velocities is not None

    operator = estimation.operators.slip_rate_to_okada_to_velocities
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_block_strain_rate_to_velocities(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.block_strain_rate_to_velocities is not None

    operator = estimation.operators.block_strain_rate_to_velocities
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_rotation_to_slip_rate(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.rotation_to_slip_rate is not None

    operator = estimation.operators.rotation_to_slip_rate
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_rotation_to_tri_slip_rate(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.rotation_to_tri_slip_rate is not None

    operator = estimation.operators.rotation_to_tri_slip_rate[0]
    rng = np.random.default_rng(seed=0)
    size = min(min(len(operator), len(operator[0])), 50)
    idx_rows = rng.choice(len(operator), size=size, replace=False)
    idx_cols = rng.choice(len(operator[0]), size=size, replace=False)

    return operator[np.ix_(idx_rows, idx_cols)]


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize(
    "config_file, eigen, tde",
    [
        ("test_japan_config", True, True),
        ("test_japan_config", False, True),
        ("test_japan_config", False, False),
        ("test_wna_config", True, True),
        ("test_wna_config", False, True),
        ("test_wna_config", False, False),
    ],
)
def test_dense_sol(config_file, eigen: bool, tde: bool):
    config_file = f"./tests/configs/{config_file}.json"
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


@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_vel_tde_eigen(config_name):
    """Test that TDE velocity components are computed correctly in eigen case.

    Verifies that vel_tde and its east/north components are computed with correct
    shapes and finite values when using eigen decomposition.
    """
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)
    n_stations = len(model.station)

    # Verify vel_tde is computed and has correct shape
    vel_tde = estimation.vel_tde
    assert vel_tde is not None
    assert vel_tde.shape == (2 * n_stations,)
    assert np.all(np.isfinite(vel_tde))

    # Verify component accessors work and have correct shapes
    east_vel_tde = estimation.east_vel_tde
    north_vel_tde = estimation.north_vel_tde
    assert east_vel_tde is not None
    assert north_vel_tde is not None
    assert east_vel_tde.shape == (n_stations,)
    assert north_vel_tde.shape == (n_stations,)
    assert np.all(np.isfinite(east_vel_tde))
    assert np.all(np.isfinite(north_vel_tde))

    # Verify interleaving is correct
    np.testing.assert_array_equal(east_vel_tde, vel_tde[0::2])
    np.testing.assert_array_equal(north_vel_tde, vel_tde[1::2])


@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_eigen_to_velocities_shape(config_name):
    """Test that eigen_to_velocities has 3 velocity components per station.

    Verifies that the operator output shape is (3 * n_stations, n_modes), ensuring
    consistency with the expected 3-component (east, north, up) velocity output.
    """
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)
    n_stations = len(model.station)

    assert estimation.operators.eigen is not None

    for i, operator in estimation.operators.eigen.eigen_to_velocities.items():
        # eigen_to_velocities outputs 3 components (east, north, up) per station
        assert operator.shape[0] == 3 * n_stations, (
            f"Mesh {i}: eigen_to_velocities has {operator.shape[0]} rows, "
            f"expected {3 * n_stations} (3 components × {n_stations} stations)"
        )


@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
@pytest.mark.parametrize("include_vertical", [False, True])
def test_end_row_eigen_consistency(config_name, include_vertical):
    """Test that index.eigen.end_row_eigen equals index.end_station_row.

    Verifies that eigen indexing is consistent with the station row end index,
    which respects the include_vertical_velocity setting.
    """
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    config.include_vertical_velocity = include_vertical
    model = celeri.build_model(config)
    n_stations = len(model.station)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.index.eigen is not None

    # Verify end_station_row is computed correctly based on include_vertical_velocity
    expected_end_row = 3 * n_stations if include_vertical else 2 * n_stations
    assert estimation.index.end_station_row == expected_end_row

    for i in range(len(estimation.index.eigen.end_row_eigen)):
        assert estimation.index.eigen.end_row_eigen[i] == estimation.index.end_station_row, (
            f"Mesh {i}: end_row_eigen[{i}]={estimation.index.eigen.end_row_eigen[i]} "
            f"doesn't match index.end_station_row={estimation.index.end_station_row}"
        )
