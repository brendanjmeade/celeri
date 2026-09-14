import numpy as np
import pytest

import celeri


@pytest.mark.array_compare(rtol=1e-4, atol=1e-9)
@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_operator_tde_to_velocities(config_name):
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.operators.tde is not None
    assert estimation.operators.tde.tde_to_velocities is not None

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

    # The dense okada operator is no longer kept on the Operators object; the
    # solve consumes only the composed rotation product. Rebuild the dense
    # matrix with the (unchanged) assembler to compare against the same
    # baseline, and tie the streamed composed operator back to it.
    assert estimation.operators.slip_rate_to_okada_to_velocities is None
    operator = celeri.get_segment_station_operator_okada(
        model.segment, model.station, model.config, progress_bar=False
    )
    composed_reference = operator @ estimation.operators.rotation_to_slip_rate
    composed = estimation.operators.rotation_to_slip_rate_to_okada_to_velocities
    np.testing.assert_allclose(
        composed,
        composed_reference,
        rtol=1e-10,
        atol=1e-12 * np.abs(composed_reference).max(),
    )

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


@pytest.mark.array_compare(rtol=1e-3, atol=1e-9)
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

    # exclude vertical station rows when scaling
    G = estimation.operators.full_dense_operator
    n_sta = estimation.index.n_stations
    mask = np.ones(G.shape[0], dtype=bool)
    mask[np.arange(2, 3 * n_sta, 3)] = False
    scale = np.abs(G[mask, :]).max(0)
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

    # Verify interleaving is correct (east, north, up pattern)
    np.testing.assert_array_equal(east_vel_tde, vel_tde[0::3])
    np.testing.assert_array_equal(north_vel_tde, vel_tde[1::3])


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

    Verifies that eigen indexing is consistent with the station row end index.
    The operator is always built with 3 components per station; include_vertical_velocity
    only affects the weighting (zero weight for vertical when False).
    """
    config_file = f"./tests/configs/{config_name}.json"
    config = celeri.get_config(config_file)
    config.include_vertical_velocity = include_vertical
    model = celeri.build_model(config)
    n_stations = len(model.station)

    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    assert estimation.index.eigen is not None

    # Verify end_station_row is always 3 * n_stations (all 3 velocity components)
    expected_end_row = 3 * n_stations
    assert estimation.index.end_station_row == expected_end_row

    for i in range(len(estimation.index.eigen.end_row_eigen)):
        assert (
            estimation.index.eigen.end_row_eigen[i] == estimation.index.end_station_row
        ), (
            f"Mesh {i}: end_row_eigen[{i}]={estimation.index.eigen.end_row_eigen[i]} "
            f"doesn't match index.end_station_row={estimation.index.end_station_row}"
        )


@pytest.mark.parametrize("config_name", ["test_japan_config", "test_wna_config"])
def test_dense_no_meshes_state_layout(config_name):
    """The no-mesh dense system has strain and Mogi columns and config weights."""
    config = celeri.get_config(f"./tests/configs/{config_name}.json")
    model = celeri.build_model(config)

    estimation = celeri.assemble_and_solve_dense(model, eigen=False, tde=False)
    index = estimation.index
    operators = estimation.operators

    assert operators.tde is None
    assert index.tde is None
    assert estimation.operator.shape[1] == index.n_operator_cols
    assert estimation.state_vector.shape == (index.n_operator_cols,)

    # Column blocks: rotations | block strain rates | Mogi volume change rates
    assert index.end_block_col == 3 * index.n_blocks
    assert index.start_block_strain_col == index.end_block_col
    assert (
        index.end_block_strain_col - index.start_block_strain_col
        == 3 * index.n_strain_blocks
    )
    assert index.start_mogi_col == index.end_block_strain_col
    assert index.end_mogi_col == index.n_operator_cols
    assert estimation.block_strain_rates.shape == (3 * index.n_strain_blocks,)
    assert estimation.mogi_volume_change_rates.shape == (index.n_mogis,)
    np.testing.assert_array_equal(
        estimation.operator[
            index.start_station_row : index.end_station_row,
            index.start_block_strain_col : index.end_block_strain_col,
        ],
        operators.block_strain_rate_to_velocities,
    )
    np.testing.assert_array_equal(
        estimation.operator[
            index.start_station_row : index.end_station_row,
            index.start_mogi_col : index.end_mogi_col,
        ],
        operators.mogi_to_velocities,
    )

    # Every derived station column must be computable from the state vector
    station = estimation.station
    assert len(station) == index.n_stations

    # Block rotation constraints use the configured weight, as in the mesh paths
    weights = estimation.weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ]
    assert weights.shape == (3 * index.n_block_constraints,)
    np.testing.assert_array_equal(weights, config.block_constraint_weight)


def test_build_and_solve_dense_variants_honor_mesh_flags():
    """build_and_solve_dense keeps the TDEs; build_and_solve_dense_no_meshes drops them."""
    config = celeri.get_config("./tests/configs/test_japan_config.json")
    config.plot_estimation_summary = False
    config.repl = False
    model = celeri.build_model(config)

    with_meshes = celeri.build_and_solve_dense(model)
    assert with_meshes.operators.tde is not None
    assert with_meshes.index.tde is not None
    assert with_meshes.mesh_estimate is not None

    without_meshes = celeri.build_and_solve_dense_no_meshes(model)
    assert without_meshes.operators.tde is None
    assert without_meshes.index.tde is None
    assert without_meshes.mesh_estimate is None
    assert without_meshes.state_vector.shape == (without_meshes.index.n_operator_cols,)
    assert without_meshes.state_vector.size < with_meshes.state_vector.size
    # Every derived output table must be computable for the no-mesh estimation
    assert len(without_meshes.station) == without_meshes.index.n_stations
    assert len(without_meshes.segment) == without_meshes.index.n_segments
    assert len(without_meshes.mogi) == without_meshes.index.n_mogis


def test_mogi_volume_change_sigma():
    """The Mogi sigma column is propagated from the state covariance, not the rates."""
    from dataclasses import replace

    config = celeri.get_config("./tests/configs/test_japan_config.json")
    model = celeri.build_model(config)
    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)
    index = estimation.index
    assert index.n_mogis > 0

    expected = np.sqrt(
        np.diag(estimation.state_covariance_matrix)[
            index.start_mogi_col : index.end_mogi_col
        ]
    )
    sigma = estimation.mogi.volume_change_sig.to_numpy()
    np.testing.assert_allclose(sigma, expected)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)
    assert not np.allclose(sigma, estimation.mogi.volume_change.to_numpy())

    no_covariance = replace(estimation, state_covariance_matrix=None)
    assert np.isnan(no_covariance.mogi.volume_change_sig.to_numpy()).all()


def test_euler_pole_errors_from_covariance():
    """Euler pole uncertainties propagate the rotation-vector covariance."""
    from dataclasses import replace

    from celeri.operators import rotation_vector_err_to_euler_pole_err

    config = celeri.get_config("./tests/configs/test_wna_config.json")
    model = celeri.build_model(config)
    estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)
    n_rotation = 3 * estimation.index.n_blocks

    covariance = estimation.state_covariance_matrix[0:n_rotation, 0:n_rotation]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.array(
            rotation_vector_err_to_euler_pole_err(
                estimation.rotation_vector_x,
                estimation.rotation_vector_y,
                estimation.rotation_vector_z,
                covariance,
            )
        )
    np.testing.assert_allclose(estimation.euler_err, expected, equal_nan=True)

    block = estimation.block
    rotating = block.euler_rate.to_numpy() > 1e-6
    assert rotating.any()
    for column in ("euler_lon_err", "euler_lat_err", "euler_rate_err"):
        values = block[column].to_numpy()[rotating]
        assert np.all(np.isfinite(values)) and np.all(values > 0)

    no_covariance = replace(estimation, state_covariance_matrix=None)
    assert np.isnan(no_covariance.block.euler_rate_err.to_numpy()).all()


def test_eigen_to_tde_bcs_available_before_full_operator():
    """The eigen boundary-condition operator is built with the other operators."""
    config = celeri.get_config("./tests/configs/test_wna_config.json")
    model = celeri.build_model(config)
    operators = celeri.build_operators(model, eigen=True, tde=True)
    assert operators.eigen is not None and operators.tde is not None

    # Available before the full dense operator is ever assembled
    assert set(operators.eigen.eigen_to_tde_bcs) == set(range(len(model.meshes)))
    for i, mesh in enumerate(model.meshes):
        expected = (
            mesh.config.eigenmode_slip_rate_constraint_weight
            * operators.tde.tde_slip_rate_constraints[i]
            @ operators.eigen.eigenvectors_to_tde_slip[i]
        )
        np.testing.assert_array_equal(operators.eigen.eigen_to_tde_bcs[i], expected)

    index = operators.index
    assert index.eigen is not None
    rows = operators.full_dense_operator[
        index.eigen.start_tde_constraint_row_eigen[
            0
        ] : index.eigen.end_tde_constraint_row_eigen[0],
        index.eigen.start_col_eigen[0] : index.eigen.end_col_eigen[0],
    ]
    np.testing.assert_array_equal(rows, operators.eigen.eigen_to_tde_bcs[0])
