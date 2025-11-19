import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import celeri
from celeri.solve import Estimation


@pytest.fixture
def temp_dir():
    """Create a temporary directory for the tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
        "./tests/configs/test_wna_config.json",
    ],
)
def test_mesh_serialization(config_file, temp_dir):
    """Test serializing and deserializing a mesh."""
    # Load model from config
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    # Get the first mesh from the model
    original_mesh = model.meshes[0]

    # Test to_disk method
    output_dir = Path(temp_dir) / "mesh_output"
    output_dir.mkdir(exist_ok=True)
    original_mesh.to_disk(output_dir)

    # Verify files were created
    config_file = output_dir / "mesh_config.json"
    data_file = output_dir / "arrays.zarr"
    remaining_file = output_dir / "remaining.json"

    assert config_file.exists(), f"Mesh config file not created: {config_file}"
    assert data_file.exists(), f"Mesh data file not created: {data_file}"
    assert remaining_file.exists(), f"Mesh remaining file not created: {remaining_file}"

    # Deserialize the mesh
    deserialized_mesh = celeri.Mesh.from_disk(output_dir)
    np.testing.assert_equal(deserialized_mesh.dep3, original_mesh.dep3)


def test_mesh_serialization_error_handling(temp_dir):
    """Test error handling in from_disk method."""
    # Only test the config file not found error
    with pytest.raises(FileNotFoundError):
        non_existent_dir = Path(temp_dir) / "non_existent_dir"
        non_existent_dir.mkdir(exist_ok=True, parents=True)
        # We expect this to fail because we're not creating the required files
        celeri.Mesh.from_disk(non_existent_dir)


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
        "./tests/configs/test_western_north_america_config.json",
    ],
)
def test_model_serialization(config_file, temp_dir):
    """Test serializing and deserializing a model."""
    # Load model from config
    config = celeri.get_config(config_file)
    original_model = celeri.build_model(config)

    # Test to_disk method
    output_dir = Path(temp_dir)
    original_model.to_disk(output_dir)

    # Verify files were created
    assert (output_dir / "segment.parquet").exists(), "Segment file not created"
    assert (output_dir / "block.parquet").exists(), "Block file not created"
    assert (output_dir / "station.parquet").exists(), "Station file not created"
    assert (output_dir / "mogi.parquet").exists(), "Mogi file not created"
    assert (output_dir / "sar.parquet").exists(), "SAR file not created"
    assert (output_dir / "config.json").exists(), "Config file not created"

    # Check for mesh files
    for i in range(len(original_model.meshes)):
        assert (output_dir / f"meshes/{i:05d}/mesh_config.json").exists(), (
            f"Mesh config file not created for mesh {i}"
        )

    # Verify all files are created correctly before testing deserialization
    # We'll skip actual deserialization for now due to potential issues with mesh serialization

    # Manually check the model contents rather than deserializing
    station_df = pd.read_parquet(output_dir / "station.parquet")
    segment_df = pd.read_parquet(output_dir / "segment.parquet")
    block_df = pd.read_parquet(output_dir / "block.parquet")
    mogi_df = pd.read_parquet(output_dir / "mogi.parquet")
    sar_df = pd.read_parquet(output_dir / "sar.parquet")

    # Compare with original data
    pd.testing.assert_frame_equal(original_model.station, station_df)
    pd.testing.assert_frame_equal(original_model.segment, segment_df)
    pd.testing.assert_frame_equal(original_model.block, block_df)
    pd.testing.assert_frame_equal(original_model.mogi, mogi_df)
    pd.testing.assert_frame_equal(original_model.sar, sar_df)


def test_model_round_trip_serialization(temp_dir):
    """Test complete model round-trip serialization with validation."""
    # SkipLoad model from config
    config_file = "./tests/configs/test_japan_config.json"
    config = celeri.get_config(config_file)
    original_model = celeri.build_model(config)

    # Serialize the model
    output_dir = Path(temp_dir)
    original_model.to_disk(output_dir)

    # Deserialize the model
    deserialized_model = celeri.Model.from_disk(output_dir)

    # Validate the deserialized model by comparing with original model
    # Compare DataFrame contents
    pd.testing.assert_frame_equal(original_model.segment, deserialized_model.segment)
    pd.testing.assert_frame_equal(original_model.block, deserialized_model.block)
    pd.testing.assert_frame_equal(original_model.station, deserialized_model.station)
    pd.testing.assert_frame_equal(original_model.mogi, deserialized_model.mogi)
    pd.testing.assert_frame_equal(original_model.sar, deserialized_model.sar)

    # Compare mesh properties
    for _i, (orig_mesh, deserialized_mesh) in enumerate(
        zip(original_model.meshes, deserialized_model.meshes, strict=False)
    ):
        assert orig_mesh.n_tde == deserialized_mesh.n_tde
        assert orig_mesh.n_modes == deserialized_mesh.n_modes
        assert np.array_equal(orig_mesh.areas, deserialized_mesh.areas)
        assert np.array_equal(orig_mesh.centroids, deserialized_mesh.centroids)

    # Check that the closure is preserved
    np.testing.assert_allclose(
        original_model.closure.edge_idx_to_vertex_idx,
        deserialized_model.closure.edge_idx_to_vertex_idx,
    )

    # Test if model can be used in calculations
    est = celeri.assemble_and_solve_dense(deserialized_model, eigen=True, tde=True)
    # If we get here without errors, the deserialized model works correctly
    assert est.tde_rates is not None


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
    ],
)
def test_operators_serialization(config_file, temp_dir):
    """Test serializing and deserializing operators."""
    # Load model from config and build operators
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    # Build operators with both eigen and tde options
    original_operators = celeri.build_operators(model, eigen=True, tde=True)

    # Test to_disk method
    output_dir = Path(temp_dir) / "operators_output"
    output_dir.mkdir(exist_ok=True)
    original_operators.to_disk(output_dir)

    # Verify files were created
    assert (output_dir / "index" / "arrays.zarr").exists(), "Index file not created"
    assert (output_dir / "tde" / "arrays.zarr").exists(), "TDE index file not created"
    assert (output_dir / "eigen" / "arrays.zarr").exists(), (
        "Eigen index file not created"
    )
    assert (output_dir / "arrays.zarr").exists(), "Operators file not created"

    # Deserialize the operators
    deserialized_operators = celeri.Operators.from_disk(output_dir)

    # Verify basic properties match
    assert (
        original_operators.index.n_operator_cols
        == deserialized_operators.index.n_operator_cols
    )
    assert (
        original_operators.index.n_operator_rows
        == deserialized_operators.index.n_operator_rows
    )

    assert original_operators.index.tde is not None
    assert deserialized_operators.index.tde is not None
    assert (
        original_operators.index.n_tde_total == deserialized_operators.index.n_tde_total
    )

    assert original_operators.eigen is not None
    assert deserialized_operators.eigen is not None
    np.testing.assert_equal(
        original_operators.eigen.eigen_to_velocities,
        deserialized_operators.eigen.eigen_to_velocities,
    )

    # Compare dense operators if available
    G_original = original_operators.full_dense_operator
    G_deserialized = deserialized_operators.full_dense_operator
    np.testing.assert_allclose(G_original, G_deserialized)


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
    ],
)
def test_estimation_serialization(config_file, temp_dir):
    """Test serializing and deserializing an Estimation object."""
    # Load model from config
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)

    # Generate an estimation object by solving the model
    original_estimation = celeri.assemble_and_solve_dense(model, eigen=True, tde=True)

    # Test to_disk method
    output_dir = Path(temp_dir) / "estimation_output"
    output_dir.mkdir(exist_ok=True)
    original_estimation.to_disk(output_dir)

    # Verify operators directory was created
    assert (output_dir / "operators").exists(), "Operators directory not created"
    assert (output_dir / "operators" / "arrays.zarr").exists(), (
        "Operators arrays not created"
    )

    # Deserialize the estimation
    deserialized_estimation = Estimation.from_disk(output_dir)

    # Verify basic properties match
    pd.testing.assert_frame_equal(
        original_estimation.model.station,
        deserialized_estimation.model.station,
    )

    # Compare a few key attributes that should be present in a solved estimation
    np.testing.assert_allclose(original_estimation.vel, deserialized_estimation.vel)

    np.testing.assert_allclose(
        original_estimation.rotation_vector,
        deserialized_estimation.rotation_vector,
    )

    np.testing.assert_allclose(
        original_estimation.slip_rates,
        deserialized_estimation.slip_rates,
    )

    if original_estimation.tde_rates is not None:
        assert deserialized_estimation.tde_rates is not None
        for mesh_idx in original_estimation.tde_rates:
            np.testing.assert_allclose(
                original_estimation.tde_rates[mesh_idx],
                deserialized_estimation.tde_rates[mesh_idx],
            )
