import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from loguru import logger

import celeri
from celeri.config import get_config
from celeri.operators import (
    _hash_elastic_operator_input,
    _OperatorBuilder,
    _store_elastic_operators,
)

test_logger = logger.bind(name="test_output_files")


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/configs/wna_outputs.json",
    ],
)
def test_smart_segment_recompute(config_file):
    """Test that selective recompute works correctly when segments change.

    This test verifies the selective recomputation feature of elastic operators:
    1. First computes operators from scratch and caches them
    2. Modifies a segment file and triggers selective recompute (only changed segments)
    3. Renames the cache file and recomputes from scratch
    4. Compares the selective recompute results with full recompute to ensure they match

    Args:
        config_file: Path to the test configuration file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Load config and redirect cache to temp directory
        config = get_config(config_file)
        config.elastic_operator_cache_dir = cache_dir

        test_logger.info("Computing operator with original segment file")
        model = celeri.build_model(config)
        operators = _OperatorBuilder(model)
        _store_elastic_operators(model, operators)

        input_hash = _hash_elastic_operator_input(
            [mesh.config for mesh in model.meshes],
            model.station,
            model.config,
        )

        cache_file = cache_dir / f"{input_hash}.hdf5"
        assert cache_file.exists(), f"Cache file should exist: {cache_file}"

        # Build model with modified segment file
        model = celeri.build_model(
            config,
            override_segment=pd.read_csv(Path("tests/data/segment/wna_segment1.csv")),
        )
        test_logger.info("Selectively recomputing operators with modified segment file")
        operators2 = _OperatorBuilder(model)
        _store_elastic_operators(model, operators2)

        assert cache_file.exists(), (
            "Cache file should still exist after selective recompute"
        )

        recomputed_file = cache_dir / "recomputed.hdf5"
        cache_file.rename(recomputed_file)

        # Compute from scratch (cache file renamed, so it will recompute everything)
        test_logger.info("Fully recomputing operators using modified segment file")
        operators3 = _OperatorBuilder(model)
        _store_elastic_operators(model, operators3)

        new_cache_file = cache_dir / f"{input_hash}.hdf5"
        assert new_cache_file.exists(), "New cache file should be created"

        with (
            h5py.File(new_cache_file, "r") as f_new,
            h5py.File(recomputed_file, "r") as f_old,
        ):
            name = "slip_rate_to_okada_to_velocities"
            assert name in f_new, (
                f"[{name}] not found in hdf5 file computed from scratch"
            )
            assert name in f_old, (
                f"[{name}] not found in hdf5 file recomputed from cache"
            )
            dataset_new = f_new[name]
            dataset_old = f_old[name]
            arr_new = np.array(dataset_new)
            arr_old = np.array(dataset_old)
            if arr_new.shape != arr_old.shape:
                print(f"[{name}] Shapes differ: {arr_new.shape} vs {arr_old.shape}")
            else:
                max_diff = np.max(np.abs(arr_new - arr_old))
                assert np.allclose(arr_new, arr_old, rtol=1e-10, atol=1e-10), (
                    f"[{name}] Arrays should be equal (within tolerance). "
                    f"Max difference: {max_diff}"
                )


def test_tde_cache_tracks_mesh_geometry(tmp_path):
    """A mesh edited at an unchanged path must not be served from the cache."""
    import shutil

    import meshio

    from celeri.mesh import Mesh

    config = get_config("tests/configs/test_japan_config.json")
    config.repl = False
    config.elastic_operator_cache_dir = tmp_path / "cache"
    config.propagate_mesh_defaults()
    mesh_file = tmp_path / "sagami.msh"
    shutil.copy("tests/data/mesh/test_japan_sagami.msh", mesh_file)
    mesh_config = config.mesh_params[2].model_copy(update={"mesh_filename": mesh_file})

    def build():
        model = celeri.build_model(
            config, override_meshes=[Mesh.from_params(mesh_config)]
        )
        operators = _OperatorBuilder(model)
        _store_elastic_operators(model, operators)
        return model, operators.tde_to_velocities[0].copy()

    model, first = build()
    cache_file = config.elastic_operator_cache_dir / (
        _hash_elastic_operator_input(
            [model.meshes[0].config], model.station, model.config
        )
        + ".hdf5"
    )
    assert cache_file.exists()

    # Deepen every node by 1 km, at the same path
    mesh = meshio.read(mesh_file)
    mesh.points[:, 2] -= 1.0
    meshio.write(mesh_file, mesh, file_format="gmsh22", binary=False)
    _, edited = build()
    assert not np.allclose(first, edited)

    # The recomputed operator equals a computation with no cache at all
    config.elastic_operator_cache_dir = tmp_path / "empty_cache"
    _, fresh = build()
    np.testing.assert_array_equal(edited, fresh)

    # A dataset written before geometry validation (no digest) is recomputed
    config.elastic_operator_cache_dir = tmp_path / "cache"
    with h5py.File(cache_file, "r+") as hdf5_file:
        dataset = hdf5_file["tde_to_velocities_0"]
        dataset[...] = 0.0
        del dataset.attrs["mesh_geometry"]
    _, restored = build()
    np.testing.assert_array_equal(restored, fresh)
    with h5py.File(cache_file, "r") as hdf5_file:
        assert "mesh_geometry" in hdf5_file["tde_to_velocities_0"].attrs


def test_kinematic_smoothing_length_scale_does_not_change_the_cache_key():
    """The smoothing acts on the kinematic operator only, so changing it must
    not invalidate the elastic-operator cache.
    """
    config = get_config("./tests/configs/test_japan_config.json")
    model = celeri.build_model(config)
    reference = _hash_elastic_operator_input(
        [mesh.config for mesh in model.meshes], model.station, model.config
    )
    for mesh in model.meshes:
        mesh.config.kinematic_smoothing_length_scale = 3.0
    changed = _hash_elastic_operator_input(
        [mesh.config for mesh in model.meshes], model.station, model.config
    )
    assert changed == reference
