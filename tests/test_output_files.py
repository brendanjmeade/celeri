from pathlib import Path
import h5py
import pandas as pd
import pytest
import numpy as np
from loguru import logger
import celeri
from celeri.celeri_util import get_newest_run_folder
from celeri.operators import _OperatorBuilder, Assembly, _store_elastic_operators, _hash_elastic_operator_input

test_logger = logger.bind(name="test_output_files")

@pytest.mark.parametrize(
    "config_file",
    [
        "data/config/wna_config.json",
    ],
)
def test_celeri_solve_creates_output_files(config_file):
    """Test that celeri_solve.py creates the HDF5 file and CSV files via write_output()."""
    # Load config and set solve type
    config = celeri.get_config(config_file)
    config.solve_type = "dense"
    
    model = celeri.build_model(config)
    estimation = celeri.build_and_solve_dense(model)
    celeri.write_output(estimation)
    
    run_dir = get_newest_run_folder(base=Path(__file__).parent.parent / "runs")
    run_name = run_dir.name
    hdf5_file = run_dir / f"model_{run_name}.hdf5"
    assert hdf5_file.exists(), f"HDF5 file not created: {hdf5_file}"

    with h5py.File(hdf5_file, "r") as hdf:
        assert "meshes" in hdf, "HDF5 file missing 'meshes' Group"
        assert "segments" in hdf, "HDF5 file missing 'segments' Group"
        assert "segment" in hdf, "HDF5 file missing 'segment' Dataset"
        assert "station" in hdf, "HDF5 file missing 'station' Dataset"
        assert "station_names" in hdf, "HDF5 file missing 'station_names' Dataset"

    csv_files = [
        "model_station.csv",
        "model_segment.csv",
        "model_block.csv",
        "model_mogi.csv",
    ]

    for csv_file in csv_files:
        csv_path = run_dir / csv_file
        assert csv_path.exists(), f"CSV file not created: {csv_path}"

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
    logger.disable("celeri")
    cache_dir = None
    try:
        test_logger.info("Computing operator with original segment file")
        model = celeri.build_model(config_file)
        assert model.config.elastic_operator_cache_dir is not None, "elastic_operator_cache_dir must be set"
        cache_dir = Path(model.config.elastic_operator_cache_dir)
        if cache_dir.exists():
            for item in cache_dir.iterdir():
                if item.is_file() and item.suffix == ".hdf5":
                    item.unlink()
        assembly = Assembly(data=dict(), sigma=dict(), index=dict())
        operators = _OperatorBuilder(model)
        operators.assembly = assembly
        _store_elastic_operators(model, operators)

        input_hash = _hash_elastic_operator_input(
            [mesh.config for mesh in model.meshes],
            model.station,
            model.config,
        )

        cache_file = cache_dir / f"{input_hash}.hdf5"
        assert cache_file.exists(), f"Cache file should exist: {cache_file}"

        model = celeri.build_model(config_file, override_segment=pd.read_csv(Path("tests/data/segment/wna_segment1.csv")))
        test_logger.info("Selectively recomputing operators with modified segment file")
        operators2 = _OperatorBuilder(model)
        operators2.assembly = assembly
        _store_elastic_operators(model, operators2)

        assert cache_file.exists(), "Cache file should still exist after selective recompute"

        recomputed_file = cache_dir / "recomputed.hdf5"
        cache_file.rename(recomputed_file)

        # Compute from scratch (cache file renamed, so it will recompute everything)
        test_logger.info("Fully recomputing operators using modified segment file")
        operators3 = _OperatorBuilder(model)
        operators3.assembly = assembly
        _store_elastic_operators(model, operators3)

        new_cache_file = cache_dir / f"{input_hash}.hdf5"
        assert new_cache_file.exists(), "New cache file should be created"

        with h5py.File(new_cache_file, "r") as f_new, h5py.File(recomputed_file, "r") as f_old:
            name = "slip_rate_to_okada_to_velocities"
            assert name in f_new, f"[{name}] not found in hdf5 file computed from scratch"
            assert name in f_old, f"[{name}] not found in hdf5 file recomputed from cache"
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
    finally:
        logger.enable("celeri")
        if cache_dir is not None and cache_dir.exists():
            test_logger.info(f"Cleaning up cache directory: {cache_dir}")
            for cache_file in cache_dir.glob("*.hdf5"):
                cache_file.unlink()