"""Tests for the streamed Okada segment operator construction.

The dense (3 * n_stations, 3 * n_segments) okada operator is no longer held in
memory: only the composed okada @ rotation_to_slip_rate product is built,
either by accumulating segment-chunk slabs (no cache configured) or by
streaming row slabs from the HDF5 cache dataset, which is filled slab-by-slab
on a cache miss and updated in place by selective recompute. These tests pin:
  1. Bitwise equality of the slab helper against the dense assembler.
  2. Numerical equivalence of both composed-product paths against the dense
     composition, across chunk-size edge cases.
  3. Cache behavior: bitwise miss/hit equality, legacy caches readable, new
     caches readable the old way, segment validation before any matrix read,
     in-place selective recompute.
  4. Serialization: the always-None dense field round-trips, and legacy run
     folders containing the dense array load without materializing it.
  5. tracemalloc guards that neither composed path approaches the dense
     matrix footprint.

Composed-operator comparisons use a matrix-scaled absolute tolerance:
rotation_to_slip_rate is [m/rad] with entries up to ~6e6, so composed entries
are O(1e6) and bare atol=1e-10 would be unrealistically tight.
"""

import tracemalloc

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr

import celeri
from celeri.operators import (
    LosOperators,
    Operators,
    _compute_okada_composed_and_cache,
    _hash_elastic_operator_input,
    _load_segments_from_hdf5,
    _project_operator_to_los,
    _stream_matmul_rows,
    build_los_operators,
    build_operators,
)
from celeri.spatial import (
    get_okada_displacement_slab,
    get_rotation_to_slip_rate_partials,
    get_segment_station_operator_okada,
)

WNA_CONFIG = "tests/configs/test_wna_config.json"
OKADA_KEY = "slip_rate_to_okada_to_velocities"


def _station_subset(config, n_stations):
    """First n stations that survive processing (process_station drops flag == 0)."""
    station = pd.read_csv(config.station_file_name)
    return station[station.flag != 0].iloc[:n_stations].reset_index(drop=True)


def _wna_model(n_stations, cache_dir=None, **build_kwargs):
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = cache_dir
    station = _station_subset(config, n_stations)
    return celeri.build_model(config, override_station=station, **build_kwargs)


def _scaled_tol(reference):
    return {"rtol": 1e-10, "atol": 1e-12 * np.abs(reference).max()}


def _cache_path(model):
    input_hash = _hash_elastic_operator_input(
        [mesh.config for mesh in model.meshes], model.station, model.config
    )
    return model.config.elastic_operator_cache_dir / f"{input_hash}.hdf5"


def _dense_okada(model):
    return get_segment_station_operator_okada(
        model.segment, model.station, model.config, progress_bar=False
    )


@pytest.fixture(scope="module")
def wna_model():
    """25-station wna model (837 segments, 31 blocks)."""
    return _wna_model(25)


@pytest.fixture(scope="module")
def wna_dense(wna_model):
    """The dense okada operator (the object the refactor keeps out of RAM)."""
    return _dense_okada(wna_model)


@pytest.fixture(scope="module")
def wna_rotation(wna_model):
    return get_rotation_to_slip_rate_partials(wna_model.segment, wna_model.block)


@pytest.fixture(scope="module")
def wna_composed_reference(wna_dense, wna_rotation):
    """The pre-refactor composition: full dense matrix @ rotation partials."""
    return wna_dense @ wna_rotation


def test_okada_slab_bitwise_equals_dense_assembler(wna_model, wna_dense):
    """The slab helper must reproduce the dense assembler columns bitwise:
    this is what keeps the arraydiff baseline and every cache dataset
    bit-identical across the refactor.
    """
    n_segments = len(wna_model.segment)
    for seg_start, seg_stop in [
        (0, 1),
        (17, 18),
        (n_segments - 1, n_segments),
        (3, 11),
    ]:
        slab = get_okada_displacement_slab(
            wna_model.segment,
            wna_model.station,
            wna_model.config,
            seg_start=seg_start,
            seg_stop=seg_stop,
        )
        assert np.array_equal(slab, wna_dense[:, 3 * seg_start : 3 * seg_stop])


@pytest.mark.parametrize("seg_batch", [1, 7, "exact", "oversize"])
def test_accumulation_matches_dense_composition(
    wna_model, wna_rotation, wna_composed_reference, seg_batch
):
    """No-cache path: segment-chunk accumulation must match the dense
    composition for degenerate, non-divisor, exact, and oversize chunk sizes
    (chunk size is driven through the memory budget).
    """
    n_stations = len(wna_model.station)
    n_segments = len(wna_model.segment)
    batch = {"exact": n_segments, "oversize": n_segments + 13}.get(seg_batch, seg_batch)
    result = _compute_okada_composed_and_cache(
        wna_model.segment,
        wna_model.station,
        wna_model.config,
        wna_rotation,
        cache=None,
        target_bytes=batch * 72 * n_stations,
    )
    assert result.shape == (3 * n_stations, wna_rotation.shape[1])
    np.testing.assert_allclose(
        result, wna_composed_reference, **_scaled_tol(wna_composed_reference)
    )


@pytest.mark.parametrize("rows_per_batch", [1, 7, 10**9])
def test_row_slab_stream_matches_dense_composition(
    tmp_path, wna_dense, wna_rotation, wna_composed_reference, rows_per_batch
):
    """Cache-hit path: row-slab streaming from an HDF5 dataset must match the
    dense composition for degenerate, non-divisor, and single-batch slabs.
    """
    with h5py.File(tmp_path / "cache.hdf5", "w") as hdf5_file:
        hdf5_file.create_dataset(OKADA_KEY, data=wna_dense)
    with h5py.File(tmp_path / "cache.hdf5", "r") as hdf5_file:
        result = _stream_matmul_rows(
            hdf5_file[OKADA_KEY],
            wna_rotation,
            rows_per_batch * 8 * wna_dense.shape[1],
        )
    np.testing.assert_allclose(
        result,
        wna_composed_reference,
        rtol=1e-12,
        atol=1e-14 * np.abs(wna_composed_reference).max(),
    )


def test_streamed_build_matches_dense_reference_end_to_end(tmp_path):
    """build_operators: the dense field is None, the composed operator matches
    the dense composition, and cache-miss and cache-hit builds are bitwise
    equal (the miss path re-streams from the completed dataset).
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops_miss = build_operators(model, eigen=True, tde=True)
    assert ops_miss.slip_rate_to_okada_to_velocities is None

    dense = _dense_okada(model)
    reference = dense @ ops_miss.rotation_to_slip_rate
    np.testing.assert_allclose(
        ops_miss.rotation_to_slip_rate_to_okada_to_velocities,
        reference,
        **_scaled_tol(reference),
    )

    ops_hit = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops_hit.rotation_to_slip_rate_to_okada_to_velocities,
        ops_miss.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_build_reads_legacy_okada_cache(tmp_path):
    """A contiguous full-matrix dataset written the pre-refactor way must be
    consumed by the streaming path. The cached matrix is scaled by 2 so the
    test proves the cache (not a recompute) was the source of the result.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops1 = build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    with h5py.File(cache, "r+") as hdf5_file:
        dense = np.array(hdf5_file[OKADA_KEY])
        del hdf5_file[OKADA_KEY]
        # data= writes the legacy contiguous layout
        hdf5_file.create_dataset(OKADA_KEY, data=2.0 * dense)

    ops2 = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        2.0 * ops1.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_new_cache_dataset_readable_the_old_way(tmp_path, wna_dense):
    """The slab-filled chunked dataset must hold exactly the dense assembly
    (no NaN holes from partial fills) and remain loadable via a plain
    np.array read, with segment metadata present.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    build_operators(model, eigen=True, tde=True)
    with h5py.File(_cache_path(model), "r") as hdf5_file:
        arr = np.array(hdf5_file[OKADA_KEY])
        assert arr.shape == wna_dense.shape
        assert not np.isnan(arr).any()
        assert np.array_equal(arr, wna_dense)
        cached_segments = _load_segments_from_hdf5(hdf5_file)
        assert cached_segments is not None
        assert cached_segments["name"].tolist() == model.segment["name"].tolist()


def test_selective_recompute_updates_cache_in_place(tmp_path):
    """A segment edit must update only the changed column triples of the cache
    dataset in place (no full rewrite, no file-size blowup) and produce a
    composed operator equal to a from-scratch build.
    """
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = tmp_path
    station = _station_subset(config, 25)
    model = celeri.build_model(config, override_station=station)
    build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    with h5py.File(cache, "r") as hdf5_file:
        unchanged_snapshot = np.array(hdf5_file[OKADA_KEY][:, 9:12])  # segment 3
    size_before = cache.stat().st_size

    changed = [0, 5, 11]
    segment = model.segment.copy(deep=True)
    for idx in changed:
        segment.loc[idx, "locking_depth"] = (
            float(segment.loc[idx, "locking_depth"]) + 1.0
        )
    model2 = celeri.build_model(
        config, override_station=station, override_segment=segment
    )
    ops_sel = build_operators(model2, eigen=True, tde=True)

    # In-place update: only metadata churn, no dataset rewrite
    assert cache.stat().st_size <= size_before + 2 * 2**20
    with h5py.File(cache, "r") as hdf5_file:
        dataset = hdf5_file[OKADA_KEY]
        assert np.array_equal(dataset[:, 9:12], unchanged_snapshot)
        for idx in changed:
            fresh = get_okada_displacement_slab(
                model2.segment,
                model2.station,
                model2.config,
                seg_start=idx,
                seg_stop=idx + 1,
            )
            assert np.array_equal(dataset[:, 3 * idx : 3 * idx + 3], fresh)

    dense2 = _dense_okada(model2)
    reference = dense2 @ ops_sel.rotation_to_slip_rate
    np.testing.assert_allclose(
        ops_sel.rotation_to_slip_rate_to_okada_to_velocities,
        reference,
        **_scaled_tol(reference),
    )


def test_segment_rename_recomputes_without_reading_matrix(tmp_path):
    """When segment names change, the cache must be invalidated by metadata
    alone: a deliberately poisoned (garbage-shaped) dataset must never be
    read, and the rebuild must succeed.
    """
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = tmp_path
    station = _station_subset(config, 25)
    model = celeri.build_model(config, override_station=station)
    build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    with h5py.File(cache, "r+") as hdf5_file:
        del hdf5_file[OKADA_KEY]
        hdf5_file.create_dataset(OKADA_KEY, data=np.zeros(1))

    segment = model.segment.copy(deep=True)
    segment.loc[0, "name"] = "renamed_segment_for_test"
    model2 = celeri.build_model(
        config, override_station=station, override_segment=segment
    )
    ops = build_operators(model2, eigen=True, tde=True)
    dense = _dense_okada(model2)
    reference = dense @ ops.rotation_to_slip_rate
    np.testing.assert_allclose(
        ops.rotation_to_slip_rate_to_okada_to_velocities,
        reference,
        **_scaled_tol(reference),
    )


def test_mismatched_dataset_triggers_recompute(tmp_path):
    """A cache whose okada dataset has the wrong shape (e.g. a crash artifact)
    must trigger a clean recompute even when segments are unchanged.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops1 = build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    with h5py.File(cache, "r+") as hdf5_file:
        del hdf5_file[OKADA_KEY]
        hdf5_file.create_dataset(OKADA_KEY, data=np.zeros(1))

    ops2 = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops1.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_corrupt_cache_file_recovers(tmp_path):
    """A cache file that is not valid HDF5 at all (e.g. truncated by a killed
    run or a full disk) must be recreated, not crash every subsequent run.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops1 = build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    cache.write_bytes(b"this is not an hdf5 file")

    ops2 = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops1.rotation_to_slip_rate_to_okada_to_velocities,
    )
    # The file was recreated as a valid cache
    with h5py.File(cache, "r") as hdf5_file:
        assert OKADA_KEY in hdf5_file


def test_cache_missing_names_dataset_recomputes(tmp_path):
    """A cache holding segments but not segments_names (crash window inside
    the metadata write) must read as old-format and recompute cleanly rather
    than raising KeyError on the missing name column.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops1 = build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    with h5py.File(cache, "r+") as hdf5_file:
        del hdf5_file["segments_names"]

    ops2 = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops1.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_interrupted_selective_recompute_recovers(tmp_path):
    """Simulate the crash window of an in-place selective recompute: metadata
    invalidated, one column already overwritten with new-geometry values, then
    the run dies and the user reverts the segment edit. The next run must
    recompute (metadata absent) instead of serving the corrupted column.
    """
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = tmp_path
    station = _station_subset(config, 25)
    model = celeri.build_model(config, override_station=station)
    ops1 = build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    # Crash state: metadata deleted (the fixed ordering deletes it first),
    # column 0 poisoned mid-update
    with h5py.File(cache, "r+") as hdf5_file:
        for key in ("segments", "segments_names"):
            del hdf5_file[key]
        hdf5_file[OKADA_KEY][:, 0:3] = 12345.0

    ops2 = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops1.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_cache_without_segment_metadata_recomputes(tmp_path):
    """A cache missing the segments metadata (e.g. a crash before the
    metadata-last write) must read as old-format and recompute cleanly.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops1 = build_operators(model, eigen=True, tde=True)
    cache = _cache_path(model)
    with h5py.File(cache, "r+") as hdf5_file:
        for key in ("segments", "segments_names"):
            if key in hdf5_file:
                del hdf5_file[key]

    ops2 = build_operators(model, eigen=True, tde=True)
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops1.rotation_to_slip_rate_to_okada_to_velocities,
    )


def _synthetic_los(station, n_los=12, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(station), n_los)
    look = rng.normal(size=(n_los, 3))
    look /= np.linalg.norm(look, axis=1, keepdims=True)
    return pd.DataFrame(
        {
            "lon": station.lon.to_numpy()[idx] + rng.uniform(-0.1, 0.1, n_los),
            "lat": station.lat.to_numpy()[idx] + rng.uniform(-0.1, 0.1, n_los),
            "los_val": rng.normal(size=n_los),
            "look_vector_east": look[:, 0],
            "look_vector_north": look[:, 1],
            "look_vector_up": look[:, 2],
        }
    )


@pytest.fixture(scope="module")
def wna_los_model():
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = None
    station = _station_subset(config, 25)
    model = celeri.build_model(config, override_station=station)
    los = _synthetic_los(model.station)
    return celeri.build_model(config, override_station=station, override_los=los)


@pytest.fixture(scope="module")
def wna_los_operators(wna_los_model):
    ops = build_operators(wna_los_model, eigen=True, tde=True)
    los_ops = build_los_operators(wna_los_model, ops)
    assert los_ops is not None
    return ops, los_ops


def test_los_rotation_to_okada_matches_dense_reference(
    wna_los_model, wna_los_operators
):
    """LOS path: the chunk-accumulated rotation_to_okada_los must match the
    dense project-then-compose reference; raw okada_to_los is no longer
    stored.
    """
    ops, los_ops = wna_los_operators
    model = wna_los_model
    assert los_ops.okada_to_los is None

    okada_at_los = get_segment_station_operator_okada(
        model.segment, model.los, model.config, progress_bar=False
    )
    look_vectors = np.column_stack(
        [
            model.los.look_vector_east.to_numpy(),
            model.los.look_vector_north.to_numpy(),
            model.los.look_vector_up.to_numpy(),
        ]
    )
    reference = _project_operator_to_los(
        okada_at_los @ ops.rotation_to_slip_rate, look_vectors
    )
    np.testing.assert_allclose(
        los_ops.rotation_to_okada_los,
        reference,
        rtol=1e-9,
        atol=1e-10 * np.abs(reference).max(),
    )


def test_los_operators_roundtrip_and_legacy_okada_to_los(tmp_path, wna_los_operators):
    """LosOperators round-trips with okada_to_los None, and a legacy los
    folder containing an okada_to_los array loads with the field skipped.
    """
    _, los_ops = wna_los_operators
    out = tmp_path / "los"
    los_ops.to_disk(out)
    store = zarr.open_group(str(out / "arrays.zarr"), mode="r")
    assert "okada_to_los" not in set(store.array_keys())
    reloaded = LosOperators.from_disk(out)
    assert reloaded.okada_to_los is None
    np.testing.assert_array_equal(
        reloaded.rotation_to_okada_los, los_ops.rotation_to_okada_los
    )

    # Retrofit a legacy-format folder with the dense LOS okada array present
    legacy = zarr.open_group(str(out / "arrays.zarr"), mode="a")
    dense_los_okada = np.ones((los_ops.n_los, 9))
    array_store = legacy.create_array(
        "okada_to_los", shape=dense_los_okada.shape, dtype=dense_los_okada.dtype
    )
    array_store[...] = dense_los_okada
    reloaded2 = LosOperators.from_disk(out)
    assert reloaded2.okada_to_los is None


def test_operators_roundtrip_with_none_okada(tmp_path):
    """Operators round-trips with the dense okada field None: nothing is
    written for it, and from_disk restores None.
    """
    model = _wna_model(25)
    ops = build_operators(model, eigen=True, tde=True)
    out = tmp_path / "operators"
    ops.to_disk(out)

    store = zarr.open_group(str(out / "arrays.zarr"), mode="r")
    keys = set(store.array_keys())
    assert "slip_rate_to_okada_to_velocities" not in keys
    assert "rotation_to_slip_rate_to_okada_to_velocities" in keys

    ops2 = Operators.from_disk(out)
    assert ops2.slip_rate_to_okada_to_velocities is None
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_from_disk_legacy_run_folder_with_dense_okada(tmp_path, wna_dense):
    """A run folder written by an older version contains the dense okada array
    in arrays.zarr; loading must skip it without materializing.
    """
    model = _wna_model(25)
    ops = build_operators(model, eigen=True, tde=True)
    out = tmp_path / "operators"
    ops.to_disk(out)

    legacy = zarr.open_group(str(out / "arrays.zarr"), mode="a")
    array_store = legacy.create_array(
        "slip_rate_to_okada_to_velocities",
        shape=wna_dense.shape,
        dtype=wna_dense.dtype,
    )
    array_store[...] = wna_dense

    ops2 = Operators.from_disk(out)
    assert ops2.slip_rate_to_okada_to_velocities is None
    np.testing.assert_array_equal(
        ops2.rotation_to_slip_rate_to_okada_to_velocities,
        ops.rotation_to_slip_rate_to_okada_to_velocities,
    )


def test_streamed_composition_peak_memory_stays_below_dense():
    """Regression guard: with a small memory budget, the accumulate path must
    never come close to materializing the dense okada matrix.
    """
    model = _wna_model(400)
    rotation = get_rotation_to_slip_rate_partials(model.segment, model.block)
    n_stations = len(model.station)
    n_segments = len(model.segment)
    dense_bytes = (3 * n_stations) * (3 * n_segments) * 8

    tracemalloc.start()
    tracemalloc.reset_peak()
    result = _compute_okada_composed_and_cache(
        model.segment,
        model.station,
        model.config,
        rotation,
        cache=None,
        target_bytes=dense_bytes // 20,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.shape == (3 * n_stations, rotation.shape[1])
    assert peak < dense_bytes / 3, (
        f"accumulate peak {peak / 2**20:.1f} MiB vs dense {dense_bytes / 2**20:.1f} MiB"
    )


def test_cache_hit_stream_peak_memory_stays_below_dense(tmp_path):
    """Regression guard for the cache-hit path: row-slab streaming must stay
    far below the dense matrix footprint.
    """
    model = _wna_model(400)
    rotation = get_rotation_to_slip_rate_partials(model.segment, model.block)
    dense = _dense_okada(model)
    dense_bytes = dense.nbytes
    with h5py.File(tmp_path / "cache.hdf5", "w") as hdf5_file:
        hdf5_file.create_dataset(OKADA_KEY, data=dense)
    del dense

    with h5py.File(tmp_path / "cache.hdf5", "r") as hdf5_file:
        tracemalloc.start()
        tracemalloc.reset_peak()
        result = _stream_matmul_rows(hdf5_file[OKADA_KEY], rotation, dense_bytes // 20)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    assert result.shape[1] == rotation.shape[1]
    assert peak < dense_bytes / 3, (
        f"stream peak {peak / 2**20:.1f} MiB vs dense {dense_bytes / 2**20:.1f} MiB"
    )
