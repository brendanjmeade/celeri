"""Tests for the batched/streaming eigen operator construction (issue #485).

The streaming MCMC path (mcmc_station_velocity_method="project_to_eigen") no
longer materializes the dense per-mesh (3 * n_stations, 3 * n_tris) TDE matrix.
These tests pin:
  1. Bitwise equality of the single-disp_matrix-call slab helper against the
     pre-#485 one-hot construction (reimplemented inline here).
  2. Numerical equivalence of the triangle-chunk accumulation and the row-batch
     projection against the dense reference, across batch-size edge cases.
  3. Cache behavior: legacy full-matrix caches are still read (in row slabs),
     and streaming mode no longer writes TDE datasets.
  4. A tracemalloc guard that the streaming path stays far below the dense
     matrix footprint.

All equivalence tests reuse a single model per module scope so that
mesh.eigenvectors are held fixed: eigenvector signs are arbitrary if recomputed
(scipy eigh/eigsh do no sign normalization), so results from two independent
build_model calls must never be compared.
"""

import warnings

import cutde.halfspace as cutde_halfspace
import h5py
import numpy as np
import pandas as pd
import pytest

import celeri
from celeri.celeri_util import get_keep_index_12, get_transverse_projection
from celeri.constants import KM2M
from celeri.operators import (
    _accumulate_eigen_to_velocities_streaming,
    _hash_elastic_operator_input,
    _OperatorBuilder,
    _project_operator_to_los,
    _project_tde_rows_to_eigen,
    _store_eigenvectors_to_tde_slip,
    build_los_operators,
    build_operators,
)
from celeri.spatial import (
    get_tde_displacement_slab_single_mesh,
    get_tde_to_velocities_single_mesh,
)

WNA_CONFIG = "tests/configs/test_wna_config.json"


def _station_subset(config, n_stations):
    """First n stations that survive processing (process_station drops flag == 0)."""
    station = pd.read_csv(config.station_file_name)
    return station[station.flag != 0].iloc[:n_stations].reset_index(drop=True)


def _wna_model(n_stations, cache_dir=None, **build_kwargs):
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = cache_dir
    station = _station_subset(config, n_stations)
    return celeri.build_model(config, override_station=station, **build_kwargs)


def _eigenvectors_to_tde_slip(model):
    builder = _OperatorBuilder(model)
    _store_eigenvectors_to_tde_slip(model, builder)
    return builder.eigenvectors_to_tde_slip


@pytest.fixture(scope="module")
def wna_model():
    """60-station wna model (1 mesh, 1841 triangles)."""
    return _wna_model(60)


@pytest.fixture(scope="module")
def wna_eigenvectors(wna_model):
    return _eigenvectors_to_tde_slip(wna_model)[0]


@pytest.fixture(scope="module")
def wna_dense(wna_model):
    """The dense per-mesh TDE matrix (the object streaming mode avoids)."""
    return get_tde_to_velocities_single_mesh(
        wna_model.meshes, wna_model.station, wna_model.config, mesh_idx=0
    )


@pytest.fixture(scope="module")
def wna_reference(wna_dense, wna_eigenvectors):
    """The exact pre-#485 projection: fancy-index copy then one big matmul."""
    keep = get_keep_index_12(wna_dense.shape[1])
    return -wna_dense[:, keep] @ wna_eigenvectors


def _one_hot_column_reference(model, tri_idx, slip):
    """Pre-#485 per-triangle computation: a full disp_matrix call dotted with a
    one-hot slip vector (this is what get_tri_displacements_single_mesh did
    before the refactor). Returns the (3 * n_obs,) interleaved displacement
    vector.
    """
    mesh = model.meshes[0]
    config = model.config
    poissons_ratio = config.material_mu / (
        2 * (config.material_mu + config.material_lambda)
    )
    projection = get_transverse_projection(
        mesh.centroids[tri_idx, 0], mesh.centroids[tri_idx, 1]
    )
    obs_lon = model.station.lon.to_numpy()
    obs_lat = model.station.lat.to_numpy()
    obs_x, obs_y = projection(obs_lon, obs_lat)
    tri_x1, tri_y1 = projection(mesh.lon1[tri_idx], mesh.lat1[tri_idx])
    tri_x2, tri_y2 = projection(mesh.lon2[tri_idx], mesh.lat2[tri_idx])
    tri_x3, tri_y3 = projection(mesh.lon3[tri_idx], mesh.lat3[tri_idx])
    tri_coords = np.array(
        [
            [tri_x1, tri_y1, KM2M * mesh.dep1[tri_idx]],
            [tri_x2, tri_y2, KM2M * mesh.dep2[tri_idx]],
            [tri_x3, tri_y3, KM2M * mesh.dep3[tri_idx]],
        ]
    )
    obs_coords = np.vstack((obs_x, obs_y, np.zeros_like(obs_x))).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mat = cutde_halfspace.disp_matrix(
            obs_pts=obs_coords, tris=np.array([tri_coords]), nu=poissons_ratio
        )
    return disp_mat.reshape((-1, 3)).dot(np.array(slip))


def test_slab_helper_bitwise_equals_one_hot_path(wna_model):
    """The single-call slab must reproduce the pre-#485 one-hot construction
    bitwise (extracting a column of the Green's tensor equals dotting with a
    one-hot slip vector exactly, since x*1 + y*0 + z*0 is exact for finite
    floats).
    """
    n_tris = wna_model.meshes[0].lon1.size
    config = wna_model.config
    for tri_idx in [0, 17, n_tris - 1]:
        slab = get_tde_displacement_slab_single_mesh(
            wna_model.station.lon.to_numpy(),
            wna_model.station.lat.to_numpy(),
            wna_model.meshes,
            config.material_lambda,
            config.material_mu,
            mesh_idx=0,
            tri_start=tri_idx,
            tri_stop=tri_idx + 1,
        )
        for col, slip in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            expected = _one_hot_column_reference(wna_model, tri_idx, slip)
            assert np.array_equal(slab[:, col], expected)


@pytest.mark.parametrize("tri_batch", [1, 3, 7, "n_tris", "oversize"])
def test_accumulation_matches_dense_reference(
    wna_model, wna_eigenvectors, wna_reference, tri_batch
):
    """Triangle-chunk accumulation must match the dense projection for batch
    sizes covering the degenerate, non-divisor, exact, and oversize cases.
    """
    n_tris = wna_model.meshes[0].lon1.size
    batch = {"n_tris": n_tris, "oversize": n_tris + 13}.get(tri_batch, tri_batch)
    result = _accumulate_eigen_to_velocities_streaming(
        wna_model.meshes,
        wna_model.station,
        wna_model.config,
        0,
        wna_eigenvectors,
        target_bytes=2**30,
        tri_batch_size=batch,
    )
    # Full 3 * n_stations rows are kept (the docstring used to claim 2 *
    # n_stations; consumers index rows via station_row_keep_index)
    assert result.shape == (3 * len(wna_model.station), wna_eigenvectors.shape[1])
    np.testing.assert_allclose(result, wna_reference, rtol=1e-6, atol=1e-9)


def test_default_batch_size_derivation(wna_model, wna_eigenvectors, wna_reference):
    """Batch size derived from a small memory budget (forces several chunks)."""
    n_obs = len(wna_model.station)
    # Budget for ~11 triangles per chunk
    result = _accumulate_eigen_to_velocities_streaming(
        wna_model.meshes,
        wna_model.station,
        wna_model.config,
        0,
        wna_eigenvectors,
        target_bytes=11 * 72 * n_obs,
    )
    np.testing.assert_allclose(result, wna_reference, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("rows_per_batch", [1, 7, 10**9])
def test_row_projection_matches_dense_reference(
    wna_dense, wna_eigenvectors, wna_reference, rows_per_batch
):
    """Row-batched projection (cache-read / non-streaming path) must match the
    dense fancy-index projection for degenerate, non-divisor, and single-batch
    row-slab sizes.
    """
    target_bytes = rows_per_batch * 8 * wna_dense.shape[1]
    result = _project_tde_rows_to_eigen(wna_dense, wna_eigenvectors, target_bytes)
    np.testing.assert_allclose(result, wna_reference, rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def japan_model():
    """3-mesh japan model with a small station subset (multi-mesh loop)."""
    config = celeri.get_config("tests/configs/test_japan_config.json")
    config.elastic_operator_cache_dir = None
    station = _station_subset(config, 25)
    return celeri.build_model(config, override_station=station)


@pytest.mark.parametrize("mesh_idx", [0, 1, 2])
def test_accumulation_matches_dense_reference_multimesh(japan_model, mesh_idx):
    """Multi-mesh model: catches per-mesh index bugs; batch 7 does not divide
    any of the three triangle counts.
    """
    eigenvectors = _eigenvectors_to_tde_slip(japan_model)[mesh_idx]
    dense = get_tde_to_velocities_single_mesh(
        japan_model.meshes, japan_model.station, japan_model.config, mesh_idx=mesh_idx
    )
    assert dense.ndim == 2
    reference = -dense[:, get_keep_index_12(dense.shape[1])] @ eigenvectors
    result = _accumulate_eigen_to_velocities_streaming(
        japan_model.meshes,
        japan_model.station,
        japan_model.config,
        mesh_idx,
        eigenvectors,
        target_bytes=2**30,
        tri_batch_size=7,
    )
    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-9)


def test_streaming_build_matches_dense_build(tmp_path):
    """End-to-end build_operators with and without discard_tde_to_velocities on
    one shared model. The dense build writes the full TDE cache; the streaming
    build then reads it in row slabs.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    ops_dense = build_operators(
        model, eigen=True, tde=True, discard_tde_to_velocities=False
    )
    ops_stream = build_operators(
        model, eigen=True, tde=True, discard_tde_to_velocities=True
    )
    assert ops_stream.tde is not None and ops_stream.tde.tde_to_velocities is None
    assert ops_dense.eigen is not None and ops_stream.eigen is not None
    for i in ops_dense.eigen.eigen_to_velocities:
        np.testing.assert_allclose(
            ops_stream.eigen.eigen_to_velocities[i],
            ops_dense.eigen.eigen_to_velocities[i],
            rtol=1e-10,
            atol=1e-10,
        )


def test_streaming_reads_legacy_full_tde_cache(tmp_path):
    """Format compatibility: a full dense tde_to_velocities_<i> dataset written
    by pre-#485 code must still be consumed by the streaming path. The cached
    matrix is scaled by 2 so the test proves the cache (not a recompute) was
    the source of the result.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    dense = get_tde_to_velocities_single_mesh(
        model.meshes, model.station, model.config, mesh_idx=0
    )
    assert dense.ndim == 2
    # Let _store_elastic_operators establish the cache file first (it would
    # rebuild a file it does not recognize), then inject the legacy dataset
    build_operators(model, eigen=True, tde=True, discard_tde_to_velocities=True)
    input_hash = _hash_elastic_operator_input(
        [mesh.config for mesh in model.meshes], model.station, model.config
    )
    with h5py.File(tmp_path / f"{input_hash}.hdf5", "a") as hdf5_file:
        hdf5_file.create_dataset("tde_to_velocities_0", data=2.0 * dense)

    ops = build_operators(model, eigen=True, tde=True, discard_tde_to_velocities=True)
    assert ops.eigen is not None
    expected = (
        -2.0
        * dense[:, get_keep_index_12(dense.shape[1])]
        @ ops.eigen.eigenvectors_to_tde_slip[0]
    )
    np.testing.assert_allclose(
        ops.eigen.eigen_to_velocities[0], expected, rtol=1e-10, atol=1e-10
    )


def test_streaming_does_not_write_tde_cache(tmp_path):
    """Streaming mode must no longer persist dense TDE matrices (at production
    scale they would be hundreds of gigabytes on disk).
    """
    model = _wna_model(25, cache_dir=tmp_path)
    build_operators(model, eigen=True, tde=True, discard_tde_to_velocities=True)
    input_hash = _hash_elastic_operator_input(
        [mesh.config for mesh in model.meshes], model.station, model.config
    )
    cache_file = tmp_path / f"{input_hash}.hdf5"
    assert cache_file.exists(), "streaming build should still cache okada operators"
    with h5py.File(cache_file, "r") as hdf5_file:
        tde_keys = [k for k in hdf5_file if k.startswith("tde_to_velocities_")]
    assert tde_keys == []


def test_dense_build_after_streaming_cache(tmp_path):
    """A streaming run leaves a cache with okada but no TDE datasets. A later
    non-streaming build with the same cache (e.g. Operators.from_disk after a
    save_operators=False run) must compute the missing TDE matrices instead of
    returning empty operators, and must write them back to the cache.
    """
    model = _wna_model(25, cache_dir=tmp_path)
    build_operators(model, eigen=True, tde=True, discard_tde_to_velocities=True)

    ops_dense = build_operators(
        model, eigen=True, tde=True, discard_tde_to_velocities=False
    )
    assert ops_dense.tde is not None and ops_dense.tde.tde_to_velocities is not None
    dense = ops_dense.tde.tde_to_velocities[0]
    reference = get_tde_to_velocities_single_mesh(
        model.meshes, model.station, model.config, mesh_idx=0
    )
    np.testing.assert_allclose(dense, reference, rtol=1e-10, atol=1e-10)

    input_hash = _hash_elastic_operator_input(
        [mesh.config for mesh in model.meshes], model.station, model.config
    )
    with h5py.File(tmp_path / f"{input_hash}.hdf5", "r") as hdf5_file:
        assert "tde_to_velocities_0" in hdf5_file


def test_segment_edit_streaming_preserves_tde_cache(tmp_path):
    """A streaming run after a segment edit rewrites the okada part of the
    cache but must NOT destroy previously cached TDE datasets (they stay valid:
    the cache hash covers stations, meshes, and materials, not segments).
    """
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = tmp_path
    station = _station_subset(config, 25)
    model = celeri.build_model(config, override_station=station)
    # Dense build populates the cache with tde_to_velocities datasets
    build_operators(model, eigen=True, tde=True, discard_tde_to_velocities=False)

    # Edit one segment (locking depth change alters the okada operator only)
    segment = model.segment.copy(deep=True)
    segment.loc[0, "locking_depth"] = float(segment.loc[0, "locking_depth"]) + 1.0
    model_edited = celeri.build_model(
        config, override_station=station, override_segment=segment
    )
    build_operators(model_edited, eigen=True, tde=True, discard_tde_to_velocities=True)

    input_hash = _hash_elastic_operator_input(
        [mesh.config for mesh in model_edited.meshes],
        model_edited.station,
        model_edited.config,
    )
    with h5py.File(tmp_path / f"{input_hash}.hdf5", "r") as hdf5_file:
        assert "tde_to_velocities_0" in hdf5_file, (
            "segment edit during streaming must not destroy cached TDE datasets"
        )


def test_streaming_peak_memory_stays_below_dense():
    """Regression guard: with a small memory budget, the streaming projection
    must never come close to materializing the dense TDE matrix.
    """
    import tracemalloc

    model = _wna_model(400)
    eigenvectors = _eigenvectors_to_tde_slip(model)[0]
    n_stations = len(model.station)
    n_tris = model.meshes[0].lon1.size
    dense_bytes = (3 * n_stations) * (3 * n_tris) * 8
    target_bytes = 4 * 2**20  # 4 MiB slab budget

    tracemalloc.start()
    tracemalloc.reset_peak()
    result = _accumulate_eigen_to_velocities_streaming(
        model.meshes, model.station, model.config, 0, eigenvectors, target_bytes
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.shape == (3 * n_stations, eigenvectors.shape[1])
    assert peak < dense_bytes / 3, (
        f"streaming peak {peak / 2**20:.0f} MiB vs dense {dense_bytes / 2**20:.0f} MiB"
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


def test_eigen_to_los_batched_matches_dense_reference():
    """The LOS path mirrors the station path: chunked accumulation (discard) and
    row-batched projection (keep) must both match the dense reference.
    """
    config = celeri.get_config(WNA_CONFIG)
    config.elastic_operator_cache_dir = None
    station = pd.read_csv(config.station_file_name).iloc[:25].reset_index(drop=True)
    model = celeri.build_model(config, override_station=station)
    los = _synthetic_los(model.station)
    model = celeri.build_model(config, override_station=station, override_los=los)

    ops = build_operators(model, eigen=True, tde=True)
    assert ops.eigen is not None
    assert model.los is not None

    # Dense reference computed the pre-#485 way
    tde = get_tde_to_velocities_single_mesh(
        model.meshes, model.los, model.config, mesh_idx=0
    )
    look_vectors = np.column_stack(
        [
            model.los.look_vector_east.to_numpy(),
            model.los.look_vector_north.to_numpy(),
            model.los.look_vector_up.to_numpy(),
        ]
    )
    tde_los = _project_operator_to_los(tde, look_vectors)
    reference = (
        -tde_los[:, get_keep_index_12(tde_los.shape[1])]
        @ ops.eigen.eigenvectors_to_tde_slip[0]
    )

    # Force several triangle chunks in the discard path
    model.config.tde_operator_memory_gb = (100 * 72 * len(model.los)) / 2**30

    los_discard = build_los_operators(model, ops, discard_tde_to_los=True)
    los_keep = build_los_operators(model, ops, discard_tde_to_los=False)
    assert los_discard is not None and los_keep is not None

    assert los_discard.tde_to_los is None
    np.testing.assert_allclose(
        los_discard.eigen_to_los[0], reference, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        los_keep.eigen_to_los[0], reference, rtol=1e-9, atol=1e-9
    )
    assert los_keep.tde_to_los is not None
    np.testing.assert_allclose(los_keep.tde_to_los[0], tde_los, rtol=1e-12, atol=1e-12)
