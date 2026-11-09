"""Vertical and near-vertical dips (issue #536).

Dips within VERTICAL_DIP_TOLERANCE_DEG of 90 are treated as vertical: segments
are snapped to 90, mesh elements take no kinematic dip slip, and the
consumers of the kinematic rates handle the resulting zero rates.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from celeri.celeri_util import is_vertical_dip, sph2cart
from celeri.constants import RADIUS_EARTH, VERTICAL_DIP_TOLERANCE_DEG
from celeri.model import _warn_near_vertical_amplification, snap_vertical_segment_dips
from celeri.operators import (
    MESH_GEOMETRY_ATTR,
    TDE_KERNEL_VERSION,
    TDE_KERNEL_VERSION_ATTR,
    _mesh_geometry_digest,
    _tde_dataset_is_current,
)
from celeri.optimize_sqp import (
    ZERO_KINEMATIC_SLIP_RATE_BOUND,
    _get_coupling,
    _get_coupling_linear,
    _SlipRateBounds,
    _update_slip_rate_bounds,
)
from celeri.solve import coupling_ratio
from celeri.solve_mcmc import _coupling_prior_mean
from celeri.spatial import get_rotation_to_slip_rate_partials, snap_near_vertical_edges


@pytest.fixture
def warnings_logged():
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    logger.remove(handler_id)


def test_is_vertical_dip():
    tol = VERTICAL_DIP_TOLERANCE_DEG
    np.testing.assert_array_equal(
        is_vertical_dip([90.0, 90.0 - tol / 2, 90.0 + tol / 2, 89.9, 92.0, 60.0]),
        [True, True, True, False, False, False],
    )


def test_near_vertical_segment_dips_are_snapped(warnings_logged):
    segment = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d", "e", "f"],
            "dip": [89.97, 90.04, 89.9, 90.0, 60.0, 92.0],
            "ds_rate_flag": [0, 1, 0, 0, 0, 0],
            "ds_rate_bound_flag": [0, 0, 0, 0, 0, 0],
        }
    )
    snapped = snap_vertical_segment_dips(segment)
    assert snapped.dip.tolist() == [90.0, 90.0, 89.9, 90.0, 60.0, 92.0]
    # The input is not modified
    assert segment.dip.tolist()[0] == 89.97
    # Segment b carries a dip-slip constraint, which a vertical segment
    # cannot satisfy
    assert any("dip-slip" in m and "b" in m for m in warnings_logged)

    # Nothing to snap: the same frame comes back
    assert snap_vertical_segment_dips(snapped) is snapped


def _one_segment(dip):
    """A north-striking segment between block 0 (west) and block 1 (east)."""
    mid_x, mid_y, mid_z = sph2cart(240.0, 40.0, RADIUS_EARTH)
    return pd.DataFrame(
        {
            "dip": [dip],
            "azimuth": [0.0],
            "lat1": [39.9],
            "lat2": [40.1],
            "mid_lon": [240.0],
            "mid_lat": [40.0],
            "mid_x": [mid_x],
            "mid_y": [mid_y],
            "mid_z": [mid_z],
            "east_labels": [1],
            "west_labels": [0],
        }
    )


def test_snapped_segment_takes_tensile_not_amplified_dip_slip():
    block = pd.DataFrame(index=range(2))
    raw = get_rotation_to_slip_rate_partials(_one_segment(89.97), block)
    snapped = get_rotation_to_slip_rate_partials(
        snap_vertical_segment_dips(_one_segment(89.97)), block
    )
    # Unsnapped: fault-normal motion goes to dip slip, amplified ~1900 times
    assert np.all(raw[2] == 0.0)
    np.testing.assert_allclose(
        np.abs(raw[1]),
        np.abs(snapped[2]) / np.cos(np.deg2rad(89.97)),
        rtol=1e-12,
    )
    # Snapped: the vertical branch, with the fault-normal motion as tensile slip
    np.testing.assert_array_equal(snapped[1], 0.0)
    assert np.any(snapped[2] != 0.0)
    np.testing.assert_array_equal(snapped[0], raw[0])


def test_near_vertical_amplification_is_warned(warnings_logged):
    segment = pd.DataFrame(
        {"name": ["steep", "vertical", "dipping"], "dip": [89.5, 90.0, 60.0]}
    )
    mesh = SimpleNamespace(dip=np.array([88.0, 90.0, 89.99, 45.0]), n_tde=4, name="m")
    _warn_near_vertical_amplification(segment, [mesh])
    assert len(warnings_logged) == 1
    message = warnings_logged[0]
    # 89.5 (x115) and 88.0 (x29) are amplified; 90 and 89.99 count as vertical
    assert "1 segments (steep)" in message
    assert "1 of 4 elements of mesh m" in message


def test_snap_near_vertical_edges_makes_sides_vertical():
    offset = 10e3 * np.tan(np.deg2rad(1e-4))
    tri = np.array([[0.0, 0.0, -1e3], [5e3, 0.0, -1e3], [offset, 1e-3, -11e3]])
    snapped = snap_near_vertical_edges(tri, [False, False, True])
    # Side 2 joins vertices 2 and 0; the deeper vertex 2 moves under vertex 0
    np.testing.assert_array_equal(snapped[2], [0.0, 0.0, -11e3])
    np.testing.assert_array_equal(snapped[:2], tri[:2])
    # The input is not modified, and unflagged sides are left alone
    assert tri[2, 0] == offset
    np.testing.assert_array_equal(snap_near_vertical_edges(tri, [False] * 3), tri)

    # A snap that would collapse the triangle (all three vertices on one
    # vertical line) is skipped
    sliver = np.array([[0.0, 0.0, -1e3], [1e-3, 0.0, -11e3], [0.0, 0.0, -6e3]])
    np.testing.assert_array_equal(
        snap_near_vertical_edges(sliver, [True, False, False]), sliver
    )


def _sqp_mesh():
    bounds = SimpleNamespace(lower=0.0, upper=1.0)
    return SimpleNamespace(
        config=SimpleNamespace(
            coupling_constraints_ss=bounds,
            coupling_constraints_ds=bounds,
            iterative_coupling_linear_slip_rate_reduction_factor=0.025,
        )
    )


def test_sqp_pins_slip_where_kinematic_rate_is_zero():
    kinematic_ss = np.array([10.0, 10.0])
    kinematic_ds = np.array([0.0, 5.0])
    coupling_ss, _ = _get_coupling_linear(np.array([5.0, 5.0]), kinematic_ss)
    coupling_ds, _ = _get_coupling_linear(np.array([3.0, 2.0]), kinematic_ds)
    assert np.isnan(coupling_ds[0])
    np.testing.assert_allclose(coupling_ds[1], 0.4)

    hint = np.full(2, 100.0)
    bounds = _SlipRateBounds(-hint, hint, -hint, hint)
    n_oob, bounds = _update_slip_rate_bounds(
        [_sqp_mesh()], 0, coupling_ss, coupling_ds, kinematic_ss, kinematic_ds, bounds
    )
    # Only the zero-kinematic element is out of bounds, and it is pinned at once
    assert n_oob == 1
    assert bounds.ds_lower[0] == -ZERO_KINEMATIC_SLIP_RATE_BOUND
    assert bounds.ds_upper[0] == ZERO_KINEMATIC_SLIP_RATE_BOUND
    np.testing.assert_array_equal(bounds.ss_lower, -hint)
    assert bounds.ds_upper[1] == 100.0

    # Once pinned it counts as satisfied
    coupling_ds, _ = _get_coupling_linear(np.array([1e-4, 2.0]), kinematic_ds)
    n_oob, _ = _update_slip_rate_bounds(
        [_sqp_mesh()], 0, coupling_ss, coupling_ds, kinematic_ss, kinematic_ds, bounds
    )
    assert n_oob == 0


def test_regularized_coupling_is_finite_for_zero_kinematic_rate():
    coupling, kinematic = _get_coupling(
        None,
        None,
        np.array([0.5, 0.5, 0.5, 3.0]),
        np.array([0.0, 0.5, -0.5, 3.0]),
        smoothing_length_scale=None,
        kinematic_slip_regularization_scale=1.0,
    )
    np.testing.assert_array_equal(kinematic, [1.0, 1.0, -1.0, 3.0])
    np.testing.assert_array_equal(coupling, [0.5, 0.5, -0.5, 1.0])


def test_coupling_ratio_is_nan_where_kinematic_rate_is_zero():
    np.testing.assert_array_equal(
        coupling_ratio(np.array([1.0, 2.0, 0.0]), np.array([0.0, 4.0, 0.0])),
        [np.nan, 0.5, np.nan],
    )


def test_coupling_prior_mean_fills_undefined_couplings(tmp_path):
    path = tmp_path / "model_meshes.csv"
    pd.DataFrame({"dip_slip_coupling": [0.2, np.nan, 0.4]}).to_csv(path)
    config = SimpleNamespace(coupling_mean_filename=path, coupling_mean=0.5)
    model = SimpleNamespace(meshes=[SimpleNamespace(config=config, n_tde=3)])
    np.testing.assert_array_equal(
        _coupling_prior_mean(model, 0, "dip_slip"), [0.2, 0.5, 0.4]
    )

    config.coupling_mean = None
    with pytest.raises(ValueError, match="non-finite"):
        _coupling_prior_mean(model, 0, "dip_slip")


def test_tde_cache_rejects_unsnapped_near_vertical_datasets(tmp_path):
    """Datasets from kernels before the near-vertical snap are reused only for
    meshes with no near-vertical sides.
    """
    mesh = SimpleNamespace(
        points=np.zeros((3, 3)),
        verts=np.array([[0, 1, 2]]),
        near_vertical_edges=np.zeros((1, 3), dtype=bool),
    )
    config = SimpleNamespace(material_lambda=3e10, material_mu=3e10)
    with h5py.File(tmp_path / "cache.hdf5", "w") as hdf5_file:
        datasets = {}
        for version in (1, 2, TDE_KERNEL_VERSION):
            dataset = hdf5_file.create_dataset(f"v{version}", data=np.zeros(1))
            dataset.attrs[MESH_GEOMETRY_ATTR] = _mesh_geometry_digest(mesh)
            if version > 1:
                dataset.attrs[TDE_KERNEL_VERSION_ATTR] = version
            datasets[version] = dataset

        for dataset in datasets.values():
            assert _tde_dataset_is_current(dataset, mesh, config)

        mesh.near_vertical_edges = np.array([[False, True, False]])
        assert not _tde_dataset_is_current(datasets[1], mesh, config)
        assert not _tde_dataset_is_current(datasets[2], mesh, config)
        assert _tde_dataset_is_current(datasets[TDE_KERNEL_VERSION], mesh, config)
