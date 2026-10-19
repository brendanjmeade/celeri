from types import SimpleNamespace

import meshio
import numpy as np
import pandas as pd
import pytest
import scipy.spatial.distance
from pydantic import ValidationError
from sklearn.gaussian_process.kernels import Matern

from celeri.mesh import (
    Mesh,
    MeshConfig,
    ScalarBound,
    _get_eigenvalues_and_eigenvectors,
    _local_length_scales,
    _nonstationary_matern_kernel,
    resolve_eigenmode_area_weighting,
)
from celeri.operators import _hash_elastic_operator_input
from celeri.solve_mcmc import _get_eigenmode_projection


def _strip_mesh(path, depths):
    """A two-column strip of triangles along latitude 40 with the given depths (km)."""
    points = []
    for depth in depths:
        for k in range(4):
            points.append((240.0 + 0.05 * k, 40.0 + 0.02 * len(points), depth))
    points = np.array(points, dtype=float)
    tris = []
    n = 4
    for row in range(len(depths) - 1):
        for k in range(n - 1):
            a, b, c, d = (
                row * n + k,
                row * n + k + 1,
                (row + 1) * n + k + 1,
                (row + 1) * n + k,
            )
            tris += [[a, b, c], [a, c, d]]
    meshio.write(
        path,
        meshio.Mesh(points, [("triangle", np.array(tris))]),
        file_format="gmsh22",
        binary=False,
    )
    return path


def _mesh_config(path, **overrides):
    params = dict(
        file_name=path.parent / "mesh_parameters.json",
        mesh_filename=str(path),
        smoothing_weight=1.0,
        n_modes_strike_slip=2,
        n_modes_dip_slip=2,
        top_slip_rate_constraint=0,
        bottom_slip_rate_constraint=0,
        side_slip_rate_constraint=0,
        matern_nu=2.5,
        matern_length_scale=0.2,
        matern_length_units="diameters",
        eigenvector_algorithm="eigh",
    )
    params.update(overrides)
    return MeshConfig.model_validate(params)


def test_positive_down_depths_are_rejected(tmp_path):
    path = _strip_mesh(tmp_path / "positive.msh", depths=(0.0, 10.0, 20.0))
    with pytest.raises(ValueError, match="negative below the surface"):
        Mesh.from_params(_mesh_config(path))


def test_more_modes_than_elements_is_rejected(tmp_path):
    path = _strip_mesh(tmp_path / "small.msh", depths=(0.0, -10.0, -20.0))
    Mesh.from_params(_mesh_config(path, n_modes_strike_slip=12, n_modes_dip_slip=12))
    with pytest.raises(ValueError, match="eigenmodes were requested"):
        Mesh.from_params(_mesh_config(path, n_modes_strike_slip=13, n_modes_dip_slip=1))


def test_scalar_bound_must_be_ordered():
    assert ScalarBound.model_validate([0, 1]).upper == 1
    assert ScalarBound.model_validate([None, 5]).lower is None
    with pytest.raises(ValidationError):
        ScalarBound.model_validate([1, 0])
    with pytest.raises(ValidationError):
        ScalarBound.model_validate({"lower": 0, "uper": 1})


# --- Eigenmodes: local (mesh-following) length scale and area weighting ---


def _points(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(0, 100e3, n), rng.uniform(0, 50e3, n), np.zeros(n)]


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, np.inf, 1.0])
def test_nonstationary_kernel_with_constant_length_is_stationary(nu):
    pts = _points()
    ell = 20e3
    distance = scipy.spatial.distance.cdist(pts, pts)
    kernel = _nonstationary_matern_kernel(distance, np.full(len(pts), ell), nu)
    np.testing.assert_allclose(
        kernel, Matern(nu=nu, length_scale=ell)(pts), rtol=0, atol=1e-12
    )


def test_nonstationary_kernel_is_positive_definite():
    pts = _points()
    # length scale varying by a factor of 10 across the region
    ell = 5e3 + 45e3 * pts[:, 0] / 100e3
    distance = scipy.spatial.distance.cdist(pts, pts)
    kernel = _nonstationary_matern_kernel(distance, ell, 2.5)
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    np.testing.assert_allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() > 0


def test_local_length_scales_follow_element_size():
    pts = _points()
    areas = np.where(pts[:, 0] < 50e3, 1e8, 1e8 / 16)  # 10 km and 2.5 km elements
    distance = scipy.spatial.distance.cdist(pts, pts)
    ell, smoothing = _local_length_scales(distance, areas, 4.0, 0.0)
    assert smoothing == 0.0
    np.testing.assert_allclose(ell, 4.0 * np.sqrt(areas))
    # smoothing keeps a uniform field and blurs a step
    ell_uniform, _ = _local_length_scales(distance, np.full(len(pts), 1e8), 4.0, None)
    np.testing.assert_allclose(ell_uniform, 40e3)
    ell_smooth, smoothing = _local_length_scales(distance, areas, 4.0, None)
    assert smoothing > 0
    assert ell_smooth.min() >= 10e3 - 1e-6 and ell_smooth.max() <= 40e3 + 1e-6
    with pytest.raises(ValueError, match="positive"):
        _local_length_scales(distance, areas, 0.0, None)


def test_area_weighted_modes_recover_kernel():
    pts = _points(n=120)
    rng = np.random.default_rng(1)
    areas = rng.uniform(0.5e8, 2e8, len(pts))
    n = len(pts)
    lam, vec, trace = _get_eigenvalues_and_eigenvectors(
        n,
        *pts.T,
        matern_nu=2.5,
        matern_length_scale=20e3,
        matern_length_units="absolute",
        areas=areas,
        area_weighting=True,
    )
    w = areas / areas.mean()
    np.testing.assert_allclose(vec.T @ (w[:, None] * vec), np.eye(n), atol=1e-10)
    np.testing.assert_allclose(
        (vec * lam) @ vec.T, Matern(nu=2.5, length_scale=20e3)(pts), atol=1e-10
    )
    np.testing.assert_allclose(trace, w.sum())
    np.testing.assert_allclose(lam.sum(), trace)


def test_area_weighting_with_uniform_areas_matches_unweighted():
    pts = _points()
    kwargs = dict(
        matern_nu=2.5, matern_length_scale=20e3, matern_length_units="absolute"
    )
    lam0, vec0, tr0 = _get_eigenvalues_and_eigenvectors(10, *pts.T, **kwargs)
    lam1, vec1, tr1 = _get_eigenvalues_and_eigenvectors(
        10, *pts.T, **kwargs, areas=np.full(len(pts), 3e7), area_weighting=True
    )
    np.testing.assert_allclose(lam1, lam0, rtol=1e-10)
    np.testing.assert_allclose(tr1, tr0)
    np.testing.assert_allclose(np.abs(vec1.T @ vec0), np.eye(10), atol=1e-8)


def test_legacy_eigenmodes_ignore_areas():
    pts = _points()
    kwargs = dict(
        matern_nu=2.5, matern_length_scale=0.2, matern_length_units="diameters"
    )
    lam0, vec0, tr0 = _get_eigenvalues_and_eigenvectors(12, *pts.T, **kwargs)
    lam1, vec1, tr1 = _get_eigenvalues_and_eigenvectors(
        12, *pts.T, **kwargs, areas=np.linspace(1e7, 1e8, len(pts))
    )
    np.testing.assert_allclose(lam1, lam0, rtol=1e-12)
    assert tr1 == tr0
    np.testing.assert_allclose(
        (vec1 * lam1) @ vec1.T, (vec0 * lam0) @ vec0.T, atol=1e-12
    )


def test_local_units_on_uniform_points_match_absolute():
    # A uniform grid of equal elements: the local length scale is constant
    x, y = np.meshgrid(np.arange(0, 60e3, 2e3), np.arange(0, 30e3, 2e3))
    pts = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    areas = np.full(len(pts), 4e6)  # 2 km elements
    lam0, vec0, _ = _get_eigenvalues_and_eigenvectors(
        15,
        *pts.T,
        matern_nu=2.5,
        matern_length_scale=16e3,
        matern_length_units="absolute",
        areas=areas,
        area_weighting=True,
    )
    lam1, vec1, _ = _get_eigenvalues_and_eigenvectors(
        15,
        *pts.T,
        matern_nu=2.5,
        matern_length_scale=8.0,
        matern_length_units="local",
        areas=areas,
        area_weighting=True,
    )
    np.testing.assert_allclose(lam1, lam0, rtol=1e-8)
    np.testing.assert_allclose(
        (vec1 * lam1) @ vec1.T, (vec0 * lam0) @ vec0.T, atol=1e-8
    )


def test_local_units_require_areas():
    pts = _points(n=20)
    with pytest.raises(ValueError, match="areas are required"):
        _get_eigenvalues_and_eigenvectors(
            3, *pts.T, matern_length_scale=5.0, matern_length_units="local"
        )


def test_area_weighting_resolution(tmp_path):
    path = _strip_mesh(tmp_path / "strip.msh", depths=(0.0, -10.0, -20.0))
    assert not resolve_eigenmode_area_weighting(_mesh_config(path))
    local = _mesh_config(path, matern_length_units="local", matern_length_scale=4.0)
    assert resolve_eigenmode_area_weighting(local)
    assert not resolve_eigenmode_area_weighting(
        _mesh_config(
            path,
            matern_length_units="local",
            matern_length_scale=4.0,
            eigenmode_area_weighting=False,
        )
    )
    assert resolve_eigenmode_area_weighting(
        _mesh_config(path, eigenmode_area_weighting=True)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"matern_length_units": "local", "matern_length_scale": 4.0},
        {"eigenmode_area_weighting": True},
    ],
)
def test_mesh_eigenmode_projection(tmp_path, overrides):
    path = _strip_mesh(tmp_path / "strip.msh", depths=(0.0, -10.0, -20.0))
    mesh = Mesh.from_params(_mesh_config(path, **overrides))
    weights = mesh.eigenmode_weights
    assert (weights is None) == (not overrides)
    model = SimpleNamespace(meshes=[mesh])
    for kind in ("strike_slip", "dip_slip"):
        projection = _get_eigenmode_projection(model, 0, kind)
        vec = mesh.eigenvectors[:, :2]
        np.testing.assert_allclose(projection.T @ vec, np.eye(2), atol=1e-10)
        if weights is None:
            assert np.shares_memory(projection, mesh.eigenvectors)


def test_operator_cache_key_ignores_eigenmode_settings(tmp_path):
    # The cache holds tde_to_velocities only; the eigenmodes are projected at
    # run time, so the new eigenmode settings must not change the key
    path = _strip_mesh(tmp_path / "strip.msh", depths=(0.0, -10.0, -20.0))
    station = pd.DataFrame(
        {
            "lon": [240.1],
            "lat": [40.1],
            "depth": [0.0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
        }
    )
    config = SimpleNamespace(material_mu=3e10, material_lambda=3e10)
    key = _hash_elastic_operator_input([_mesh_config(path)], station, config)
    local = _mesh_config(
        path,
        matern_length_units="local",
        matern_length_scale=4.0,
        matern_length_smoothing=5e3,
        eigenmode_area_weighting=True,
    )
    local_key = _hash_elastic_operator_input([local], station, config)
    reverted = local.model_copy(
        update=dict(matern_length_units="diameters", matern_length_scale=0.2)
    )
    assert _hash_elastic_operator_input([reverted], station, config) == key
    assert local_key != key  # the length scale itself is still part of the key
