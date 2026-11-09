"""Poisson's ratio of the TDE and Mogi operators (issue #536).

Both used mu / (2 (lambda + mu)) instead of lambda / (2 (lambda + mu)); the two
agree only for lambda == mu, so these tests use lambda != mu.
"""

import warnings
from types import SimpleNamespace

import cutde.halfspace as cutde_halfspace
import h5py
import numpy as np
import pandas as pd
import pytest

from celeri.celeri_util import get_transverse_projection
from celeri.constants import KM2M
from celeri.operators import (
    MESH_GEOMETRY_ATTR,
    TDE_KERNEL_VERSION,
    TDE_KERNEL_VERSION_ATTR,
    _mesh_geometry_digest,
    _tde_dataset_is_current,
)
from celeri.spatial import (
    get_mogi_to_velocities_partials,
    get_tde_displacement_slab_single_mesh,
    poissons_ratio,
)

MU = 3e10


def test_poissons_ratio_from_lame_parameters():
    assert poissons_ratio(MU, MU) == 0.25
    assert poissons_ratio(2 * MU, MU) == pytest.approx(1 / 3)
    assert poissons_ratio(0.5 * MU, MU) == pytest.approx(1 / 6)


def _one_triangle_mesh():
    """The attributes of a Mesh that the TDE slab function and the cache
    digest read, for a single dipping triangle.
    """
    lon = np.array([240.0, 240.1, 240.05])
    lat = np.array([40.0, 40.0, 40.08])
    dep = np.array([-1.0, -1.0, -12.0])
    return SimpleNamespace(
        centroids=np.array([[lon.mean(), lat.mean(), dep.mean()]]),
        lon1=lon[:1],
        lat1=lat[:1],
        dep1=dep[:1],
        lon2=lon[1:2],
        lat2=lat[1:2],
        dep2=dep[1:2],
        lon3=lon[2:],
        lat3=lat[2:],
        dep3=dep[2:],
        points=np.c_[lon, lat, dep],
        verts=np.array([[0, 1, 2]]),
    )


@pytest.mark.parametrize("lambda_over_mu", [1.0, 2.0, 0.5])
def test_tde_slab_uses_poissons_ratio_of_lame_parameters(lambda_over_mu):
    mesh = _one_triangle_mesh()
    obs_lon = np.array([239.9, 240.05, 240.2, 240.05])
    obs_lat = np.array([40.05, 39.9, 40.1, 40.3])

    slab = get_tde_displacement_slab_single_mesh(
        obs_lon,
        obs_lat,
        [mesh],
        lambda_over_mu * MU,
        MU,
        mesh_idx=0,
        tri_start=0,
        tri_stop=1,
    )

    # Direct cutde call with nu = lambda / (2 (lambda + mu))
    nu = lambda_over_mu / (2 * (lambda_over_mu + 1))
    projection = get_transverse_projection(mesh.centroids[0, 0], mesh.centroids[0, 1])
    obs_x, obs_y = projection(obs_lon, obs_lat)
    tri = np.array(
        [
            [*projection(mesh.lon1[0], mesh.lat1[0]), KM2M * mesh.dep1[0]],
            [*projection(mesh.lon2[0], mesh.lat2[0]), KM2M * mesh.dep2[0]],
            [*projection(mesh.lon3[0], mesh.lat3[0]), KM2M * mesh.dep3[0]],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = cutde_halfspace.disp_matrix(
            obs_pts=np.c_[obs_x, obs_y, np.zeros_like(obs_x)],
            tris=tri[np.newaxis],
            nu=nu,
        ).reshape(3 * obs_lon.size, 3)
    np.testing.assert_allclose(slab, expected, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("lambda_over_mu", [1.0, 2.0, 0.5])
def test_mogi_uses_poissons_ratio_of_lame_parameters(lambda_over_mu):
    depth_km = 5.0
    mogi = pd.DataFrame({"lon": [240.0], "lat": [40.0], "depth": [depth_km]})
    station = pd.DataFrame({"lon": np.array([240.0]), "lat": np.array([40.0])})
    config = SimpleNamespace(material_lambda=lambda_over_mu * MU, material_mu=MU)

    operator = get_mogi_to_velocities_partials(mogi, station, config)

    # Directly above the source: u_up = (1 - nu) / pi * d / d^3
    nu = lambda_over_mu / (2 * (lambda_over_mu + 1))
    depth = KM2M * depth_km
    np.testing.assert_allclose(
        operator[2, 0], (1 - nu) / (np.pi * depth**2), rtol=1e-12
    )


@pytest.fixture
def cache_file(tmp_path):
    with h5py.File(tmp_path / "cache.hdf5", "w") as hdf5_file:
        yield hdf5_file


def _dataset(hdf5_file, mesh, version=None):
    dataset = hdf5_file.create_dataset(f"d{len(hdf5_file)}", data=np.zeros((3, 3)))
    dataset.attrs[MESH_GEOMETRY_ATTR] = _mesh_geometry_digest(mesh)
    if version is not None:
        dataset.attrs[TDE_KERNEL_VERSION_ATTR] = version
    return dataset


def test_tde_cache_rejects_wrong_poissons_ratio_datasets(cache_file):
    """Datasets from the old kernel (no version stamp) used the wrong Poisson's
    ratio; they are identical to the current kernel's only when lambda == mu.
    """
    mesh = _one_triangle_mesh()
    equal = SimpleNamespace(material_lambda=MU, material_mu=MU)
    unequal = SimpleNamespace(material_lambda=2 * MU, material_mu=MU)

    legacy = _dataset(cache_file, mesh)
    assert _tde_dataset_is_current(legacy, mesh, equal)
    assert not _tde_dataset_is_current(legacy, mesh, unequal)

    current = _dataset(cache_file, mesh, version=TDE_KERNEL_VERSION)
    assert _tde_dataset_is_current(current, mesh, equal)
    assert _tde_dataset_is_current(current, mesh, unequal)

    # The geometry digest is still required
    other = _one_triangle_mesh()
    other.points = other.points + 0.01
    assert not _tde_dataset_is_current(current, other, equal)
