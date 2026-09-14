import meshio
import numpy as np
import pytest

from celeri.constants import GEOID
from celeri.mesh import Mesh, MeshConfig


def _dipping_strip(
    path, lon0=240.0, lat0=40.0, dip=60.0, hanging_wall_az=90.0, reverse=False
):
    """A planar 6 x 3 node fault mesh striking north from (lon0, lat0), dipping `dip`
    degrees towards azimuth `hanging_wall_az`, written in gmsh format.
    """
    trace_lat = lat0 + np.linspace(0.0, 1.0, 6)
    points = []
    for depth_km in (0.0, 10.0, 20.0):
        offset = depth_km * 1e3 / np.tan(np.deg2rad(dip))
        for lat in trace_lat:
            lon, lat_off, _ = GEOID.fwd(lon0, lat, hanging_wall_az, offset)
            points.append((lon % 360.0, lat_off, -depth_km))
    points = np.array(points)
    tris = []
    for row in range(2):
        for k in range(5):
            a, b, c, d = (
                row * 6 + k,
                row * 6 + k + 1,
                (row + 1) * 6 + k + 1,
                (row + 1) * 6 + k,
            )
            tris += [[a, b, c], [a, c, d]]
    tris = np.array(tris)
    if reverse:
        tris = tris[:, ::-1]
    meshio.write(
        path,
        meshio.Mesh(points, [("triangle", tris)]),
        file_format="gmsh22",
        binary=False,
    )
    return path


def _load(path):
    config = MeshConfig.model_validate(
        dict(
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
    )
    return Mesh.from_params(config)


@pytest.mark.parametrize(
    "hanging_wall_az, expected_strike", [(90.0, 0.0), (270.0, 180.0)]
)
def test_strike_and_dip_are_measured_in_a_local_frame(
    tmp_path, hanging_wall_az, expected_strike
):
    mesh = _load(
        _dipping_strip(
            tmp_path / "strip.msh", dip=60.0, hanging_wall_az=hanging_wall_az
        )
    )

    np.testing.assert_allclose(mesh.dip, 60.0, atol=0.3)
    strike_error = (mesh.strike - expected_strike + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(strike_error, 0.0, atol=1.0)


def test_vertex_order_is_normalised(tmp_path):
    forward = _load(_dipping_strip(tmp_path / "forward.msh"))
    reversed_ = _load(_dipping_strip(tmp_path / "reversed.msh", reverse=True))

    assert np.all(forward.nv[:, 2] > 0) and np.all(reversed_.nv[:, 2] > 0)
    np.testing.assert_allclose(reversed_.dip, forward.dip)
    np.testing.assert_allclose(reversed_.strike, forward.strike)
    np.testing.assert_array_equal(reversed_.verts, forward.verts)


def test_centroids_and_orientation_across_the_prime_meridian(tmp_path):
    reference = _load(_dipping_strip(tmp_path / "ref.msh", lon0=10.0))
    crossing = _load(_dipping_strip(tmp_path / "cross.msh", lon0=359.95))

    assert np.all((crossing.lon_centroid < 1.0) | (crossing.lon_centroid > 359.0))
    np.testing.assert_allclose(crossing.dip, reference.dip, atol=0.05)
    strike_diff = (crossing.strike - reference.strike + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(strike_diff, 0.0, atol=0.05)
