import meshio
import numpy as np
import pytest
from loguru import logger

from celeri.constants import GEOID
from celeri.mesh import Mesh, MeshConfig, triangle_winding_sign
from celeri.spatial import get_tde_displacement_slab_single_mesh


def _dipping_strip(
    path,
    lon0=240.0,
    lat0=40.0,
    dip=60.0,
    hanging_wall_az=90.0,
    reverse=False,
    reverse_first=False,
):
    """A planar 6 x 3 node fault mesh striking north from (lon0, lat0), dipping `dip`
    degrees towards azimuth `hanging_wall_az`, written in gmsh format.

    With the default vertex order the normals point down for a hanging wall to
    the east (azimuth 90) and up for a hanging wall to the west (azimuth 270);
    `reverse` flips every triangle and `reverse_first` only the first one.
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
    if reverse_first:
        tris[0] = tris[0, ::-1]
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
    "hanging_wall_az, reverse, expected_strike",
    [(90.0, True, 0.0), (270.0, False, 180.0)],
)
def test_strike_and_dip_are_measured_in_a_local_frame(
    tmp_path, hanging_wall_az, reverse, expected_strike
):
    # Both strips are wound with upward normals so that the reported strike
    # and dip are those of the fault plane itself
    mesh = _load(
        _dipping_strip(
            tmp_path / "strip.msh",
            dip=60.0,
            hanging_wall_az=hanging_wall_az,
            reverse=reverse,
        )
    )

    assert np.all(mesh.nv[:, 2] > 0)
    np.testing.assert_allclose(mesh.dip, 60.0, atol=0.3)
    strike_error = (mesh.strike - expected_strike + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(strike_error, 0.0, atol=1.0)


def test_vertex_order_is_preserved_and_sets_the_dip_slip_sign(tmp_path):
    """The file's winding is kept. A downward normal reports the same plane
    with dip in (90, 180] and the opposite strike, and the two dip-slip
    sign factors (the kinematic 1/cos(dip) and cutde's dip-slip column)
    reverse together, so the physics of the element does not depend on the
    winding while its stored dip-slip numbers do.
    """
    forward = _load(_dipping_strip(tmp_path / "forward.msh", hanging_wall_az=270.0))
    reversed_ = _load(
        _dipping_strip(tmp_path / "reversed.msh", hanging_wall_az=270.0, reverse=True)
    )

    assert np.all(forward.nv[:, 2] > 0) and np.all(reversed_.nv[:, 2] < 0)
    np.testing.assert_array_equal(reversed_.verts, forward.verts[:, ::-1])
    np.testing.assert_allclose(reversed_.dip, 180.0 - forward.dip, atol=1e-9)
    np.testing.assert_allclose(
        (reversed_.strike - forward.strike) % 360.0, 180.0, atol=1e-9
    )
    np.testing.assert_allclose(
        1.0 / np.cos(np.deg2rad(reversed_.dip)),
        -1.0 / np.cos(np.deg2rad(forward.dip)),
    )

    obs_lon = np.array([240.3, 240.6, 239.8, 240.0])
    obs_lat = np.array([40.2, 40.7, 40.5, 41.2])
    columns = {}
    for name, mesh in (("forward", forward), ("reversed", reversed_)):
        columns[name] = get_tde_displacement_slab_single_mesh(
            obs_lon,
            obs_lat,
            [mesh],
            3e10,
            3e10,
            mesh_idx=0,
            tri_start=0,
            tri_stop=mesh.n_tde,
        )
    scale = np.max(np.abs(columns["forward"]))
    # strike-slip, dip-slip, tensile columns of every element
    np.testing.assert_allclose(
        columns["reversed"][:, 0::3], columns["forward"][:, 0::3], atol=1e-12 * scale
    )
    np.testing.assert_allclose(
        columns["reversed"][:, 1::3], -columns["forward"][:, 1::3], atol=1e-12 * scale
    )
    np.testing.assert_allclose(
        columns["reversed"][:, 2::3], columns["forward"][:, 2::3], atol=1e-12 * scale
    )


def test_mixed_winding_is_reported(tmp_path):
    messages = []
    handler_id = logger.add(
        lambda message: messages.append(str(message)), level="WARNING"
    )
    try:
        mesh = _load(_dipping_strip(tmp_path / "mixed.msh", reverse_first=True))
    finally:
        logger.remove(handler_id)

    assert int(np.sum(mesh.nv[:, 2] < 0)) == mesh.n_tde - 1
    assert any(
        "mixed vertex winding" in message
        and f"{mesh.n_tde - 1} of {mesh.n_tde}" in message
        for message in messages
    )


def test_centroids_and_orientation_across_the_prime_meridian(tmp_path):
    reference = _load(_dipping_strip(tmp_path / "ref.msh", lon0=10.0))
    crossing = _load(_dipping_strip(tmp_path / "cross.msh", lon0=359.95))

    assert np.all((crossing.lon_centroid < 1.0) | (crossing.lon_centroid > 359.0))
    np.testing.assert_allclose(crossing.dip, reference.dip, atol=0.05)
    strike_diff = (crossing.strike - reference.strike + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(strike_diff, 0.0, atol=0.05)


def test_vertical_mesh_is_not_reported_as_mixed_winding(tmp_path):
    """A vertical mesh has horizontal normals whose up component is round-off
    of either sign; it must not be reported as mixed winding.
    """
    messages = []
    handler_id = logger.add(
        lambda message: messages.append(str(message)), level="WARNING"
    )
    try:
        mesh = _load(_dipping_strip(tmp_path / "vertical.msh", dip=90.0))
    finally:
        logger.remove(handler_id)

    np.testing.assert_allclose(mesh.dip, 90.0, atol=1e-6)
    assert np.all(triangle_winding_sign(mesh.nv) == 1.0)
    assert not any("mixed vertex winding" in message for message in messages)
