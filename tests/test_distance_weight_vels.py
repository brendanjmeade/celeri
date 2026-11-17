"""Tests for celeri-distance-weight-vels (celeri.scripts.celeri_distance_weight_vels)."""

import meshio
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from celeri.scripts.celeri_distance_weight_vels import (
    RADIUS_EARTH_KM,
    closest_points_on_triangles,
    distance_to_meshes_km,
    distance_to_segments_km,
    distance_weights,
    fault_distances,
    main,
    mesh_on_side,
    read_mesh_triangles,
    scale_sigmas,
    segment_on_side,
    select_segments,
)

KM_PER_DEG = np.pi * RADIUS_EARTH_KM / 180
KM_PER_DEG_LON = KM_PER_DEG * np.cos(np.radians(23.5))


def _segments(dip=90.0):
    # A north-south trace along lon 121 from lat 23 to 24, split in two, plus
    # an unrelated east-west segment far away.
    return pd.DataFrame(
        {
            "name": ["Fault_a", "Fault_b", "Other"],
            "lon1": [121.0, 121.0, 100.0],
            "lat1": [23.0, 23.5, 10.0],
            "lon2": [121.0, 121.0, 101.0],
            "lat2": [23.5, 24.0, 10.0],
            "dip": [dip, dip, 90.0],
        }
    )


def _stations():
    # On the trace; 0.05 deg east and west of it (~5.1 km at lat 23.5);
    # far away; beyond the northern endpoint.
    return pd.DataFrame(
        {
            "lon": [121.0, 121.05, 120.95, 125.0, 121.0],
            "lat": [23.5, 23.5, 23.5, 23.5, 24.05],
            "name": ["on", "east", "west", "far", "north_tip"],
            "east_vel": 0.0,
            "north_vel": 0.0,
            "east_sig": [2.0, 2.0, 2.0, 2.0, 2.0],
            "north_sig": [3.0, 3.0, 3.0, 3.0, 3.0],
            "flag": 1,
            "up_vel": 0.0,
            "up_sig": 4.0,
        }
    )


def _dipping_mesh_points(dip_east=True):
    """Planar fault under lon 121, lat 23-24, 45 deg dip, 0-10 km depth.

    Returns (lon, lat, depth) nodes and two triangles.
    """
    dlon = 10.0 / KM_PER_DEG_LON * (1 if dip_east else -1)
    points = np.array(
        [
            [121.0, 23.0, 0.0],
            [121.0, 24.0, 0.0],
            [121.0 + dlon, 24.0, -10.0],
            [121.0 + dlon, 23.0, -10.0],
        ]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    return points, cells


def _write_mesh(path, dip_east=True):
    points, cells = _dipping_mesh_points(dip_east)
    meshio.write(path, meshio.Mesh(points, [("triangle", cells)]), file_format="gmsh")
    return path


# --- segments ---------------------------------------------------------------


def test_select_segments_wildcards():
    seg = _segments()
    assert select_segments(seg, ["Fault_*"]).tolist() == [True, True, False]
    assert select_segments(seg, None).all()
    with pytest.raises(ValueError):
        select_segments(seg, ["Nothing*"])


def test_distance_to_segments():
    seg = _segments().iloc[:2]
    st = _stations()
    d, nearest = distance_to_segments_km(st.lon, st.lat, seg)
    expected_east = 0.05 * KM_PER_DEG_LON
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    assert d[1] == pytest.approx(expected_east, rel=1e-3)
    assert d[2] == pytest.approx(expected_east, rel=1e-3)
    # beyond the endpoint the distance is to the endpoint
    assert d[4] == pytest.approx(0.05 * KM_PER_DEG, rel=1e-6)
    assert nearest[4] == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_compass_side_ignores_segment_direction(reverse):
    seg = _segments().iloc[:2].copy()
    if reverse:
        seg[["lon1", "lat1", "lon2", "lat2"]] = seg[
            ["lon2", "lat2", "lon1", "lat1"]
        ].to_numpy()
    st = _stations().iloc[:3]
    _, nearest = distance_to_segments_km(st.lon, st.lat, seg)
    assert segment_on_side(st.lon, st.lat, seg, nearest, "east").tolist() == [
        True,
        True,
        False,
    ]
    assert segment_on_side(st.lon, st.lat, seg, nearest, "west").tolist() == [
        True,
        False,
        True,
    ]
    assert segment_on_side(st.lon, st.lat, seg, nearest, "both").all()


def test_compass_side_north_south():
    seg = _segments().iloc[[2]].reset_index(drop=True)
    lon, lat = np.array([100.5, 100.5]), np.array([10.1, 9.9])
    _, nearest = distance_to_segments_km(lon, lat, seg)
    assert segment_on_side(lon, lat, seg, nearest, "north").tolist() == [True, False]
    assert segment_on_side(lon, lat, seg, nearest, "south").tolist() == [False, True]
    with pytest.raises(ValueError, match="parallel"):
        segment_on_side(lon, lat, seg, nearest, "east")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("dip", "hanging"), [(45.0, "east"), (135.0, "west")])
def test_hanging_wall_follows_celeri_dip_convention(reverse, dip, hanging):
    # Slightly oblique trace, so the west-first endpoint order is well defined.
    # celeri: the fault dips to the right of the west->east strike for dip < 90.
    # West->east here is south->north, so dip 45 dips east.
    seg = pd.DataFrame(
        {
            "name": ["F"],
            "lon1": [121.0],
            "lat1": [23.0],
            "lon2": [121.01],
            "lat2": [24.0],
            "dip": [dip],
        }
    )
    if reverse:
        seg[["lon1", "lat1", "lon2", "lat2"]] = seg[
            ["lon2", "lat2", "lon1", "lat1"]
        ].to_numpy()
    lon, lat = np.array([121.06, 120.95]), np.array([23.5, 23.5])
    _, nearest = distance_to_segments_km(lon, lat, seg)
    hw = segment_on_side(lon, lat, seg, nearest, "hanging-wall")
    fw = segment_on_side(lon, lat, seg, nearest, "footwall")
    assert hw.tolist() == ([True, False] if hanging == "east" else [False, True])
    assert fw.tolist() == [not x for x in hw]


def test_hanging_wall_rejects_vertical_segments():
    seg = _segments(dip=90.0).iloc[:2]
    st = _stations().iloc[:3]
    _, nearest = distance_to_segments_km(st.lon, st.lat, seg)
    with pytest.raises(ValueError, match="vertical"):
        segment_on_side(st.lon, st.lat, seg, nearest, "hanging-wall")


# --- meshes -----------------------------------------------------------------


def test_closest_points_on_triangles_regions():
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    p = np.array(
        [
            [0.2, 0.2, 1.0],  # above the interior
            [-1.0, -1.0, 0.0],  # vertex a
            [2.0, -0.5, 0.0],  # vertex b
            [-0.5, 2.0, 0.0],  # vertex c
            [0.5, -1.0, 0.0],  # edge ab
            [-1.0, 0.5, 0.0],  # edge ac
            [1.0, 1.0, 0.0],  # edge bc
        ]
    )
    n = len(p)
    q = closest_points_on_triangles(
        p, np.tile(a, (n, 1)), np.tile(b, (n, 1)), np.tile(c, (n, 1))
    )
    expected = [[0.2, 0.2, 0], a, b, c, [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]]
    assert_allclose(q, expected, atol=1e-12)


def test_read_mesh_triangles(tmp_path):
    triangles = read_mesh_triangles(_write_mesh(tmp_path / "f.msh"))
    assert triangles.shape == (2, 3, 3)
    radius = np.linalg.norm(triangles, axis=-1)
    assert radius.max() == pytest.approx(RADIUS_EARTH_KM)
    assert radius.min() == pytest.approx(RADIUS_EARTH_KM - 10.0)


def test_read_mesh_rejects_positive_depths(tmp_path):
    points, cells = _dipping_mesh_points()
    points[:, 2] *= -1
    path = tmp_path / "bad.msh"
    meshio.write(path, meshio.Mesh(points, [("triangle", cells)]), file_format="gmsh")
    with pytest.raises(ValueError, match="negative below the surface"):
        read_mesh_triangles(path)


def test_distance_to_dipping_mesh(tmp_path):
    triangles = read_mesh_triangles(_write_mesh(tmp_path / "f.msh"))
    # On the trace; 5 km east (above the 45 deg fault: 5 sin 45 km away);
    # 5 km west (footwall: 5 km to the top edge); far away.
    lon = np.array(
        [121.0, 121.0 + 5 / KM_PER_DEG_LON, 121.0 - 5 / KM_PER_DEG_LON, 125.0]
    )
    lat = np.full(4, 23.5)
    d, nearest, closest = distance_to_meshes_km(lon, lat, triangles, reach_km=20.0)
    # the flat 111 km top edge sags 0.24 km below the surface at lat 23.5
    sag = RADIUS_EARTH_KM * (1 - np.cos(np.radians(0.5)))
    assert d[0] == pytest.approx(sag, rel=1e-2)
    assert d[1] == pytest.approx(5 / np.sqrt(2), abs=sag)
    assert d[2] == pytest.approx(5.0, abs=sag)
    assert d[3] > 20.0
    # the reach only limits which distances are exact
    d_all, _, _ = distance_to_meshes_km(lon[:3], lat[:3], triangles, reach_km=np.inf)
    assert_allclose(d_all, d[:3], atol=1e-9)

    hw = mesh_on_side(
        lon[:3], lat[:3], triangles, nearest[:3], closest[:3], "hanging-wall"
    )
    fw = mesh_on_side(lon[:3], lat[:3], triangles, nearest[:3], closest[:3], "footwall")
    assert hw.tolist() == [True, True, False]
    # the on-trace station is 0.24 km above the sagging flat edge: hanging wall
    assert fw.tolist() == [False, False, True]
    east = mesh_on_side(lon[:3], lat[:3], triangles, nearest[:3], closest[:3], "east")
    assert east.tolist() == [True, True, False]


def test_mesh_hanging_wall_west_dipping(tmp_path):
    triangles = read_mesh_triangles(_write_mesh(tmp_path / "f.msh", dip_east=False))
    lon = np.array([121.0 + 5 / KM_PER_DEG_LON, 121.0 - 5 / KM_PER_DEG_LON])
    lat = np.full(2, 23.5)
    _, nearest, closest = distance_to_meshes_km(lon, lat, triangles, reach_km=20.0)
    assert mesh_on_side(
        lon, lat, triangles, nearest, closest, "hanging-wall"
    ).tolist() == [False, True]
    assert mesh_on_side(lon, lat, triangles, nearest, closest, "east").tolist() == [
        True,
        False,
    ]


# --- weights and sigmas -----------------------------------------------------


def test_linear_weights():
    assert_allclose(distance_weights([0, 5, 10, 20], 11, "linear", 10), [11, 6, 1, 1])
    with pytest.raises(ValueError):
        distance_weights([0], 11, "linear", 0)
    with pytest.raises(ValueError):
        distance_weights([0], 0.5, "linear", 10)


@pytest.mark.parametrize("taper", ["linear", "gaussian", "exponential", "cosine"])
def test_tapers_hit_max_and_one(taper):
    d = np.linspace(0, 60, 601)
    w = distance_weights(d, 50, taper, 5.0)
    assert w[0] == pytest.approx(50)
    assert np.all(np.diff(w) <= 1e-12)  # non-increasing
    cutoff = {"linear": 5, "gaussian": 15, "exponential": 25, "cosine": 5}[taper]
    assert np.all(w[d >= cutoff] == 1)
    assert np.all(w[d < cutoff] > 1)


def test_scale_sigmas():
    st = _stations()
    w = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
    out = scale_sigmas(st, w)
    assert out.east_sig.tolist() == [1.0, 2.0, 2.0, 2.0, 2.0]
    assert out.north_sig[0] == pytest.approx(1.5)
    assert out.up_sig[0] == pytest.approx(2.0)
    uni = scale_sigmas(st, w, uniform_base=True)
    assert uni.east_sig.tolist() == [0.5, 1.0, 1.0, 1.0, 1.0]
    assert st.east_sig[0] == 2.0  # input untouched


def test_fault_distances_needs_one_source():
    with pytest.raises(ValueError):
        fault_distances(_stations())


# --- command line -----------------------------------------------------------


def test_main_segments(tmp_path):
    station_file = tmp_path / "stations.csv"
    segment_file = tmp_path / "segments.csv"
    _stations().to_csv(station_file, index=False)
    _segments().to_csv(segment_file, index=False)
    out = tmp_path / "out.csv"
    main(
        [
            str(station_file),
            str(out),
            "--segment-file",
            str(segment_file),
            "--segment-names",
            "Fault_*",
            "--max-weight",
            "100",
            "--width-km",
            "10",
            "--side",
            "east",
            "--uniform-base",
        ]
    )
    res = pd.read_csv(out)
    src = _stations()
    assert len(res) == len(src)
    sig = ["east_sig", "north_sig", "up_sig"]
    assert res.drop(columns=sig).equals(src.drop(columns=sig))
    assert res.east_sig[0] == pytest.approx(0.1)  # on the trace: w = 100
    assert 0.1 < res.east_sig[1] < 1.0  # east, within 10 km
    assert res.east_sig[2] == 1.0  # west: not weighted
    assert res.east_sig[3] == 1.0  # far


def test_main_meshes(tmp_path):
    station_file = tmp_path / "stations.csv"
    _stations().to_csv(station_file, index=False)
    mesh = _write_mesh(tmp_path / "f.msh")
    out = tmp_path / "out.csv"
    main(
        [
            str(station_file),
            str(out),
            "--mesh-files",
            str(mesh),
            "--max-weight",
            "4",
            "--taper",
            "gaussian",
            "--width-km",
            "2",
            "--side",
            "hanging-wall",
        ]
    )
    res = pd.read_csv(out)
    # on the trace: w ~ 4 (sig 2 -> ~1); not exactly, because of the edge sag
    assert res.east_sig[0] == pytest.approx(1.0, rel=1e-2)
    assert res.north_sig[0] == pytest.approx(1.5, rel=1e-2)
    assert 1.0 < res.east_sig[1] < 2.0  # hanging wall, ~3.6 km from the fault
    assert res.east_sig[2] == 2.0  # footwall
    assert res.east_sig[3] == 2.0  # far


def test_main_argument_errors(tmp_path):
    station_file = tmp_path / "stations.csv"
    _stations().to_csv(station_file, index=False)
    mesh = str(_write_mesh(tmp_path / "f.msh"))
    with pytest.raises(SystemExit):  # no fault source
        main([str(station_file), str(tmp_path / "o.csv")])
    with pytest.raises(SystemExit):  # output overwrites input
        main([str(station_file), str(station_file), "--mesh-files", mesh])
    with pytest.raises(SystemExit):  # names without a segment file
        main(
            [
                str(station_file),
                str(tmp_path / "o.csv"),
                "--mesh-files",
                mesh,
                "--segment-names",
                "F*",
            ]
        )


def test_main_default_output_name_and_no_overwrite(tmp_path):
    station_file = tmp_path / "stations.csv"
    segment_file = tmp_path / "segments.csv"
    _stations().to_csv(station_file, index=False)
    _segments().to_csv(segment_file, index=False)
    args = [str(station_file), "--segment-file", str(segment_file)]
    args += ["--segment-names", "Fault_*", "--max-weight", "100", "--side", "east"]
    main([*args, "--uniform-base"])
    out = tmp_path / "stations_w100_linear10km_east_u.csv"
    assert out.exists()
    main([*args, "--taper", "gaussian", "--width-km", "5"])
    assert (tmp_path / "stations_w100_gaussian5km_east.csv").exists()
    # an existing output is never replaced without --overwrite
    before = out.read_text()
    with pytest.raises(SystemExit):
        main([*args, "--uniform-base"])
    assert out.read_text() == before
    main([*args, "--uniform-base", "--overwrite"])


def test_cutoff_is_not_an_option(tmp_path):
    station_file = tmp_path / "stations.csv"
    _stations().to_csv(station_file, index=False)
    mesh = str(_write_mesh(tmp_path / "f.msh"))
    with pytest.raises(SystemExit):
        main([str(station_file), "--mesh-files", mesh, "--cutoff-km", "8"])
