#!/usr/bin/env python3
r"""Weight stations near chosen faults by shrinking their velocity sigmas.

Reads a station file and writes a copy in which every station within the
cutoff distance of the chosen faults has ``east_sig``/``north_sig``/``up_sig``
divided by ``sqrt(w(d))``, so its weight ``1/sig^2`` is multiplied by
``w(d)``. Every station keeps a single row; nothing else in the file changes.

The faults are given either as a segment file (distance to the surface trace
of the selected segments) or as one or more Gmsh ``.msh`` meshes (3D distance
from the station, at the surface, to the nearest triangle). A station above a
shallow-dipping mesh is therefore "near" the fault even when it is far from
the trace.

The weight is ``max_weight`` on the fault and decays to 1 at a cutoff C:

    w(d) = 1 + (max_weight - 1) * (f(d) - f(C)) / (1 - f(C))    for d < C
    w(d) = 1                                                    for d >= C

with taper shape f and width L (``--width-km``):

    linear       f(d) = max(0, 1 - d / L)
    gaussian     f(d) = exp(-d^2 / (2 L^2))
    exponential  f(d) = exp(-d / L)
    cosine       f(d) = (1 + cos(pi d / L)) / 2 for d < L, else 0

The rescaling makes w exactly ``max_weight`` at d = 0 and exactly 1 at the
cutoff C, so the weight has no jump. C is fixed by the taper: L (linear,
cosine, which already reach 0 there), 3 L (gaussian) and 5 L (exponential),
where the gaussian and exponential tails are ~1% and the rescaling barely
changes the curve.

With ``--side`` only stations on one side of the fault are weighted; the
others keep weight 1. ``hanging-wall`` / ``footwall`` use the segment dip (or
the mesh geometry) and fail for vertical faults. ``east`` / ``west`` /
``north`` / ``south`` use the local compass direction and suit faults that
strike roughly north-south or east-west.

The output is meant for ``"mcmc_station_weighting": "station_sigma"``, where
the MCMC likelihood reads the station sigmas.

Examples
--------
    celeri-distance-weight-vels station.csv \
        --segment-file segment.csv --segment-names "Longitudinal_Valley*" \
        --max-weight 100 --width-km 10 --side hanging-wall
    # writes station_w100_linear10km_hanging-wall.csv next to station.csv

    celeri-distance-weight-vels station.csv station_weighted.csv \
        --mesh-files subduction.msh crustal.msh \
        --max-weight 10 --taper gaussian --width-km 5
"""

import argparse
import os
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RADIUS_EARTH_KM = 6371.0
SIGMA_COLUMNS = ("east_sig", "north_sig", "up_sig")
SIDES = ("both", "hanging-wall", "footwall", "east", "west", "north", "south")
TAPERS = ("linear", "gaussian", "exponential", "cosine")
CUTOFF_WIDTHS = {
    "linear": 1.0,
    "gaussian": 3.0,
    "exponential": 5.0,
    "cosine": 1.0,
}
# celeri treats dips within 0.05 degrees of 90 as vertical (model.snap_vertical_segment_dips)
VERTICAL_TOLERANCE_DEG = 0.05
_MAX_PAIRS_PER_CHUNK = 2_000_000  # (station, segment/triangle) pairs evaluated at once
_ON_FAULT_TOLERANCE_KM = 1e-6  # points this close to the fault count as on either side
_PARALLEL_TOLERANCE = (
    1e-9  # |normal . compass| below this: fault parallel to the direction
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def unit_vectors(lon, lat):
    """Unit position vectors (..., 3) for longitude/latitude in degrees."""
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _angle(u, v):
    """Angle (radians) between unit vectors along the last axis."""
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.sum(u * v, axis=-1))


def _local_direction(position, direction: str):
    """Local unit vector ('up', 'east' or 'north') at unit/ECEF positions."""
    p = position / np.linalg.norm(position, axis=-1, keepdims=True)
    if direction == "up":
        return p
    lon = np.arctan2(p[..., 1], p[..., 0])
    lat = np.arcsin(np.clip(p[..., 2], -1.0, 1.0))
    if direction == "east":
        return np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    if direction == "north":
        return np.stack(
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            axis=-1,
        )
    raise ValueError(f"unknown direction {direction!r}")


def _compass(side: str):
    """(local direction, sign) for a compass side."""
    return {
        "east": ("east", 1.0),
        "west": ("east", -1.0),
        "north": ("north", 1.0),
        "south": ("north", -1.0),
    }[side]


# ---------------------------------------------------------------------------
# Segments: distance to the surface trace
# ---------------------------------------------------------------------------


def select_segments(segment: pd.DataFrame, patterns=None) -> np.ndarray:
    """Boolean mask of segments whose name matches any wildcard pattern.

    With no patterns every segment is selected.
    """
    if not patterns:
        return np.ones(len(segment), dtype=bool)
    names = segment["name"].astype(str).str.strip()
    mask = np.array([any(fnmatchcase(n, p) for p in patterns) for n in names])
    if not mask.any():
        raise ValueError(f"No segment name matches {list(patterns)}")
    return mask


def _segment_geometry(segment: pd.DataFrame):
    """Endpoints, great-circle normal and degenerate flag of each segment.

    The normal is ``cross(west endpoint, east endpoint)``, i.e. the endpoints
    are ordered as in ``celeri.model.order_endpoints_sphere``, so that the
    down-dip side of a segment is ``-normal`` for dip < 90 and ``+normal`` for
    dip > 90, as in celeri.
    """
    a = unit_vectors(segment.lon1.to_numpy(), segment.lat1.to_numpy())
    b = unit_vectors(segment.lon2.to_numpy(), segment.lat2.to_numpy())
    normal = np.cross(a, b)
    length = np.linalg.norm(normal, axis=-1)
    degenerate = length < 1e-12  # zero-length segment: endpoints only
    normal = normal / np.where(degenerate, 1.0, length)[:, None]
    normal = np.where(normal[:, 2:3] < 0, -normal, normal)
    return a, b, normal, degenerate


def distance_to_segments_km(lon, lat, segment: pd.DataFrame):
    """Great-circle distance (km) from each point to the nearest segment trace.

    Each segment is the minor arc from (lon1, lat1) to (lon2, lat2): the
    distance is to its great circle where the foot of the perpendicular falls
    on the arc, and to the nearer endpoint otherwise.

    Returns
    -------
    distance_km : (n,) array
    nearest : (n,) int array, row position in ``segment`` of the nearest segment
    """
    p = unit_vectors(lon, lat).reshape(-1, 3)
    a, b, normal, degenerate = _segment_geometry(segment)
    chunk = max(1, _MAX_PAIRS_PER_CHUNK // max(len(a), 1))
    angle = np.empty(len(p))
    nearest = np.empty(len(p), dtype=int)
    for start in range(0, len(p), chunk):
        pc = p[start : start + chunk, None, :]
        to_endpoints = np.minimum(_angle(pc, a[None]), _angle(pc, b[None]))
        along = np.sum(pc * normal[None], axis=-1)
        foot = pc - along[..., None] * normal[None]
        # the foot lies on the arc when it is between a and b along the circle
        # (sign-independent: compare against the circle's own orientation)
        circle = np.cross(a, b)[None]
        on_arc = (
            (np.sum(np.cross(a[None], foot) * circle, axis=-1) >= 0)
            & (np.sum(np.cross(foot, b[None]) * circle, axis=-1) >= 0)
            & ~degenerate[None]
        )
        to_circle = np.arcsin(np.clip(np.abs(along), 0.0, 1.0))
        d = np.where(on_arc, to_circle, to_endpoints)
        nearest[start : start + chunk] = d.argmin(1)
        angle[start : start + chunk] = d.min(1)
    return angle * RADIUS_EARTH_KM, nearest


def segment_on_side(lon, lat, segment: pd.DataFrame, nearest, side: str) -> np.ndarray:
    """True for points on the requested side of their nearest segment.

    The side is the sign of the point's component along the segment's
    great-circle normal. For "hanging-wall"/"footwall" the normal is oriented
    down dip using the segment ``dip`` column (celeri convention: the fault
    dips to the right of the west-to-east strike for dip < 90, to the left
    for dip > 90). For a compass side it is oriented with the local east (or
    north) unit vector at the segment midpoint. Points on the trace count as
    on the side.
    """
    p = unit_vectors(lon, lat).reshape(-1, 3)
    if side == "both":
        return np.ones(len(p), dtype=bool)
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    a, b, normal, degenerate = _segment_geometry(segment)
    if np.any(degenerate[nearest]):
        raise ValueError("Cannot assign a side for a zero-length segment")

    if side in ("hanging-wall", "footwall"):
        if "dip" not in segment:
            raise ValueError(f"side={side!r} needs a 'dip' column in the segment file")
        dip = segment["dip"].to_numpy(dtype=float)
        vertical = np.abs(dip - 90.0) <= VERTICAL_TOLERANCE_DEG
        if np.any(vertical[nearest]):
            names = sorted(
                set(segment["name"].astype(str).to_numpy()[nearest][vertical[nearest]])
            )
            raise ValueError(
                f"side={side!r} is undefined for vertical segments: {', '.join(names)}"
            )
        toward = -np.sign(np.cos(np.radians(dip)))  # +1: down dip is +normal
        if side == "footwall":
            toward = -toward
    else:
        mid = a + b
        direction, sign = _compass(side)
        component = np.sum(normal * _local_direction(mid, direction), axis=-1)
        toward = sign * np.where(
            np.abs(component) < _PARALLEL_TOLERANCE, 0.0, np.sign(component)
        )
        if np.any(toward[nearest] == 0):
            other = "east/west" if direction == "north" else "north/south"
            raise ValueError(
                f"side={side!r} is undefined for a segment parallel to that "
                f"direction; use {other} or hanging-wall/footwall"
            )
    n = normal[nearest] * toward[nearest, None]
    return np.sum(p * n, axis=-1) >= -_ON_FAULT_TOLERANCE_KM / RADIUS_EARTH_KM


# ---------------------------------------------------------------------------
# Meshes: 3D distance to the fault surface
# ---------------------------------------------------------------------------


def read_mesh_triangles(filename) -> np.ndarray:
    """Triangles (n, 3, 3) of a Gmsh mesh as Earth-centred coordinates in km.

    The mesh nodes are (lon, lat, depth) with depth in km, negative below the
    surface, as in celeri. Elements are flat, so a long surface edge sags below
    the curved Earth (about 0.24 km at the middle of a 111 km edge, 0.1 m for a
    2 km edge); distances near the trace include that sag.
    """
    import meshio

    old_stdout = sys.stdout  # meshio prints a newline when reading .msh
    sys.stdout = Path(os.devnull).open("w")
    try:
        mesh = meshio.read(filename)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    points = np.asarray(mesh.points, dtype=float)
    if points[:, 2].min() >= 0.0 and points[:, 2].max() > 0.0:
        raise ValueError(
            f"Mesh {filename} has only non-negative depths; celeri expects mesh "
            "depths in km, negative below the surface"
        )
    cells = np.asarray(mesh.get_cells_type("triangle"))
    if cells.size == 0:
        raise ValueError(f"Mesh {filename} has no triangles")
    xyz = (RADIUS_EARTH_KM + points[:, 2])[:, None] * unit_vectors(
        points[:, 0], points[:, 1]
    )
    return xyz[cells]


def closest_points_on_triangles(p, a, b, c):
    """Closest point on each triangle (a, b, c) to each point p, all (m, 3).

    Vectorized version of the region test in Ericson, Real-Time Collision
    Detection (2005), section 5.1.5.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.sum(ab * ap, -1)
    d2 = np.sum(ac * ap, -1)
    bp = p - b
    d3 = np.sum(ab * bp, -1)
    d4 = np.sum(ac * bp, -1)
    cp = p - c
    d5 = np.sum(ab * cp, -1)
    d6 = np.sum(ac * cp, -1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = vb / denom
        w = vc / denom
        out = a + v[:, None] * ab + w[:, None] * ac  # interior

        # edges (later assignments take precedence over earlier ones)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        out = np.where(m[:, None], b + t_bc[:, None] * (c - b), out)
        t_ac = d2 / (d2 - d6)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out = np.where(m[:, None], a + t_ac[:, None] * ac, out)
        t_ab = d1 / (d1 - d3)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out = np.where(m[:, None], a + t_ab[:, None] * ab, out)

    # vertices
    out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, out)
    return out


def distance_to_meshes_km(lon, lat, triangles: np.ndarray, reach_km: float):
    """3D distance (km) from surface points to the nearest mesh triangle.

    Only triangles that could be within ``reach_km`` are tested, found with a
    KD-tree on the triangle centroids. Distances up to ``reach_km`` are
    exact. A point with no triangle within ``reach_km`` gets a distance
    larger than ``reach_km`` (an upper bound, not the exact value).

    Returns
    -------
    distance_km : (n,) array
    nearest : (n,) int array, index of the nearest triangle
    closest : (n, 3) array, closest point on that triangle (km, Earth-centred)
    """
    p = RADIUS_EARTH_KM * unit_vectors(lon, lat).reshape(-1, 3)
    centroid = triangles.mean(axis=1)
    r_max = np.linalg.norm(triangles - centroid[:, None, :], axis=-1).max()
    tree = cKDTree(centroid)
    d_centroid, nearest = tree.query(p)
    closest = centroid[nearest].copy()
    distance = d_centroid.copy()  # upper bound: the centroid is on the triangle

    # A triangle within min(d_centroid, reach) of p has its centroid within
    # that distance + r_max, so no closer triangle is missed.
    radius = np.minimum(d_centroid, reach_km) + r_max
    todo = np.flatnonzero(d_centroid - r_max <= reach_km)
    candidates = tree.query_ball_point(p[todo], radius[todo], return_sorted=False)
    counts = np.array([len(c) for c in candidates], dtype=int)
    pair_point = np.repeat(todo, counts)
    pair_tri = (
        np.concatenate(candidates).astype(int) if counts.sum() else np.zeros(0, int)
    )

    for start in range(0, len(pair_point), _MAX_PAIRS_PER_CHUNK):
        ip = pair_point[start : start + _MAX_PAIRS_PER_CHUNK]
        it = pair_tri[start : start + _MAX_PAIRS_PER_CHUNK]
        tri = triangles[it]
        q = closest_points_on_triangles(p[ip], tri[:, 0], tri[:, 1], tri[:, 2])
        d = np.linalg.norm(p[ip] - q, axis=-1)
        # per point, keep the pair with the smallest distance
        order = np.lexsort((d, ip))
        ip, it, q, d = ip[order], it[order], q[order], d[order]
        first = np.unique(ip, return_index=True)[1]
        ip, it, q, d = ip[first], it[first], q[first], d[first]
        better = d < distance[ip]
        ip, it, q, d = ip[better], it[better], q[better], d[better]
        distance[ip] = d
        nearest[ip] = it
        closest[ip] = q
    return distance, nearest, closest


def mesh_on_side(lon, lat, triangles, nearest, closest, side: str) -> np.ndarray:
    """True for surface points on the requested side of their nearest triangle.

    The triangle normal is oriented up (toward the hanging wall) for
    "hanging-wall"/"footwall", or along local east/north for a compass side;
    a point is on the side when its offset from the closest point on the
    triangle has a non-negative component along that normal.
    """
    p = RADIUS_EARTH_KM * unit_vectors(lon, lat).reshape(-1, 3)
    if side == "both":
        return np.ones(len(p), dtype=bool)
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    tri = triangles[nearest]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    centroid = tri.mean(axis=1)
    if side in ("hanging-wall", "footwall"):
        component = np.sum(normal * _local_direction(centroid, "up"), axis=-1)
        if np.any(np.abs(component) < np.sin(np.radians(VERTICAL_TOLERANCE_DEG))):
            raise ValueError(
                f"side={side!r} is undefined where the nearest mesh element is vertical"
            )
        sign = 1.0 if side == "hanging-wall" else -1.0
    else:
        direction, sign = _compass(side)
        component = np.sum(normal * _local_direction(centroid, direction), axis=-1)
        if np.any(np.abs(component) < _PARALLEL_TOLERANCE):
            raise ValueError(f"side={side!r} is undefined for some mesh elements")
    normal = normal * (sign * np.sign(component))[:, None]
    return np.sum((p - closest) * normal, axis=-1) >= -_ON_FAULT_TOLERANCE_KM


# ---------------------------------------------------------------------------
# Weights and sigmas
# ---------------------------------------------------------------------------


def taper_function(distance_km, taper: str, width_km: float):
    """Unscaled taper f(d): 1 at d = 0, decreasing toward 0."""
    d = np.asarray(distance_km, dtype=float)
    if taper == "linear":
        return np.clip(1.0 - d / width_km, 0.0, 1.0)
    if taper == "gaussian":
        return np.exp(-0.5 * (d / width_km) ** 2)
    if taper == "exponential":
        return np.exp(-d / width_km)
    if taper == "cosine":
        return np.where(
            d < width_km,
            0.5 * (1.0 + np.cos(np.pi * np.minimum(d, width_km) / width_km)),
            0.0,
        )
    raise ValueError(f"taper must be one of {TAPERS}, got {taper!r}")


def cutoff_km(taper: str, width_km: float) -> float:
    """Distance where the weight of a taper reaches 1."""
    return CUTOFF_WIDTHS[taper] * width_km


def distance_weights(distance_km, max_weight: float, taper: str, width_km: float):
    """Weight ``max_weight`` at d = 0, tapering to exactly 1 at the cutoff."""
    if width_km <= 0:
        raise ValueError("width_km must be positive")
    if max_weight < 1:
        raise ValueError("max_weight must be >= 1")
    cutoff = cutoff_km(taper, width_km)
    d = np.asarray(distance_km, dtype=float)
    f_cut = float(taper_function(cutoff, taper, width_km))
    ramp = (taper_function(d, taper, width_km) - f_cut) / (1.0 - f_cut)
    ramp = np.where(d < cutoff, np.clip(ramp, 0.0, 1.0), 0.0)
    return 1.0 + (max_weight - 1.0) * ramp


def scale_sigmas(
    station: pd.DataFrame, weight, uniform_base: bool = False
) -> pd.DataFrame:
    """Copy of the station table with sigmas divided by sqrt(weight).

    With ``uniform_base`` every sigma is first set to 1, so the weight is the
    only difference between stations.
    """
    out = station.copy()
    for column in SIGMA_COLUMNS:
        if column not in out:
            continue
        base = np.ones(len(out)) if uniform_base else out[column].to_numpy(dtype=float)
        out[column] = base / np.sqrt(weight)
    return out


def fault_distances(
    station, *, segment=None, triangles=None, side="both", reach_km=np.inf
):
    """Distance (km) to the faults and side flag for each station."""
    lon = station.lon.to_numpy()
    lat = station.lat.to_numpy()
    if (segment is None) == (triangles is None):
        raise ValueError("give exactly one of segment or triangles")
    if segment is not None:
        distance, nearest = distance_to_segments_km(lon, lat, segment)
        selected = segment_on_side(lon, lat, segment, nearest, side)
    else:
        distance, nearest, closest = distance_to_meshes_km(
            lon, lat, triangles, reach_km
        )
        selected = mesh_on_side(lon, lat, triangles, nearest, closest, side)
    return distance, selected


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="celeri-distance-weight-vels",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    parser.add_argument("input_vel_file", help="input station CSV")
    parser.add_argument(
        "output_vel_file",
        nargs="?",
        default=None,
        help="output station CSV (default: named from the options, next to the "
        "input, e.g. stations_w100_linear10km_east.csv)",
    )
    faults = parser.add_mutually_exclusive_group(required=True)
    faults.add_argument("--segment-file", help="segment CSV with the fault traces")
    faults.add_argument("--mesh-files", nargs="+", help="Gmsh .msh fault meshes")
    parser.add_argument(
        "--segment-names",
        nargs="+",
        default=None,
        help='segment-name wildcards, e.g. "Longitudinal_Valley*" (default: all segments)',
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=10.0,
        help="weight on the fault (default 10)",
    )
    parser.add_argument(
        "--taper", choices=TAPERS, default="linear", help="taper shape (default linear)"
    )
    parser.add_argument(
        "--width-km",
        type=float,
        default=10.0,
        help="taper width L in km (default 10); the weight reaches 1 at L "
        "(linear, cosine), 3L (gaussian) or 5L (exponential)",
    )
    parser.add_argument(
        "--side",
        choices=SIDES,
        default="both",
        help="weight only stations on this side of the fault (default both)",
    )
    parser.add_argument(
        "--uniform-base",
        action="store_true",
        help="set every sigma to 1/sqrt(w) instead of scaling the file's sigmas",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    args = parser.parse_args(argv)
    if args.max_weight < 1:
        parser.error("--max-weight must be >= 1")
    if args.width_km <= 0:
        parser.error("--width-km must be positive")
    if args.segment_names and not args.segment_file:
        parser.error("--segment-names needs --segment-file")
    if args.output_vel_file is None:
        args.output_vel_file = str(default_output_path(args))
    if Path(args.output_vel_file).resolve() == Path(args.input_vel_file).resolve():
        parser.error("output_vel_file must differ from input_vel_file")
    if Path(args.output_vel_file).exists() and not args.overwrite:
        parser.error(
            f"{args.output_vel_file} already exists; pass --overwrite to replace it"
        )
    return parser, args


def default_output_path(args) -> Path:
    """Output name built from the options, next to the input file.

    ``<stem>_w<max>_<taper><width>km[_<side>][_u].csv``; ``_u`` marks
    ``--uniform-base``.
    """
    src = Path(args.input_vel_file)
    tag = f"_w{args.max_weight:g}_{args.taper}{args.width_km:g}km"
    if args.side != "both":
        tag += f"_{args.side}"
    if args.uniform_base:
        tag += "_u"
    return src.with_name(src.stem + tag + ".csv")


def main(argv=None):
    parser, args = _parse_args(argv)
    cutoff = cutoff_km(args.taper, args.width_km)

    station = pd.read_csv(args.input_vel_file)
    missing = [c for c in ("lon", "lat", *SIGMA_COLUMNS[:2]) if c not in station]
    if missing:
        parser.error(f"station file has no {', '.join(missing)} column")
    if not args.uniform_base:
        sig = station[[c for c in SIGMA_COLUMNS if c in station]].to_numpy(dtype=float)
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            parser.error(
                "station sigmas must be finite and positive (or pass --uniform-base)"
            )

    try:
        if args.segment_file:
            segment = pd.read_csv(args.segment_file)
            trace = segment[select_segments(segment, args.segment_names)].reset_index(
                drop=True
            )
            distance, selected = fault_distances(station, segment=trace, side=args.side)
            source = f"{len(trace)} segments"
        else:
            triangles = np.concatenate(
                [read_mesh_triangles(f) for f in args.mesh_files]
            )
            distance, selected = fault_distances(
                station, triangles=triangles, side=args.side, reach_km=cutoff
            )
            source = f"{len(args.mesh_files)} mesh(es), {len(triangles)} triangles"
    except ValueError as err:
        parser.error(str(err))

    weight = distance_weights(distance, args.max_weight, args.taper, args.width_km)
    weight = np.where(selected, weight, 1.0)
    out_station = scale_sigmas(station, weight, args.uniform_base)
    out_station.to_csv(args.output_vel_file, index=False)

    near = distance < cutoff
    print(
        f"{source}; {int(near.sum())} of {len(station)} stations within {cutoff:g} km"
    )
    if args.side != "both":
        print(
            f"side={args.side}: {int((near & selected).sum())} weighted, "
            f"{int((near & ~selected).sum())} stations on the other side left at weight 1"
        )
    print(
        f"{args.taper} taper, L = {args.width_km:g} km; weights 1 to {weight.max():.4g}"
    )
    print(f"wrote {args.output_vel_file} ({len(out_station)} rows)")


if __name__ == "__main__":
    main()
