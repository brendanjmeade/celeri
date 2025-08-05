#!/usr/bin/env python3
# TODO (Adrian): Adapt to model refactor

import argparse
import json
from pathlib import Path

import addict
import meshio
import numpy as np
import pandas as pd
import pyproj
from loguru import logger
from scipy.spatial.distance import cdist

import celeri

# Global constants
GEOID = pyproj.Geod(ellps="WGS84")
KM2M = 1.0e3
M2MM = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)
DEG_PER_MYR_TO_RAD_PER_YR = 1 / 1e6  # TODO: What should this conversion be?
N_MESH_DIM = 3


def sph2cart(lon, lat, radius):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def snap_segments(segment, meshes):
    """Replace segments tracing meshes with the actual top edges of those meshes"""
    # For each mesh, find associated segments
    cut_segment_idx = []
    all_edge_segment = make_default_segment(0)
    for i in range(len(meshes)):
        these_segments = np.where(
            (segment.mesh_flag != 0) & (segment.mesh_file_index == i + 1)
        )[0]
        cut_segment_idx = np.append(cut_segment_idx, these_segments)
        # Get top coordinates of the mesh
        top_el_indices = np.where(meshes[i].top_elements)
        edges = np.sort(meshes[i].ordered_edge_nodes[:-1], axis=1)
        top_verts = np.sort(meshes[i].verts[top_el_indices], axis=1)
        # Concatenate edges with vertex pairs
        edges1 = np.vstack((edges, top_verts[:, 0:2]))
        # Find unique edges
        unique_edges1, unique_indices1, unique_counts1 = np.unique(
            edges1, axis=0, return_index=True, return_counts=True
        )
        # But keep those edges that appear twice
        top_edge_indices1 = unique_indices1[np.where(unique_counts1 == 2)]
        # Same process with 2nd and 3rd columns of the mesh vertex array
        edges2 = np.vstack((edges, top_verts[:, 1:3]))
        unique_edges2, unique_indices2, unique_counts2 = np.unique(
            edges2, axis=0, return_index=True, return_counts=True
        )
        top_edge_indices2 = unique_indices2[np.where(unique_counts2 == 2)]
        # Final selection
        top_edge_indices = np.sort(np.hstack((top_edge_indices1, top_edge_indices2)))
        # Get new segment coordinates from these indices
        edge_segs = make_default_segment(len(top_edge_indices))
        edge_segs.lon1 = meshes[i].meshio_object.points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 0], 0
        ]
        edge_segs.lat1 = meshes[i].meshio_object.points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 0], 1
        ]
        edge_segs.lon2 = meshes[i].meshio_object.points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 1], 0
        ]
        edge_segs.lat2 = meshes[i].meshio_object.points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 1], 1
        ]
        edge_segs.locking_depth = -15
        edge_segs.mesh_flag = +1
        edge_segs.mesh_file_index = +i + 1
        all_edge_segment = all_edge_segment.append(edge_segs)

    # Get indices of segments to keep
    keep_segment_idx = np.setdiff1d(range(len(segment.lon1)), cut_segment_idx)
    # Isolate kept segments and reindex
    keep_segment = segment.loc[keep_segment_idx]
    new_index = range(len(keep_segment_idx))
    keep_segment.index = new_index
    # Find hanging endpoints; these mark terminations of mesh-replaced segments
    lons = np.hstack((keep_segment.lon1, keep_segment.lon2))
    lats = np.hstack((keep_segment.lat1, keep_segment.lat2))
    coords = np.array([lons, lats])
    unique_coords, indices, counts = np.unique(
        coords, axis=1, return_index=True, return_counts=True
    )
    hanging_idx = indices[np.where(counts == 1)]
    # Calculate distance to all mesh edge coordinates
    # Can't just use the terminations because we might have triple junctions in the middle of a mesh
    elons = np.hstack((all_edge_segment.lon1, all_edge_segment.lon2))
    elats = np.hstack((all_edge_segment.lat1, all_edge_segment.lat2))
    ecoords = np.array([elons, elats])
    hang_to_mesh_dist = cdist(coords[:, hanging_idx].T, ecoords.T)
    # Find closest edge coordinate
    closest_edge_idx = np.argmin(hang_to_mesh_dist, axis=1)
    # Replace segment coordinates with closest mesh coordinate
    # Using a loop because we need to evaluate whether to replace endpoint 1 or 2
    for i in range(len(closest_edge_idx)):
        if hanging_idx[i] < len(keep_segment.lon1):
            keep_segment.loc[hanging_idx[i], "lon1"] = ecoords[0, closest_edge_idx[i]]
            keep_segment.loc[hanging_idx[i], "lat1"] = ecoords[1, closest_edge_idx[i]]
        else:
            keep_segment.loc[hanging_idx[i] - len(keep_segment.lon1), "lon2"] = ecoords[
                0, closest_edge_idx[i]
            ]
            keep_segment.loc[hanging_idx[i] - len(keep_segment.lon1), "lat2"] = ecoords[
                1, closest_edge_idx[i]
            ]
    # Merge with mesh edge segments
    new_segment = keep_segment.append(all_edge_segment)
    new_index = range(len(new_segment))
    new_segment.index = new_index
    return new_segment


def make_default_segment(length):
    """Create a default segment Dict of specified length"""
    default_segment = pd.DataFrame(
        columns=pd.Index(
            [
                "name",
                "lon1",
                "lat1",
                "lon2",
                "lat2",
                "dip",
                "locking_depth",
                "locking_depth_flag",
                "dip_sig",
                "dip_flag",
                "ss_rate",
                "ss_rate_sig",
                "ss_rate_flag",
                "ds_rate",
                "ds_rate_sig",
                "ds_rate_flag",
                "ts_rate",
                "ts_rate_sig",
                "ts_rate_flag",
                "mesh_file_index",
                "mesh_flag",
            ]
        )
    )
    # Set everything to zeros, then we'll fill in a few specific values
    length_vec = range(length)
    for key in default_segment.keys():
        default_segment[key] = np.zeros_like(length_vec)
    default_segment.locking_depth = +15
    default_segment.dip = +90
    for i in range(len(default_segment.name)):
        default_segment.name[i] = "segment_" + str(i)
    return default_segment


@logger.catch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "segment_file_name",
        type=Path,
        help="Name of segment file.",
    )
    parser.add_argument(
        "mesh_parameters_file_name",
        type=Path,
        help="Name of mesh parameter file.",
    )
    args = addict.Dict(vars(parser.parse_args()))

    # Read segment data
    segment = pd.read_csv(args.segment_file_name)
    segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]
    logger.success(f"Read: {args.segment_file_name}")

    # Read mesh data - List of dictionary version
    meshes = []
    with args.mesh_parameters_file_name.open() as f:
        mesh_param = json.load(f)
        logger.success(f"Read: {args.mesh_parameters_file_name}")

    if len(mesh_param) > 0:
        for i in range(len(mesh_param)):
            meshes.append(addict.Dict())
            meshes[i].meshio_object = meshio.read(mesh_param[i]["mesh_filename"])
            meshes[i].file_name = mesh_param[i]["mesh_filename"]
            meshes[i].verts = meshes[i].meshio_object.get_cells_type("triangle")

            # Expand mesh coordinates
            meshes[i].lon1 = meshes[i].meshio_object.points[meshes[i].verts[:, 0], 0]
            meshes[i].lon2 = meshes[i].meshio_object.points[meshes[i].verts[:, 1], 0]
            meshes[i].lon3 = meshes[i].meshio_object.points[meshes[i].verts[:, 2], 0]
            meshes[i].lat1 = meshes[i].meshio_object.points[meshes[i].verts[:, 0], 1]
            meshes[i].lat2 = meshes[i].meshio_object.points[meshes[i].verts[:, 1], 1]
            meshes[i].lat3 = meshes[i].meshio_object.points[meshes[i].verts[:, 2], 1]
            meshes[i].dep1 = meshes[i].meshio_object.points[meshes[i].verts[:, 0], 2]
            meshes[i].dep2 = meshes[i].meshio_object.points[meshes[i].verts[:, 1], 2]
            meshes[i].dep3 = meshes[i].meshio_object.points[meshes[i].verts[:, 2], 2]
            meshes[i].centroids = np.mean(
                meshes[i].meshio_object.points[meshes[i].verts, :], axis=1
            )
            # Cartesian coordinates in meters
            meshes[i].x1, meshes[i].y1, meshes[i].z1 = sph2cart(
                meshes[i].lon1,
                meshes[i].lat1,
                RADIUS_EARTH + KM2M * meshes[i].dep1,
            )
            meshes[i].x2, meshes[i].y2, meshes[i].z2 = sph2cart(
                meshes[i].lon2,
                meshes[i].lat2,
                RADIUS_EARTH + KM2M * meshes[i].dep2,
            )
            meshes[i].x3, meshes[i].y3, meshes[i].z3 = sph2cart(
                meshes[i].lon3,
                meshes[i].lat3,
                RADIUS_EARTH + KM2M * meshes[i].dep3,
            )

            # Cartesian triangle centroids
            meshes[i].x_centroid = (meshes[i].x1 + meshes[i].x2 + meshes[i].x3) / 3.0
            meshes[i].y_centroid = (meshes[i].y1 + meshes[i].y2 + meshes[i].y3) / 3.0
            meshes[i].z_centroid = (meshes[i].z1 + meshes[i].z2 + meshes[i].z3) / 3.0

            # Cross products for orientations
            tri_leg1 = np.transpose(
                [
                    np.deg2rad(meshes[i].lon2 - meshes[i].lon1),
                    np.deg2rad(meshes[i].lat2 - meshes[i].lat1),
                    (1 + KM2M * meshes[i].dep2 / RADIUS_EARTH)
                    - (1 + KM2M * meshes[i].dep1 / RADIUS_EARTH),
                ]
            )
            tri_leg2 = np.transpose(
                [
                    np.deg2rad(meshes[i].lon3 - meshes[i].lon1),
                    np.deg2rad(meshes[i].lat3 - meshes[i].lat1),
                    (1 + KM2M * meshes[i].dep3 / RADIUS_EARTH)
                    - (1 + KM2M * meshes[i].dep1 / RADIUS_EARTH),
                ]
            )
            meshes[i].nv = np.cross(tri_leg1, tri_leg2)
            azimuth, elevation, r = cart2sph(
                meshes[i].nv[:, 0],
                meshes[i].nv[:, 1],
                meshes[i].nv[:, 2],
            )
            meshes[i].strike = celeri.wrap2360(-np.rad2deg(azimuth))
            meshes[i].dip = 90 - np.rad2deg(elevation)
            meshes[i].dip_flag = meshes[i].dip != 90
            meshes[i].smoothing_weight = mesh_param[i]["smoothing_weight"]
            meshes[i].top_slip_rate_constraint = mesh_param[i][
                "top_slip_rate_constraint"
            ]
            meshes[i].bot_slip_rate_constraint = mesh_param[i][
                "bot_slip_rate_constraint"
            ]
            meshes[i].side_slip_rate_constraint = mesh_param[i][
                "side_slip_rate_constraint"
            ]
            meshes[i].n_tde = meshes[i].lon1.size
            celeri.get_mesh_edge_elements(meshes)
            logger.success(f"Read: {mesh_param[i]['mesh_filename']}")
        celeri.get_mesh_perimeter(meshes)

        new_segment = snap_segments(segment, meshes)
        segpath = args.segment_file_name.resolve()
        new_segment_file_name = segpath.with_name(segpath.stem + "_snapped.csv")
        new_segment.to_csv(new_segment_file_name)
        logger.success(f"Wrote: {new_segment_file_name}")


if __name__ == "__main__":
    main()
