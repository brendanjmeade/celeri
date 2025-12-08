#!/usr/bin/env python3
# TODO (Adrian): Adapt to model refactor

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import cdist

from celeri.mesh import Mesh, MeshConfig

# Global constants
DEG_PER_MYR_TO_RAD_PER_YR = 1 / 1e6  # TODO: What should this conversion be?


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
        edge_segs.lon1 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 0], 0
        ]
        edge_segs.lat1 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 0], 1
        ]
        edge_segs.lon2 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 1], 0
        ]
        edge_segs.lat2 = meshes[i].points[
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
    args = SimpleNamespace(**vars(parser.parse_args()))

    # Read segment data
    segment = pd.read_csv(args.segment_file_name)
    segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]
    logger.success(f"Read: {args.segment_file_name}")

    meshes = []
    with args.mesh_parameters_file_name.open() as f:
        mesh_param = json.load(f)
        logger.success(f"Read: {args.mesh_parameters_file_name}")

    if len(mesh_param) > 0:
        for i in range(len(mesh_param)):
            mesh_config_dict = mesh_param[i].copy()
            mesh_config_dict["file_name"] = args.mesh_parameters_file_name
            mesh_config = MeshConfig.model_validate(mesh_config_dict)

            mesh = Mesh.from_params(mesh_config)
            meshes.append(mesh)
            logger.success(f"Read: {mesh_param[i]['mesh_filename']}")

        new_segment = snap_segments(segment, meshes)
        segpath = args.segment_file_name.resolve()
        new_segment_file_name = segpath.with_name(segpath.stem + "_snapped.csv")
        new_segment.to_csv(new_segment_file_name)
        logger.success(f"Wrote: {new_segment_file_name}")


if __name__ == "__main__":
    main()
