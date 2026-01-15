#!/usr/bin/env python3
# %%

import argparse
from pathlib import Path

import gmsh
import numpy as np
import pyproj
import shapely

import celeri

# Global constants
GEOID = pyproj.Geod(ellps="WGS84")
KM2M = 1.0e3
M2MM = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file_name",
        type=Path,
        help="Name of config file.",
    )

    parser.add_argument(
        "lon_min",
        type=float,
        help="Minimum longitude of the bounding box",
    )
    parser.add_argument(
        "lon_max",
        type=float,
        help="Maximum longitude of the bounding box",
    )
    parser.add_argument(
        "lat_min",
        type=float,
        help="Minimum latitude of the bounding box",
    )
    parser.add_argument(
        "lat_max",
        type=float,
        help="Maximum latitude of the bounding box",
    )

    parser.add_argument(
        "--bbox_threshold",
        default=0.8,
        help="Fraction of block nodes that must lie within bounding box to be meshed (0-1)",
        required=False,
    )

    parser.add_argument(
        "-el",
        "--el_length",
        default=1,
        help="Element length (scalar)",
        required=False,
    )

    parser.add_argument(
        "-z",
        "--depth",
        default=-25,
        help="Element depth in km (scalar)",
        required=False,
    )
    args = dict(vars(parser.parse_args()))
    model = celeri.build_model(args["config_file_name"])

    # Get mesh directory
    mesh_dir = model.meshes[0].file_name.parent.absolute()

    # Get element length
    el_length = float(args["el_length"])

    # Get element depth
    mesh_depth = -np.abs(float(args["depth"]))

    # Returning a copy of the closure class lets us access data within it
    thisclosure = model.closure

    # Define bounding polygon as a matplotlib Path for contains_points testing
    bbox_path = shapely.geometry.Polygon(
        shapely.geometry.LineString(
            np.array(
                [
                    [args["lon_min"], args["lat_min"]],
                    [args["lon_max"], args["lat_min"]],
                    [args["lon_max"], args["lat_max"]],
                    [args["lon_min"], args["lat_max"]],
                ]
            )
        )
    )
    bbox_lines = shapely.linestrings(
        [
            [[args["lon_min"], args["lat_min"]], [args["lon_max"], args["lat_min"]]],
            [[args["lon_max"], args["lat_min"]], [args["lon_max"], args["lat_max"]]],
            [[args["lon_max"], args["lat_max"]], [args["lon_min"], args["lat_max"]]],
            [[args["lon_min"], args["lat_max"]], [args["lon_min"], args["lat_min"]]],
        ]
    ).tolist()

    # Convert bbox threshold to fraction, if need be
    if args["bbox_threshold"] > 1:
        bbox_threshold = args["bbox_threshold"] / 100
    else:
        bbox_threshold = args["bbox_threshold"]

    # Initialize Gmsh
    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.clear()

    # Write all segment vertices
    # Only those by blocks within the bounding box will be referenced in constructing lines
    seg_endpoint1 = np.array([model.segment.lon1.T, model.segment.lat1.T])
    seg_endpoint2 = np.array([model.segment.lon2.T, model.segment.lat2.T])

    all_seg_endpoints = np.unique(
        np.vstack(
            (
                seg_endpoint1.T,
                seg_endpoint2.T,
            )
        ),
        axis=0,
    )
    for i in range(len(all_seg_endpoints)):
        gmsh.model.geo.addPoint(
            all_seg_endpoints[i, 0], all_seg_endpoints[i, 1], mesh_depth, el_length, i
        )
    gmsh.model.geo.addPoint(
        args["lon_min"],
        args["lat_min"],
        mesh_depth,
        5 * el_length,
        len(all_seg_endpoints) + 0,
    )
    gmsh.model.geo.addPoint(
        args["lon_max"],
        args["lat_min"],
        mesh_depth,
        5 * el_length,
        len(all_seg_endpoints) + 1,
    )
    gmsh.model.geo.addPoint(
        args["lon_max"],
        args["lat_max"],
        mesh_depth,
        5 * el_length,
        len(all_seg_endpoints) + 2,
    )
    gmsh.model.geo.addPoint(
        args["lon_min"],
        args["lat_max"],
        mesh_depth,
        5 * el_length,
        len(all_seg_endpoints) + 3,
    )
    # Initial counter for supplementary points
    # These are points at the intersection of segments and the bounding box
    supp_points_idx = len(all_seg_endpoints) + 4

    # Initialize geometry item counters
    tot_lines = 0
    # Loop through all closed blocks
    for block_idx in range(len(thisclosure.polygons)):
        # Ordered coordinates from block closure
        # Cutting off final coordinate because it's a duplicate of the first
        block_coords = thisclosure.polygons[block_idx].vertices[:-1, :]
        n_lines = len(block_coords)
        # Find coordinates within bounding polygon
        block_coords_within = np.where(
            bbox_path.contains(shapely.points(block_coords))
        )[0]
        # Get percentage of nodes within box
        block_coords_within_frac = len(block_coords_within) / len(block_coords)
        # Check to see if any coordinates are within bounding box
        if block_coords_within_frac > 0:
            # Mesh the whole block if enough coordinates are within the box
            if block_coords_within_frac > bbox_threshold:
                # Find indices of block boundaries in unique endpoint array
                block_in_seg_idx = np.nonzero(
                    np.all(all_seg_endpoints == block_coords[:, np.newaxis], axis=2)
                )[1]
                # Define lines around the perimeter
                for j in range(n_lines - 1):
                    gmsh.model.geo.addLine(
                        block_in_seg_idx[j], block_in_seg_idx[j + 1], tot_lines + j
                    )
                # Final line that completes the block
                gmsh.model.geo.addLine(
                    block_in_seg_idx[-1], block_in_seg_idx[0], tot_lines + n_lines - 1
                )

                # Define curve loop
                block_line_loop = list(range(tot_lines, tot_lines + n_lines))
                # For some reason we need to start the line loop indexing at 1
                # Using 0 as an index writes all zeros to the line loop. Gmsh bug?
                gmsh.model.geo.addCurveLoop(block_line_loop, block_idx + 1)
                # Update total lines
                tot_lines += n_lines
                # Define surface
                gmsh.model.geo.addPlaneSurface([block_idx + 1], block_idx + 1)
            else:
                # For blocks that have some nodes within the bounding box,
                # use those nodes along with the nearest bounding box coordinates

                # Rearrange block_coords_within indices to remove jumps
                jump = np.flatnonzero(np.diff(block_coords_within) > 1)
                if len(jump) > 0:
                    block_coords_within = np.concatenate(
                        (
                            block_coords_within[jump[0] + 1 :],
                            block_coords_within[0 : jump[0] + 1],
                        )
                    )

                # Split 1: segment made of first inside point and previous block coordinate
                if block_coords_within[0] != 0:
                    split_1_outside = block_coords_within[0] - 1
                else:
                    split_1_outside = n_lines - 1
                # Define the line from the first inside point and previous block coordinate
                cross_line1 = shapely.geometry.LineString(
                    [
                        block_coords[block_coords_within[0], :],
                        block_coords[split_1_outside, :],
                    ]
                )
                # Find intersections between this line and all bounding box sides
                box_cross_1 = [
                    cross_line1.intersection(bbox_lines[0]),
                    cross_line1.intersection(bbox_lines[1]),
                    cross_line1.intersection(bbox_lines[2]),
                    cross_line1.intersection(bbox_lines[3]),
                ]
                # Identify point of intersection
                box_intersection_1 = next(
                    x for x in box_cross_1 if not shapely.is_empty(x)
                )

                # Split 2: segment made of last inside point and next block coordinate
                if block_coords_within[-1] != n_lines:
                    split_2_outside = block_coords_within[-1] + 1
                else:
                    split_2_outside = 0
                # Define the line from the last inside point and next block coordinate
                cross_line2 = shapely.geometry.LineString(
                    [
                        block_coords[block_coords_within[-1], :],
                        block_coords[split_2_outside, :],
                    ]
                )
                # Find intersections between this line and all bounding box sides
                box_cross_2 = [
                    cross_line2.intersection(bbox_lines[0]),
                    cross_line2.intersection(bbox_lines[1]),
                    cross_line2.intersection(bbox_lines[2]),
                    cross_line2.intersection(bbox_lines[3]),
                ]
                # Identify point of intersection
                box_intersection_2 = next(
                    x for x in box_cross_2 if not shapely.is_empty(x)
                )

                # Add intersection points to Gmsh list
                gmsh.model.geo.addPoint(
                    box_intersection_1.coords.xy[0][0],
                    box_intersection_1.coords.xy[1][0],
                    mesh_depth,
                    el_length,
                    supp_points_idx,
                )
                gmsh.model.geo.addPoint(
                    box_intersection_2.coords.xy[0][0],
                    box_intersection_2.coords.xy[1][0],
                    mesh_depth,
                    el_length,
                    supp_points_idx + 1,
                )
                # Add block lines to Gmsh list

                # Find indices of block boundaries in unique endpoint array
                # Only using block boundaries within bounding box
                block_in_seg_idx = np.nonzero(
                    np.all(
                        all_seg_endpoints
                        == block_coords[block_coords_within, np.newaxis],
                        axis=2,
                    )
                )[1]
                # Keep only those within the bounding box

                # Define lines around the perimeter
                n_lines = len(block_in_seg_idx)
                for j in range(n_lines - 1):
                    gmsh.model.geo.addLine(
                        block_in_seg_idx[j], block_in_seg_idx[j + 1], tot_lines + j
                    )

                # Add lines connecting the extreme points within the bounding box to the intersections
                gmsh.model.geo.addLine(
                    supp_points_idx, block_in_seg_idx[0], tot_lines + n_lines - 1
                )
                gmsh.model.geo.addLine(
                    block_in_seg_idx[-1], supp_points_idx + 1, tot_lines + n_lines + 0
                )

                # Check to see if intersection 2 is along the same BB side as intersection 1
                # If it's the same side, there's no need to place a BB coordinate in between
                # If it's a different side, need to place a BB corner in between the intersections

                bb_cross_idx1 = np.where(~shapely.is_empty(box_cross_1))[0][0]
                bb_cross_idx2 = np.where(~shapely.is_empty(box_cross_2))[0][0]

                if bb_cross_idx1 != bb_cross_idx2:
                    gmsh.model.geo.addLine(
                        supp_points_idx,
                        len(all_seg_endpoints) + bb_cross_idx2,
                        tot_lines + n_lines + 1,
                    )
                    gmsh.model.geo.addLine(
                        len(all_seg_endpoints) + bb_cross_idx2,
                        supp_points_idx + 1,
                        tot_lines + n_lines + 2,
                    )
                    # Write line loop and surface
                    # Define curve loop
                    block_line_loop = list(range(tot_lines, tot_lines + n_lines - 1))
                    block_line_loop.extend(
                        [
                            tot_lines + n_lines + 0,
                            -(tot_lines + n_lines + 2),
                            -(tot_lines + n_lines + 1),
                            tot_lines + n_lines - 1,
                        ]
                    )
                    gmsh.model.geo.addCurveLoop(block_line_loop, block_idx + 1)
                    tot_lines += n_lines + 3
                else:
                    # Along bounding box boundary
                    gmsh.model.geo.addLine(
                        supp_points_idx + 1, supp_points_idx, tot_lines + n_lines + 1
                    )
                    # Write line loop and surface
                    # Define curve loop
                    block_line_loop = list(range(tot_lines, tot_lines + n_lines - 1))
                    block_line_loop.extend(
                        [
                            tot_lines + n_lines + 0,
                            tot_lines + n_lines + 1,
                            tot_lines + n_lines - 1,
                        ]
                    )
                    gmsh.model.geo.addCurveLoop(block_line_loop, block_idx + 1)
                    tot_lines += n_lines + 2
                supp_points_idx += 2
                # Define surface
                gmsh.model.geo.addPlaneSurface([block_idx + 1], block_idx + 1)

    # Finish writing geo attributes
    gmsh.model.geo.synchronize()
    filestem = args["config_file_name"].stem
    gmsh.write(str(mesh_dir / f"{filestem}_cmi.geo_unrolled"))

    # Generate mesh
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)
    # Write the mesh for later reading in celeri
    gmsh.write(str(mesh_dir / f"{filestem}_cmi.msh"))
    gmsh.finalize()


if __name__ == "__main__":
    main()
