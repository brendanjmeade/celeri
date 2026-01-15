#!/usr/bin/env python3
# %%

import argparse
import json
from pathlib import Path

import gmsh
import numpy as np
import pyproj

import celeri
from celeri.celeri_util import cart2sph, sph2cart

import matplotlib.pyplot as plt

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
        "-el",
        "--el_length",
        nargs="*",
        default=["5"],
        help="Element length (1 value for constant, 2 for top, bottom)",
        required=False,
    )
    args = dict(vars(parser.parse_args()))
    model = celeri.build_model(args["config_file_name"])

    # Get mesh directory
    mesh_dir = model.meshes[0].file_name.parent
    # Get stem of segment file name
    seg_file_stem = model.config.segment_file_name.stem
    n_meshes = len(model.meshes)  # Number of preexisting meshes
    # Identify meshes that replace segments
    # First catch segments that have their mesh_flag = 1 but no mesh
    model.segment.loc[
        (model.segment.mesh_file_index == -1) & (model.segment.mesh_flag == 1),
        "mesh_flag",
    ] = 0
    segment_mesh_file_index = np.unique(
        model.segment.mesh_file_index[model.segment.mesh_flag == 1]
    ).astype(int)
    n_segment_meshes = len(segment_mesh_file_index)
    standalone_mesh_file_index = np.setdiff1d(
        np.arange(n_meshes), segment_mesh_file_index
    )

    # Returning a copy of the closure class lets us access data within it
    thisclosure = model.closure

    # Meshing segments with `ribbon_mesh` flag > 0

    # Get indices/coordinates of segments with ribbon_mesh flag
    seg_mesh_idx = np.where(model.segment.create_ribbon_mesh > 0)[0]

    # Break if no segments need meshing
    if len(seg_mesh_idx) == 0:
        print("No segments with create_ribbon_mesh > 0")
    else:
        # Calculate bottom coordinates
        width_projected = model.segment.locking_depth / np.tan(
            np.deg2rad(model.segment.dip)
        )
        lon1_bot = np.zeros(len(model.segment))
        lon2_bot = np.zeros(len(model.segment))
        lat1_bot = np.zeros(len(model.segment))
        lat2_bot = np.zeros(len(model.segment))

        for i in range(len(model.segment)):
            lon1_bot[i], lat1_bot[i], _ = GEOID.fwd(
                model.segment.lon1[i],
                model.segment.lat1[i],
                model.segment.azimuth[i] + 90,
                1e3 * width_projected[i],
            )
            lon2_bot[i], lat2_bot[i], _ = GEOID.fwd(
                model.segment.lon2[i],
                model.segment.lat2[i],
                model.segment.azimuth[i] + 90,
                1e3 * width_projected[i],
            )

        # Get block labels of the segments that should be meshes
        sm_west_label = model.segment.loc[seg_mesh_idx, "west_labels"]
        sm_east_label = model.segment.loc[seg_mesh_idx, "east_labels"]
        sm_block_labels = np.sort(np.array([sm_east_label, sm_west_label]), axis=0)

        # Unique block pairs
        sm_block_labels_unique, sm_block_labels_unique_idx = np.unique(
            sm_block_labels, axis=1, return_inverse=True
        )

        # Number of unique block pairs corresponds to number of meshes to be generated
        # This forces meshes to be split at triple+ junctions
        # This results in a more straightforward assignment of block labels to mesh elements
        unique_mesh_idx = np.arange(np.shape(sm_block_labels_unique)[1])
        # Allocate space to hold indices of segmeshes
        segmesh_file_index: list[int | None] = [None] * len(unique_mesh_idx)

        # Loop through unique meshes and find indices of ordered coordinates
        for i in range(len(unique_mesh_idx)):
            # Find the segments associated with this mesh
            # This is done by finding the segments that have this block label pair
            this_seg_mesh_idx = seg_mesh_idx[
                np.isin(sm_block_labels[0, :], sm_block_labels_unique[0, i])
                & np.isin(sm_block_labels[1, :], sm_block_labels_unique[1, i])
            ]
            # print(model.segment.name[this_seg_mesh_idx])
            # Get the ordered coordinates from the closure array, using the first block label

            # Concatenated endpoint arrays
            this_coord1 = np.array(
                [
                    model.segment.loc[this_seg_mesh_idx, "lon1"],
                    model.segment.loc[this_seg_mesh_idx, "lat1"],
                ]
            )
            this_coord2 = np.array(
                [
                    model.segment.loc[this_seg_mesh_idx, "lon2"],
                    model.segment.loc[this_seg_mesh_idx, "lat2"],
                ]
            )
            seg_coords = np.zeros((2 * len(this_seg_mesh_idx), 2))
            seg_coords[0::2, :] = this_coord1.T
            seg_coords[1::2, :] = this_coord2.T
            # Allocate space for bottom coordinates
            seg_coords_bot = np.zeros((2 * len(this_seg_mesh_idx), 3))

            # Bottom coordinate logic based on mesh generation method

            # Panel approach: Use projected bottom coordinates based on top coordinates, depths, and dips
            if np.max(model.segment.create_ribbon_mesh[this_seg_mesh_idx]) == 1:
                # Bottom coordinates are the top lon, lat but placed at the locking depth

                seg_coords_bot[0::2, :] = np.array(
                    [
                        lon1_bot[this_seg_mesh_idx],
                        lat1_bot[this_seg_mesh_idx],
                        model.segment.loc[this_seg_mesh_idx, "locking_depth"],
                    ]
                ).T
                seg_coords_bot[1::2, :] = np.array(
                    [
                        lon2_bot[this_seg_mesh_idx],
                        lat2_bot[this_seg_mesh_idx],
                        model.segment.loc[this_seg_mesh_idx, "locking_depth"],
                    ]
                ).T

            # True ribbon approach: Bottom coordinates as duplicates of top coordinates, but placed at locking depth and rotated by average dip
            elif np.max(model.segment.create_ribbon_mesh[this_seg_mesh_idx]) == 2:
                # Define rotation axis as average strike of these segments
                av_strike = np.sum(
                    model.segment.loc[this_seg_mesh_idx, "azimuth"]
                    * model.segment.loc[this_seg_mesh_idx, "length"]
                ) / np.sum(model.segment.loc[this_seg_mesh_idx, "length"])
                # Project coordinates from surface trace, perpendicular to this single averaged cluster strike
                # av_dip = np.mean(model.segment.loc[this_seg_mesh_idx, "dip"])
                lon1_bot_proj, lat1_bot_proj, _ = GEOID.fwd(
                    seg_coords[0::2, 0],
                    seg_coords[0::2, 1],
                    (av_strike + 90) * np.ones_like(seg_coords[0::2, 0]),
                    KM2M
                    * model.segment.loc[this_seg_mesh_idx, "locking_depth"]
                    / np.tan(np.radians(model.segment.loc[this_seg_mesh_idx, "dip"])),
                )
                # Add projected coordinates to bottom array
                seg_coords_bot[0::2, 0] = lon1_bot_proj
                seg_coords_bot[0::2, 1] = lat1_bot_proj
                # Add locking depth
                seg_coords_bot[0::2, 2] = model.segment.loc[
                    this_seg_mesh_idx, "locking_depth"
                ]
                lon2_bot_proj, lat2_bot_proj, _ = GEOID.fwd(
                    seg_coords[1::2, 0],
                    seg_coords[1::2, 1],
                    (av_strike + 90) * np.ones_like(seg_coords[0::2, 0]),
                    KM2M
                    * model.segment.loc[this_seg_mesh_idx, "locking_depth"]
                    / np.tan(np.radians(model.segment.loc[this_seg_mesh_idx, "dip"])),
                )
                # Add projected coordinates to bottom array
                seg_coords_bot[1::2, 0] = lon2_bot_proj
                seg_coords_bot[1::2, 1] = lat2_bot_proj
                # Add locking depth
                seg_coords_bot[1::2, 2] = model.segment.loc[
                    this_seg_mesh_idx, "locking_depth"
                ]
            # Ordered coordinates from block closure
            block_coords = thisclosure.polygons[sm_block_labels_unique[0, i]].vertices
            # Find the ordered indices into block_coords
            seg_in_block_idx = np.unique(
                np.nonzero(np.all(block_coords == seg_coords[:, np.newaxis], axis=2))[1]
            )
            # Check to see that indices are sequential
            # If not, it means that the first ordered coordinate happens to be in the middle of the segmesh
            check_seq = (np.diff(seg_in_block_idx) != 1).nonzero()[0]
            if len(check_seq) > 0:
                seg_in_block_idx = np.concatenate(
                    (
                        seg_in_block_idx[check_seq[0] + 1 :],
                        seg_in_block_idx[1 : check_seq[0] + 1],
                    )
                )
            ordered_coords = block_coords[seg_in_block_idx, :]
            # Ordered segment indices, needed to get averaged bottom coordinates
            ordered_seg_idx = np.nonzero(
                np.all(seg_coords == ordered_coords[:, np.newaxis], axis=2)
            )[1]
            # Bottom indices 1 are first, odds, and last
            bot_idx1 = np.zeros((len(ordered_coords),))
            bot_idx1[1:-1] = np.arange(1, len(ordered_seg_idx) - 1, 2)
            bot_idx1[-1] = len(ordered_seg_idx) - 1
            # Bottom indices 2 are first, evens, and last
            bot_idx2 = np.zeros((len(ordered_coords),))

            # BJM

            print(f"\n{ordered_seg_idx = }\n")
            if i == 27:
                plt.figure()
                plt.plot(
                    [
                        model.segment.loc[:, "lon1"],
                        model.segment.loc[:, "lon2"],
                    ],
                    [
                        model.segment.loc[:, "lat1"],
                        model.segment.loc[:, "lat2"],
                    ],
                    ".k",
                )

                plt.plot(
                    [
                        model.segment.loc[this_seg_mesh_idx, "lon1"],
                        model.segment.loc[this_seg_mesh_idx, "lon2"],
                    ],
                    [
                        model.segment.loc[this_seg_mesh_idx, "lat1"],
                        model.segment.loc[this_seg_mesh_idx, "lat2"],
                    ],
                    "+r",
                )
                plt.show()

            bot_idx2[1:-1] = np.arange(2, len(ordered_seg_idx) - 1, 2)
            bot_idx2[-1] = len(ordered_seg_idx) - 1
            bot_coords1 = seg_coords_bot[ordered_seg_idx[bot_idx1.astype(int)], :]
            bot_coords2 = seg_coords_bot[ordered_seg_idx[bot_idx2.astype(int)], :]
            # Bottom coordinates are averages of "internal" endpoints (hanging ends are also averaged, but they're duplicates)
            bot_coords = (bot_coords1 + bot_coords2) / 2

            # Top coordinates are ordered block coordinates with zero depths appended
            top_coords = np.hstack((ordered_coords, np.zeros((len(ordered_coords), 1))))
            # Use top and bottom coordinates to make a mesh
            filename = mesh_dir / f"{seg_file_stem}_segmesh{unique_mesh_idx[i]}.msh"

            # Combined coordinates making a continuous perimeter loop
            all_coords = np.vstack((top_coords, np.flipud(bot_coords)))
            # Number of geometric objects
            n_coords = np.shape(all_coords)[0]
            n_surf = int((n_coords - 2) / 2)
            el_length = [float(i) for i in args["el_length"]]
            if len(el_length) == 2:
                el_length_bot = el_length[1] * np.ones(int(n_coords / 2))
                el_length_top = el_length[0] * np.ones(int(n_coords / 2))
                all_el_length = np.hstack((el_length_top, el_length_bot))
            else:
                all_el_length = el_length[0] * np.ones(int(n_coords))

            # Convert to Cartesian coordinates
            cx, cy, cz = sph2cart(
                all_coords[:, 0], all_coords[:, 1], 6371 - all_coords[:, 2]
            )

            if gmsh.isInitialized() == 0:
                gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.clear()
            # Define points
            for j in range(n_coords):
                gmsh.model.geo.addPoint(cx[j], cy[j], cz[j], all_el_length[j], j)
            # Define lines
            # Start with lines around the perimeter
            for j in range(n_coords - 1):
                # j ends as n_coords - 2
                gmsh.model.geo.addLine(j, j + 1, j)
            gmsh.model.geo.addLine(n_coords - 1, 0, n_coords - 1)
            # Add interior lines
            for k in range(n_surf - 1):
                # k ends as n_surf - 2
                gmsh.model.geo.addLine(n_coords - k - 2, k + 1, n_coords + k)

            # Define curve loops
            # All but last
            for m in range(n_surf - 1):
                # m ends as n_surf - 2
                gmsh.model.geo.addCurveLoop(
                    [m, -(n_coords + m), n_coords - 2 - m, n_coords - 1 + m], m + 1
                )
            # Last
            gmsh.model.geo.addCurveLoop(
                [n_surf - 1, n_surf, n_surf + 1, n_coords + n_surf - 2], n_surf
            )
            # Define surfaces
            for m in range(n_surf):
                gmsh.model.geo.addSurfaceFilling([m + 1], m + 1)
            # Finish writing geo attributes
            gmsh.model.geo.synchronize()

            # Combine interior panels
            gmsh.model.mesh.setCompound(2, list(range(1, n_surf)))

            # Generate mesh
            gmsh.model.mesh.generate(2)
            # Access node coordinates and convert back to spherical
            nodetags, nodecoords, _ = gmsh.model.mesh.getNodes(-1, -1)
            lon, lat, r = cart2sph(nodecoords[0::3], nodecoords[1::3], nodecoords[2::3])
            lon = np.rad2deg(lon)
            lat = np.rad2deg(lat)
            dep = r - 6371
            nodecoords[0::3] = lon
            nodecoords[1::3] = lat
            nodecoords[2::3] = dep
            # Reassign spherical node coordinates
            for j in range(len(nodetags)):
                gmsh.model.mesh.setNode(nodetags[j], nodecoords[3 * j : 3 * j + 3], [])
            # Write the mesh for later reading in celeri
            gmsh.write(str(filename))
            # gmsh.write(filename + ".geo_unrolled")
            gmsh.finalize()

            # Update segment DataFrame
            # mesh_file_index may differ from the _segmesh number
            # mesh_file_index is an index into the list of meshes in the mesh_param
            model.segment.loc[this_seg_mesh_idx, "mesh_file_index"] = (
                n_segment_meshes + i
            )  # 0-based indexing means we start at n_segment_meshes
            # Define index of this segmesh
            # This is different from the above index, because we use it for reordering mesh_param
            segmesh_file_index[i] = n_meshes + i
            model.segment.loc[this_seg_mesh_idx, "mesh_flag"] = 1
            model.segment.loc[this_seg_mesh_idx, "create_ribbon_mesh"] = 0

            # Print status
            print(
                f"Segment(s) {np.array2string(this_seg_mesh_idx)} meshed as {filename}"
            )

    # Updating mesh parameters and config

    # Assign default parameters to newly created meshes
    for j in range(len(unique_mesh_idx)):
        filename = mesh_dir / f"{seg_file_stem}_segmesh{unique_mesh_idx[j]}.msh"
        new_entry = celeri.MeshConfig(file_name=model.config.mesh_parameters_file_name)
        new_entry.mesh_filename = filename
        model.config.mesh_params.append(new_entry)

    # Reorder mesh_param list so that standalone meshes are placed last
    mesh_reorder_index = np.concatenate(
        (segment_mesh_file_index, segmesh_file_index, standalone_mesh_file_index)
    )

    model.config.mesh_params = [model.config.mesh_params[i] for i in mesh_reorder_index]
    # Write updated mesh_param json
    new_mesh_param_name = model.config.mesh_parameters_file_name.with_name(
        model.config.mesh_parameters_file_name.stem + "_segmesh.json"
    )

    data = [
        mesh_config.model_dump(mode="json") for mesh_config in model.config.mesh_params
    ]
    with new_mesh_param_name.open("w") as mf:
        json.dump(data, mf, indent=4)

    # Write updated segment csv
    new_segment_file_name = model.config.segment_file_name.with_name(
        model.config.segment_file_name.stem + "_segmesh.csv"
    )
    model.segment.to_csv(new_segment_file_name)

    # Write updated config json
    new_config_file_name = Path(args["config_file_name"]).with_name(
        Path(args["config_file_name"]).stem + "_segmesh.json"
    )

    # Reference new segment file, with mesh options set to reflect new meshes
    model.config.segment_file_name = Path(new_segment_file_name)
    # Reference new mesh parameter file, including newly created meshes
    model.config.mesh_parameters_file_name = Path(new_mesh_param_name)
    # Strip mesh params from config, because if we need to edit params, we want to do so in one place (*mesh_params_segmesh.json)
    delattr(model.config, "mesh_params")

    with new_config_file_name.open("w") as cf:
        cf.write(model.config.model_dump_json(indent=4))

    # # Visualize meshes

    # # Get a default plotting parameter dictionary
    # class BlankEstimation:
    #     strike_slip_rates = 0
    #     dip_slip_rates = 0
    #     tensile_slip_rates = 0

    # estimation = BlankEstimation()
    # p = celeri.plot.get_default_plotting_options(
    #     model.config, estimation, model.station
    # )

    # # Read in revised inputs
    # model = celeri.build_model(new_config_file_name)
    # celeri.plot.plot_fault_geometry(p, model.segment, model.meshes)


if __name__ == "__main__":
    main()
