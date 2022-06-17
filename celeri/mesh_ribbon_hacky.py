import matplotlib.pyplot as plt
import numpy as np
import os
import meshio
import pandas as pd
import IPython
from pathlib import Path
from scipy.spatial.distance import cdist


def write_geo_file(
    geo_file_name,
    top_coordinates,
    bottom_coordinates,
    top_mesh_reference_size,
    bottom_mesh_reference_size,
    smooth_trace,
):
    n_top_coordinates = top_coordinates.shape[0]
    n_bottom_coordinates = bottom_coordinates.shape[0]

    # Write the .geo file that gmsh will run
    fid = open(geo_file_name, "w")

    for i in range(n_top_coordinates):
        fid.write(
            f"Point({i + 1}) = {{"
            f"{top_coordinates[i, 0]:0.4f}, "
            f"{top_coordinates[i, 1]:0.4f}, "
            f"{top_coordinates[i, 2]:0.4f}, "
            f"{top_mesh_reference_size}}};\n"
        )

    for i in range(n_bottom_coordinates):
        fid.write(
            f"Point({n_top_coordinates + i + 1}) = {{"
            f"{bottom_coordinates[i, 0]:0.4f}, "
            f"{bottom_coordinates[i, 1]:0.4f}, "
            f"{bottom_coordinates[i, 2]:0.4f}, "
            f"{bottom_mesh_reference_size}}};\n"
        )

    if smooth_trace:
        fid.write(f"CatmullRom(1) = {{1:{n_top_coordinates}}};\n")
        fid.write(
            f"CatmullRom(2) = {{{n_top_coordinates + 1}:{n_top_coordinates + n_bottom_coordinates}}};\n"
        )
        fid.write(f"CatmullRom(3) = {{{n_top_coordinates},{n_top_coordinates + 1}}};\n")
        fid.write(
            f"CatmullRom(4) = {{{n_top_coordinates + n_bottom_coordinates},1}};\n"
        )
        fid.write("Line Loop(1) = {1, 3, 2, 4};\n")
        fid.write("Ruled Surface(1) = {1};\n")
    else:
        fid.write(f"Line(1) = {{1:{n_top_coordinates}}};\n")
        fid.write(
            f"Line(2) = {{{n_top_coordinates + 1}:{n_top_coordinates + n_bottom_coordinates}}};\n"
        )
        fid.write(f"Line(3) = {{{n_top_coordinates},{n_top_coordinates + 1}}};\n")
        fid.write(f"Line(4) = {{{n_top_coordinates + n_bottom_coordinates},1}};\n")
        fid.write("Line Loop(1) = {1, 3, 2, 4};\n")
        fid.write("Ruled Surface(1) = {1};\n")
    fid.close()


def resample_trace(df, resample_length):
    longitude_diff = np.diff(df.lons)
    latitude_diff = np.diff(df.lats)
    approximate_segment_lengths = np.sqrt(longitude_diff ** 2.0 + latitude_diff ** 2.0)

    resampled_lons = []
    resampled_lats = []
    resampled_locking_depths = []
    resampled_dips = []
    for i in range(len(df) - 1):
        if approximate_segment_lengths[i] > resample_length:
            n_segments = np.ceil(
                approximate_segment_lengths[i] / resample_length
            ).astype(int)
            print(f"Divide the segment from {i} and {i + 1} into {n_segments} segments")
            new_lons = np.linspace(df.lons[i], df.lons[i + 1], n_segments + 1)
            new_lats = np.linspace(df.lats[i], df.lats[i + 1], n_segments + 1)
            for j in range(n_segments):
                resampled_lons.append(new_lons[j])
                resampled_lats.append(new_lats[j])
                resampled_locking_depths.append(df.locking_depth[i])
                resampled_dips.append(df.dip[i])
        else:
            resampled_lons.append(df.lons[i])
            resampled_lats.append(df.lats[i])
            resampled_locking_depths.append(df.locking_depth[i])
            resampled_dips.append(df.dip[i])

    return resampled_lons, resampled_lats, resampled_locking_depths, resampled_dips


def main():
    geo_file_name = "mesh_test.geo"
    gmsh_excutable_location = "/opt/homebrew/bin/gmsh"
    smooth_trace = False

    # NAF parameters
    top_mesh_reference_size = 0.01
    bottom_mesh_reference_size = 0.1
    depth_scaling = 100.0

    # EAF parameters
    top_mesh_reference_size = 0.01
    bottom_mesh_reference_size = 0.1
    depth_scaling = 100.0

    locking_depth_override_flag = True
    locking_depth_override_value = 40.0
    resample_flag = True
    resample_length = 0.01

    # TEMP: Read .csv dataframe from Emily
    df = pd.read_csv("naf_sorted_top_coordinates.csv")
    df_segment = pd.read_csv("emed0026_segment.csv")

    # Convert tips to radians
    df.dip = np.deg2rad(df.dip)
    df_segment.dip = np.deg2rad(df_segment.dip)

    # Deeper locking depth setting
    if bool(locking_depth_override_flag):
        df.locking_depth = locking_depth_override_value
        df_segment.locking_depth = locking_depth_override_value

    # Which segments should be ribbon meshed
    keep_segment_idx = np.where(df_segment["create_ribbon_mesh"].values == 2)[0]
    df_segment_keep = df_segment.loc[keep_segment_idx]
    new_index = range(len(keep_segment_idx))
    df_segment_keep.index = new_index

    # --- START OF SORTING APPROACH ---
    # Longitudinal sorting of surface trace points
    all_lons = np.concatenate(
        (df_segment_keep.lon1.values, df_segment_keep.lon2.values), axis=0
    )  # Creates an array with lon 1 and lon 2
    all_lats = np.concatenate(
        (df_segment_keep.lat1.values, df_segment_keep.lat2.values), axis=0
    )  # Creates an array with lat 1 and lat 2
    duplicated_locking_depth = np.concatenate(
        (df_segment_keep.locking_depth.values, df_segment_keep.locking_depth.values),
        axis=0,
    )
    duplicated_dip = np.concatenate(
        (df_segment_keep.dip.values, df_segment_keep.dip.values), axis=0
    )

    coords = pd.DataFrame(
        {
            "all_lons": all_lons,
            "all_lats": all_lats,
            "duplicated_locking_depth": duplicated_locking_depth,
            "dip": duplicated_dip,
        }
    )
    coords_array = coords.to_numpy()

    coords_array_sorted = np.unique(coords_array, axis=0)
    data = {
        "lons": coords_array_sorted[:, 0],
        "lats": coords_array_sorted[:, 1],
        "locking_depth": coords_array_sorted[:, 2],
        "dip": coords_array_sorted[:, 3],
    }
    coords_sorted = pd.DataFrame(
        data, columns=["lons", "lats", "locking_depth", "dip"]
    )  # store data in a Pandas Dataframe (again!)
    df = coords_sorted
    # --- END OF SORTING APPROACH ---

    if resample_flag:  # No resampling case
        (
            resampled_lons,
            resampled_lats,
            resampled_locking_depths,
            resampled_dips,
        ) = resample_trace(df, resample_length)

        top_coordinates = np.zeros((len(resampled_lons), 3))
        bottom_coordinates = np.zeros((len(resampled_lons), 3))
        top_coordinates[:, 0] = np.array(resampled_lons)
        top_coordinates[:, 1] = np.array(resampled_lats)
        bottom_coordinates[:, 0] = np.array(resampled_lons)
        bottom_coordinates[:, 1] = np.array(resampled_lats)
        bottom_coordinates[:, 2] = -np.array(resampled_locking_depths)
        bottom_coordinates[:, 2] = bottom_coordinates[:, 2] / depth_scaling

    else:  # No resampling case
        top_coordinates = np.zeros((len(df), 3))
        bottom_coordinates = np.zeros((len(df), 3))
        top_coordinates[:, 0] = df.lons.values
        top_coordinates[:, 1] = df.lats.values
        bottom_coordinates[:, 0] = df.lons.values
        bottom_coordinates[:, 1] = df.lats.values
        bottom_coordinates[:, 2] = -df.locking_depth.values / depth_scaling

    # Try adding dip effects
    # May have to resample the bottom trace this time.

    # Reorder bottom coordinates so that all coordinates "circulate"
    bottom_coordinates = np.flipud(bottom_coordinates)

    # Write a .geo file for gmsh to run
    write_geo_file(
        geo_file_name,
        top_coordinates,
        bottom_coordinates,
        top_mesh_reference_size,
        bottom_mesh_reference_size,
        smooth_trace,
    )

    # Run gmsh
    # -2 means two dimensional mesh
    # -v 0 means verbosity level 0
    msh_file_name = f"{Path(geo_file_name).stem}.msh"
    os.system(
        f"{gmsh_excutable_location} -2 {geo_file_name} -o {msh_file_name} -v 0 > /dev/null"
    )
    # Plot gmsh generated mesh with ad hoc scaling
    os.system(f"{gmsh_excutable_location} {msh_file_name} &")

    # Read the mesh file that gmsh created
    meshio_object = meshio.read(msh_file_name)
    n_triangles = meshio_object.cells_dict["triangle"].shape[0]
    print(f"Created mesh with {n_triangles} triangles")

    # Reset the depths to eliminate ad-hoc scaling
    meshio_object.points[:, 2] = depth_scaling * meshio_object.points[:, 2]
    meshio_object.write(msh_file_name, file_format="gmsh22")

    # Plot the gmsh generated mesh with out ad hoc scaling
    os.system(f"{gmsh_excutable_location} {msh_file_name} &")

    # Drop into repl
    # IPython.embed(banner1="")


if __name__ == "__main__":
    main()
