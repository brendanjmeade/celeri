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


def get_hanging_segments(df):
    # Find hanging endpoints; these mark terminations of mesh-replaced segments
    all_lons = np.hstack((df.lon1, df.lon2))
    all_lats = np.hstack((df.lat1, df.lat2))
    all_lons_lats = np.array([all_lons, all_lats])
    _, indices, counts = np.unique(
        all_lons_lats, axis=1, return_index=True, return_counts=True
    )
    hanging_index = indices[np.where(counts == 1)]
    hanging_vertex = all_lons_lats[:, hanging_index]
    hanging_vertex = np.transpose(hanging_vertex)
    print(hanging_vertex.shape)

    # Find indices of hanging segments
    point_1_match_index = np.argmin(
        cdist(
            np.array([df["lon1"].values, df["lat1"].values]).T,
            hanging_vertex,
        ),
        axis=0,
    )

    point_2_match_index = np.argmin(
        cdist(
            np.array([df["lon2"].values, df["lat2"].values]).T,
            hanging_vertex,
        ),
        axis=0,
    )

    hanging_segment_index = np.unique(
        np.concatenate((point_1_match_index, point_2_match_index)).flatten()
    )

    # Identify the endpoint that has the hanging coordinates
    hanging_segment_vertex = np.zeros((hanging_segment_index.size, 2))
    hanging_segment_not_vertex = np.zeros((hanging_segment_index.size, 2))
    hanging_segment_lon1_lon2 = []
    for i in range(hanging_segment_index.size):
        current_hanging_vertex = hanging_vertex[i, :]
        point_1 = np.array(
            [
                df["lon1"][hanging_segment_index[i]],
                df["lat1"][hanging_segment_index[i]],
            ]
        )
        point_2 = np.array(
            [
                df["lon2"][hanging_segment_index[i]],
                df["lat2"][hanging_segment_index[i]],
            ]
        )

        if np.allclose(point_1, current_hanging_vertex):
            hanging_segment_vertex[i, :] = point_1
            hanging_segment_not_vertex[i, :] = point_2
            hanging_segment_lon1_lon2.append("lon1")
        elif np.allclose(point_2, current_hanging_vertex):
            hanging_segment_vertex[i, :] = point_2
            hanging_segment_not_vertex[i, :] = point_1
            hanging_segment_lon1_lon2.append("lon2")

    hanging_segment = {}
    hanging_segment["hanging_vertex"] = hanging_segment_vertex
    hanging_segment["not_hanging_vertex"] = hanging_segment_not_vertex
    hanging_segment["lon1_lon2"] = hanging_segment_lon1_lon2
    hanging_segment["index"] = hanging_segment_index

    return hanging_segment


def plot_hanging_segments(df, hanging_segment_index):
    plt.figure()

    # Plot all segments for current ribbon
    for i in range(len(df)):
        plt.plot(
            [df["lon1"][i], df["lon2"][i]],
            [df["lat1"][i], df["lat2"][i]],
            "-g",
        )

    # Plot hanging segments for current ribbon
    for i in range(hanging_segment_index.size):
        plt.plot(
            [
                df["lon1"][hanging_segment_index[i]],
                df["lon2"][hanging_segment_index[i]],
            ],
            [
                df["lat1"][hanging_segment_index[i]],
                df["lat2"][hanging_segment_index[i]],
            ],
            "-r",
        )
        plt.plot(
            df["lon1"][hanging_segment_index[i]],
            df["lat1"][hanging_segment_index[i]],
            "r+",
        )
        plt.plot(
            df["lon2"][hanging_segment_index[i]],
            df["lat2"][hanging_segment_index[i]],
            "r+",
        )
    plt.gca().set_aspect("equal")
    plt.show(block=True)


def main():
    geo_file_name = "mesh_test.geo"
    gmsh_excutable_location = "/opt/homebrew/bin/gmsh"
    smooth_trace = False
    top_mesh_reference_size = 0.01
    bottom_mesh_reference_size = 0.2
    locking_depth_override_flag = True
    locking_depth_override_value = 40.0
    depth_scaling = 100.0
    resample_flag = True
    resample_length = 0.01

    # TEMP: Read .csv dataframe from Emily
    df = pd.read_csv("naf_sorted_top_coordinates.csv")
    df_segment = pd.read_csv("emed0026_segment.csv")

    # Convert tips to radians
    df_segment.dip = np.deg2rad(df_segment.dip)

    # Which segments should be ribbon meshed
    keep_segment_idx = np.where(df_segment["create_ribbon_mesh"].values == 1)[0]
    df_segment_keep = df_segment.loc[keep_segment_idx]
    new_index = range(len(keep_segment_idx))
    df_segment_keep.index = new_index

    # Find hanging segments
    hanging_segment = get_hanging_segments(df_segment_keep)
    print(f"{hanging_segment['hanging_vertex']=}")
    print(f"{hanging_segment['not_hanging_vertex']=}")

    # Loop to build ordered segments in order
    # segment_ordered_lon = np.zeros(len(df_segment_keep) + 1)
    # segment_ordered_lat = np.zeros(len(df_segment_keep) + 1)
    point_1s = np.array(
        [df_segment_keep["lon1"].values, df_segment_keep["lat1"].values]
    ).T
    point_2s = np.array(
        [df_segment_keep["lon2"].values, df_segment_keep["lat2"].values]
    ).T

    segment_ordered_lon = np.zeros(5)
    segment_ordered_lat = np.zeros(5)
    segment_ordered_lon[0] = hanging_segment["hanging_vertex"][0, 0]
    segment_ordered_lat[0] = hanging_segment["hanging_vertex"][0, 1]
    segment_ordered_lon[1] = hanging_segment["not_hanging_vertex"][0, 0]
    segment_ordered_lat[1] = hanging_segment["not_hanging_vertex"][0, 1]

    segment_order_index = np.zeros(5)
    segment_order_index[0] = hanging_segment["index"][0]
    # # for i in range(len(df_segment_keep) - 1):
    for i in range(2, 5):
        print(i)
        current_point = np.array(
            [segment_ordered_lon[i - 1], segment_ordered_lat[i - 1]]
        )[None, :]
        print(current_point)

        # Find segment connected to current segment
        point_1_match_index = np.argmin(cdist(current_point, point_1s), axis=1)

        # Find segment connected to current segment
        point_2_match_index = np.argmin(cdist(current_point, point_2s), axis=1)
        match_segment_index = np.unique(
            np.concatenate((point_1_match_index, point_2_match_index)).flatten()
        )

        print(match_segment_index)

    # point_2_match_index = np.argmin(
    #     cdist(
    #         all_lons_lats[:, hanging_index].T,
    #         np.array([df["lon2"].values, df["lat2"].values]).T,
    #     ),
    #     axis=1,
    # )

    # connected_segment_index = np.unique(
    #     np.concatenate((point_1_match_index, point_2_match_index)).flatten()
    # )

    # Check to make sure it's not itself or a prior connected segment

    # Add index to list
    # new_index = 0
    # segment_order_index.append(new_index)

    IPython.embed(banner1="")
    return

    # Deeper locking depth setting
    if bool(locking_depth_override_flag):
        df.locking_depth = locking_depth_override_value

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
