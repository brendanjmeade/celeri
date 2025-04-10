import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd
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
    fid = Path(geo_file_name).open("w")

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


# JPL removed dips, since dip is really a property of the segment, not the coordinates
def resample_trace(lons, lats, locking_depth, resample_length):
    longitude_diff = np.diff(lons)
    latitude_diff = np.diff(lats)
    approximate_segment_lengths = np.sqrt(longitude_diff**2.0 + latitude_diff**2.0)

    resampled_lons = []
    resampled_lats = []
    resampled_locking_depths = []

    for i in range(len(lons) - 1):
        if approximate_segment_lengths[i] > resample_length:
            n_segments = np.ceil(
                approximate_segment_lengths[i] / resample_length
            ).astype(int)
            print(f"Divide the segment from {i} and {i + 1} into {n_segments} segments")
            new_lons = np.linspace(lons[i], lons[i + 1], n_segments + 1)
            new_lats = np.linspace(lats[i], lats[i + 1], n_segments + 1)
            for j in range(n_segments):
                resampled_lons.append(new_lons[j])
                resampled_lats.append(new_lats[j])
                resampled_locking_depths.append(locking_depth[i])
        else:
            resampled_lons.append(lons[i])
            resampled_lats.append(lats[i])
            resampled_locking_depths.append(locking_depth[i])

    return resampled_lons, resampled_lats, resampled_locking_depths


def get_hanging_segments(df):
    # Find hanging endpoints; these mark terminations of mesh-replaced segments
    all_lons = np.hstack((df.lon1, df.lon2))
    all_lats = np.hstack((df.lat1, df.lat2))
    all_lons_lats = np.array([all_lons, all_lats])
    _, indices, counts = np.unique(
        all_lons_lats, axis=1, return_index=True, return_counts=True
    )
    hanging_index = indices[np.where(counts == 1)]
    hanging_point = all_lons_lats[:, hanging_index]
    hanging_point = np.transpose(hanging_point)
    print(hanging_point.shape)

    # Find indices of hanging segments
    point_1_match_index = np.argmin(
        cdist(
            np.array([df["lon1"].values, df["lat1"].values]).T,
            hanging_point,
        ),
        axis=0,
    )

    point_2_match_index = np.argmin(
        cdist(
            np.array([df["lon2"].values, df["lat2"].values]).T,
            hanging_point,
        ),
        axis=0,
    )

    hanging_segment_index = np.unique(
        np.concatenate((point_1_match_index, point_2_match_index)).flatten()
    )

    # Identify the endpoint that has the hanging coordinates
    hanging_segment_point = np.zeros((hanging_segment_index.size, 2))
    hanging_segment_not_point = np.zeros((hanging_segment_index.size, 2))
    hanging_segment_lon1_lon2 = []
    for i in range(hanging_segment_index.size):
        current_hanging_point = hanging_point[i, :]
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

        if np.allclose(point_1, current_hanging_point):
            hanging_segment_point[i, :] = point_1
            hanging_segment_not_point[i, :] = point_2
            hanging_segment_lon1_lon2.append("lon1")
        elif np.allclose(point_2, current_hanging_point):
            hanging_segment_point[i, :] = point_2
            hanging_segment_not_point[i, :] = point_1
            hanging_segment_lon1_lon2.append("lon2")

    hanging_segment = {}
    hanging_segment["hanging_point"] = hanging_segment_point
    hanging_segment["not_hanging_point"] = hanging_segment_not_point
    hanging_segment["lon1_lon2"] = hanging_segment_lon1_lon2
    hanging_segment["index"] = hanging_segment_index

    return hanging_segment


def order_endpoints(segment):
    """Endpoint ordering function, placing west point first.

    Could go back to cross product-based ordering if we call process_segment
    """
    segment_copy = copy.deepcopy(segment)
    swap_endpoint_idx = np.where(segment_copy.lon1 > segment_copy.lon2)
    segment_copy.lon1.values[swap_endpoint_idx] = segment.lon2.values[swap_endpoint_idx]
    segment_copy.lat1.values[swap_endpoint_idx] = segment.lat2.values[swap_endpoint_idx]
    segment_copy.lon2.values[swap_endpoint_idx] = segment.lon1.values[swap_endpoint_idx]
    segment_copy.lat2.values[swap_endpoint_idx] = segment.lat1.values[swap_endpoint_idx]
    return segment_copy


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
    top_mesh_reference_size = 1.0
    bottom_mesh_reference_size = 1.0
    locking_depth_override_flag = True
    locking_depth_override_value = 40.0
    depth_scaling = 100.0
    resample_flag = True
    resample_length = 0.01

    # TEMP: Read .csv dataframe from Emily
    df = pd.read_csv("emed0026_segment.csv")

    # Order endpoints of segments, just placing the western point first
    df = order_endpoints(df)

    # Convert tips to radians
    df.dip = np.deg2rad(df.dip)

    # Which segments should be ribbon meshed
    keep_idx = np.where(df["create_ribbon_mesh"].values == 1)[0]
    df_keep = df.loc[keep_idx]
    new_index = range(len(keep_idx))
    df_keep.index = new_index

    # Find hanging segments
    hanging_segment = get_hanging_segments(df_keep)

    # Loop to build ordered segments in order
    segment_ordered_lon = np.zeros(len(df_keep))
    segment_ordered_lat = np.zeros(len(df_keep))
    segment_ordered_dep = np.zeros(len(df_keep))
    point_1s = np.array([df_keep["lon1"], df_keep["lat1"]]).T
    point_2s = np.array([df_keep["lon2"], df_keep["lat2"]]).T
    segment_ordered_lon[0] = hanging_segment["hanging_point"][0, 0]
    segment_ordered_lat[0] = hanging_segment["hanging_point"][0, 1]
    segment_ordered_dep[0] = df_keep["locking_depth"][hanging_segment["index"][0]]
    segment_ordered_lon[1] = hanging_segment["not_hanging_point"][0, 0]
    segment_ordered_lat[1] = hanging_segment["not_hanging_point"][0, 1]
    segment_ordered_dep[1] = df_keep["locking_depth"][hanging_segment["index"][0]]
    segment_order_index = np.zeros(len(df_keep))
    segment_order_index[0] = hanging_segment["index"][0]
    print(f"{segment_order_index=}")

    # Crawl along segments
    for i in range(2, len(df_keep)):
        print(" ")
        print(f"{i=}")
        current_point = np.array(
            [segment_ordered_lon[i - 1], segment_ordered_lat[i - 1]]
        )[None, :]
        print(f"{current_point=}")

        # Find distance between current_point and all other end points
        point_1_distances = cdist(current_point, point_1s)
        point_2_distances = cdist(current_point, point_2s)
        print(f"{point_1_distances.min()=}")
        print(f"{point_2_distances.min()=}")

        # Find index of closest point
        point_1_min_idx = np.where(point_1_distances == point_1_distances.min())[1]
        point_2_min_idx = np.where(point_2_distances == point_2_distances.min())[1]
        print(f"{point_1_min_idx=}")
        print(f"{point_2_min_idx=}")

        point_1 = np.array(
            [df_keep["lon1"][point_1_min_idx], df_keep["lat1"][point_1_min_idx]]
        )
        point_2 = np.array(
            [df_keep["lon2"][point_2_min_idx], df_keep["lat2"][point_2_min_idx]]
        )
        print(f"{point_1.T=}")
        print(f"{point_2.T=}")

        # TODO: #105 There is something wrong with the logic of the next ~10 lines
        if point_1_distances.min() == 0.0:
            match_segment_index = point_1_min_idx
        else:
            match_segment_index = point_2_min_idx
        print(f"{match_segment_index=}")

        # Select the segment that is not the current segment
        match_segment_index = match_segment_index[
            np.where(match_segment_index != segment_order_index[i - 2])[0]
        ]
        match_segment_index = match_segment_index.astype(int)
        print(f"{segment_order_index[i-2]=}")
        print(f"{match_segment_index=}")

        segment_order_index[i - 1] = match_segment_index

        # End points from selected segment
        point_1 = np.array(
            [
                df_keep["lon1"][match_segment_index],
                df_keep["lat1"][match_segment_index],
            ]
        )
        point_2 = np.array(
            [
                df_keep["lon2"][match_segment_index],
                df_keep["lat2"][match_segment_index],
            ]
        )
        print(f"{point_1.T=}")
        print(f"{point_2.T=}")

        # Select (lon1, lat1) or (lon2, lat2) as the next current point
        # if np.allclose(point_1.flatten(), current_point.flatten()):
        if np.array_equal(point_1.flatten(), current_point.flatten()):
            print("(lon1, lat1)")
            segment_ordered_lon[i] = df_keep["lon2"][match_segment_index]
            segment_ordered_lat[i] = df_keep["lat2"][match_segment_index]
        # elif np.allclose(point_2.flatten(), current_point.flatten()):
        elif np.array_equal(point_2.flatten(), current_point.flatten()):
            print("(lon2, lat2)")
            segment_ordered_lon[i] = df_keep["lon1"][match_segment_index]
            segment_ordered_lat[i] = df_keep["lat1"][match_segment_index]
        # Set coordinate depth to average of locking depths of segments it's a part of
        segment_ordered_dep[i] = np.mean(
            [segment_ordered_dep[i - 1], df_keep["locking_depth"][match_segment_index]]
        )

    # plt.figure()
    # for i in range(segment_ordered_lon.size):
    #     plt.plot(segment_ordered_lon[i], segment_ordered_lat[i], ".r")
    #     plt.text(segment_ordered_lon[i], segment_ordered_lat[i], str(i))
    # plt.show()

    # IPython.embed(banner1="")
    # return

    # Deeper locking depth setting
    if bool(locking_depth_override_flag):
        df.locking_depth = locking_depth_override_value

    if resample_flag:  # Resampling case
        (resampled_lons, resampled_lats, resampled_locking_depths) = resample_trace(
            segment_ordered_lon,
            segment_ordered_lat,
            segment_ordered_dep,
            resample_length,
        )

        top_coordinates = np.zeros((len(resampled_lons), 3))
        bottom_coordinates = np.zeros((len(resampled_lons), 3))
        top_coordinates[:, 0] = np.array(resampled_lons)
        top_coordinates[:, 1] = np.array(resampled_lats)
        bottom_coordinates[:, 0] = np.array(resampled_lons)
        bottom_coordinates[:, 1] = np.array(resampled_lats)
        bottom_coordinates[:, 2] = -np.array(resampled_locking_depths)
        # bottom_coordinates[:, 2] = bottom_coordinates[:, 2] / depth_scaling

    else:  # No resampling case
        top_coordinates = np.zeros((len(df_keep), 3))
        bottom_coordinates = np.zeros((len(df_keep), 3))
        top_coordinates[:, 0] = np.array(segment_ordered_lon)
        top_coordinates[:, 1] = np.array(segment_ordered_lat)
        bottom_coordinates[:, 0] = np.array(segment_ordered_lon)
        bottom_coordinates[:, 1] = np.array(segment_ordered_lat)
        bottom_coordinates[:, 2] = -np.array(segment_ordered_dep) / depth_scaling

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
