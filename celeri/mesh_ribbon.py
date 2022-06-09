import matplotlib.pyplot as plt
import numpy as np
import os
import meshio
import pandas as pd
import IPython
from pathlib import Path


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


def main():
    geo_file_name = "mesh_test.geo"
    gmsh_excutable_location = "/opt/homebrew/bin/gmsh"
    smooth_trace = False
    top_mesh_reference_size = 0.1
    bottom_mesh_reference_size = 2.0

    # TEMP: Read .csv dataframe from Emily
    df = pd.read_csv("naf_sorted_top_coordinates.csv")
    top_coordinates = np.zeros((len(df), 3))
    bottom_coordinates = np.zeros((len(df), 3))
    top_coordinates[:, 0] = df.lons.values
    top_coordinates[:, 1] = df.lats.values

    # Find shortest segment length
    longitude_diff = np.diff(top_coordinates[:, 0])
    latitude_diff = np.diff(top_coordinates[:, 1])
    approximate_segment_lengths = np.sqrt(longitude_diff ** 2.0 + latitude_diff ** 2.0)

    reference_length = 0.01
    resampled_lons = []
    resampled_lats = []
    resampled_locking_depths = []
    resampled_dips = []

    for i in range(len(df) - 1):
        if approximate_segment_lengths[i] > reference_length:
            n_segments = np.ceil(
                approximate_segment_lengths[i] / reference_length
            ).astype(int)
            print(f"need to divide {i} and {i + 1} in {n_segments}")
            new_longitudes = np.linspace(df.lons[i], df.lons[i + 1], n_segments + 1)
            new_latitudes = np.linspace(df.lats[i], df.lats[i + 1], n_segments + 1)
            for j in range(n_segments):
                resampled_lons.append(new_longitudes[j])
                resampled_lats.append(new_latitudes[j])
                resampled_locking_depths.append(df.locking_depth[i])
                resampled_dips.append(df.dip[i])

            print(df.lons[i], df.lons[i + 1])
            print(new_longitudes)
        else:
            resampled_lons.append(df.lons[i])
            resampled_lats.append(df.lats[i])
            resampled_locking_depths.append(df.locking_depth[i])
            resampled_dips.append(df.dip[i])

    # top_coordinates = np.zeros((len(resampled_lons), 3))
    # bottom_coordinates = np.zeros((len(resampled_lons), 3))
    # top_coordinates[:, 0] = np.array(resampled_lons)
    # top_coordinates[:, 1] = np.array(resampled_lats)
    # bottom_coordinates[:, 0] = np.array(resampled_lons)
    # bottom_coordinates[:, 1] = np.array(resampled_lats)
    # bottom_coordinates[:, 2] = -np.array(resampled_locking_depths)

    plt.figure()
    plt.plot(resampled_lons, resampled_lats, "ro")
    plt.plot(df.lons, df.lats, "b.")
    plt.show()

    IPython.embed(banner1="")

    bottom_coordinates[:, 0] = df.lons.values
    bottom_coordinates[:, 1] = df.lats.values
    bottom_coordinates[:, 2] = -df.locking_depth.values
    bottom_coordinates = np.flipud(bottom_coordinates)

    # Write a .geo file gmsh to run
    write_geo_file(
        geo_file_name,
        top_coordinates,
        bottom_coordinates,
        top_mesh_reference_size,
        bottom_mesh_reference_size,
        smooth_trace,
    )

    # Run gmsh
    msh_file_name = f"{Path(geo_file_name).stem}.msh"
    os.system(
        f"{gmsh_excutable_location} -2 {geo_file_name} -o {msh_file_name} -v 0 > /dev/null"
    )

    # Read the mesh file that gmsh created
    meshio_object = meshio.read(msh_file_name)

    # Plot the gmsh generated mesh
    os.system(f"{gmsh_excutable_location} {msh_file_name} &")

    # IPython.embed(banner1="")


if __name__ == "__main__":
    main()
