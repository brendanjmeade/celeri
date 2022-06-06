import matplotlib.pyplot as plt
import numpy as np
import os
import meshio
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


def plot_mesh(meshio_object, top_coordinates, bottom_coordinates):
    # Plot mesh
    triangle_indices = meshio_object.cells_dict["triangle"]
    line_indices = meshio_object.cells_dict["line"]

    plt.figure(figsize=(20, 10))
    ax = plt.axes(projection="3d")

    # Plot each mesh element
    for i in range(triangle_indices.shape[0]):
        ax.plot3D(
            [
                meshio_object.points[triangle_indices[i, 0], 0],
                meshio_object.points[triangle_indices[i, 1], 0],
                meshio_object.points[triangle_indices[i, 2], 0],
                meshio_object.points[triangle_indices[i, 0], 0],
            ],
            [
                meshio_object.points[triangle_indices[i, 0], 1],
                meshio_object.points[triangle_indices[i, 1], 1],
                meshio_object.points[triangle_indices[i, 2], 1],
                meshio_object.points[triangle_indices[i, 0], 1],
            ],
            [
                meshio_object.points[triangle_indices[i, 0], 2],
                meshio_object.points[triangle_indices[i, 1], 2],
                meshio_object.points[triangle_indices[i, 2], 2],
                meshio_object.points[triangle_indices[i, 0], 2],
            ],
            "-k",
            linewidth=0.5,
        )

    # Plot mesh perimeter
    for i in range(line_indices.shape[0]):
        ax.plot3D(
            [
                meshio_object.points[line_indices[i, 0], 0],
                meshio_object.points[line_indices[i, 1], 0],
            ],
            [
                meshio_object.points[line_indices[i, 0], 1],
                meshio_object.points[line_indices[i, 1], 1],
            ],
            [
                meshio_object.points[line_indices[i, 0], 2],
                meshio_object.points[line_indices[i, 1], 2],
            ],
            "-r",
            linewidth=2.0,
        )

    # Plot the initial coordinates
    for i in range(top_coordinates.shape[0]):
        ax.plot3D(
            top_coordinates[i, 0], top_coordinates[i, 1], top_coordinates[i, 2], ".b"
        )

    for i in range(bottom_coordinates.shape[0]):
        ax.plot3D(
            bottom_coordinates[i, 0],
            bottom_coordinates[i, 1],
            bottom_coordinates[i, 2],
            ".g",
        )

    plt.title(f"meshed {triangle_indices.shape[0]} triangles")
    plt.show()


def get_top_and_bottom_ribbon_coordinates():
    """
    This is just a placeholder for some operation that we'll do on a segment
    file based on flags in the segment file.
    """

    top_coordinates = np.array(
        [
            [1.2538000e00, -2.5934000e00, 0.0000000e00],
            [2.4805000e00, -2.1099000e00, 0.0000000e00],
            [3.1064000e00, -5.0550000e-01, 0.0000000e00],
            [4.0358000e00, 1.8681000e00, 0.0000000e00],
            [4.5867000e00, 3.3626000e00, 0.0000000e00],
            [4.9646000e00, 4.1758000e00, 0.0000000e00],
        ]
    )

    bottom_coordinates = np.array(
        [
            [1.2538000e00, -2.5934000e00, -5.0000000e00],
            [2.4805000e00, -2.1099000e00, -5.0000000e00],
            [3.1064000e00, -5.0550000e-01, -5.0000000e00],
            [4.0358000e00, 1.8681000e00, -5.0000000e00],
            [4.5867000e00, 3.3626000e00, -5.0000000e00],
            [4.9646000e00, 4.1758000e00, -5.0000000e00],
        ]
    )
    bottom_coordinates = np.flipud(bottom_coordinates)

    return top_coordinates, bottom_coordinates


def main():
    geo_file_name = "mesh_test.geo"
    gmsh_excutable_location = "/opt/homebrew/bin/gmsh"
    smooth_trace = True
    top_mesh_reference_size = 0.2
    bottom_mesh_reference_size = 2.0

    # Red top and bottom coordinates
    top_coordinates, bottom_coordinates = get_top_and_bottom_ribbon_coordinates()

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
    plot_mesh(meshio_object, top_coordinates, bottom_coordinates)

    # IPython.embed(banner1="")


if __name__ == "__main__":
    main()
