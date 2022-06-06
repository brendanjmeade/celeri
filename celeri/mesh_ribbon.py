import matplotlib.pyplot as plt
import numpy as np
import os
import meshio
import IPython
from pathlib import Path

geo_file_name = "mesh_test.geo"
gmsh_excutable_location = "/opt/homebrew/bin/gmsh"
smooth_trace = True
reference_mesh_size = 0.5

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

n_top_coordinates = top_coordinates.shape[0]
n_bottom_coordinates = bottom_coordinates.shape[0]

# Write the Gmsh geometry file
fid = open(geo_file_name, "w")
fid.write(f"charl = {reference_mesh_size};\n")

for i in range(n_top_coordinates):
    fid.write(
        f"Point({i + 1}) = {{"
        f"{top_coordinates[i, 0]:0.4f}, "
        f"{top_coordinates[i, 1]:0.4f}, "
        f"{top_coordinates[i, 2]:0.4f}, "
        f"charl}};\n"
    )

for i in range(n_bottom_coordinates):
    fid.write(
        f"Point({n_top_coordinates + i + 1}) = {{"
        f"{bottom_coordinates[i, 0]:0.4f}, "
        f"{bottom_coordinates[i, 1]:0.4f}, "
        f"{bottom_coordinates[i, 2]:0.4f}, "
        f"2.0}};\n"
    )

if smooth_trace:
    fid.write(f"CatmullRom(1) = {{1:{n_top_coordinates}}};\n")
    fid.write(
        f"CatmullRom(2) = {{{n_top_coordinates + 1}:{n_top_coordinates + n_bottom_coordinates}}};\n"
    )
    fid.write(f"CatmullRom(3) = {{{n_top_coordinates},{n_top_coordinates + 1}}};\n")
    fid.write(f"CatmullRom(4) = {{{n_top_coordinates + n_bottom_coordinates},1}};\n")
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

# Run gmsh
msh_file_name = f"{Path(geo_file_name).stem}.msh"
os.system(
    f"{gmsh_excutable_location} -2 {geo_file_name} -o {msh_file_name} -v 0 > /dev/null"
)


meshio_object = meshio.read(msh_file_name)
print(meshio_object)

# IPython.embed(banner1="")

# Read and plot mesh
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
    ax.plot3D(top_coordinates[i, 0], top_coordinates[i, 1], top_coordinates[i, 2], ".b")

for i in range(bottom_coordinates.shape[0]):
    ax.plot3D(
        bottom_coordinates[i, 0],
        bottom_coordinates[i, 1],
        bottom_coordinates[i, 2],
        ".g",
    )


plt.title(f"meshed {triangle_indices.shape[0]} triangles")
plt.show()

# IPython.embed(banner1="")
