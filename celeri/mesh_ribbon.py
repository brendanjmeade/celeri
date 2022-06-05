import matplotlib.pyplot as plt
import numpy as np
import os

# import pandas as pd

from pathlib import Path

geo_file_name = "mesh_test.geo"
smooth_trace = False
reference_mesh_size = 0.05

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
        f"charl}};\n"
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
os.system(f"gmsh -2 {geo_file_name} -o {Path(geo_file_name).stem}.msh -v 0 > /dev/null")
