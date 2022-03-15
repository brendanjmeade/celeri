import copy
import pygmt
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload

import celeri

plt.rcParams["text.usetex"] = False

command_file_name = "../data/command/western_north_america_command.json"
command, segment, block, meshes, station, mogi, sar = celeri.read_data(
    command_file_name
)
station = celeri.process_station(station, command)
segment = celeri.process_segment(segment, command, meshes)
sar = celeri.process_sar(sar, command)
closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)

# Plot TDE displacements from a single element
tri_idx = 0
vel_east, vel_north, vel_up = celeri.get_tri_displacements(
    station.lon.to_numpy(),
    station.lat.to_numpy(),
    meshes,
    command.material_lambda,
    command.material_mu,
    tri_idx=0,
    strike_slip=1,
    dip_slip=0,
    tensile_slip=0,
)
plt.figure()
plt.plot(meshes[0].lon1[tri_idx], meshes[0].lat1[tri_idx], ".r")
plt.plot(meshes[0].lon2[tri_idx], meshes[0].lat2[tri_idx], ".r")
plt.plot(meshes[0].lon3[tri_idx], meshes[0].lat3[tri_idx], ".r")
plt.quiver(station.lon, station.lat, vel_east, vel_north, scale=1e-3)
plt.show(block=False)

# Create a plot with coast, Miller projection (J) over the continental US
min_lon = -135.0
max_lon = -110.0
min_lat = 30.0
max_lat = 50.0
region = [min_lon, max_lon, min_lat, max_lat]
topo_data = "@earth_relief_30s"
# topo_data = '@earth_relief_01m'
projection = "J-65/12c"

fig = pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE="plain")
pygmt.config(MAP_FRAME_PEN="0.25p")
pygmt.config(MAP_TICK_PEN="0.25p")

pygmt.makecpt(
    cmap="topo",
    # cmap='terra',
    # series='-4000/4000/20',
    # series='-11000/4000/500',
    continuous=True,
)

fig.grdimage(
    grid=topo_data,
    region=region,
    # shading=True,
    projection=projection,
)

fig.coast(
    region=region,
    projection=projection,
    # land="lightgray",
    # water="white",
    area_thresh=4000,
    shorelines="0.25p,200/200/200",
    frame="p",
)

# Plot block boundaries
for i in range(closure.n_polygons()):
    fig.plot(
        x=closure.polygons[i].vertices[:, 0],
        y=closure.polygons[i].vertices[:, 1],
        pen="0.25p,0/0/0",
    )

# Plot GPS stations
fig.plot(x=station.lon, y=station.lat, style="c0.05", color="yellow", pen="0.1p,black")
fig.show()

# Stained glass block polygons
fig = pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE="plain")
pygmt.config(MAP_FRAME_PEN="0.25p")
pygmt.config(MAP_TICK_PEN="0.25p")

pygmt.makecpt(cmap="gray", series="-4000/4000/20", continuous=True)

fig.grdimage(
    grid=topo_data,
    region=region,
    shading=True,
    projection=projection,
)

# Plot blocks as transparent polygons
for i in range(closure.n_polygons()):
    red = 100 + 50 * np.random.rand(1)
    green = 150 + 100 * np.random.rand(1)
    blue = 150 + 100 * np.random.rand(1)
    color_string = str(red[0]) + "/" + str(green[0]) + "/" + str(blue[0]) + "@75"
    fig.plot(
        x=closure.polygons[i].vertices[:, 0],
        y=closure.polygons[i].vertices[:, 1],
        color=color_string,
    )

# Plot block boundaries
for i in range(closure.n_polygons()):
    fig.plot(
        x=closure.polygons[i].vertices[:, 0],
        y=closure.polygons[i].vertices[:, 1],
        pen="0.1p,0/0/0",
    )

fig.coast(
    region=region,
    projection=projection,
    area_thresh=4000,
    shorelines="0.25p,255/255/255",
    frame="p",
)
fig.show()

# Visualize mesh perimeter
celeri.get_ordered_edge_nodes(meshes)
plt.figure(figsize=(12, 7))
plt.plot(
    meshes[0].meshio_object.points[meshes[0].ordered_edge_nodes[:, 0], 0],
    meshes[0].meshio_object.points[meshes[0].ordered_edge_nodes[:, 0], 1],
)

plt.figure()
ax = plt.gca()
print(meshes[0].n_tde)
celeri.get_mesh_edge_elements(meshes)
celeri.plot_meshes(meshes, np.random.rand(meshes[0].n_tde), ax)
plt.gca().set_aspect("equal", adjustable="box")
plt.show(block=False)
