import copy
import pygmt
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload

import celeri

command_file_name = "../data/command/western_north_america_command.json"
command, segment, block, meshes, station, mogi, sar = celeri.read_data(
    command_file_name
)
# station = celeri.process_station(station, command)
# segment = celeri.process_segment(segment, command, meshes)
station_file_name = "model_station.csv"
segment_file_name = "model_segment.csv"
station = pd.read_csv(command.station_file_name)
station = station.loc[:, ~station.columns.str.match("Unnamed")]
segment = pd.read_csv(command.segment_file_name)
segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]
closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)

# Create a plot with coast, Miller projection (J) over the continental US
min_lon = -135.0
max_lon = -110.0
min_lat = 30.0
max_lat = 50.0
region = [min_lon, max_lon, min_lat, max_lat]
topo_data = "@earth_relief_30s"
projection = "J-65/12c"


# Basic slip rate figure
fig = pygmt.Figure()
pygmt.config(MAP_FRAME_TYPE="plain")
pygmt.config(MAP_FRAME_PEN="0.25p")
pygmt.config(MAP_TICK_PEN="0.25p")
pygmt.makecpt(cmap="gray", series="-4000/4000/20", continuous=True)
fig.basemap(region=region, projection=projection, frame=True)

fig.grdimage(grid=topo_data, region=region, shading=True, projection=projection)

# Plot block boundaries
for i in range(closure.n_polygons()):
    fig.plot(
        x=closure.polygons[i].vertices[:, 0],
        y=closure.polygons[i].vertices[:, 1],
        pen="0.1p,0/0/0",
    )


fig.plot(x=station.lon, y=station.lat, style="c0.05", color="yellow", pen="0.1p,black")

fig.show()
