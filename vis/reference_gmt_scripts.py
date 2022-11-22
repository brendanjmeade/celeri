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
station = pd.read_csv(station_file_name)
station = station.loc[:, ~station.columns.str.match("Unnamed")]
segment = pd.read_csv(segment_file_name)
segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]
closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)

def get_region(segment, block, station):
    
    


def write_tri_slip_file(meshes):
        for i in range(len(meshes)):
            




# Create a plot with coast, Miller projection (J) over the continental US
min_lon = -170.0
max_lon = 170.0
min_lat = 00.0
max_lat = 80.0
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

# fig.grdimage(grid=topo_data, region=region, shading=True, projection=projection)

# Plot segments
n_segment = len(segment)
lon_list = np.nan * np.zeros(3 * n_segment)
lat_list = np.nan * np.zeros(3 * n_segment)
lon_list[0::3] = segment.lon1
lon_list[1::3] = segment.lon2
lat_list[0::3] = segment.lat1
lat_list[1::3] = segment.lat2
fig.plot(x=lon_list, y=lat_list, pen="0.1p,255/0/0")

fig.plot(x=station.lon, y=station.lat, style="c0.05", color="yellow", pen="0.1p,black")

fig.show()
