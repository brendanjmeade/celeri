#!/usr/bin/env python3

import IPython
import pygmt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import celeri


def main():
    # Read in data from an output folder
    output_folder_name = "../runs/2022-05-03-21-28-56"
    segment = pd.read_csv(output_folder_name + "/model_segment.csv")
    station = pd.read_csv(output_folder_name + "/model_station.csv")

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
    pygmt.makecpt(cmap="topo", continuous=True)

    # fig.grdimage(grid=topo_data, region=region, projection=projection)
    fig.coast(
        region=region,
        projection=projection,
        area_thresh=4000,
        shorelines="0.25p,200/200/200",
        frame="p",
    )

    # Plot block boundaries with one call to plot
    lons = np.full([3 * len(segment)], np.nan)
    lats = np.full([3 * len(segment)], np.nan)
    lons[0::3] = segment.lon1.values
    lons[1::3] = segment.lon2.values
    lats[0::3] = segment.lat1.values
    lats[1::3] = segment.lat2.values
    fig.plot(x=lons, y=lats, pen="0.25p,0/0/0")

    # Plot GPS stations
    fig.plot(
        x=station.lon, y=station.lat, style="c0.05", color="yellow", pen="0.1p,black"
    )
    fig.show()

    # Plot mesh
    command_file_name = "../data/command/western_north_america_command.json"
    command = celeri.get_command(command_file_name)
    segment, block, meshes, station, mogi, sar = celeri.read_data(command)
    station = celeri.process_station(station, command)
    segment = celeri.process_segment(segment, command, meshes)

    # matplotlib version
    # plt.figure()
    # ax = plt.gca()
    # print(meshes[0].n_tde)
    # celeri.get_mesh_edge_elements(meshes)
    # celeri.plot_meshes(meshes, np.random.rand(meshes[0].n_tde), ax)
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.show()

    # pygmt version
    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(MAP_FRAME_PEN="0.25p")
    pygmt.config(MAP_TICK_PEN="0.25p")
    pygmt.makecpt(cmap="topo", continuous=True)

    fig.coast(
        region=region,
        projection=projection,
        area_thresh=4000,
        shorelines="0.25p,200/200/200",
        frame="p",
    )
    fig.plot(x=[-130, -125, -127], y=[35, 40, 39], color="red", pen="1.1p,black")
    fig.plot(x=[-128, -127, -129], y=[35, 40, 39], color="red", pen="1.1p,black")

    x = np.array([-130, -125, -127, np.nan, -128, -127, -129])
    y = np.array([35, 40, 39, np.nan, 35, 40, 39])

    data = [
        "> -Z0",
        "0 0",
        "2 2",
        "4 0",
        "> -Z3",
        "0 0",
        "2 2",
        "0 4",
    ]
    # fig.plot(x=x, y=y, color="red", pen="1.1p,black")
    # fig.plot(x=x, y=y, color="red", pen="1.1p,black")

    fig.plot(data=data, color="red", pen="1.1p,black")
    fig.show()

    # Drop into REPL
    # IPython.embed(banner1="")


if __name__ == "__main__":
    main()
