import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def plot_input_summary(
    segment: pd.DataFrame,
    station: pd.DataFrame,
    block: pd.DataFrame,
    meshes: List,
    mogi: pd.DataFrame,
    sar: pd.DataFrame,
    lon_range: Tuple,
    lat_range: Tuple,
    quiver_scale: float,
):
    """Plot overview figures showing observed and modeled velocities as well
    as velocity decomposition and estimates slip rates.

    Args:
        segment (pd.DataFrame): Fault segments
        station (pd.DataFrame): GPS observations
        block (pd.DataFrame): Block interior points and priors
        meshes (List): Mesh geometries and properties
        mogi (pd.DataFrame): Mogi sources
        sar (pd.DataFrame): SAR observations
        lon_range (Tuple): Latitude range (min, max)
        lat_range (Tuple): Latitude range (min, max)
        quiver_scale (float): Scaling for velocity arrows
    """

    def common_plot_elements(segment: pd.DataFrame, lon_range: Tuple, lat_range: Tuple):
        """Elements common to all subplots

        Args:
            segment (pd.DataFrame): Fault segments
            lon_range (Tuple): Longitude range (min, max)
            lat_range (Tuple): Latitude range (min, max)
        """
        for i in range(len(segment)):
            if segment.dip[i] == 90.0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-k",
                    linewidth=0.5,
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=0.5,
                )

        plt.xlim([lon_range[0], lon_range[1]])
        plt.ylim([lat_range[0], lat_range[1]])
        plt.gca().set_aspect("equal", adjustable="box")

    max_sigma_cutoff = 99.0
    n_subplot_rows = 4
    n_subplot_cols = 3
    subplot_index = 0

    plt.figure(figsize=(12, 16))

    subplot_index += 1
    ax1 = plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
    plt.title("observed velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        station.east_vel,
        station.north_vel,
        scale=quiver_scale,
        scale_units="inches",
        color="red",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("LOS")
    common_plot_elements(segment, lon_range, lat_range)

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("blocks\nprior rotation")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(block)):
        if block.rotation_flag.values[i] > 0:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "r+")
            plt.text(
                block.interior_lon[i],
                block.interior_lat[i],
                f"lon = {block.euler_lon[i]:.3f}\nlat = {block.euler_lat[i]:.3f}\nrate = {block.rotation_rate[i]:.3f}",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "bx")

    # Plot blocks that can have interior strain
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("blocks\ninterior strain allowed")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(block)):
        if block.strain_rate_flag[i] > 0:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "r+")
            plt.text(
                block.interior_lon[i],
                block.interior_lat[i],
                f"interior\nstrain allowed",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "bx")

    # Plot mogi sources
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("Mogi sources")
    common_plot_elements(segment, lon_range, lat_range)
    plt.plot(mogi.lon, mogi.lat, "r+")

    # Skip a subplot
    subplot_index += 1

    # Plot a priori slip rate constraints
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("strike-slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if segment.ss_rate_flag[i] == 1:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{segment.ss_rate[i]:.1f}({segment.ss_rate_sig[i]:.1f})",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("dip-slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if segment.ds_rate_flag[i] == 1:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{segment.ds_rate[i]:.1f}({segment.ds_rate_sig[i]:.1f})",
                color="blue",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("tensile-slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if segment.ts_rate_flag[i] == 1:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{segment.ts_rate[i]:.1f}({segment.ts_rate_sig[i]:.1f})",
                color="green",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    # TODO: #49 Plot mesh geometries and slip rate constraints
    plt.suptitle("inputs")
    plt.show()
