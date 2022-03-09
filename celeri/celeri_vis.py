import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections
from typing import List, Dict, Tuple

import celeri

EPS = np.finfo(float).eps


def test_plot():
    plt.figure()
    plt.plot(np.random.rand(3), "-r")
    plt.show()


def plot_matrix_abs_log(matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log10(np.abs(matrix + EPS)), cmap="plasma_r")
    plt.colorbar()
    plt.show()


def plot_meshes(meshes: List, fill_value: np.array, ax):
    for i in range(len(meshes)):
        x_coords = meshes[i].meshio_object.points[:, 0]
        y_coords = meshes[i].meshio_object.points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)

        if not ax:
            ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", cmap="rainbow"
        )
        pc.set_array(fill_value)
        ax.add_collection(pc)
        ax.autoscale()
        plt.colorbar(pc, label="slip (mm/yr)")

        # Add mesh edge
        x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
        y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
        plt.plot(x_edge, y_edge, color="black", linewidth=1)


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

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(meshes)):
        is_constrained_edge = np.zeros(meshes[i].n_tde)
        is_constrained_edge[meshes[i].top_elements] = meshes[i].top_slip_rate_constraint
        is_constrained_edge[meshes[i].bot_elements] = meshes[i].bot_slip_rate_constraint
        is_constrained_edge[meshes[i].side_elements] = meshes[
            i
        ].side_slip_rate_constraint
        x_coords = meshes[i].meshio_object.points[:, 0]
        y_coords = meshes[i].meshio_object.points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)
        ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", linewidth=0.25, cmap="Oranges"
        )
        pc.set_array(is_constrained_edge)
        ax.add_collection(pc)
        # ax.autoscale()

    plt.suptitle("inputs")
    plt.show()


def plot_estimation_summary(
    segment: pd.DataFrame,
    station: pd.DataFrame,
    meshes: List,
    estimation: Dict,
    lon_range: Tuple,
    lat_range: Tuple,
    quiver_scale: float,
):
    """Plot overview figures showing observed and modeled velocities as well
    as velocity decomposition and estimates slip rates.

    Args:
        segment (pd.DataFrame): Fault segments
        station (pd.DataFrame): GPS observations
        meshes (List): List of mesh dictionaries
        estimation (Dict): All estimated values
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
    plt.title("model velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel,
        estimation.north_vel,
        scale=quiver_scale,
        scale_units="inches",
        color="blue",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("residual velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_residual,
        estimation.north_vel_residual,
        scale=quiver_scale,
        scale_units="inches",
        color="green",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("rotation velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_rotation,
        estimation.north_vel_rotation,
        scale=quiver_scale,
        scale_units="inches",
        color="orange",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("elastic segment velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_elastic_segment,
        estimation.north_vel_elastic_segment,
        scale=quiver_scale,
        scale_units="inches",
        color="magenta",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("elastic tde velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_tde,
        estimation.north_vel_tde,
        scale=quiver_scale,
        scale_units="inches",
        color="black",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("segment strike-slip \n (negative right-lateral)")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if estimation.strike_slip_rate_sigma[i] < max_sigma_cutoff:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.strike_slip_rates[i]:.1f}({estimation.strike_slip_rate_sigma[i]:.1f})",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.strike_slip_rates[i]:.1f}(*)",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("segment dip-slip \n (positive convergences)")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if estimation.dip_slip_rate_sigma[i] < max_sigma_cutoff:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.dip_slip_rates[i]:.1f}({estimation.dip_slip_rate_sigma[i]:.1f})",
                color="blue",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.dip_slip_rates[i]:.1f}(*)",
                color="blue",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("segment tensile-slip \n (negative convergences)")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if estimation.tensile_slip_rate_sigma[i] < max_sigma_cutoff:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.tensile_slip_rates[i]:.1f}({estimation.tensile_slip_rate_sigma[i]:.1f})",
                color="green",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.tensile_slip_rates[i]:.1f}(*)",
                color="green",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("TDE slip (strike-slip)")
    common_plot_elements(segment, lon_range, lat_range)
    # plot_meshes(meshes, estimation.tde_strike_slip_rates, plt.gca())
    fill_value = estimation.tde_strike_slip_rates
    fill_value_range = [np.min(fill_value), np.max(fill_value)]
    ax = plt.gca()
    for i in range(len(meshes)):
        x_coords = meshes[i].meshio_object.points[:, 0]
        y_coords = meshes[i].meshio_object.points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)

        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", cmap="rainbow"
        )
        if i == 0:
            tde_slip_component_start = 0
            tde_slip_component_end = meshes[i].n_tde
        else:
            tde_slip_component_start = tde_slip_component_end
            tde_slip_component_end = tde_slip_component_start + meshes[i].n_tde
        pc.set_array(fill_value[tde_slip_component_start:tde_slip_component_end])
        pc.set_clim(fill_value_range)
        ax.add_collection(pc)
        # ax.autoscale()
        if i == len(meshes) - 1:
            plt.colorbar(pc, label="slip (mm/yr)")

        # Add mesh edge
        x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
        y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
        plt.plot(x_edge, y_edge, color="black", linewidth=1)

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("TDE slip (dip-slip)")
    common_plot_elements(segment, lon_range, lat_range)
    # plot_meshes(meshes, estimation.tde_dip_slip_rates, plt.gca())
    fill_value = estimation.tde_dip_slip_rates
    fill_value_range = [np.min(fill_value), np.max(fill_value)]
    ax = plt.gca()
    for i in range(len(meshes)):
        x_coords = meshes[i].meshio_object.points[:, 0]
        y_coords = meshes[i].meshio_object.points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)

        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", cmap="rainbow"
        )
        if i == 0:
            tde_slip_component_start = 0
            tde_slip_component_end = meshes[i].n_tde
        else:
            tde_slip_component_start = tde_slip_component_end
            tde_slip_component_end = tde_slip_component_start + meshes[i].n_tde
        pc.set_array(fill_value[tde_slip_component_start:tde_slip_component_end])
        pc.set_clim(fill_value_range)
        ax.add_collection(pc)
        # ax.autoscale()
        if i == len(meshes) - 1:
            plt.colorbar(pc, label="slip (mm/yr)")

        # Add mesh edge
        x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
        y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
        plt.plot(x_edge, y_edge, color="black", linewidth=1)

    plt.show()


def plot_segment_displacements(
    segment,
    station,
    command,
    segment_idx,
    strike_slip,
    dip_slip,
    tensile_slip,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    quiver_scale,
):
    u_east, u_north, u_up = celeri.get_okada_displacements(
        segment.lon1.values[segment_idx],
        segment.lat1[segment_idx],
        segment.lon2[segment_idx],
        segment.lat2[segment_idx],
        segment.locking_depth[segment_idx],
        segment.burial_depth[segment_idx],
        segment.dip[segment_idx],
        command.material_lambda,
        command.material_mu,
        strike_slip,
        dip_slip,
        tensile_slip,
        station.lon,
        station.lat,
    )
    plt.figure()
    plt.plot(
        [segment.lon1[segment_idx], segment.lon2[segment_idx]],
        [segment.lat1[segment_idx], segment.lat2[segment_idx]],
        "-r",
    )
    plt.quiver(
        station.lon,
        station.lat,
        u_east,
        u_north,
        scale=quiver_scale,
        scale_units="inches",
    )
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Okada displacements: longitude and latitude")
    plt.show()


def plot_strain_rate_components_for_block(closure, segment, station, block_idx):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    vel_east, vel_north, vel_up = celeri.get_strain_rate_displacements(
        station,
        segment,
        block_idx=block_idx,
        strain_rate_lon_lon=1,
        strain_rate_lat_lat=0,
        strain_rate_lon_lat=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 2)
    vel_east, vel_north, vel_up = celeri.get_strain_rate_displacements(
        station,
        segment,
        block_idx=block_idx,
        strain_rate_lon_lon=0,
        strain_rate_lat_lat=1,
        strain_rate_lon_lat=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 3)
    vel_east, vel_north, vel_up = celeri.get_strain_rate_displacements(
        station,
        segment,
        block_idx=block_idx,
        strain_rate_lon_lon=0,
        strain_rate_lat_lat=0,
        strain_rate_lon_lat=1,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )
    plt.show()


def plot_rotation_components(closure, station):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    vel_east, vel_north, vel_up = celeri.get_rotation_displacements(
        station.lon.values,
        station.lat.values,
        omega_x=1,
        omega_y=0,
        omega_z=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 2)
    vel_east, vel_north, vel_up = celeri.get_rotation_displacements(
        station.lon.values,
        station.lat.values,
        omega_x=0,
        omega_y=1,
        omega_z=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 3)
    vel_east, vel_north, vel_up = celeri.get_rotation_displacements(
        station.lon.values,
        station.lat.values,
        omega_x=0,
        omega_y=0,
        omega_z=1,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )
    plt.show()
