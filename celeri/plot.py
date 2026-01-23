from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cartopy.io.shapereader as shpreader
import colorcet as cc
import matplotlib.collections as mc
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)

from celeri.constants import EPS
from celeri.mesh import Mesh
from celeri.model import Model
from celeri.optimize_sqp import _get_coupling
from celeri.solve import Estimation
from celeri.spatial import (
    get_block_centroid,
    get_okada_displacements,
    get_rotation_displacements,
    get_strain_rate_displacements,
)


@dataclass(kw_only=True)
class PlotParams:
    """Plot parameters for visualization configuration."""

    # Figure configuration
    figsize_vectors: tuple[int, int] = field(default_factory=lambda: (12, 6))
    fontsize: int = 16

    # Geographic ranges and ticks
    lon_range: tuple[float, float] = (0.0, 0.0)
    lat_range: tuple[float, float] = (0.0, 0.0)
    lon_ticks: np.ndarray = field(default_factory=lambda: np.array([]))
    lat_ticks: np.ndarray = field(default_factory=lambda: np.array([]))

    # Slip rate limits
    slip_rate_min: float = 0.0
    slip_rate_max: float = 0.0

    # Land appearance
    land_color: str = "lightgray"
    land_linewidth: float = 0.5
    land_zorder: int = 0

    # Key/legend rectangle
    key_rectangle_anchor: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )
    key_rectangle_width: float = 3.0
    key_rectangle_height: float = 1.55

    # Key arrow configuration
    key_arrow_lon: float = 0.0
    key_arrow_lat: float = 0.0
    key_arrow_magnitude: float = 0.0
    key_arrow_text: str | None = None
    key_arrow_color: str = "k"

    # Key background
    key_background_color: str = "white"
    key_linewidth: float = 1.0
    key_edgecolor: str = "k"

    # Arrow magnitude and appearance
    arrow_magnitude_min: float = 0.0
    arrow_magnitude_max: float = 0.0
    arrow_colormap: Any = None
    arrow_scale_default: float = 0.0
    arrow_width: float = 0.0025
    arrow_linewidth: float = 0.5
    arrow_edgecolor: str = "k"

    # Segment lines
    segment_line_width_outer: float = 2.0
    segment_line_width_inner: float = 1.0
    segment_line_color_outer: str = "k"
    segment_line_color_inner: str = "w"

    # Coastlines
    coast_lon: np.ndarray = field(default_factory=lambda: np.array([]))
    coast_lat: np.ndarray = field(default_factory=lambda: np.array([]))


def plot_block_labels(segment, block, station, closure):
    plt.figure()
    plt.title("West and east labels")
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )

    for i in range(len(segment)):
        plt.text(
            segment.mid_lon_plate_carree.values[i],
            segment.mid_lat_plate_carree.values[i],
            str(segment["west_labels"][i]) + "," + str(segment["east_labels"][i]),
            fontsize=8,
            color="m",
            horizontalalignment="center",
            verticalalignment="center",
        )

    for i in range(len(station)):
        plt.text(
            station.lon.values[i],
            station.lat.values[i],
            str(station.block_label[i]),
            fontsize=8,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )

    for i in range(len(block)):
        plt.text(
            block.interior_lon.values[i],
            block.interior_lat.values[i],
            str(block.block_label[i]),
            fontsize=8,
            color="g",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.gca().set_aspect("equal")
    plt.show()


def plot_input_summary(
    model: Model,
    lon_range: tuple | None = None,
    lat_range: tuple | None = None,
    quiver_scale: float = 1e2,
    save: bool = True,
):
    """Plot overview figures showing observed and modeled velocities as well
    as velocity decomposition and estimates slip rates.

    Args:
        model: Model object containing all the data
        lon_range (Tuple): Latitude range (min, max)
        lat_range (Tuple): Latitude range (min, max)
        quiver_scale (float): Scaling for velocity arrows
        save (bool): Whether to save the figure to disk
    """
    if lon_range is None:
        lon_range = model.config.lon_range
    if lat_range is None:
        lat_range = model.config.lat_range

    segment = model.segment
    block = model.block
    station = model.station
    mogi = model.mogi
    meshes = model.meshes

    def common_plot_elements(segment: pd.DataFrame, lon_range: tuple, lat_range: tuple):
        """Elements common to all subplots.

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
                "interior\nstrain allowed",
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
    plt.title("Mogi Sources")
    common_plot_elements(segment, lon_range, lat_range)
    if len(mogi.lon > 0):
        plt.plot(mogi.lon, mogi.lat, "r+")

    # Skip a subplot
    subplot_index += 1

    # Plot a priori slip rate constraints
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("Strike-Slip Rate Constraints")
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
    plt.title("Dip-Slip Rate Constraints")
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
    plt.title("Tensile-Slip Rate Constraints")
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
    plt.title("Slip Rate Constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(meshes)):
        is_constrained_tde = np.zeros(meshes[i].n_tde)
        is_constrained_tde[meshes[i].top_elements] = meshes[
            i
        ].config.top_slip_rate_constraint
        is_constrained_tde[meshes[i].bot_elements] = meshes[
            i
        ].config.bot_slip_rate_constraint
        is_constrained_tde[meshes[i].side_elements] = meshes[
            i
        ].config.side_slip_rate_constraint
        x_coords = meshes[i].points[:, 0]
        y_coords = meshes[i].points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)
        ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = mc.PolyCollection(verts, edgecolor="none", linewidth=0.25, cmap="Oranges")
        pc.set_array(is_constrained_tde)
        ax.add_collection(pc)

    plt.suptitle("inputs")

    if save and model.config.output_path.exists():
        plt.savefig(model.config.output_path / "plot_input_summary.png", dpi=500)
        plt.savefig(model.config.output_path / "plot_input_summary.pdf")
        logger.success(
            f"Wrote figures {model.config.output_path}/plot_input_summary.(pdf, png)"
        )
    else:
        logger.info("No output_path specified, figures not saved to disk")

    plt.show(block=False)


def plot_estimation_summary(
    estimation: Estimation,
    lon_range: tuple | None = None,
    lat_range: tuple | None = None,
    quiver_scale: float | None = None,
    save: bool = True,
):
    """Plot overview figures showing observed and modeled velocities as well
    as velocity decomposition and estimates slip rates.

    Args:
        estimation (Dict): All estimated values
        lon_range (Tuple): Latitude range (min, max)
        lat_range (Tuple): Latitude range (min, max)
        quiver_scale (float): Scaling for velocity arrows
        save (bool): Whether to save the figure to disk
    """
    model = estimation.model

    if lon_range is None:
        lon_range = model.config.lon_range
    if lat_range is None:
        lat_range = model.config.lat_range
    if quiver_scale is None:
        quiver_scale = model.config.quiver_scale

    segment = model.segment
    station = model.station
    meshes = model.meshes

    def common_plot_elements(segment: pd.DataFrame, lon_range: tuple, lat_range: tuple):
        """Elements common to all subplots
        Args:
            segment (pd.DataFrame): Fault segments
            lon_range (Tuple): Longitude range (min, max)
            lat_range (Tuple): Latitude range (min, max).
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

    n_subplot_rows = 4
    n_subplot_cols = 3
    subplot_index = 1

    plt.figure(figsize=(18, 24))
    ax1 = plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
    plt.title("Observed Velocities")
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
    plt.title("Model Velocities")
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
    plt.title("Residual Velocities")
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
    plt.title("Rotation Velocities")
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
    plt.title("Elastic Segment Velocities")
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

    if model.config.solve_type != "dense_no_meshes":
        if len(meshes) > 0:
            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("Elastic TDE Velocities")
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

    if model.config.solve_type != "dense_no_meshes":
        if len(meshes) > 0:
            tde_strike_slip_dict = estimation.tde_strike_slip_rates
            tde_dip_slip_dict = estimation.tde_dip_slip_rates
            assert tde_strike_slip_dict is not None
            assert tde_dip_slip_dict is not None

            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("TDE Slip (strike-slip, mm/yr)")
            common_plot_elements(segment, lon_range, lat_range)

            all_values = np.concatenate(
                [np.ravel(vals) for vals in tde_strike_slip_dict.values()]
            )
            vmin = np.min(all_values)
            vmax = np.max(all_values)

            ax = plt.gca()
            pc = None
            for i in range(len(meshes)):
                pc = plot_mesh(
                    meshes[i],
                    fill_value=tde_strike_slip_dict[i],
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                )
            plt.colorbar(pc, ax=ax, shrink=0.7)

            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )

            plt.title("TDE Slip (dip-slip, mm/yr)")
            common_plot_elements(segment, lon_range, lat_range)

            all_values = np.concatenate(
                [np.ravel(vals) for vals in tde_dip_slip_dict.values()]
            )
            vmin = np.min(all_values)
            vmax = np.max(all_values)

            ax = plt.gca()
            pc = None
            for i in range(len(meshes)):
                pc = plot_mesh(
                    meshes[i],
                    fill_value=tde_dip_slip_dict[i],
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                )

            cbar = plt.colorbar(pc, ax=ax, shrink=0.7)
            n_ticks = 7
            ticks = np.linspace(vmin, vmax, n_ticks)
            cbar.set_ticks(ticks)

    # Plot strike-slip and dip-slip coupling
    if model.config.solve_type != "dense_no_meshes":
        if len(meshes) > 0 and estimation.mesh_estimate is not None:
            mesh_estimate = estimation.mesh_estimate

            # Strike-slip coupling
            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("Strike-Slip Coupling")
            common_plot_elements(segment, lon_range, lat_range)

            ax = plt.gca()
            pc = None
            for i in range(len(meshes)):
                mesh_data = mesh_estimate[mesh_estimate["mesh_idx"] == i]
                coupling_values = np.asarray(mesh_data["strike_slip_coupling"])
                pc = plot_mesh(
                    meshes[i],
                    fill_value=coupling_values,
                    ax=ax,
                )
            if pc is not None:
                plt.colorbar(pc, ax=ax, shrink=0.7)

            # Dip-slip coupling
            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("Dip-Slip Coupling")
            common_plot_elements(segment, lon_range, lat_range)

            ax = plt.gca()
            pc = None
            for i in range(len(meshes)):
                mesh_data = mesh_estimate[mesh_estimate["mesh_idx"] == i]
                coupling_values = np.asarray(mesh_data["dip_slip_coupling"])
                pc = plot_mesh(
                    meshes[i],
                    fill_value=coupling_values,
                    ax=ax,
                )
            if pc is not None:
                plt.colorbar(pc, ax=ax, shrink=0.7)

    # Plot segment strike-slip rates
    subplot_index += 1
    ax = plt.subplot(
        n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
    )
    plt.title("Segment Strike-Slip Rates")
    plot_segment_strike_slip_rates(
        segment,
        estimation.strike_slip_rates,
        ax,
        lon_range=lon_range,
        lat_range=lat_range,
        slip_rate_width_scale=0.25,
        show_legend=True,
        fontsize=8,
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
    plt.title("Residual Velocity Histogram")
    residual_velocity_vector = np.concatenate(
        (estimation.east_vel_residual, estimation.north_vel_residual)
    )
    mean_average_error = np.mean(np.abs(residual_velocity_vector))
    mean_squared_error = (
        np.sum(residual_velocity_vector**2.0) / residual_velocity_vector.size
    )

    # Create histogram of residual velocities
    plt.hist(residual_velocity_vector, 50)
    plt.xlabel("residual velocity (mm/yr)")
    plt.ylabel("N")
    plt.title(
        f"mae = {mean_average_error:.2f} (mm/yr), mse = {mean_squared_error:.2f} (mm/yr)^2"
    )

    if save and model.config.output_path.exists():
        plt.savefig(model.config.output_path / "plot_estimation_summary.png", dpi=500)
        plt.savefig(model.config.output_path / "plot_estimation_summary.pdf")
        logger.success(
            f"Wrote figures {model.config.output_path}/plot_estimation_summary.(pdf, png)"
        )
    plt.show(block=False)


def plot_matrix_abs_log(matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log10(np.abs(matrix + EPS)), cmap="plasma_r")
    plt.colorbar()
    plt.show()


def plot_mesh(
    mesh: Mesh,
    fill_value: np.ndarray,
    ax,
    *,
    vmin=None,
    vmax=None,
    cmap="seismic",
    center=0,
    set_limits=False,
):
    """Plot a 2D or cross-sectional mesh. Determines if a mesh
    is cross-sectional by checking the correlation coefficient
    between the longitude and latitude of the centroids. If cross-sectional,
    the mesh is plotted with the x-axis as the along-strike direction and
    the y-axis as the depth.

    Parameters
    ----------
    mesh : Mesh
        Mesh object containing vertices, centroids, points, and ordered edge nodes.
    fill_value : np.ndarray
        Array of values used to color the mesh polygons.
    ax : matplotlib Axes
        Matplotlib axes to draw the mesh on. If None, uses current axes.
    vmin : float, optional
        Minimum value for color normalization, defaults to array minimum.
    vmax : float, optional
        Maximum value for color normalization, defaults to array maximum.
    cmap : str, optional
        Colormap to use for coloring polygons (default 'seismic').
    center : float, optional
        Center value for diverging colormap normalization (default 0).
    set_limits : bool, optional
        Whether to set axis limits and aspect ratio (default False).

    Returns
    -------
    pc : matplotlib.collections.PolyCollection
        The PolyCollection object representing the mesh polygons.
    """
    import matplotlib.colors as colors

    if not ax:
        ax = plt.gca()

    if vmin is None:
        vmin = np.min([-1e-10, np.min(fill_value)])
    if vmax is None:
        vmax = np.max([1e-10, np.max(fill_value)])

    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    points = mesh.points
    cross_section = (
        np.abs(np.corrcoef(mesh.centroids[:, 0], mesh.centroids[:, 1])[0, 1]) > 0.9
    )
    if cross_section:
        lons = points[:, 0]
        lats = points[:, 1]
        lon_mean = np.mean(lons)
        lat_mean = np.mean(lats)
        lons_centered = lons - lon_mean
        lats_centered = lats - lat_mean

        coords = np.column_stack([lons_centered, lats_centered])
        cov = np.cov(coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = eigenvectors[:, np.argmax(eigenvalues)]

        along_strike_deg = coords @ pc1

        km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mean))
        km_per_deg_lat = 110.574
        scale_factor = np.sqrt(
            (pc1[0] * km_per_deg_lon) ** 2 + (pc1[1] * km_per_deg_lat) ** 2
        )
        dim0_coords = along_strike_deg * scale_factor
        dim1_coords = points[:, 2]
    else:
        dim0_coords = points[:, 0]
        dim1_coords = points[:, 1]

    vertex_array = np.asarray(mesh.verts)

    xy = np.c_[dim0_coords, dim1_coords]
    verts = xy[vertex_array]
    pc = mc.PolyCollection(
        verts,
        edgecolor="none",
        cmap=cmap,
        norm=norm,
    )

    pc.set_array(fill_value)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    x_edge = dim0_coords[mesh.ordered_edge_nodes[:, 0]]
    y_edge = dim1_coords[mesh.ordered_edge_nodes[:, 0]]
    x_edge = np.append(x_edge, dim0_coords[mesh.ordered_edge_nodes[0, 0]])
    y_edge = np.append(y_edge, dim1_coords[mesh.ordered_edge_nodes[0, 0]])
    ax.plot(x_edge, y_edge, color="black", linewidth=0.5)

    if set_limits:
        ax.set_aspect("equal", adjustable="box")
        margin = 0.01 * max(
            dim0_coords.max() - dim0_coords.min(), dim1_coords.max() - dim1_coords.min()
        )
        ax.set_xlim(dim0_coords.min() - margin, dim0_coords.max() + margin)
        ax.set_ylim(dim1_coords.min() - margin, dim1_coords.max() + margin)

    return pc


def plot_segment_displacements(
    segment,
    station,
    config,
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
    u_east, u_north, u_up = get_okada_displacements(
        segment.lon1.values[segment_idx],
        segment.lat1[segment_idx],
        segment.lon2[segment_idx],
        segment.lat2[segment_idx],
        segment.locking_depth[segment_idx],
        segment.dip[segment_idx],
        segment.azimuth[segment_idx],
        config.material_lambda,
        config.material_mu,
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
    uplimit = np.max(np.abs(u_up))
    plt.scatter(
        station.lon, station.lat, c=u_up, s=10, cmap="bwr", vmin=-uplimit, vmax=uplimit
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


def plot_strain_rate_components_for_block(closure, model, station, block_idx):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)

    block_centroid_lon, block_centroid_lat = get_block_centroid(
        model.segment, block_idx
    )

    vel_east, vel_north, _ = get_strain_rate_displacements(
        station.lon,
        station.lat,
        block_centroid_lon,
        block_centroid_lat,
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
    vel_east, vel_north, _ = get_strain_rate_displacements(
        station.lon,
        station.lat,
        block_centroid_lon,
        block_centroid_lat,
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
    vel_east, vel_north, _ = get_strain_rate_displacements(
        station.lon,
        station.lat,
        block_centroid_lon,
        block_centroid_lat,
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
    vel_east, vel_north, vel_up = get_rotation_displacements(
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
    vel_east, vel_north, vel_up = get_rotation_displacements(
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
    vel_east, vel_north, vel_up = get_rotation_displacements(
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


def get_default_plotting_options(config, estimation, station):
    """Parameters
    ----------
    estimation : dictionary
    station : dataframe
    config : dictionary

    Returns
    -------
    p : dictionary

    The returned dictionary includes the following keys and their default values:
        - figsize_vectors: (12, 6) - Default figure size for vector plots.
        - fontsize: 16 - Default font size.
        - lon_range: - Inferred from config.
        - lat_range: - Inferred from config.
        - lon_ticks: - Inferred from config.
        - lat_ticks: - Inferred from config.
        - slip_rate_min: - Inferred from estimation.
        - slip_rate_max: - Inferred from estimation.
        - land_color: "lightgray" - Color of land areas.
        - land_linewidth: 0.5 - Line width for land boundaries.
        - land_zorder: 0 - Z-order for land boundaries.
        - key_rectangle_anchor: [0, -90] - Anchor point for the key rectangle.
        - key_rectangle_width: 3.0 - Width of the key rectangle.
        - key_rectangle_height: 1.55 - Height of the key rectangle.
        - key_arrow_lon: 5.0 - Inferred from config.
        - key_arrow_lat: -85.0 - Inferred from config.
        - key_arrow_magnitude: - Magnitude for the key arrow.
        - key_arrow_text: - Text for the key arrow.
        - key_arrow_color: "k" - Color for the key arrow.
        - key_background_color: "white" - Background color for the key.
        - key_linewidth: 1.0 - Line width for the key.
        - key_edgecolor: "k" - Edge color for the key.
        - arrow_magnitude_min: 0.0 - Minimum arrow magnitude.
        - arrow_magnitude_max: - Maximum arrow magnitude (inferred from config).
        - arrow_colormap: cm.plasma - Colormap for arrows.
        - arrow_scale_default: - Inferred from config.
        - arrow_width: 0.0025 - Width for arrows.
        - arrow_linewidth: 0.5 - Line width for arrows.
        - arrow_edgecolor: "k" - Edge color for arrows.
        - segment_line_width_outer: 2.0 - Outer line width for segments.
        - segment_line_width_inner: 1.0 - Inner line width for segments.
        - segment_line_color_outer: "k" - Outer line color for segments.
        - segment_line_color_inner: "w" - Inner line color for segments.

    """
    slip_rate_scale = 0.5 * np.max(
        (
            np.abs(estimation.strike_slip_rates),
            np.abs(estimation.dip_slip_rates),
            np.abs(estimation.tensile_slip_rates),
        )
    )

    vel_scale = np.round(
        0.6
        * np.max(
            (
                np.abs(station.east_vel),
                np.abs(station.north_vel),
            )
        )
    )
    vel_scale = round(vel_scale / 5) * 5

    p = PlotParams()
    p.figsize_vectors = (12, 6)
    p.fontsize = 16
    p.lon_range = config.lon_range
    p.lat_range = config.lat_range
    p.lon_ticks = np.linspace(config.lon_range[0], config.lon_range[1], 3)
    p.lat_ticks = np.linspace(config.lat_range[0], config.lat_range[1], 3)
    p.slip_rate_min = -slip_rate_scale
    p.slip_rate_max = slip_rate_scale
    p.key_arrow_lon = np.mean(config.lon_range)
    p.key_arrow_lat = np.min(config.lat_range) + 0.05 * (
        config.lat_range[1] - config.lat_range[0]
    )
    p.key_arrow_magnitude = vel_scale
    p.key_arrow_text = f"{vel_scale:d} mm/yr"
    p.arrow_magnitude_max = 0.35 * vel_scale
    p.arrow_colormap = cm.plasma  # type: ignore
    p.arrow_scale_default = vel_scale

    return p


def plot_common_elements(p, segment, lon_range, lat_range):
    """Plots common map elements such as segments and axis settings.

    This function plots map segments as lines and sets the longitude and latitude
    ranges and ticks for the plot. It also adjusts the aspect ratio and sets labels
    and tick parameters.

    Parameters
    ----------
    p (dictionary): Plotting parameters
    segment (DataFrame): A DataFrame containing segment information with columns
                         'lon1', 'lon2', 'lat1', and 'lat2' representing the
                         starting and ending coordinates of each segment.
    lon_range (tuple): A tuple specifying the longitude range as (min_lon, max_lon).
    lat_range (tuple): A tuple specifying the latitude range as (min_lat, max_lat).

    The function performs the following steps:
    1. Plots the outer segment lines in black with a specified line width.
    2. Plots the inner segment lines in white with a specified line width.
    3. Sets the x and y axis limits to the provided longitude and latitude ranges.
    4. Sets the x and y axis ticks to predefined values.
    5. Adjusts the aspect ratio to be equal.
    6. Sets the x and y axis labels and their font sizes.
    7. Sets the tick parameters, including the label size.
    """
    for i in range(len(segment)):
        plt.plot(
            [segment.lon1[i], segment.lon2[i]],
            [segment.lat1[i], segment.lat2[i]],
            "-k",
            linewidth=p.segment_line_width_outer,
        )
    for i in range(len(segment)):
        plt.plot(
            [segment.lon1[i], segment.lon2[i]],
            [segment.lat1[i], segment.lat2[i]],
            "-w",
            linewidth=p.segment_line_width_inner,
        )

    plt.xlim([lon_range[0], lon_range[1]])
    plt.ylim([lat_range[0], lat_range[1]])
    plt.xticks(p.lon_ticks)
    plt.yticks(p.lat_ticks)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("longitude (degrees)", fontsize=p.fontsize)
    plt.ylabel("latitude (degrees)", fontsize=p.fontsize)
    plt.tick_params(labelsize=p.fontsize)


def plot_vel_arrows_elements(p, lon, lat, east_velocity, north_velocity, arrow_scale):
    """Plots velocity vectors as arrows on a map along with other map elements.

    This function plots velocity vectors given eastward and northward components
    of velocity, scaling the arrows appropriately. It also draws land boundaries,
    a white background rectangle for the key, and an arrow legend.

    Parameters
    ----------
    p (dictionary): Plotting parameters
    lon (ndarray): Array of longitudes.
    lat (ndarray): Array latitudes.
    east_velocity (ndarray): Array of eastward velocity components.
    north_velocity (ndarray): Array of northward velocity components.
    arrow_scale (float): Scaling factor for the arrows.

    The function uses global plotting parameters defined in a dictionary `p`,
    which should include keys for arrow properties, land properties, and key
    rectangle and arrow legend properties.

    The function performs the following steps:
    1. Calculates the magnitude of the velocity vectors.
    2. Normalizes the velocity magnitudes for colormap scaling.
    3. Plots the velocity vectors using `plt.quiver`.
    4. Draws land boundaries using `plt.fill`.
    5. Draws a white background rectangle for the key using `mpatches.Rectangle`.
    6. Adds an arrow legend using `plt.quiverkey`.
    7. Sets the aspect ratio of the plot to be equal.
    8. Displays the plot using `plt.show`.

    Minimal example:
    >>> east_velocity = np.array([1.0, 2.0, 3.0])
    >>> north_velocity = np.array([1.0, 2.0, 3.0])
    >>> arrow_scale = 1.0
    >>> plot_vel_arrows_elements(east_velocity, north_velocity, arrow_scale)
    """
    # Draw velocity vectors
    velocity_magnitude = np.sqrt(east_velocity**2.0 + north_velocity**2.0)
    norm = Normalize()
    norm.autoscale(velocity_magnitude)
    norm.vmin = p.arrow_magnitude_min
    norm.vmax = p.arrow_magnitude_max
    colormap = p.arrow_colormap
    quiver_handle = plt.quiver(
        lon,
        lat,
        east_velocity,
        north_velocity,
        scale=p.arrow_scale_default * arrow_scale,
        width=p.arrow_width,
        scale_units="inches",
        color=colormap(norm(velocity_magnitude)),
        linewidth=p.arrow_linewidth,
        edgecolor=p.arrow_edgecolor,
    )

    # Draw land
    plt.fill(
        p.coast_lon,
        p.coast_lat,
        color=p.land_color,
        linewidth=p.land_linewidth,
        zorder=p.land_zorder,
    )

    # Draw white background rectangle
    rect = mpatches.Rectangle(
        p.key_rectangle_anchor,
        p.key_rectangle_width,
        p.key_rectangle_height,
        fill=True,
        color=p.key_background_color,
        linewidth=p.key_linewidth,
        ec=p.key_edgecolor,
    )
    plt.gca().add_patch(rect)

    # Draw arrow legend
    plt.quiverkey(
        quiver_handle,
        p.key_arrow_lon,
        p.key_arrow_lat,
        p.key_arrow_magnitude,
        p.key_arrow_text,
        coordinates="data",
        color=p.key_arrow_color,
        fontproperties={"size": p.fontsize},
    )

    plt.gca().set_aspect("equal")
    plt.show()


def plot_vels(
    p, segment, lon, lat, east_vel, north_vel, arrow_scale, title_string="velocities"
):
    """Plots a map of velocity vectors with common map elements.

    This function creates a plot with a specified title, plotting segments and
    velocity vectors on a map. The plot includes common map elements such as
    axis settings and segment lines.

    Parameters
    ----------
    p (PlotParams): A PlotParams object containing plotting parameters.
    segment (DataFrame): A DataFrame containing segment information with columns
                         'lon1', 'lon2', 'lat1', and 'lat2' representing the
                         starting and ending coordinates of each segment.
    east_vel (ndarray): Array of eastward velocity components.
    north_vel (ndarray): Array of northward velocity components.
    arrow_scale (float): Arrow length scale factor.
    title_string (str): The title of the plot.

    The function performs the following steps:
    1. Creates a figure with a specified size.
    2. Sets the plot title with the specified font size.
    3. Plots common map elements including segment lines and axis settings.
    4. Plots velocity vectors as arrows with scaling and color mapping.
    """
    plt.figure(figsize=p.figsize_vectors)
    plt.title(title_string, fontsize=p.fontsize)
    plot_common_elements(p, segment, p.lon_range, p.lat_range)
    plot_vel_arrows_elements(p, lon, lat, east_vel, north_vel, arrow_scale=arrow_scale)


def plot_residuals(p, segment, station):
    """Plots the residuals of east and north velocity estimates as scatter plots with
    mean absolute error (MAE) and mean squared error (MSE) for a given station.

    The function creates two plots:
    1. A scatter plot of the station locations colored by MAE.
    2. A scatter plot of the station locations colored by MSE.

    Parameters
    ----------
    p : object
        An object containing plot configurations such as figure size, longitude range,
        and latitude range.
    segment : object
        An object representing a specific segment of data or region to be plotted.
    station : DataFrame
        A pandas DataFrame containing the station data with columns 'lon' and 'lat' for
        longitude and latitude respectively.

    Returns
    -------
    None
    """
    mae = np.abs(station.model_east_vel_residual.values) + np.abs(
        station.model_north_vel_residual.values
    )
    mse = np.sqrt(
        station.model_east_vel_residual.values**2.0
        + station.model_north_vel_residual.values**2.0
    )

    plt.figure(figsize=p.figsize_vectors)
    plt.scatter(
        station.lon,
        station.lat,
        s=25,
        edgecolors="k",
        c=mae,
        cmap="YlOrRd",
        linewidths=0.1,
    )
    plot_common_elements(p, segment, p.lon_range, p.lat_range)
    plt.clim(0, 10)
    plt.title("mean average error")
    plt.show()

    plt.figure(figsize=p.figsize_vectors)
    plt.scatter(
        station.lon,
        station.lat,
        s=25,
        edgecolors="k",
        c=mse,
        cmap="YlOrRd",
        linewidths=0.1,
    )
    plot_common_elements(p, segment, p.lon_range, p.lat_range)
    plt.clim(0, 10)
    plt.title("mean squared error")
    plt.show()


def plot_segment_rates(p, segment, estimation, rate_type, rate_scale=1):
    """Plots the slip rates (strike-slip, dip-slip, tensile-slip, or a combination of dip and tensile slip)
    for given segments on a map.

    The function creates a plot with segments color-coded and line width scaled based on the slip rates.
    The colors represent:
    - Red: Negative rates (right-lateral for strike-slip, normal for dip-slip, convergence for tensile-slip, convergence for combination)
    - Blue: Positive rates (left-lateral for strike-slip, reverse for dip-slip, extension for tensile-slip, extension for combination)

    A legend is added to distinguish the negative and positive rates.

    Parameters
    ----------
    p : object
        An object containing plot configurations such as figure size, fonts, colors, and map boundaries.
    segment : DataFrame
        A pandas DataFrame containing segment data with columns 'lon1', 'lon2', 'lat1', and 'lat2' for
        the start and end coordinates of each segment.
    rate_type : str
        A string indicating the type of slip rate to plot. Can be one of:
        - 'ss'   : strike-slip rates
        - 'ds'   : dip-slip rates
        - 'ts'   : tensile-slip rates
        - 'dsts' : combination of dip-slip and tensile-slip rates
    rate_scale : float, optional
        A scaling factor for the slip rate line widths, default is 1.

    Returns
    -------
    None
    """
    plt.figure(figsize=p.figsize_vectors)

    if rate_type == "ss":
        plt.title("strike-slip rates", fontsize=p.fontsize)
        label_text_negative = "right-lateral"
        label_text_positive = "left-lateral"
    elif rate_type == "ds":
        plt.title("dip-slip rates", fontsize=p.fontsize)
        label_text_negative = "normal"
        label_text_positive = "reverse"
    elif rate_type == "ts":
        plt.title("tensile-slip rates", fontsize=p.fontsize)
        label_text_negative = "convergence"
        label_text_positive = "extension"
    elif rate_type == "dsts":
        plt.title("dip+tensile-slip rates", fontsize=p.fontsize)
        label_text_negative = "convergence"
        label_text_positive = "extension"
    else:
        raise ValueError(
            f"Invalid rate_type: {rate_type}. Must be one of 'ss', 'ds', 'ts', or 'dsts'."
        )

    plot_common_elements(p, segment, p.lon_range, p.lat_range)

    plt.fill(
        p.coast_lon,
        p.coast_lat,
        color=p.land_color,
        linewidth=p.land_linewidth,
        zorder=p.land_zorder,
    )

    for i in range(len(segment)):
        if rate_type == "ss":
            if estimation.strike_slip_rates[i] < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale * estimation.strike_slip_rates[i],
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale * estimation.strike_slip_rates[i],
                )

        if rate_type == "ds":
            if estimation.dip_slip_rates[i] < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale * estimation.dip_slip_rates[i],
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale * estimation.dip_slip_rates[i],
                )

        if rate_type == "ts":
            if estimation.tensile_slip_rates[i] < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale * estimation.tensile_slip_rates[i],
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale * estimation.tensile_slip_rates[i],
                )

        if rate_type == "dsts":
            if (estimation.dip_slip_rates[i] - estimation.tensile_slip_rates[i]) < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale
                    * (estimation.dip_slip_rates[i] - estimation.tensile_slip_rates[i]),
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale
                    * (estimation.dip_slip_rates[i] - estimation.tensile_slip_rates[i]),
                )

    # Legend
    black_segments = mlines.Line2D(
        [],
        [],
        color="red",
        marker="s",
        linestyle="None",
        markersize=10,
        label=label_text_negative,
    )
    red_segments = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="s",
        linestyle="None",
        markersize=10,
        label=label_text_positive,
    )
    plt.legend(
        handles=[black_segments, red_segments],
        loc="lower right",
        fontsize=p.fontsize,
        framealpha=1.0,
        edgecolor="k",
    ).get_frame().set_boxstyle("Square")  # type: ignore


def plot_fault_geometry(p, segment, meshes):
    """Plots the fault geometry (segments and triangular dislocation element meshes) on a map.

    The function creates a plot with segments color-coded and line width scaled based on the slip rates.
    The colors represent:
    - Black : Standard segments
    - Red : Segments replaced by triangular dislocation element meshes

    Parameters
    ----------
    p : object
        An object containing plot configurations such as figure size, fonts, colors, and map boundaries.
    segment : DataFrame
        A pandas DataFrame containing segment data with columns 'lon1', 'lon2', 'lat1', and 'lat2' for
        the start and end coordinates of each segment.
    meshes : Dict

    Returns
    -------
    None
    """
    plt.figure(figsize=p.figsize_vectors)

    plot_common_elements(p, segment, p.lon_range, p.lat_range)

    plt.fill(
        p.coast_lon,
        p.coast_lat,
        color=p.land_color,
        linewidth=p.land_linewidth,
        zorder=p.land_zorder,
    )

    for i in range(len(meshes)):
        x_coords = meshes[i].points[:, 0]
        y_coords = meshes[i].points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)

        ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = mc.PolyCollection(verts, edgecolor="none", alpha=0.2, facecolor="red")
        ax.add_collection(pc)

        # Add mesh edge
        x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
        y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
        plt.plot(x_edge, y_edge, color="red", linewidth=1, linestyle="--")

    for i in range(len(segment)):
        if segment.mesh_file_index[i] == -1:
            plt.plot(
                [segment.lon1[i], segment.lon2[i]],
                [segment.lat1[i], segment.lat2[i]],
                "-k",
                linewidth=1,
            )
        else:
            plt.plot(
                [segment.lon1[i], segment.lon2[i]],
                [segment.lat1[i], segment.lat2[i]],
                "-r",
                linewidth=1,
            )
    plt.show()


def plot_mesh_mode(meshes, eigenvectors_to_tde_slip, mesh_idx, mode_idx, start_idx=0):
    plt.figure()
    plt.scatter(
        meshes[mesh_idx].lon_centroid,
        meshes[mesh_idx].lat_centroid,
        s=10,
        c=eigenvectors_to_tde_slip[mesh_idx][start_idx::2, mode_idx],
    )
    plt.plot(meshes[mesh_idx].x_perimeter, meshes[mesh_idx].y_perimeter, "-k")
    plt.gca().set_aspect("equal")
    plt.title(f"{meshes[0].file_name}, {mesh_idx=}, {mode_idx=}")
    plt.show()


def plot_tde_boundary_condition_labels(meshes, mesh_idx):
    top_indices = np.asarray(np.where(meshes[mesh_idx].top_elements))
    bot_indices = np.asarray(np.where(meshes[mesh_idx].bot_elements))
    side_indices = np.asarray(np.where(meshes[mesh_idx].side_elements))
    print(f"{top_indices.size=}")
    print(f"{bot_indices.size=}")
    print(f"{side_indices.size=}")

    plt.figure()
    plt.scatter(
        meshes[mesh_idx].lon_centroid,
        meshes[mesh_idx].lat_centroid,
        s=0.1,
        c="k",
    )
    plt.plot(
        meshes[mesh_idx].lon_centroid[top_indices].flatten(),
        meshes[mesh_idx].lat_centroid[top_indices].flatten(),
        "rx",
        label="top",
    )
    plt.plot(
        meshes[mesh_idx].lon_centroid[bot_indices].flatten(),
        meshes[mesh_idx].lat_centroid[bot_indices].flatten(),
        "gs",
        label="bot",
    )
    plt.plot(
        meshes[mesh_idx].lon_centroid[side_indices].flatten(),
        meshes[mesh_idx].lat_centroid[side_indices].flatten(),
        "c^",
        label="side",
    )

    plt.plot(meshes[mesh_idx].x_perimeter, meshes[mesh_idx].y_perimeter, "-k")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.title(f"{meshes[mesh_idx].file_name}, {mesh_idx=}")
    plt.show()


def plot_land(lon_min, lat_min, lon_max, lat_max):
    """Plot filled gray land within the given extent using plt."""
    extent_box = box(lon_min, lat_min, lon_max, lat_max)

    land_shp = shpreader.natural_earth(
        resolution="10m", category="physical", name="land"
    )
    reader = shpreader.Reader(land_shp)

    for geom in reader.geometries():
        if not geom.intersects(extent_box):
            continue
        clipped = geom.intersection(extent_box)
        if isinstance(clipped, Polygon):
            patch = mpatches.Polygon(
                clipped.exterior.coords, facecolor="lightgray", edgecolor="none"
            )
            plt.gca().add_patch(patch)
        elif isinstance(clipped, MultiPolygon):
            for part in clipped.geoms:
                patch = mpatches.Polygon(
                    part.exterior.coords, facecolor="lightgray", edgecolor="none"
                )
                plt.gca().add_patch(patch)


def plot_coastlines(lon_min, lat_min, lon_max, lat_max):
    """Plot coastlines as black lines within the given extent using plt."""
    extent_box = box(lon_min, lat_min, lon_max, lat_max)

    coast_shp = shpreader.natural_earth(
        resolution="10m", category="physical", name="coastline"
    )
    reader = shpreader.Reader(coast_shp)

    coastlines = []

    for geom in reader.geometries():
        if not geom.intersects(extent_box):
            continue
        clipped = geom.intersection(extent_box)

        # Recursively extract line coordinates
        def extract_lines(g):
            if isinstance(g, LineString):
                coastlines.append(list(g.coords))
            elif isinstance(g, MultiLineString):
                for part in g.geoms:
                    extract_lines(part)
            elif isinstance(g, GeometryCollection):
                for part in g.geoms:
                    extract_lines(part)

        extract_lines(clipped)

    line_collection = mc.LineCollection(coastlines, colors="black", linewidths=0.5)
    plt.gca().add_collection(line_collection)


def plot_segment_strike_slip_rates(
    segment: pd.DataFrame,
    strike_slip_rates: np.ndarray,
    ax,
    *,
    lon_range: tuple | None = None,
    lat_range: tuple | None = None,
    slip_rate_width_scale: float = 0.25,
    show_legend: bool = True,
    fontsize: int = 10,
):
    """Plot estimated strike-slip rates on fault segments.

    Segments are color-coded and scaled by slip rate:
    - Orange: right-lateral (negative rates)
    - Blue: left-lateral (positive rates)

    Parameters
    ----------
    segment : pd.DataFrame
        DataFrame containing segment data with columns 'lon1', 'lon2', 'lat1', 'lat2'.
    strike_slip_rates : np.ndarray
        Array of strike-slip rates for each segment.
    ax : matplotlib Axes
        Matplotlib axes to draw on.
    lon_range : tuple, optional
        Longitude range (min, max). If None, auto-scales to data.
    lat_range : tuple, optional
        Latitude range (min, max). If None, auto-scales to data.
    slip_rate_width_scale : float, optional
        Scale factor for line width based on slip rate (default 0.25).
    show_legend : bool, optional
        Whether to show the legend (default True).
    fontsize : int, optional
        Font size for legend text (default 10).
    """
    for i in range(len(segment)):
        rate = strike_slip_rates[i]
        ax.plot(
            [segment.lon1.iloc[i], segment.lon2.iloc[i]],
            [segment.lat1.iloc[i], segment.lat2.iloc[i]],
            "-",
            color="tab:orange" if rate < 0 else "tab:blue",
            linewidth=slip_rate_width_scale * abs(rate),
        )

    if lon_range is not None:
        ax.set_xlim([lon_range[0], lon_range[1]])
    if lat_range is not None:
        ax.set_ylim([lat_range[0], lat_range[1]])
    ax.set_aspect("equal", adjustable="box")

    if show_legend:
        orange_segments = mlines.Line2D(
            [],
            [],
            color="tab:orange",
            marker="s",
            linestyle="None",
            markersize=8,
            label="right-lateral",
        )
        blue_segments = mlines.Line2D(
            [],
            [],
            color="tab:blue",
            marker="s",
            linestyle="None",
            markersize=8,
            label="left-lateral",
        )
        ax.legend(
            handles=[orange_segments, blue_segments],
            loc="lower right",
            fontsize=fontsize,
            framealpha=1.0,
            edgecolor="k",
        ).get_frame()


def plot_coupling(estimation: Estimation, *, mesh_idx: int):
    operators = estimation.operators
    block = estimation.model.block
    index = operators.index
    meshes = estimation.model.meshes

    assert operators.eigen is not None
    assert index.eigen is not None

    # Multiply rotation vector components by TDE slip rate partials
    kinematic = (
        operators.rotation_to_tri_slip_rate[mesh_idx]
        @ estimation.state_vector[0 : 3 * len(block)]
    )

    elastic = (
        operators.eigen.eigenvectors_to_tde_slip[mesh_idx]
        @ estimation.state_vector[
            index.eigen.start_col_eigen[mesh_idx] : index.eigen.end_col_eigen[mesh_idx]
        ]
    )

    # Calculate final coupling and smoothed kinematic
    tde_coupling_ss, kinematic_tde_rates_ss_smooth = _get_coupling(
        meshes[mesh_idx].lon_centroid,
        meshes[mesh_idx].lat_centroid,
        elastic[0::2],
        kinematic[0::2],
        smoothing_length_scale=meshes[
            mesh_idx
        ].config.iterative_coupling_smoothing_length_scale,
        kinematic_slip_regularization_scale=meshes[
            mesh_idx
        ].config.iterative_coupling_kinematic_slip_regularization_scale,
    )

    tde_coupling_ds, kinematic_tde_rates_ds_smooth = _get_coupling(
        meshes[mesh_idx].lon_centroid,
        meshes[mesh_idx].lat_centroid,
        elastic[1::2],
        kinematic[1::2],
        smoothing_length_scale=meshes[
            mesh_idx
        ].config.iterative_coupling_smoothing_length_scale,
        kinematic_slip_regularization_scale=meshes[
            mesh_idx
        ].config.iterative_coupling_kinematic_slip_regularization_scale,
    )

    # Strike-slip
    plt.figure(figsize=(15, 2))
    plt.subplot(1, 4, 1)
    plot_mesh(meshes[mesh_idx], fill_value=kinematic[0::2], ax=plt.gca())
    plt.title("ss kinematic")

    plt.subplot(1, 4, 2)
    plot_mesh(meshes[mesh_idx], fill_value=kinematic_tde_rates_ss_smooth, ax=plt.gca())
    plt.title("ss kinematic (smooth)")

    plt.subplot(1, 4, 3)
    plot_mesh(meshes[mesh_idx], fill_value=elastic[0::2], ax=plt.gca())
    plt.title("ss elastic")

    plt.subplot(1, 4, 4)
    plot_mesh(meshes[mesh_idx], fill_value=tde_coupling_ss, ax=plt.gca())
    plt.title("ss coupling")

    # Dip-slip
    plt.figure(figsize=(15, 2))
    plt.subplot(1, 4, 1)
    plot_mesh(meshes[mesh_idx], fill_value=kinematic[1::2], ax=plt.gca())
    plt.title("ds kinematic")

    plt.subplot(1, 4, 2)
    plot_mesh(meshes[mesh_idx], fill_value=kinematic_tde_rates_ds_smooth, ax=plt.gca())
    plt.title("ds kinematic (smooth)")

    plt.subplot(1, 4, 3)
    plot_mesh(meshes[mesh_idx], fill_value=elastic[1::2], ax=plt.gca())
    plt.title("ds elastic")

    plt.subplot(1, 4, 4)
    plot_mesh(meshes[mesh_idx], fill_value=tde_coupling_ds, ax=plt.gca())
    plt.title("ds coupling")

    plt.show()


def _plot_common_evolution_elements():
    plt.xlim([-90, 90])
    plt.ylim([-90, 90])
    plt.xticks([-90, 0, 90])
    plt.yticks([-90, 0, 90])
    plt.gca().set_aspect("equal")


def _plot_evolution(mesh: Mesh, field1: np.ndarray, field2: np.ndarray):
    LINE_COLOR = "lightgray"
    for i in range(mesh.n_tde):
        plt.plot(
            field1[i, :],
            field2[i, :],
            "-",
            linewidth=0.1,
            color=LINE_COLOR,
            zorder=1,
        )
    plt.plot(field1[:, -1], field2[:, -1], ".k", markersize=0.5)


def plot_coupling_evolution(estimation: Estimation, *, mesh_idx: int):
    mesh = estimation.model.meshes[mesh_idx]

    if estimation.trace is None:
        raise ValueError("Estimation trace is not available for plotting.")

    store_ss_kinematic = estimation.trace["ss_kinematic"]
    store_ss_elcon = estimation.trace["ss_estimated"]
    store_ss_lower = estimation.trace["ss_lower"]
    store_ss_upper = estimation.trace["ss_upper"]
    store_ds_kinematic = estimation.trace["ds_kinematic"]
    store_ds_elcon = estimation.trace["ds_estimated"]
    store_ds_lower = estimation.trace["ds_lower"]
    store_ds_upper = estimation.trace["ds_upper"]

    def plot_background():
        REGULARIZATION_RATE = 1.0
        levels = 101
        j = np.linspace(-100, 100, 1000)
        b = np.linspace(-100, 100, 1000)
        j_grid, b_grid = np.meshgrid(j, b)
        j_grid_orig = np.copy(j_grid)
        b_grid_orig = np.copy(b_grid)
        coupling, _ = _get_coupling(
            0,
            0,
            b_grid.flatten(),
            j_grid.flatten(),
            smoothing_length_scale=0.0,
            kinematic_slip_regularization_scale=REGULARIZATION_RATE,
        )
        coupling_grid = np.reshape(coupling, (1000, 1000))
        coupling_grid[coupling_grid > 1.0] = np.nan
        coupling_grid[coupling_grid < 0.0] = np.nan

        # Create half colormap
        # Retrieve a colorcet colormap
        # full_cmap = cc.cm["coolwarm_r"]  # Replace with your desired colormap
        # full_cmap = cc.cm["CET_D8_r"]  # Replace with your desired colormap
        # full_cmap = cc.cm["cwr_r"]  # Replace with your desired colormap
        full_cmap = cc.cm["bmy_r"]  # Replace with your desired colormap

        # Extract half of the colormap
        n_colors = full_cmap.N  # Total number of colors in the colormap
        half_cmap = LinearSegmentedColormap.from_list(
            "half_cmap", full_cmap(np.linspace(0, 0.5, n_colors // 2))
        )
        # cmap = half_cmap.reversed()
        cmap = half_cmap

        ch = plt.contourf(
            j_grid_orig, b_grid_orig, coupling_grid, cmap=cmap, levels=levels
        )
        return ch

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    ch = plot_background()
    _plot_evolution(mesh, store_ss_kinematic[mesh_idx], store_ss_elcon[mesh_idx])
    _plot_common_evolution_elements()
    plt.xlabel("$v$ strike-slip kinematic (mm/yr)")
    plt.ylabel("$v$ strike-slip elastic (mm/yr)")
    cax = inset_axes(
        plt.gca(),
        width="20%",
        height="30%",
        loc="upper right",
        bbox_to_anchor=(0.0, 0.0, 0.07, 0.95),  # Position in axes fraction
        bbox_transform=plt.gca().transAxes,
        borderpad=0,
    )
    plt.colorbar(ch, cax=cax, ticks=[0.0, 1.0], label="coupling")

    plt.subplot(2, 2, 2)
    ch = plot_background()
    _plot_evolution(mesh, store_ss_kinematic[mesh_idx], store_ss_lower[mesh_idx])
    _plot_evolution(mesh, store_ss_kinematic[mesh_idx], store_ss_upper[mesh_idx])
    _plot_common_evolution_elements()
    plt.xlabel("$v$ strike-slip kinematic (mm/yr)")
    plt.ylabel("$v$ strike-slip bounds (mm/yr)")
    cax = inset_axes(
        plt.gca(),
        width="20%",
        height="30%",
        loc="upper right",
        bbox_to_anchor=(0.0, 0.0, 0.07, 0.95),  # Position in axes fraction
        bbox_transform=plt.gca().transAxes,
        borderpad=0,
    )
    plt.colorbar(ch, cax=cax, ticks=[0.0, 1.0], label="coupling")

    plt.subplot(2, 2, 3)
    ch = plot_background()
    _plot_evolution(mesh, store_ds_kinematic[mesh_idx], store_ds_elcon[mesh_idx])
    _plot_common_evolution_elements()
    plt.xlabel("$v$ dip-slip kinematic (mm/yr)")
    plt.ylabel("$v$ dip-slip elastic (mm/yr)")
    cax = inset_axes(
        plt.gca(),
        width="20%",
        height="30%",
        loc="upper right",
        bbox_to_anchor=(0.0, 0.0, 0.07, 0.95),  # Position in axes fraction
        bbox_transform=plt.gca().transAxes,
        borderpad=0,
    )
    plt.colorbar(ch, cax=cax, ticks=[0.0, 1.0], label="coupling")

    plt.subplot(2, 2, 4)
    ch = plot_background()
    _plot_evolution(mesh, store_ds_kinematic[mesh_idx], store_ds_lower[mesh_idx])
    _plot_evolution(mesh, store_ds_kinematic[mesh_idx], store_ds_upper[mesh_idx])
    _plot_common_evolution_elements()
    plt.xlabel("$v$ dip-slip kinematic (mm/yr)")
    plt.ylabel("$v$ dip-slip bounds (mm/yr)")
    cax = inset_axes(
        plt.gca(),
        width="20%",
        height="30%",
        loc="upper right",
        bbox_to_anchor=(0.0, 0.0, 0.07, 0.95),  # Position in axes fraction
        bbox_transform=plt.gca().transAxes,
        borderpad=0,
    )
    plt.colorbar(ch, cax=cax, ticks=[0.0, 1.0], label="coupling")
    plt.suptitle(f"{mesh.name}")
    plt.show()
