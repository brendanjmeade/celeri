"""Identify and plot meshes that cross a posterior resolvability threshold.

Given an MCMC estimation, computes the minimum posterior standard deviation
across triangular elements for each mesh's coupling field (a dimensionless
fraction, typically bounded to [0, 1]).  Meshes whose minimum element-wise
std falls below a cutoff are considered "resolved" — the data contain enough
information to constrain at least part of their coupling.

Only meshes with a ``coupling_{mesh_idx}_{ss|ds}`` variable in the MCMC
posterior are considered; elastic-only meshes are silently skipped.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import matplotlib.figure
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from celeri.plot import (
    PlotParams,
    plot_coastlines,
    plot_common_elements,
    plot_land,
    plot_vel_arrows_elements,
)
from celeri.solve import Estimation


def mesh_posterior_stds(
    estimation: Estimation,
    kind: Literal["ss", "ds"],
    *,
    reduce: Literal["min", "median", "max"] = "min",
) -> pd.Series:
    """Minimum posterior std of each mesh's coupling fraction.

    For each mesh that has a ``coupling_{mesh_idx}_{kind}`` variable in the
    MCMC posterior, computes the standard deviation of the dimensionless
    coupling fraction across chains and draws, then takes the **minimum**
    across triangular elements.  A low value means the data can resolve at
    least part of that mesh's coupling field.

    Meshes parameterized with elastic slip rates (mm/yr) rather than coupling
    fractions do not have a ``coupling_*`` posterior variable and are silently
    skipped.

    Parameters
    ----------
    estimation
        A solved estimation with an MCMC trace.
    kind
        Slip direction: ``"ss"`` (strike-slip) or ``"ds"`` (dip-slip).
    reduce
        Element-wise aggregation: ``"min"`` (default), ``"median"``, or
        ``"max"``.

    Returns
    -------
    pd.Series
        Indexed by mesh index, values are minimum posterior std of the
        dimensionless coupling fraction, sorted ascending.
    """
    if estimation.mcmc_trace is None:
        raise ValueError("Estimation has no MCMC trace.")

    stds: dict[int, float] = {}
    for mesh_idx in range(len(estimation.model.meshes)):
        name = f"coupling_{mesh_idx}_{kind}"
        if name in estimation.mcmc_trace.posterior:
            element_stds = estimation.mcmc_trace.posterior[name].std(["chain", "draw"])
            stds[mesh_idx] = float(getattr(element_stds, reduce)().values)

    return pd.Series(stds, dtype=float).sort_values()


def resolved_mesh_indices(
    estimation: Estimation,
    kind: Literal["ss", "ds"],
    std_cutoff: float,
    *,
    reduce: Literal["min", "median", "max"] = "min",
) -> list[int]:
    """Return indices of meshes whose posterior std is below the cutoff.

    See :func:`mesh_posterior_stds` for details on what is measured and which
    meshes are included.

    Parameters
    ----------
    estimation
        A solved estimation with an MCMC trace.
    kind
        Slip direction: ``"ss"`` (strike-slip) or ``"ds"`` (dip-slip).
    std_cutoff
        Maximum posterior standard deviation (dimensionless coupling fraction)
        to consider a mesh "resolved".
    reduce
        Passed to :func:`mesh_posterior_stds`.

    Returns
    -------
    list[int]
        Mesh indices (into ``estimation.model.meshes``) that are resolved.
    """
    stds = mesh_posterior_stds(estimation, kind=kind, reduce=reduce)
    return list(stds[stds < std_cutoff].index)


def plot_resolved_meshes(
    estimation: Estimation,
    plot_params: PlotParams,
    *,
    kind: Literal["ss", "ds"],
    std_cutoff: float,
    reduce: Literal["min", "median", "max"] = "min",
    draw: int | None = None,
    chain: int | None = None,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    slip_rate_width_scale: float = 0.25,
    arrow_scale: float = 1.0,
    plot_slip_rates: bool = True,
    plot_residuals: bool = True,
    figsize: tuple[int, int] | None = None,
) -> matplotlib.figure.Figure:
    """Plot meshes that cross the posterior resolvability threshold.

    Creates a map showing:

    - Coastlines and land
    - Segment slip rates colored by sign (optional)
    - Diamond markers at the centroids of resolved meshes
    - Velocity residual arrows (optional)

    See :func:`mesh_posterior_stds` for details on the resolvability criterion.

    Parameters
    ----------
    estimation
        A solved estimation with an MCMC trace.
    plot_params
        Plotting parameters (``PlotParams`` instance).
    kind
        Slip direction: ``"ss"`` (strike-slip) or ``"ds"`` (dip-slip).
    std_cutoff
        Maximum posterior std (dimensionless coupling fraction) to consider a
        mesh "resolved".
    reduce
        Passed to :func:`mesh_posterior_stds`.
    draw
        If given, use this specific MCMC draw; otherwise use the posterior
        mean.
    chain
        Chain index when selecting a specific MCMC draw.  ``None`` (the
        default) uses all chains.
    lon_range
        Override for the longitude plot range.  Falls back to
        ``plot_params.lon_range``.
    lat_range
        Override for the latitude plot range.  Falls back to
        ``plot_params.lat_range``.
    slip_rate_width_scale
        Line width scaling factor for segment slip rates.
    arrow_scale
        Scaling factor for velocity residual arrows.
    plot_slip_rates
        Whether to plot segment strike-slip rates.
    plot_residuals
        Whether to plot station velocity residual arrows.
    figsize
        Figure size override; defaults to ``plot_params.figsize_vectors``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    est = (
        estimation.mcmc_draw(draw=draw, chain=chain) if draw is not None else estimation
    )

    resolved_idxs = resolved_mesh_indices(
        estimation, kind=kind, std_cutoff=std_cutoff, reduce=reduce
    )
    meshes = estimation.model.meshes
    mesh_lats = [meshes[idx].lat_centroid.mean() for idx in resolved_idxs]
    mesh_lons = [meshes[idx].lon_centroid.mean() for idx in resolved_idxs]

    lon_range = lon_range or plot_params.lon_range
    lat_range = lat_range or plot_params.lat_range
    figsize = figsize or plot_params.figsize_vectors

    fig = plt.figure(figsize=figsize)

    plot_common_elements(plot_params, est.model.segment, lon_range, lat_range)
    plot_land(lon_range[0], lat_range[0], lon_range[1], lat_range[1])
    plot_coastlines(lon_range[0], lat_range[0], lon_range[1], lat_range[1])

    if plot_slip_rates:
        _plot_strike_slip_rates(est, slip_rate_width_scale, plot_params.fontsize)

    plt.xlim(*lon_range)
    plt.ylim(*lat_range)

    if resolved_idxs:
        lons_360 = np.array(mesh_lons) % 360
        plt.scatter(
            lons_360,
            mesh_lats,
            marker="D",
            zorder=1000,
            color="green",
            s=300,
            label="Resolved meshes",
        )
        for i, idx in enumerate(resolved_idxs):
            plt.annotate(str(idx), xy=(lons_360[i], mesh_lats[i]), zorder=2000)

    if plot_residuals:
        p_arrows = _position_arrow_key(plot_params, lon_range, lat_range)
        plot_vel_arrows_elements(
            p_arrows,
            est.model.station.lon,
            est.model.station.lat,
            est.station.model_east_vel_residual,
            est.station.model_north_vel_residual,
            arrow_scale=arrow_scale,
        )

    return fig


def _position_arrow_key(
    p: PlotParams,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> PlotParams:
    """Return a copy of *p* with the arrow key positioned in the lower-right."""
    lon_span = lon_range[1] - lon_range[0]
    lat_span = lat_range[1] - lat_range[0]
    return replace(
        p,
        key_rectangle_anchor=np.array(
            [
                lon_range[1] - 0.35 * lon_span,
                lat_range[0] + 0.01 * lat_span,
            ]
        ),
        key_rectangle_width=0.32 * lon_span,
        key_rectangle_height=0.12 * lat_span,
        key_arrow_lon=lon_range[1] - 0.19 * lon_span,
        key_arrow_lat=lat_range[0] + 0.05 * lat_span,
    )


def _plot_strike_slip_rates(
    est: Estimation,
    width_scale: float,
    fontsize: int,
) -> None:
    """Overlay segment strike-slip rates as colored lines."""
    segment = est.model.segment
    for i in range(len(segment)):
        slip = est.strike_slip_rates[i]
        color = "tab:orange" if slip < 0 else "tab:blue"
        plt.plot(
            [segment.lon1[i], segment.lon2[i]],
            [segment.lat1[i], segment.lat2[i]],
            "-",
            color=color,
            linewidth=width_scale * abs(slip),
        )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="tab:orange",
            marker="s",
            linestyle="None",
            markersize=10,
            label="right-lateral",
        ),
        mlines.Line2D(
            [],
            [],
            color="tab:blue",
            marker="s",
            linestyle="None",
            markersize=10,
            label="left-lateral",
        ),
    ]
    plt.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=fontsize,
        framealpha=1.0,
        edgecolor="k",
        fancybox=False,
    )
