"""Identify and plot meshes that cross a posterior resolvability threshold.

Given an MCMC estimation, computes the minimum posterior standard deviation
across triangular elements for each mesh's coupling field. Meshes with
posterior std below a cutoff are considered "resolved" — the data contain
enough information to constrain their coupling.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.figure
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import celeri
from celeri.plot import PlotParams, plot_common_elements, plot_vel_arrows_elements
from celeri.solve import Estimation

CouplingKind = Literal["ss", "ds"]


def mesh_posterior_stds(
    estimation: Estimation,
    kind: CouplingKind = "ss",
) -> pd.Series:
    """Minimum posterior std of each mesh's coupling field.

    For each mesh that has a coupling posterior in the MCMC trace, computes the
    standard deviation across chains and draws, then takes the minimum across
    triangular elements. A low value indicates the mesh is well-resolved.

    Parameters
    ----------
    estimation
        A solved estimation with an MCMC trace.
    kind
        Coupling component: ``"ss"`` (strike-slip) or ``"ds"`` (dip-slip).

    Returns
    -------
    pd.Series
        Indexed by mesh index, values are minimum posterior std, sorted
        ascending.
    """
    if estimation.mcmc_trace is None:
        raise ValueError("Estimation has no MCMC trace.")

    stds: dict[int, float] = {}
    for mesh_idx in range(len(estimation.model.meshes)):
        name = f"coupling_{mesh_idx}_{kind}"
        if name in estimation.mcmc_trace.posterior:
            stds[mesh_idx] = float(
                estimation.mcmc_trace.posterior[name]
                .std(["chain", "draw"])
                .min()
                .values
            )

    return pd.Series(stds, dtype=float).sort_values()


def resolved_mesh_indices(
    estimation: Estimation,
    kind: CouplingKind = "ss",
    std_cutoff: float = 0.1,
) -> list[int]:
    """Return indices of meshes whose posterior std is below the cutoff.

    Parameters
    ----------
    estimation
        A solved estimation with an MCMC trace.
    kind
        Coupling component: ``"ss"`` (strike-slip) or ``"ds"`` (dip-slip).
    std_cutoff
        Maximum posterior standard deviation to consider a mesh "resolved".

    Returns
    -------
    list[int]
        Mesh indices (into ``estimation.model.meshes``) that are resolved.
    """
    stds = mesh_posterior_stds(estimation, kind=kind)
    return list(stds[stds < std_cutoff].index)


def plot_resolved_meshes(
    estimation: Estimation,
    p: PlotParams,
    *,
    kind: CouplingKind = "ss",
    std_cutoff: float = 0.1,
    draw: int | None = None,
    chain: int = 0,
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

    Parameters
    ----------
    estimation
        A solved estimation with an MCMC trace.
    p
        Plotting parameters (``PlotParams`` instance).
    kind
        Coupling component: ``"ss"`` (strike-slip) or ``"ds"`` (dip-slip).
    std_cutoff
        Maximum posterior std to consider a mesh "resolved".
    draw
        If given, use this specific MCMC draw; otherwise use the posterior
        mean.
    chain
        Chain index when selecting a specific MCMC draw.
    lon_range
        Override for the longitude plot range.  Falls back to ``p.lon_range``.
    lat_range
        Override for the latitude plot range.  Falls back to ``p.lat_range``.
    slip_rate_width_scale
        Line width scaling factor for segment slip rates.
    arrow_scale
        Scaling factor for velocity residual arrows.
    plot_slip_rates
        Whether to plot segment strike-slip rates.
    plot_residuals
        Whether to plot station velocity residual arrows.
    figsize
        Figure size override; defaults to ``p.figsize_vectors``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    est = (
        estimation.mcmc_draw(draw=draw, chain=chain) if draw is not None else estimation
    )

    resolved_idxs = resolved_mesh_indices(estimation, kind=kind, std_cutoff=std_cutoff)
    meshes = estimation.model.meshes
    mesh_lats = [meshes[idx].lat_centroid.mean() for idx in resolved_idxs]
    mesh_lons = [meshes[idx].lon_centroid.mean() for idx in resolved_idxs]

    lon_range = lon_range or p.lon_range
    lat_range = lat_range or p.lat_range
    figsize = figsize or p.figsize_vectors

    fig = plt.figure(figsize=figsize)

    plot_common_elements(p, est.model.segment, lon_range, lat_range)
    celeri.plot_land(lon_range[0], lat_range[0], lon_range[1], lat_range[1])
    celeri.plot_coastlines(lon_range[0], lat_range[0], lon_range[1], lat_range[1])

    if plot_slip_rates:
        _plot_strike_slip_rates(est, slip_rate_width_scale, p.fontsize)

    plt.xlim(*lon_range)
    plt.ylim(*lat_range)

    if resolved_idxs:
        lons_360 = np.array(mesh_lons) % 360
        plt.scatter(
            lons_360,
            mesh_lats,
            marker="D",
            zorder=1000,
            color="red",
            s=100,
            label="Resolved meshes",
        )
        for i, idx in enumerate(resolved_idxs):
            plt.annotate(str(idx), xy=(lons_360[i], mesh_lats[i]), zorder=2000)

    if plot_residuals:
        plot_vel_arrows_elements(
            p,
            est.model.station.lon,
            est.model.station.lat,
            est.station.model_east_vel_residual,
            est.station.model_north_vel_residual,
            arrow_scale=arrow_scale,
        )

    return fig


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
    ).get_frame().set_boxstyle("Square")  # type: ignore[union-attr]
