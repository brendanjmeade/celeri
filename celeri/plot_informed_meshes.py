from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import celeri
from celeri.plot import PlotParams, plot_common_elements, plot_vel_arrows_elements
from celeri.solve import Estimation


def mesh_posterior_stds(
    estimation: Estimation,
    kind: str = "ss",
) -> pd.Series:
    stds = {}
    for mesh_idx in range(len(estimation.model.meshes)):
        name = f"coupling_{mesh_idx}_{kind}"
        if name in estimation.mcmc_trace.posterior:
            stds[mesh_idx] = float(
                estimation.mcmc_trace.posterior[name]
                .std(["chain", "draw"])
                .min()
                .values
            )

    stds = pd.Series(stds).sort_values()
    return stds


def resolved_mesh_indices(
    estimation: Estimation,
    kind: str = "ss",
    std_cutoff: float = 0.1,
) -> pd.Index:
    stds = mesh_posterior_stds(estimation, kind=kind)
    return stds[stds < std_cutoff].index


def plot_resolved_meshes(
    estimation: Estimation,
    p: PlotParams,
    kind: str = "ss",
    std_cutoff: float = 0.1,
    draw: int | None = None,
) -> None:
    mesh = estimation.model.meshes[0]  # noqa: F841

    low_std_mesh_idxs = resolved_mesh_indices(
        estimation, kind=kind, std_cutoff=std_cutoff
    )
    stds = mesh_posterior_stds(estimation, kind=kind)
    mesh_lat = [
        estimation.model.meshes[idx].lat_centroid.mean()
        for idx in stds[stds < std_cutoff].index
    ]
    mesh_lon = [
        estimation.model.meshes[idx].lon_centroid.mean()
        for idx in stds[stds < std_cutoff].index
    ]

    # Posterior draw, or `None` for the posterior mean
    if draw is not None:
        est = estimation.mcmc_draw(draw=draw, chain=0)
    else:
        est = estimation

    p.lon_range = (230, 240)
    p.lat_range = (35, 41)
    p.figsize_vectors = (24, 20)

    plt.figure(figsize=p.figsize_vectors)
    plot_common_elements(p, est.model.segment, p.lon_range, p.lat_range)
    celeri.plot_land(p.lon_range[0], p.lat_range[0], p.lon_range[1], p.lat_range[1])
    celeri.plot_coastlines(
        p.lon_range[0], p.lat_range[0], p.lon_range[1], p.lat_range[1]
    )

    if True:
        slip_rate_width_scale = 0.25
        # slip_rate_width_scale = 300
        for i in range(len(est.model.segment)):
            slip = est.strike_slip_rates[i]
            if slip < 0:
                # if grad.values[i] > 0:
                plt.plot(
                    [est.model.segment.lon1[i], est.model.segment.lon2[i]],
                    [est.model.segment.lat1[i], est.model.segment.lat2[i]],
                    "-",
                    color="tab:orange",
                    linewidth=slip_rate_width_scale * abs(slip),
                    # linewidth=slip_rate_width_scale * grad.values[i],
                )
            else:
                plt.plot(
                    [est.model.segment.lon1[i], est.model.segment.lon2[i]],
                    [est.model.segment.lat1[i], est.model.segment.lat2[i]],
                    "-",
                    color="tab:blue",
                    # linewidth=slip_rate_width_scale * grad.values[i],
                    linewidth=slip_rate_width_scale * abs(slip),
                )

        # Legend
        blue_segments = mlines.Line2D(
            [],
            [],
            color="tab:orange",
            marker="s",
            linestyle="None",
            markersize=10,
            label="right-lateral (10 mm/yr)",
        )
        red_segments = mlines.Line2D(
            [],
            [],
            color="tab:blue",
            marker="s",
            linestyle="None",
            markersize=10,
            label="left-lateral (10 mm/yr)",
        )
        plt.legend(
            handles=[blue_segments, red_segments],
            loc="lower left",
            fontsize=p.fontsize,
            framealpha=1.0,
            edgecolor="k",
        ).get_frame().set_boxstyle("Square")  # type: ignore[union-attr]

    if False:
        plt.scatter(
            est.model.station.lon,
            est.model.station.lat,
            c=0,  # was: grad.values.sum(1)
            cmap="seismic",
        )
        plt.clim(-0.1, 0.1)

    plt.xlim(236, 246)
    plt.ylim(31.5, 41)

    # plt.xlim(230, 250)
    # plt.ylim(31, 45)

    # plt.xlim(230, 250)
    # plt.ylim(31, 45)
    # plt.xlim(240, 243)
    # plt.ylim(33, 36)

    # plt.ylim(31, 40)
    # plt.xlim(237, 247)

    plt.xlim(242, 246)
    plt.ylim(31.5, 35)

    plt.scatter(
        np.array(mesh_lon) % 360,
        mesh_lat,
        marker="D",
        zorder=1000,
        color="red",
        s=100,
    )
    for i, idx in enumerate(low_std_mesh_idxs):
        plt.annotate(f"{idx}", xy=(mesh_lon[i] % 360, mesh_lat[i]), zorder=2000)

    if True:
        plot_vel_arrows_elements(
            p,
            est.model.station,
            # estimation.model.station.east_vel,
            # estimation.model.station.north_vel,
            est.station.model_east_vel_residual,
            est.station.model_north_vel_residual,
            # estimation.station.model_east_vel_rotation,
            # estimation.station.model_north_vel_rotation,
            # estimation.station.model_east_vel,
            # estimation.station.model_north_vel,
            arrow_scale=1,
        )
