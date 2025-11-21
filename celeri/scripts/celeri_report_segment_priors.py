#!/usr/bin/env python3
"""Calculate density-based weights for GPS station data.

This script computes weights that are inversely proportional to local station density,
allowing for more balanced influence of data points in spatial estimation procedures.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Report soft and hard constraints in a segment file"
    )
    parser.add_argument(
        "seg_file",
        type=str,
        help="Input segment CSV file",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="Plot constraint locations",
    )

    args = parser.parse_args()

    # File paths
    seg_file = args.seg_file
    plot_flag = args.plot

    # Read station data
    print(f"\nReading segment data from: {seg_file}")
    df = pd.read_csv(seg_file)
    print(f"Loaded {len(df)} segments")

    n_segs_with_prior = 0
    n_min_max_errors = 0
    seg_lons_with_prior = []
    seg_lats_with_prior = []

    for i in range(len(df)):
        conditions = {
            "soft strike-slip": df.ss_rate_flag[i],
            "soft dip-slip": df.ds_rate_flag[i],
            "soft tensile-slip": df.ts_rate_flag[i],
            "hard strike-slip": df.ss_rate_bound_flag[i],
            "hard dip-slip": df.ds_rate_bound_flag[i],
            "hard tensile-slip": df.ts_rate_bound_flag[i],
        }

        # Filter non-zero conditions
        non_zero = {k: v for k, v in conditions.items() if v != 0}

        if non_zero:
            n_segs_with_prior += 1
            print(f"\nSEGMENT NAME: {df.name[i]}, ROW: {i}")
            seg_lons_with_prior.append(df.lon1[i])
            seg_lons_with_prior.append(df.lon2[i])
            seg_lats_with_prior.append(df.lat1[i])
            seg_lats_with_prior.append(df.lat2[i])

            # Print soft constraints
            if df.ss_rate_flag[i] != 0:
                print(
                    f"Soft strike-slip constraint, val: {df.ss_rate[i]}, sigma: {df.ss_rate_sig[i]} (mm/yr)"
                )
            if df.ds_rate_flag[i] != 0:
                print(
                    f"Soft dip-slip constraint, val: {df.ds_rate[i]}, sigma: {df.ds_rate_sig[i]} (mm/yr)"
                )
            if df.ts_rate_flag[i] != 0:
                print(
                    f"Soft tensile-slip constraint, val: {df.ts_rate[i]}, sigma: {df.ts_rate_sig[i]} (mm/yr)"
                )

            # Print hard constraints
            if df.ss_rate_bound_flag[i] != 0:
                print(
                    f"Hard strike-slip bounds, min: {df.ss_rate_bound_min[i]}, max: {df.ss_rate_bound_max[i]} (mm/yr)"
                )
                if df.ss_rate_bound_min[i] >= df.ss_rate_bound_max[i]:
                    n_min_max_errors += 1
                    print(
                        f"*** STOP: MIN > MAX *** Hard strike-slip bounds , min: {df.ss_rate_bound_min[i]}, max: {df.ss_rate_bound_max[i]} (mm/yr)"
                    )

            if df.ds_rate_bound_flag[i] != 0:
                print(
                    f"Hard dip-slip bounds, min: {df.ds_rate_bound_min[i]}, max: {df.ds_rate_bound_max[i]} (mm/yr)"
                )
                if df.ds_rate_bound_min[i] >= df.ds_rate_bound_max[i]:
                    n_min_max_errors += 1
                    print(
                        f"*** STOP: MIN > MAX *** Hard dip-slip bounds , min: {df.ds_rate_bound_min[i]}, max: {df.ds_rate_bound_max[i]} (mm/yr)"
                    )
            if df.ts_rate_bound_flag[i] != 0:
                print(
                    f"Hard tensile-slip bounds, min: {df.ts_rate_bound_min[i]}, max: {df.ts_rate_bound_max[i]} (mm/yr)"
                )
                if df.ts_rate_bound_min[i] >= df.ts_rate_bound_max[i]:
                    n_min_max_errors += 1
                    print(
                        f"*** STOP: MIN > MAX *** Hard tensile-slip bounds , min: {df.ts_rate_bound_min[i]}, max: {df.ts_rate_bound_max[i]} (mm/yr)"
                    )

            # Check for tensile-slip constraints on dipping fault
            if df.dip[i] != 90.0:
                if df.ts_rate_flag[i] != 0:
                    raise SystemExit(
                        "*** STOP: SOFT TENSILE-SLIP CONSTRAINT ON NON-VERTICAL FAULT ***"
                    )
                if df.ts_rate_bound_flag[i] != 0:
                    raise SystemExit(
                        "*** STOP: HARD TENSILE-SLIP CONSTRAINT ON NON-VERTICAL FAULT ***"
                    )

            # Check for dip-slip constraints on a vertical fault
            if df.dip[i] == 90.0:
                if df.ds_rate_flag[i] != 0:
                    raise SystemExit(
                        "*** STOP: SOFT DIP-SLIP CONSTRAINT ON VERTICAL FAULT ***"
                    )
                if df.ds_rate_bound_flag[i] != 0:
                    raise SystemExit(
                        "*** STOP: HARD DIP-SLIP CONSTRAINT ON VERTICAL FAULT ***"
                    )

    print(f"\nFound {n_segs_with_prior} of {len(df)} segments with priors")
    print(f"\nFound {n_min_max_errors} segments where min >= max\n")

    # Plot
    if plot_flag == 1:
        if n_segs_with_prior > 0:
            plt.figure(figsize=(10, 10))
            for i in range(len(df)):
                conditions = {
                    "soft strike-slip": df.ss_rate_flag[i],
                    "soft dip-slip": df.ds_rate_flag[i],
                    "soft tensile-slip": df.ts_rate_flag[i],
                    "hard strike-slip": df.ss_rate_bound_flag[i],
                    "hard dip-slip": df.ds_rate_bound_flag[i],
                    "hard tensile-slip": df.ts_rate_bound_flag[i],
                }
                non_zero = {k: v for k, v in conditions.items() if v != 0}
                if non_zero:
                    plt.plot(
                        [df.lon1[i], df.lon2[i]],
                        [df.lat1[i], df.lat2[i]],
                        "-r",
                        linewidth=5.0,
                    )
                    plt.text(
                        0.5 * (df.lon1[i] + df.lon2[i]),
                        0.5 * (df.lat1[i] + df.lat2[i]),
                        f"{df.name[i]} ({i})",
                        fontsize=8,
                    )
                else:
                    plt.plot(
                        [df.lon1[i], df.lon2[i]],
                        [df.lat1[i], df.lat2[i]],
                        "-k",
                        linewidth=0.5,
                    )

            lon_min = np.min(seg_lons_with_prior)
            lon_max = np.max(seg_lons_with_prior)
            lat_min = np.min(seg_lats_with_prior)
            lat_max = np.max(seg_lats_with_prior)
            delta_lon = lon_max - lon_min
            delta_lat = lat_max - lat_min
            delta_scale = 0.25

            plt.xlim(
                lon_min - delta_scale * delta_lon, lon_max + delta_scale * delta_lon
            )
            plt.ylim(
                lat_min - delta_scale * delta_lat, lat_max + delta_scale * delta_lat
            )
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show(block=True)


if __name__ == "__main__":
    main()
