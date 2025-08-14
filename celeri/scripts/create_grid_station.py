#!/usr/bin/env python3

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a gridded station file with longitude and latitude points"
    )
    parser.add_argument(
        "lon_min",
        type=float,
        help="Minimum longitude of the bounding box",
    )
    parser.add_argument(
        "lon_max",
        type=float,
        help="Maximum longitude of the bounding box",
    )
    parser.add_argument(
        "lat_min",
        type=float,
        help="Minimum latitude of the bounding box",
    )
    parser.add_argument(
        "lat_max",
        type=float,
        help="Maximum latitude of the bounding box",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=100,
        help="Number of points along each dimension (default: 100, creates 100x100 grid)",
    )
    return parser.parse_args()


def create_grid_stations(lon_min, lon_max, lat_min, lat_max, n_points):
    """Create a grid of station coordinates within the specified bounding box."""
    # Generate the grid coordinates
    lons = np.linspace(lon_min, lon_max, n_points)
    lats = np.linspace(lat_min, lat_max, n_points)
    lons, lats = np.meshgrid(lons, lats)
    lons = lons.flatten()
    lats = lats.flatten()

    # Create the dataframe with all required columns
    n_stations = len(lons)
    df = pd.DataFrame(
        {
            "lon": lons,
            "lat": lats,
            "corr": np.zeros(n_stations),
            "other1": np.zeros(n_stations, dtype=int),
            "name": [f"GRID_{i:06d}_GPS" for i in range(n_stations)],
            "east_vel": np.zeros(n_stations),
            "north_vel": np.zeros(n_stations),
            "east_sig": np.ones(n_stations),
            "north_sig": np.ones(n_stations),
            "flag": np.zeros(n_stations, dtype=int),
            "up_vel": np.zeros(n_stations),
            "up_sig": np.ones(n_stations),
            "east_adjust": np.zeros(n_stations),
            "north_adjust": np.zeros(n_stations),
            "up_adjust": np.zeros(n_stations),
        }
    )

    return df


def main():
    args = parse_args()

    # Validate bounding box
    if args.lon_min >= args.lon_max:
        raise ValueError(
            f"lon_min ({args.lon_min}) must be less than lon_max ({args.lon_max})"
        )
    if args.lat_min >= args.lat_max:
        raise ValueError(
            f"lat_min ({args.lat_min}) must be less than lat_max ({args.lat_max})"
        )

    # Log the parameters
    logger.info("Creating grid with bounding box:")
    logger.info(f"  Longitude: {args.lon_min} to {args.lon_max}")
    logger.info(f"  Latitude: {args.lat_min} to {args.lat_max}")
    logger.info(
        f"  Grid size: {args.n_points} x {args.n_points} = {args.n_points**2} total stations"
    )

    # Create the grid stations
    station_df = create_grid_stations(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, args.n_points
    )

    # Generate output filename with UUID (no dashes)
    uuid_str = str(uuid.uuid4()).replace("-", "")
    output_filename = f"{uuid_str}_station.csv"

    # Write to CSV with trailing comma to match celeri format
    with Path(output_filename).open("w") as f:
        # Write header with trailing comma
        f.write(",".join(station_df.columns) + ",\n")
        # Write data rows with trailing comma
        for _, row in station_df.iterrows():
            values = []
            for col in station_df.columns:
                val = row[col]
                if isinstance(val, float):
                    if val == int(val):
                        values.append(str(int(val)))
                    else:
                        values.append(f"{val:.6g}")
                else:
                    values.append(str(val))
            f.write(",".join(values) + ",\n")

    logger.info(f"Created gridded station file: {output_filename}")
    logger.info(f"Total stations: {len(station_df)}")
    logger.info(
        f"Longitude range: {station_df['lon'].min():.4f} to {station_df['lon'].max():.4f}"
    )
    logger.info(
        f"Latitude range: {station_df['lat'].min():.4f} to {station_df['lat'].max():.4f}"
    )


if __name__ == "__main__":
    main()
