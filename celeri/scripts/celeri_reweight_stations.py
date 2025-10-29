#!/usr/bin/env python3
"""Calculate density-based weights for GPS station data.

This script computes weights that are inversely proportional to local station density,
allowing for more balanced influence of data points in spatial estimation procedures.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great circle distance between points on Earth.

    Parameters
    ----------
    lon1, lat1 : array-like
        Longitude and latitude of first point(s) in degrees
    lon2, lat2 : array-like
        Longitude and latitude of second point(s) in degrees

    Returns
    -------
    distance : array
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km

    # Convert to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def calculate_density_weights(
    df, k_neighbors=10, method="knn", bandwidth=None, weight_type="inverse_square"
):
    """Calculate weights inversely proportional to local station density.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'lon' and 'lat' columns
    k_neighbors : int
        Number of nearest neighbors to consider for density estimation
    method : str
        Method for density estimation ('knn' or 'gaussian')
    bandwidth : float
        Bandwidth for Gaussian kernel (in km) if method='gaussian'
    weight_type : str
        'inverse' for 1/density, 'inverse_square' for 1/density², or 'log_inverse' for 1/log10(density)

    Returns
    -------
    weights : numpy.array
        Normalized weights inversely proportional to density
    densities : numpy.array
        Raw density estimates
    """
    # Extract coordinates
    coords = df[["lon", "lat"]].values
    n_stations = len(coords)

    if method == "knn":
        # Build KD-tree for efficient nearest neighbor search
        # Convert lat/lon to approximate Cartesian for KD-tree
        # Using simple equirectangular projection scaled by cos(mean_lat)
        mean_lat = np.mean(df["lat"])
        coords_proj = np.column_stack(
            [coords[:, 0] * np.cos(np.radians(mean_lat)), coords[:, 1]]
        )

        tree = KDTree(coords_proj)

        # Find k nearest neighbors for each point
        distances, indices = tree.query(coords_proj, k=min(k_neighbors + 1, n_stations))

        # Calculate density as inverse of mean distance to k nearest neighbors
        # (excluding the point itself, which is at distance 0)
        densities = np.zeros(n_stations)
        for i in range(n_stations):
            # Convert projected distances back to approximate km
            neighbor_coords = coords[indices[i, 1:], :]  # Skip first (self)
            actual_distances = haversine_distance(
                coords[i, 0], coords[i, 1], neighbor_coords[:, 0], neighbor_coords[:, 1]
            )

            # Density is inverse of mean distance
            mean_dist = np.mean(actual_distances) if len(actual_distances) > 0 else 1.0
            densities[i] = 1.0 / (mean_dist + 1.0)  # Add 1 km to avoid division issues

    elif method == "gaussian":
        # Gaussian kernel density estimation
        if bandwidth is None:
            bandwidth = 50.0  # Default 50 km bandwidth

        densities = np.zeros(n_stations)

        for i in range(n_stations):
            # Calculate distances to all other points
            distances = haversine_distance(
                coords[i, 0], coords[i, 1], coords[:, 0], coords[:, 1]
            )

            # Apply Gaussian kernel
            kernel_values = np.exp(-0.5 * (distances / bandwidth) ** 2)
            densities[i] = np.sum(kernel_values)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate weights based on specified type
    if weight_type == "inverse":
        # Original method: weights as inverse of density
        # Normalize densities to avoid numerical issues
        densities_normalized = densities / np.max(densities)
        # Weights are inverse of normalized density
        weights = 1.0 / (
            densities_normalized + 0.1
        )  # Add small constant to avoid division by zero
    elif weight_type == "inverse_square":
        # Inverse square method: weights as 1/density²
        # Normalize densities to avoid numerical issues
        densities_normalized = densities / np.max(densities)
        # Weights are inverse square of normalized density
        weights = 1.0 / (
            (densities_normalized + 0.01) ** 2
        )  # Add small constant to avoid division by zero
        weights = (
            densities_normalized + 0.0001
        )  # Add small constant to avoid division by zero

    elif weight_type == "log_inverse":
        # Log-based method: weights as inverse of log10(density)
        # Add small constant to avoid log(0)
        densities_safe = densities + 1e-10
        # Take log10 of densities
        log_densities = np.log10(densities_safe)
        # Normalize log densities
        log_densities_normalized = (log_densities - np.min(log_densities)) / (
            np.max(log_densities) - np.min(log_densities)
        )
        # Weights are inverse of normalized log density
        weights = 1.0 / (log_densities_normalized + 0.1)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Normalize weights to have mean of 1.0
    weights = weights / np.mean(weights)

    return weights, densities


def plot_station_weights(df, weights, save_path=None):
    """Create visualization of station locations colored by their weights.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'lon' and 'lat' columns
    weights : array-like
        Weight values for each station
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Station locations with weights as colors
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        df["lon"],
        df["lat"],
        c=weights,
        s=30,
        cmap="viridis_r",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("Longitude (degrees)", fontsize=12)
    ax1.set_ylabel("Latitude (degrees)", fontsize=12)
    ax1.set_title(
        "Station Density Weights\n(Yellow = Low density/High weight, Purple = High density/Low weight)",
        fontsize=13,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Add colorbar
    plt.colorbar(scatter1, ax=ax1, label="Weight")

    # Right plot: Weight distribution histogram
    ax2 = axes[1]
    ax2.hist(weights, bins=30, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Weight Value", fontsize=12)
    ax2.set_ylabel("Number of Stations", fontsize=12)
    ax2.set_title("Distribution of Weights", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Add statistics to the histogram
    stats_text = f"Mean: {np.mean(weights):.2f}\n"
    stats_text += f"Median: {np.median(weights):.2f}\n"
    stats_text += f"Std: {np.std(weights):.2f}\n"
    stats_text += f"Min: {np.min(weights):.2f}\n"
    stats_text += f"Max: {np.max(weights):.2f}"
    ax2.text(
        0.65,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show(block=False)

    return fig


def main():
    """Main execution function."""
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate density-based weights for GPS stations"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/station/5000_xu_wna_merge_station.csv",
        help="Input station CSV file",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=["inverse", "inverse_square", "log_inverse"],
        default="inverse_square",
        help="Weight calculation method: inverse (1/density), inverse_square (1/density²), or log_inverse (1/log10(density))",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=15,
        help="Number of nearest neighbors for density estimation",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["knn", "gaussian"],
        default="knn",
        help="Density estimation method",
    )
    parser.add_argument(
        "--replace-sigmas",
        action="store_true",
        default=True,
        help="Replace sigma columns with weights",
    )
    args = parser.parse_args()

    # File paths
    station_file = args.input
    # Create output filename based on input
    import os

    base_name = os.path.basename(station_file).replace(".csv", "")
    output_dir = os.path.dirname(station_file)
    output_file = os.path.join(output_dir, f"{base_name}_reweighted.csv")

    print("=" * 60)
    print("Station Density Weighting Tool")
    print("=" * 60)

    # Read station data
    print(f"\nReading station data from: {station_file}")
    df = pd.read_csv(station_file)
    print(f"Loaded {len(df)} stations")

    # Display data summary
    print("\nCoordinate ranges:")
    print(f"  Longitude: {df['lon'].min():.3f} to {df['lon'].max():.3f}")
    print(f"  Latitude:  {df['lat'].min():.3f} to {df['lat'].max():.3f}")

    # Calculate weights using specified method
    print("\nCalculating density-based weights...")
    print(f"  Method: {args.method}")
    print(f"  Weight type: {args.weight_type}")
    print(f"  K-neighbors: {args.k_neighbors}")

    weights, densities = calculate_density_weights(
        df,
        k_neighbors=args.k_neighbors,
        method=args.method,
        weight_type=args.weight_type,
    )

    # Print weight statistics
    print("\nWeight Statistics:")
    print(f"  Mean weight:   {np.mean(weights):.3f}")
    print(f"  Median weight: {np.median(weights):.3f}")
    print(f"  Std weight:    {np.std(weights):.3f}")
    print(f"  Min weight:    {np.min(weights):.3f}")
    print(f"  Max weight:    {np.max(weights):.3f}")
    print(f"  Weight ratio (max/min): {np.max(weights) / np.min(weights):.2f}")

    # Create output dataframe
    df_output = df.copy()

    if args.replace_sigmas:
        # Replace sigma columns with weights
        print("\nReplacing sigma columns with density-based weights...")
        df_output["east_sig"] = weights
        df_output["north_sig"] = weights
        if "up_sig" in df_output.columns:
            df_output["up_sig"] = weights

        # Don't include separate weight and density columns
        if "weight" in df_output.columns:
            df_output = df_output.drop(columns=["weight"])
        if "density" in df_output.columns:
            df_output = df_output.drop(columns=["density"])

        print(
            f"  Updated east_sig, north_sig, and up_sig with weights (mean={np.mean(weights):.3f})"
        )
    else:
        # Add weights to dataframe without replacing sigmas
        df_output["weight"] = weights
        df_output["density"] = densities

    # Identify stations with extreme weights
    high_weight_threshold = np.percentile(weights, 95)
    low_weight_threshold = np.percentile(weights, 5)

    print("\nStations with highest weights (lowest density):")
    df_temp = df.copy()
    df_temp["weight"] = weights
    high_weight_stations = df_temp[
        df_temp["weight"] >= high_weight_threshold
    ].sort_values("weight", ascending=False)
    for _idx, row in high_weight_stations.head(5).iterrows():
        print(
            f"  {row['name']}: weight={row['weight']:.2f}, lon={row['lon']:.3f}, lat={row['lat']:.3f}"
        )

    print("\nStations with lowest weights (highest density):")
    low_weight_stations = df_temp[
        df_temp["weight"] <= low_weight_threshold
    ].sort_values("weight")
    for _idx, row in low_weight_stations.head(5).iterrows():
        print(
            f"  {row['name']}: weight={row['weight']:.2f}, lon={row['lon']:.3f}, lat={row['lat']:.3f}"
        )

    # Save updated dataframe
    print(f"\nSaving reweighted data to: {output_file}")
    df_output.to_csv(output_file, index=False)
    print(f"  Saved {len(df_output)} stations with updated weights")

    # Create visualization
    print("\nGenerating visualization...")
    plot_station_weights(df_temp, weights, save_path="station_density_weights.png")

    print("\nAnalysis complete!")
    print(f"Output saved: {output_file}")
    print("Visualization saved: station_density_weights.png")


if __name__ == "__main__":
    main()
