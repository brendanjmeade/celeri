#!/usr/bin/env python3
"""
Comprehensive segment checking and fixing tool for Celeri input files.

This script checks fault segment files for various issues and optionally fixes them:
- Zero-degree dips (converted to 90 degrees)
- Terminating endpoints (segments not connected to others)
- Duplicate segments (same start/end points or same endpoint pairs)
- Axis-aligned segments (perfectly vertical or horizontal)
- Very short segments (below threshold length)

Usage:
    python celeri_check_fix_segments.py segment_file.csv [options]
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
import uuid
import warnings
from typing import Tuple, Dict, List, Optional

# Earth radius for distance calculations
EARTH_RADIUS_KM = 6371.0088


class SegmentChecker:
    """Class to handle all segment checking operations."""

    def __init__(self, df: pd.DataFrame, tolerance: float = 1e-5):
        """
        Initialize the segment checker.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with segment data
        tolerance : float
            Tolerance in degrees for point matching
        """
        self.df = df.copy()
        self.tolerance = tolerance
        self.issues = {}

    def haversine_length_km(self, lon1, lat1, lon2, lat2):
        """
        Vectorized great-circle distance between two lon/lat arrays (degrees).

        Returns distance in kilometers using the haversine formula.
        """
        lon1r = np.deg2rad(lon1)
        lat1r = np.deg2rad(lat1)
        lon2r = np.deg2rad(lon2)
        lat2r = np.deg2rad(lat2)

        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        # Normalize to shortest angular separation
        dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

        sin_dlat_2 = np.sin(dlat / 2.0)
        sin_dlon_2 = np.sin(dlon / 2.0)
        a = (sin_dlat_2 * sin_dlat_2 +
             np.cos(lat1r) * np.cos(lat2r) * sin_dlon_2 * sin_dlon_2)
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return EARTH_RADIUS_KM * c

    def compute_segment_lengths(self):
        """Add a 'length_km' column to the DataFrame."""
        self.df["length_km"] = self.haversine_length_km(
            self.df["lon1"].values, self.df["lat1"].values,
            self.df["lon2"].values, self.df["lat2"].values
        )
        return self.df

    def check_zero_dips(self) -> pd.DataFrame:
        """Find segments with zero-degree dips."""
        if "dip" not in self.df.columns:
            return pd.DataFrame()

        zero_dip_mask = self.df["dip"] == 0
        zero_dip_df = self.df[zero_dip_mask].copy()
        self.issues["zero_dips"] = zero_dip_df
        return zero_dip_df

    def check_terminating_endpoints(self) -> pd.DataFrame:
        """Find endpoints that appear only once (terminating endpoints)."""
        endpoints = []

        # Collect all endpoints with their segment index
        for idx, row in self.df.iterrows():
            endpoints.append({
                "lon": row["lon1"],
                "lat": row["lat1"],
                "segment_idx": idx,
                "endpoint": "start"
            })
            endpoints.append({
                "lon": row["lon2"],
                "lat": row["lat2"],
                "segment_idx": idx,
                "endpoint": "end"
            })

        # Create DataFrame of all endpoints
        ep_df = pd.DataFrame(endpoints)

        # Round coordinates for matching
        ep_df["lon_round"] = np.round(ep_df["lon"] / self.tolerance) * self.tolerance
        ep_df["lat_round"] = np.round(ep_df["lat"] / self.tolerance) * self.tolerance

        # Count occurrences of each unique point
        point_counts = ep_df.groupby(["lon_round", "lat_round"]).size()

        # Find terminating points (appearing exactly once)
        terminating_points = point_counts[point_counts == 1]

        # Get details of terminating endpoints
        terminating_list = []
        for (lon_r, lat_r), count in terminating_points.items():
            mask = (ep_df["lon_round"] == lon_r) & (ep_df["lat_round"] == lat_r)
            matching = ep_df[mask]

            for _, ep in matching.iterrows():
                terminating_list.append({
                    "lon": ep["lon"],
                    "lat": ep["lat"],
                    "segment_idx": ep["segment_idx"],
                    "endpoint_type": ep["endpoint"],
                    "segment_name": self.df.iloc[ep["segment_idx"]].get(
                        "name", f"seg_{ep['segment_idx']}"
                    )
                })

        terminating_df = pd.DataFrame(terminating_list)
        self.issues["terminating_endpoints"] = terminating_df
        return terminating_df

    def check_duplicate_segments(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check for two types of duplicate segments:
        1. Degenerate segments where start == end
        2. Segments with same endpoint pairs (order-insensitive)
        """
        # Check for degenerate segments (start == end)
        degenerate = []
        for idx, row in self.df.iterrows():
            lon_diff = abs(row["lon1"] - row["lon2"])
            lat_diff = abs(row["lat1"] - row["lat2"])

            if lon_diff <= self.tolerance and lat_diff <= self.tolerance:
                degenerate.append({
                    "segment_idx": idx,
                    "segment_name": row.get("name", f"seg_{idx}"),
                    "lon1": row["lon1"],
                    "lat1": row["lat1"],
                    "lon2": row["lon2"],
                    "lat2": row["lat2"],
                    "lon_diff": lon_diff,
                    "lat_diff": lat_diff
                })

        degenerate_df = pd.DataFrame(degenerate)

        # Check for endpoint duplicates (order-insensitive)
        key_to_indices = defaultdict(list)
        key_to_coords = {}

        for idx, row in self.df.iterrows():
            # Discretize endpoints
            a = (int(round(row["lon1"] / self.tolerance)),
                 int(round(row["lat1"] / self.tolerance)))
            b = (int(round(row["lon2"] / self.tolerance)),
                 int(round(row["lat2"] / self.tolerance)))

            # Order-insensitive key
            key = tuple(sorted((a, b)))
            key_to_indices[key].append(idx)

            if key not in key_to_coords:
                (ax, ay), (bx, by) = key
                key_to_coords[key] = (
                    (ax * self.tolerance, ay * self.tolerance),
                    (bx * self.tolerance, by * self.tolerance)
                )

        # Build records for duplicate groups
        records = []
        group_id = 1
        for key, idxs in key_to_indices.items():
            if len(idxs) <= 1:
                continue

            (lonA, latA), (lonB, latB) = key_to_coords[key]
            for idx in idxs:
                row = self.df.iloc[idx]
                records.append({
                    "group_id": group_id,
                    "group_lonA": lonA,
                    "group_latA": latA,
                    "group_lonB": lonB,
                    "group_latB": latB,
                    "group_size": len(idxs),
                    "segment_idx": idx,
                    "segment_name": row.get("name", f"seg_{idx}"),
                    "lon1": row["lon1"],
                    "lat1": row["lat1"],
                    "lon2": row["lon2"],
                    "lat2": row["lat2"]
                })
            group_id += 1

        endpoint_dup_df = pd.DataFrame(records)

        self.issues["degenerate_segments"] = degenerate_df
        self.issues["endpoint_duplicates"] = endpoint_dup_df

        return degenerate_df, endpoint_dup_df

    def check_axis_aligned_segments(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify perfectly vertical and horizontal segments.

        Returns:
        --------
        (vertical_df, horizontal_df)
        """
        vertical, horizontal = [], []

        for idx, row in self.df.iterrows():
            lon_diff = abs(row["lon1"] - row["lon2"])
            lat_diff = abs(row["lat1"] - row["lat2"])

            if lon_diff <= self.tolerance:
                vertical.append({
                    "segment_idx": idx,
                    "segment_name": row.get("name", f"seg_{idx}"),
                    "lon1": row["lon1"],
                    "lat1": row["lat1"],
                    "lon2": row["lon2"],
                    "lat2": row["lat2"],
                    "lon_diff": lon_diff
                })

            if lat_diff <= self.tolerance:
                horizontal.append({
                    "segment_idx": idx,
                    "segment_name": row.get("name", f"seg_{idx}"),
                    "lon1": row["lon1"],
                    "lat1": row["lat1"],
                    "lon2": row["lon2"],
                    "lat2": row["lat2"],
                    "lat_diff": lat_diff
                })

        vertical_df = pd.DataFrame(vertical)
        horizontal_df = pd.DataFrame(horizontal)

        self.issues["vertical_segments"] = vertical_df
        self.issues["horizontal_segments"] = horizontal_df

        return vertical_df, horizontal_df

    def check_short_segments(self, threshold_km: float = 0.01) -> pd.DataFrame:
        """Find segments with length below threshold."""
        if "length_km" not in self.df.columns:
            self.compute_segment_lengths()

        short = []
        for idx, row in self.df.iterrows():
            if row["length_km"] <= threshold_km:
                short.append({
                    "segment_idx": idx,
                    "segment_name": row.get("name", f"seg_{idx}"),
                    "lon1": row["lon1"],
                    "lat1": row["lat1"],
                    "lon2": row["lon2"],
                    "lat2": row["lat2"],
                    "length_km": row["length_km"]
                })

        short_df = pd.DataFrame(short)
        self.issues["short_segments"] = short_df
        return short_df

    def run_all_checks(self, short_threshold_km: float = 0.01) -> Dict:
        """Run all checks and return results."""
        print("\n" + "=" * 80)
        print("RUNNING SEGMENT CHECKS")
        print("=" * 80)

        # Check zero dips
        zero_dip_df = self.check_zero_dips()
        if len(zero_dip_df) > 0:
            print(f"\n✗ Found {len(zero_dip_df)} segments with zero-degree dips")
        else:
            print("\n✓ No segments with zero-degree dips")

        # Compute lengths
        self.compute_segment_lengths()
        if len(self.df) > 0:
            lengths = self.df["length_km"].values
            print(f"\nSegment length statistics (km):")
            print(f"  min={np.nanmin(lengths):.6f}")
            print(f"  median={np.nanmedian(lengths):.3f}")
            print(f"  max={np.nanmax(lengths):.1f}")

        # Check for short segments
        short_df = self.check_short_segments(threshold_km=short_threshold_km)
        if len(short_df) > 0:
            print(f"\n✗ Found {len(short_df)} very short segments (≤{short_threshold_km} km)")
        else:
            print(f"\n✓ No very short segments (≤{short_threshold_km} km)")

        # Check for duplicate segments
        degenerate_df, endpoint_dup_df = self.check_duplicate_segments()
        if len(degenerate_df) > 0:
            print(f"\n✗ Found {len(degenerate_df)} degenerate segments (start==end)")
        else:
            print("\n✓ No degenerate segments")

        if len(endpoint_dup_df) > 0:
            n_groups = endpoint_dup_df["group_id"].nunique() if len(endpoint_dup_df) > 0 else 0
            print(f"\n✗ Found {len(endpoint_dup_df)} segments in {n_groups} duplicate groups")
        else:
            print("\n✓ No duplicate endpoint pairs")

        # Check for axis-aligned segments
        vertical_df, horizontal_df = self.check_axis_aligned_segments()
        if len(vertical_df) > 0:
            print(f"\n✗ Found {len(vertical_df)} perfectly vertical segments")
        else:
            print("\n✓ No perfectly vertical segments")

        if len(horizontal_df) > 0:
            print(f"\n✗ Found {len(horizontal_df)} perfectly horizontal segments")
        else:
            print("\n✓ No perfectly horizontal segments")

        # Check for terminating endpoints
        terminating_df = self.check_terminating_endpoints()
        if len(terminating_df) > 0:
            print(f"\n✗ Found {len(terminating_df)} terminating endpoints")
        else:
            print("\n✓ No terminating endpoints")

        print("\n" + "=" * 80)

        return self.issues


class SegmentFixer:
    """Class to handle segment fixing operations."""

    def __init__(self, df: pd.DataFrame, tolerance: float = 1e-5):
        """Initialize the segment fixer."""
        self.df = df.copy()
        self.tolerance = tolerance
        self.fixes_applied = []

    def fix_zero_dips(self) -> bool:
        """Convert all zero-degree dips to 90-degree dips."""
        if "dip" not in self.df.columns:
            return False

        zero_mask = self.df["dip"] == 0
        n_fixed = zero_mask.sum()

        if n_fixed > 0:
            self.df.loc[zero_mask, "dip"] = 90
            self.fixes_applied.append(f"Fixed {n_fixed} zero-degree dips to 90 degrees")
            return True
        return False

    def fix_axis_aligned_segments(self, perturbation: float = 0.001) -> bool:
        """
        Perturb axis-aligned segments while maintaining connectivity.
        """
        # Build connectivity map
        connectivity = defaultdict(list)
        for idx, row in self.df.iterrows():
            key1 = (round(row["lon1"] / self.tolerance) * self.tolerance,
                   round(row["lat1"] / self.tolerance) * self.tolerance)
            key2 = (round(row["lon2"] / self.tolerance) * self.tolerance,
                   round(row["lat2"] / self.tolerance) * self.tolerance)
            connectivity[key1].append((idx, "start"))
            connectivity[key2].append((idx, "end"))

        # Track perturbed endpoints
        perturbed_endpoints = {}
        n_vertical_fixed = 0
        n_horizontal_fixed = 0

        # Fix vertical segments
        for idx, row in self.df.iterrows():
            lon_diff = abs(row["lon1"] - row["lon2"])
            if lon_diff <= self.tolerance:
                endpoint_key = (round(row["lon2"] / self.tolerance) * self.tolerance,
                               round(row["lat2"] / self.tolerance) * self.tolerance)

                if endpoint_key not in perturbed_endpoints:
                    new_lon = row["lon2"] + perturbation
                    perturbed_endpoints[endpoint_key] = (new_lon, row["lat2"])

                    # Update all connected segments
                    for seg_idx, endpoint_type in connectivity[endpoint_key]:
                        if endpoint_type == "start":
                            self.df.loc[seg_idx, "lon1"] = new_lon
                        else:
                            self.df.loc[seg_idx, "lon2"] = new_lon
                    n_vertical_fixed += 1

        # Fix horizontal segments
        for idx, row in self.df.iterrows():
            lat_diff = abs(row["lat1"] - row["lat2"])
            if lat_diff <= self.tolerance:
                endpoint_key = (round(row["lon2"] / self.tolerance) * self.tolerance,
                               round(row["lat2"] / self.tolerance) * self.tolerance)

                if endpoint_key not in perturbed_endpoints:
                    new_lat = row["lat2"] + perturbation
                    perturbed_endpoints[endpoint_key] = (row["lon2"], new_lat)

                    # Update all connected segments
                    for seg_idx, endpoint_type in connectivity[endpoint_key]:
                        if endpoint_type == "start":
                            self.df.loc[seg_idx, "lat1"] = new_lat
                        else:
                            self.df.loc[seg_idx, "lat2"] = new_lat
                    n_horizontal_fixed += 1
                else:
                    # Use existing perturbed values
                    new_lon, new_lat = perturbed_endpoints[endpoint_key]
                    for seg_idx, endpoint_type in connectivity[endpoint_key]:
                        if endpoint_type == "start":
                            self.df.loc[seg_idx, "lon1"] = new_lon
                            self.df.loc[seg_idx, "lat1"] = new_lat
                        else:
                            self.df.loc[seg_idx, "lon2"] = new_lon
                            self.df.loc[seg_idx, "lat2"] = new_lat

        if n_vertical_fixed > 0 or n_horizontal_fixed > 0:
            self.fixes_applied.append(
                f"Perturbed {n_vertical_fixed} vertical and {n_horizontal_fixed} horizontal segments"
            )
            return True
        return False

    def apply_fixes(self, fix_dips: bool = True, fix_aligned: bool = False,
                    perturbation: float = 0.001) -> pd.DataFrame:
        """Apply requested fixes to the segments."""
        print("\n" + "=" * 80)
        print("APPLYING FIXES")
        print("=" * 80)

        any_fixed = False

        if fix_dips:
            if self.fix_zero_dips():
                any_fixed = True

        if fix_aligned:
            if self.fix_axis_aligned_segments(perturbation=perturbation):
                any_fixed = True

        if any_fixed:
            print("\nFixes applied:")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        else:
            print("\nNo fixes were necessary")

        print("=" * 80)

        return self.df


def plot_segment_issues(df: pd.DataFrame, issues: Dict, output_file: Optional[str] = None):
    """Create visualization of segment issues."""

    # Set up matplotlib backend
    if output_file:
        matplotlib.use('Agg')  # Non-interactive for file output
    else:
        try:
            matplotlib.use('TkAgg')  # Interactive
        except:
            matplotlib.use('Qt5Agg')  # Fallback

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Common plot settings
    LINEWIDTH = 0.5
    FONTSIZE = 10

    titles = [
        "All Segments",
        "Terminating Endpoints",
        "Problematic Segments",
        "Segment Length Distribution"
    ]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=FONTSIZE + 2)
        ax.set_xlabel("Longitude", fontsize=FONTSIZE)
        ax.set_ylabel("Latitude", fontsize=FONTSIZE)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Plot 1: All segments
    ax = axes[0]
    for i in range(len(df)):
        ax.plot([df.lon1.values[i], df.lon2.values[i]],
                [df.lat1.values[i], df.lat2.values[i]],
                '-', color='tab:blue', linewidth=LINEWIDTH, alpha=0.5)

    # Plot 2: Terminating endpoints
    ax = axes[1]
    for i in range(len(df)):
        ax.plot([df.lon1.values[i], df.lon2.values[i]],
                [df.lat1.values[i], df.lat2.values[i]],
                '-', color='lightgray', linewidth=LINEWIDTH, alpha=0.3)

    if "terminating_endpoints" in issues and len(issues["terminating_endpoints"]) > 0:
        term_df = issues["terminating_endpoints"]
        ax.scatter(term_df["lon"].values, term_df["lat"].values,
                  color='red', s=20, zorder=100,
                  label=f'{len(term_df)} terminating endpoints')
        ax.legend(fontsize=FONTSIZE)

    # Plot 3: Problematic segments
    ax = axes[2]
    for i in range(len(df)):
        ax.plot([df.lon1.values[i], df.lon2.values[i]],
                [df.lat1.values[i], df.lat2.values[i]],
                '-', color='lightgray', linewidth=LINEWIDTH, alpha=0.3)

    legend_items = []

    # Highlight different types of problems
    if "vertical_segments" in issues and len(issues["vertical_segments"]) > 0:
        for _, seg in issues["vertical_segments"].iterrows():
            idx = seg["segment_idx"]
            ax.plot([df.lon1.values[idx], df.lon2.values[idx]],
                   [df.lat1.values[idx], df.lat2.values[idx]],
                   '-', color='red', linewidth=2, alpha=0.8)
        legend_items.append(f'{len(issues["vertical_segments"])} vertical')

    if "horizontal_segments" in issues and len(issues["horizontal_segments"]) > 0:
        for _, seg in issues["horizontal_segments"].iterrows():
            idx = seg["segment_idx"]
            ax.plot([df.lon1.values[idx], df.lon2.values[idx]],
                   [df.lat1.values[idx], df.lat2.values[idx]],
                   '--', color='red', linewidth=2, alpha=0.8)
        legend_items.append(f'{len(issues["horizontal_segments"])} horizontal')

    if "short_segments" in issues and len(issues["short_segments"]) > 0:
        for _, seg in issues["short_segments"].iterrows():
            idx = seg["segment_idx"]
            ax.plot([df.lon1.values[idx], df.lon2.values[idx]],
                   [df.lat1.values[idx], df.lat2.values[idx]],
                   '-', color='orange', linewidth=2, alpha=0.8)
        legend_items.append(f'{len(issues["short_segments"])} very short')

    if "degenerate_segments" in issues and len(issues["degenerate_segments"]) > 0:
        for _, seg in issues["degenerate_segments"].iterrows():
            ax.plot(seg["lon1"], seg["lat1"], 'o', color='purple',
                   markersize=8, markeredgecolor='darkpurple')
        legend_items.append(f'{len(issues["degenerate_segments"])} degenerate')

    if legend_items:
        ax.text(0.02, 0.98, '\n'.join(legend_items),
                transform=ax.transAxes, fontsize=FONTSIZE,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: Length distribution histogram
    ax = axes[3]
    if "length_km" in df.columns:
        lengths = df["length_km"].values
        ax.hist(lengths, bins=50, color='tab:blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Segment Length (km)", fontsize=FONTSIZE)
        ax.set_ylabel("Count", fontsize=FONTSIZE)
        ax.set_title("Segment Length Distribution", fontsize=FONTSIZE + 2)
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = (f"Min: {np.min(lengths):.3f} km\n"
                     f"Median: {np.median(lengths):.1f} km\n"
                     f"Max: {np.max(lengths):.0f} km\n"
                     f"Total: {len(lengths)} segments")
        ax.text(0.70, 0.95, stats_text, transform=ax.transAxes,
                fontsize=FONTSIZE, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_file}")
    else:
        plt.show(block=False)
        plt.pause(0.1)


def print_detailed_report(issues: Dict, verbose: bool = False):
    """Print detailed report of all issues found."""

    print("\n" + "=" * 80)
    print("DETAILED ISSUE REPORT")
    print("=" * 80)

    # Zero dips
    if "zero_dips" in issues and len(issues["zero_dips"]) > 0:
        df = issues["zero_dips"]
        print(f"\nZERO-DEGREE DIPS ({len(df)} segments)")
        print("-" * 40)
        if verbose:
            for idx, row in df.head(10).iterrows():
                print(f"  Segment {idx}: {row.get('name', f'seg_{idx}')}")

    # Short segments
    if "short_segments" in issues and len(issues["short_segments"]) > 0:
        df = issues["short_segments"]
        print(f"\nVERY SHORT SEGMENTS ({len(df)} segments)")
        print("-" * 40)
        if verbose:
            for _, row in df.head(10).iterrows():
                print(f"  {row['segment_name']}: {row['length_km']:.6f} km")

    # Degenerate segments
    if "degenerate_segments" in issues and len(issues["degenerate_segments"]) > 0:
        df = issues["degenerate_segments"]
        print(f"\nDEGENERATE SEGMENTS ({len(df)} segments with start==end)")
        print("-" * 40)
        if verbose:
            for _, row in df.iterrows():
                print(f"  {row['segment_name']}: ({row['lon1']:.6f}, {row['lat1']:.6f})")

    # Duplicate endpoint pairs
    if "endpoint_duplicates" in issues and len(issues["endpoint_duplicates"]) > 0:
        df = issues["endpoint_duplicates"]
        n_groups = df["group_id"].nunique() if len(df) > 0 else 0
        print(f"\nDUPLICATE ENDPOINT PAIRS ({len(df)} segments in {n_groups} groups)")
        print("-" * 40)
        if verbose:
            for gid in sorted(df["group_id"].unique())[:5]:
                g = df[df["group_id"] == gid]
                print(f"  Group {gid} ({len(g)} segments):")
                for _, row in g.iterrows():
                    print(f"    • {row['segment_name']}")

    # Axis-aligned segments
    if "vertical_segments" in issues and len(issues["vertical_segments"]) > 0:
        df = issues["vertical_segments"]
        print(f"\nVERTICAL SEGMENTS ({len(df)} segments)")
        print("-" * 40)
        if verbose:
            for _, row in df.head(10).iterrows():
                print(f"  {row['segment_name']}: Δlon={row['lon_diff']:.9f}")

    if "horizontal_segments" in issues and len(issues["horizontal_segments"]) > 0:
        df = issues["horizontal_segments"]
        print(f"\nHORIZONTAL SEGMENTS ({len(df)} segments)")
        print("-" * 40)
        if verbose:
            for _, row in df.head(10).iterrows():
                print(f"  {row['segment_name']}: Δlat={row['lat_diff']:.9f}")

    # Terminating endpoints
    if "terminating_endpoints" in issues and len(issues["terminating_endpoints"]) > 0:
        df = issues["terminating_endpoints"]
        print(f"\nTERMINATING ENDPOINTS ({len(df)} endpoints)")
        print("-" * 40)
        if verbose:
            # Group by segment for cleaner output
            seg_counts = df.groupby("segment_idx").size()
            print(f"  Segments with terminating endpoints: {len(seg_counts)}")
            for seg_idx in list(seg_counts.index)[:10]:
                seg_endpoints = df[df["segment_idx"] == seg_idx]
                seg_name = seg_endpoints.iloc[0]["segment_name"]
                print(f"    • {seg_name}:")
                for _, ep in seg_endpoints.iterrows():
                    print(f"      - {ep['endpoint_type']} at ({ep['lon']:.6f}, {ep['lat']:.6f})")

    print("\n" + "=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check and fix issues in Celeri segment files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check segments without fixing
  python celeri_check_fix_segments.py segments.csv

  # Fix zero dips and save output
  python celeri_check_fix_segments.py segments.csv --fix-dips

  # Fix all issues and create plot
  python celeri_check_fix_segments.py segments.csv --fix-all --plot

  # Verbose output with custom tolerance
  python celeri_check_fix_segments.py segments.csv --verbose --tolerance 1e-4
        """
    )

    # Required arguments
    parser.add_argument("csv_file",
                       help="Path to CSV file containing segments")

    # Fix options
    fix_group = parser.add_argument_group("fixing options")
    fix_group.add_argument("--fix-dips", action="store_true",
                          help="Fix zero-degree dips (convert to 90 degrees)")
    fix_group.add_argument("--fix-aligned", action="store_true",
                          help="Fix axis-aligned segments by perturbation")
    fix_group.add_argument("--fix-all", action="store_true",
                          help="Apply all available fixes")
    fix_group.add_argument("--perturbation", type=float, default=0.001,
                          help="Perturbation amount for fixing aligned segments (default: 0.001 degrees)")

    # Check options
    check_group = parser.add_argument_group("checking options")
    check_group.add_argument("--tolerance", type=float, default=1e-5,
                            help="Tolerance in degrees for point matching (default: 1e-5)")
    check_group.add_argument("--short-threshold", type=float, default=0.01,
                            help="Threshold in km for short segments (default: 0.01)")

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument("--output", help="Output CSV file for fixed segments")
    output_group.add_argument("--plot", action="store_true",
                             help="Display interactive plot of issues")
    output_group.add_argument("--plot-file", help="Save plot to file")
    output_group.add_argument("--verbose", action="store_true",
                             help="Print detailed information about issues")
    output_group.add_argument("--quiet", action="store_true",
                             help="Suppress most output")

    args = parser.parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Check required columns
    required_cols = ["lon1", "lat1", "lon2", "lat2"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if not args.quiet:
        print(f"\nLoaded {len(df)} segments from {args.csv_file}")

    # Run checks
    checker = SegmentChecker(df, tolerance=args.tolerance)
    issues = checker.run_all_checks(short_threshold_km=args.short_threshold)

    # Print detailed report if requested
    if args.verbose:
        print_detailed_report(issues, verbose=True)
    elif not args.quiet:
        print_detailed_report(issues, verbose=False)

    # Apply fixes if requested
    df_fixed = df
    need_output = False

    if args.fix_all or args.fix_dips or args.fix_aligned:
        fixer = SegmentFixer(df, tolerance=args.tolerance)
        df_fixed = fixer.apply_fixes(
            fix_dips=args.fix_all or args.fix_dips,
            fix_aligned=args.fix_all or args.fix_aligned,
            perturbation=args.perturbation
        )
        need_output = len(fixer.fixes_applied) > 0

        if need_output and not args.quiet:
            # Re-run checks on fixed data
            print("\n" + "=" * 80)
            print("VERIFYING FIXES")
            print("=" * 80)
            checker_fixed = SegmentChecker(df_fixed, tolerance=args.tolerance)
            issues_fixed = checker_fixed.run_all_checks(short_threshold_km=args.short_threshold)

    # Save output if fixes were applied
    if need_output:
        if args.output:
            output_path = Path(args.output)
        else:
            # Generate output filename with UUID
            input_path = Path(args.csv_file)
            uid = uuid.uuid4().hex[:8]
            output_path = input_path.with_name(f"{input_path.stem}_fixed_{uid}.csv")

        df_fixed.to_csv(output_path, index=False, float_format="%.6f")
        print(f"\n✓ Fixed segments saved to: {output_path}")
    elif args.output:
        print("\nNo fixes were applied; no output file created")

    # Create plot if requested
    if args.plot or args.plot_file:
        # Use fixed dataframe if fixes were applied
        plot_df = df_fixed if need_output else df
        plot_issues = issues  # Always show original issues

        if args.plot_file:
            plot_segment_issues(plot_df, plot_issues, output_file=args.plot_file)
        else:
            print("\nDisplaying interactive plot...")
            plot_segment_issues(plot_df, plot_issues)
            print("Close the plot window to exit...")
            plt.show()

    # Final summary
    total_issues = sum(len(v) for v in issues.values() if isinstance(v, pd.DataFrame))

    if not args.quiet:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if total_issues == 0:
            print("✓ No issues found in segment file")
        else:
            print(f"✗ Found {total_issues} total issues across all checks")

            if not (args.fix_all or args.fix_dips or args.fix_aligned):
                print("\nTo fix issues, use:")
                print("  --fix-dips     Fix zero-degree dips")
                print("  --fix-aligned  Fix axis-aligned segments")
                print("  --fix-all      Apply all fixes")

        print("=" * 80)

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())