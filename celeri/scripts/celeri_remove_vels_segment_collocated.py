#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point


def main():
    # Check if correct number of arguments provided
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python process_csv.py <station.csv> <segment.csv> [buffer_distance]"
        )
        print("Default buffer distance: 0.002")
        sys.exit(1)

    station_file = sys.argv[1]
    segment_file = sys.argv[2]
    buffer_distance = float(sys.argv[3]) if len(sys.argv) == 4 else 0.002

    # Read CSV files into pandas dataframes
    try:
        station_df = pd.read_csv(station_file)
        print(f"Successfully loaded station file: {station_file}")

        segment_df = pd.read_csv(segment_file)
        print(f"Successfully loaded segment file: {segment_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)

    print(f"Using buffer distance: {buffer_distance}\n")
    print("-" * 80)

    # Lists to store matching segments and stations
    matching_segments = []
    matching_stations = []
    matching_station_indices = set()  # Track indices of stations to remove

    # Loop over each line segment
    for idx, segment in segment_df.iterrows():
        # Create line segment geometry
        line = LineString(
            [(segment["lon1"], segment["lat1"]), (segment["lon2"], segment["lat2"])]
        )

        # Create buffer around the line segment
        buffer = line.buffer(buffer_distance)

        # Check which stations fall within the buffer
        stations_near_segment = []

        for station_idx, station in station_df.iterrows():
            station_point = Point(station["lon"], station["lat"])

            if buffer.contains(station_point):
                distance_to_line = line.distance(station_point)
                stations_near_segment.append(
                    {
                        "station": station,
                        "station_index": station_idx,
                        "distance": distance_to_line,
                    }
                )

        # If stations were found near this segment, print and store results
        if stations_near_segment:
            matching_segments.append(segment)

            print(
                f"Segment {idx}: ({segment['lon1']:.4f}, {segment['lat1']:.4f}) to "
                f"({segment['lon2']:.4f}, {segment['lat2']:.4f})"
            )

            if "name" in segment:
                print(f"  Segment name: {segment['name']}")

            print(f"  Found {len(stations_near_segment)} nearby station(s):")

            for station_info in stations_near_segment:
                station = station_info["station"]
                station_idx = station_info["station_index"]
                distance = station_info["distance"]
                matching_stations.append(station)
                matching_station_indices.add(station_idx)

                print(
                    f"    - Station at ({station['lon']:.4f}, {station['lat']:.4f}), "
                    f"distance: {distance:.6f}"
                )
                if "name" in station or "station_name" in station:
                    name_col = "name" if "name" in station else "station_name"
                    print(f"      Name: {station[name_col]}")

            print()

    print("-" * 80)
    print(f"\nSummary: Found {len(matching_segments)} segments with nearby stations")
    print(f"Total unique stations near segments: {len(matching_station_indices)}")

    # Filter out stations near segments and write new CSV
    filtered_station_df = station_df.drop(index=list(matching_station_indices))

    # Create output filename
    base_name = os.path.splitext(station_file)[0]
    output_file = f"{base_name}_near_segment_removed_{buffer_distance}.csv"

    # Write the filtered dataframe to CSV without index
    filtered_station_df.to_csv(output_file, index=False)
    print(f"\nFiltered station file written to: {output_file}")
    print(f"Original stations: {len(station_df)}")
    print(f"Stations removed: {len(matching_station_indices)}")
    print(f"Remaining stations: {len(filtered_station_df)}")

    # Create plot if matches were found
    if matching_segments:
        plt.figure(figsize=(12, 10))

        # Plot all matching line segments in cyan
        for segment in matching_segments:
            plt.plot(
                [segment["lon1"], segment["lon2"]],
                [segment["lat1"], segment["lat2"]],
                "c-",
                linewidth=2,
                alpha=0.7,
            )

        # Plot all matching stations as red dots
        for station in matching_stations:
            plt.plot(station["lon"], station["lat"], "ro", markersize=6)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Line Segments with Stations within {buffer_distance} units")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        # Save and show the plot
        plt.savefig("segments_with_nearby_stations.png", dpi=300, bbox_inches="tight")
        print("\nPlot saved as 'segments_with_nearby_stations.png'")
        plt.show()
    else:
        print("\nNo matching segments found - no plot created")


if __name__ == "__main__":
    main()
