"""Plot segment/block diagnostics to check model segment and block integrity."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import celeri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_file",
        help="Path to the model config JSON file (e.g. wna_config_constraints.json).",
    )
    parser.add_argument(
        "--block-label",
        type=int,
        default=92,
        help="Block label whose stations are highlighted (default: 92).",
    )
    args = parser.parse_args()

    model = celeri.build_model(args.config_file)

    # Plot segments
    plt.figure(figsize=(10, 10))
    for i in range(len(model.segment)):
        plt.plot(
            [model.segment.lon1[i], model.segment.lon2[i]],
            [model.segment.lat1[i], model.segment.lat2[i]],
            "-b",
            linewidth=0.5,
        )

    # Plot block interior points with labels
    for i in range(len(model.block)):
        plt.text(
            model.block.interior_lon[i],
            model.block.interior_lat[i],
            f"{model.block.block_label[i]}",
        )

    # Highlight stations on a specific block
    current_block_idx = np.where(model.station.block_label == args.block_label)[0]
    plt.plot(
        model.station.lon[current_block_idx],
        model.station.lat[current_block_idx],
        "r+",
    )

    plt.show()


if __name__ == "__main__":
    main()
