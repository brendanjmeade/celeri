#!/usr/bin/env python3
import argparse
from pathlib import Path

import gmsh
import numpy as np


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert MSH file to STL with vertical scaling"
    )
    parser.add_argument("input_file", help="Input MSH file path", type=Path)
    parser.add_argument(
        "--vertical_scale_factor",
        type=float,
        default=0.01,
        help="Scale factor for Z coordinates (default: 0.01)",
    )
    args = parser.parse_args()

    # Get input and generate output filenames
    input_file = Path(args.input_file)
    output_file = input_file.with_name(f"{input_file.stem}_scaled.stl")

    gmsh.initialize()
    gmsh.open(input_file)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(nodeCoords).reshape(-1, 3)

    # Scale Z by the provided factor
    coords[:, 2] *= args.vertical_scale_factor

    for i, tag in enumerate(nodeTags):
        x, y, z = coords[i]
        gmsh.model.mesh.setNode(int(tag), [x, y, z], [])

    gmsh.write(output_file)
    gmsh.finalize()

    print(
        f"Converted {input_file} to {output_file} with vertical scale factor {args.vertical_scale_factor}"
    )


if __name__ == "__main__":
    main()
