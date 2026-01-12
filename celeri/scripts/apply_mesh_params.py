#!/usr/bin/env python3
# %%

import argparse
import json
from pathlib import Path

import celeri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "template_file_name",
        type=Path,
        help="Name of template mesh_param file.",
    )

    parser.add_argument(
        "destination_file_name",
        type=Path,
        help="Name of mesh_param file to be updated.",
    )

    parser.add_argument(
        "--start_idx",
        "-s",
        type=int,
        default="0",
        help="Starting index to apply parameters",
        required=False,
    )

    parser.add_argument(
        "--end_idx",
        "-e",
        type=int,
        default="-1",
        help="Ending index to apply parameters",
        required=False,
    )

    args = dict(vars(parser.parse_args()))
    start_idx = int(args["start_idx"])
    end_idx = int(args["end_idx"])

    # Read template and destination files
    template = celeri.MeshConfig.from_file(args["template_file_name"])
    destination = celeri.MeshConfig.from_file(args["destination_file_name"])

    # Define the last dict entry to be changed
    range_end = len(destination) + end_idx + 1
    # For each set of mesh_params to be changed
    for i in range(start_idx, range_end):
        # For each mesh_param in the template list (except filenames)
        for name in type(template[0]).model_fields:
            if (name != "file_name") & (name != "mesh_filename"):
                template_value = getattr(template[0], name)
                setattr(destination[i], name, template_value)

    # Write the updated destination file
    data = [mesh_config.model_dump(mode="json") for mesh_config in destination]
    with args["destination_file_name"].open("w") as destination_file:
        json.dump(data, destination_file, indent=4)


if __name__ == "__main__":
    main()
