#!/usr/bin/env python3
# %%

import argparse
import json
from pathlib import Path

import numpy as np

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
        "--alt_template",
        "-at",
        type=Path,
        default=Path(),
        help="Name of alternate template mesh_param file.",
        required=False,
    )

    parser.add_argument(
        "--dip_threshold",
        "-dt",
        type=float,
        default="85",
        help="Dip value above which alternate template is applied.",
        required=False,
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

    # Check to see if multiple templates have been passed
    if args["alt_template"] == Path():
        # If not, just use a single template
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
    else:
        # If so, need to read in the mesh and check its geometry
        # For each set of mesh_params to be changed
        alt_template = celeri.MeshConfig.from_file(args["alt_template"])
        meshes = []
        meshes = [celeri.Mesh.from_params(mesh_param) for mesh_param in destination]
        for i in range(start_idx, range_end):
            this_mesh = meshes[i]
            # Check the dip threshold
            # If the mesh's mean element dip is greater than the threshold
            if np.abs(90 - np.mean(this_mesh.dip)) < np.abs(90 - args["dip_threshold"]):
                # Use the alternate template

                # For each mesh_param in the template list (except filenames)
                for name in type(alt_template[0]).model_fields:
                    if (name != "file_name") & (name != "mesh_filename"):
                        template_value = getattr(alt_template[0], name)
                        # Set the destination to the alternate template value
                        setattr(destination[i], name, template_value)
            else:
                # Use the standard template

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
