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

    # Create output filename with "_templated" suffix
    dest_path = args["destination_file_name"]
    output_file_name = (
        dest_path.parent / f"{dest_path.stem}_templated{dest_path.suffix}"
    )

    # Track statistics
    standard_count = 0
    alternate_count = 0
    standard_indices = []
    alternate_indices = []

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
            standard_count += 1
            standard_indices.append(i)

        # Print summary statistics
        print("\nSummary Statistics")
        print("------------------")
        print(f"Total meshes processed: {standard_count}")
        print(f"Index range: {start_idx} to {range_end - 1}")
        print(f"Output file: {output_file_name}")
    else:
        # If so, load the alternate template
        alt_template = celeri.MeshConfig.from_file(args["alt_template"])
        # Need to read in the meshes and check their geometry
        meshes = []
        meshes = [celeri.Mesh.from_params(mesh_param) for mesh_param in destination]
        # For each set of mesh_params to be changed
        for i in range(start_idx, range_end):
            # Check the dip threshold
            # If the mesh's maximum element dip is greater than the threshold
            # Taking the max of the magnitudes of the deviation from 90, so that dips expressed > 90 don't bias mean
            if 90 - np.max(np.abs(90 - meshes[i].dip)) > args["dip_threshold"]:
                # Use the alternate template

                # For each mesh_param in the template list (except filenames)
                for name in type(alt_template[0]).model_fields:
                    if (name != "file_name") & (name != "mesh_filename"):
                        template_value = getattr(alt_template[0], name)
                        # Set the destination to the alternate template value
                        setattr(destination[i], name, template_value)
                alternate_count += 1
                alternate_indices.append(i)
            else:
                # Use the standard template

                # For each mesh_param in the template list (except filenames)
                for name in type(template[0]).model_fields:
                    if (name != "file_name") & (name != "mesh_filename"):
                        template_value = getattr(template[0], name)
                        setattr(destination[i], name, template_value)
                standard_count += 1
                standard_indices.append(i)

        # Print summary statistics
        total_count = standard_count + alternate_count
        print("\nSummary Statistics")
        print("------------------")
        print(f"Total meshes processed: {total_count}")
        print(f"Index range: {start_idx} to {range_end - 1}")
        print(f"Dip threshold: {args['dip_threshold']}°")
        print(f"\nStandard template applied: {standard_count} meshes")
        if standard_indices:
            print(f"  Indices: {standard_indices}")
        print(f"Alternate template applied: {alternate_count} meshes")
        if alternate_indices:
            print(f"  Indices: {alternate_indices}")
        print(f"Output file: {output_file_name}")

    # Write the output file (does not overwrite original)
    # Use relative paths based on the output file's parent directory
    output_dir = output_file_name.parent.resolve()
    context = {"paths_relative_to": output_dir}
    data = [
        mesh_config.model_dump(mode="json", context=context)
        for mesh_config in destination
    ]
    with output_file_name.open("w") as output_file:
        json.dump(data, output_file, indent=4)


if __name__ == "__main__":
    main()
