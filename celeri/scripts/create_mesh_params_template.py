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
        default=Path("template_mesh_params.json"),
        help="Name of template mesh_param file.",
        nargs="?",
    )

    args = dict(vars(parser.parse_args()))

    # Get default mesh_params from MeshConfig in mesh.py
    template = celeri.MeshConfig(file_name=args["template_file_name"])

    # Write template file

    # Create json dump as a list of length one, to agree with what celeri.mesh.from_file expects
    data = [template.model_dump(mode="json")]
    with args["template_file_name"].open("w") as template_file:
        json.dump(data, template_file, indent=4)


if __name__ == "__main__":
    main()
