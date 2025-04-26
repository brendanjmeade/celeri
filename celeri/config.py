import glob
import json
import os

import addict
import numpy as np

type Config = addict.Dict


def _get_new_folder_name() -> str:
    """Generate a new folder name based on existing numeric folder names.

    This function scans the current directory for folders with numeric names,
    identifies the highest number, and returns a new folder name that is one
    greater than the highest number, formatted as a zero-padded 10-digit string.

    Returns:
        str: A new folder name as a zero-padded 10-digit string.

    Raises:
        ValueError: If no numeric folder names are found in the current directory.

    Example:
        If the current directory contains folders named "0000000001", "0000000002",
        and "0000000003", the function will return "0000000004".
    """
    # Get all folder names
    folder_names = glob.glob("./../runs/*/")

    # Remove trailing slashes
    folder_names = [folder_name.rstrip(os.sep) for folder_name in folder_names]

    # Remove anything before numerical folder name
    folder_names = [folder_name[-10:] for folder_name in folder_names]

    # Check to see if the folder name is a native run number
    folder_names_runs = list()
    for folder_name in folder_names:
        try:
            folder_names_runs.append(int(folder_name))
        except ValueError:
            pass

    # Get new folder name
    if len(folder_names_runs) == 0:
        new_folder_name = "0000000001"
    else:
        new_folder_number = np.max(folder_names_runs) + 1
        new_folder_name = f"{new_folder_number:010d}"

    return new_folder_name


def get_config(command_file_name) -> Config:
    # NOTE: Rename to `read_command`?
    """Read *command.json file and return contents as a dictionary.

    Args:
        command_file_name (string): Path to command file

    Returns:
        command (Dict): Dictionary with content of command file
    """
    with open(command_file_name) as f:
        command = json.load(f)
    command = addict.Dict(command)  # Convert to dot notation dictionary
    command.file_name = command_file_name

    # Add run_name and output_path
    # command.run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    command.run_name = _get_new_folder_name()
    command.output_path = os.path.join(command.base_runs_folder, command.run_name)
    # command.file_name = command_file_name

    # Sort command keys alphabetically for readability
    command = addict.Dict(sorted(command.items()))

    return command
