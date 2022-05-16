from typing import Dict
import IPython
import argparse
import numpy as np
import os
import addict
from rich.console import Console
from rich.table import Table
import glob
import pandas as pd
import celeri


def main(args: Dict):
    # Process arguments
    # Conditions for report of a single run
    if (args.report_type == "report") or (args.report_type == None):
        print("Reporting on single run.")
        if args.folder_name_1 == None:
            print("No folder specified.  Selecting most recent run folder.")
            list_of_folders = filter(os.path.isdir, glob.glob("./../runs/*"))
            list_of_folders = sorted(list_of_folders, key=os.path.getmtime)
            folder_name_1 = list_of_folders[-1]
            print(f"Most recent run folder is: {folder_name_1}.")
        # TODO: If condition when folder_name_1 is specified

    elif args.report_type == "diff":
        print("Reporting on difference between two runs.")

    else:
        print("Invalid comingation of arguments.  See celeri_report --help.")

    # Conditions for a diff report of two runs

    # Find and read the data in folder_1
    command_file_name_1 = glob.glob(os.path.join(folder_name_1, "*_command.json"))[0]
    station_file_name_1 = glob.glob(os.path.join(folder_name_1, "model_station.csv"))[0]
    command_1 = celeri.get_command(command_file_name_1)

    # Check for empty los file.  This is common
    if command_1.los_file_name == {}:
        command_1.los_file_name = "none"

    # Calculation basic velocity statistics
    station_1 = pd.read_csv(station_file_name_1)
    station_1_vels = np.array([station_1.north_vel, station_1.east_vel]).flatten()
    station_1_model_vels = np.array(
        [station_1.model_north_vel, station_1.model_east_vel]
    ).flatten()
    station_1_residuals = station_1_vels - station_1_model_vels
    mae_1 = np.mean(np.abs(station_1_residuals))
    mse_1 = np.mean(station_1_residuals ** 2.0)

    # Build table for reporting on a single model run
    console = Console()
    table = Table(show_header=True, header_style="bold #ffffff")
    table.add_column("property", justify="left")
    table.add_column("value", justify="left")
    table.add_row(
        "[bold white]run folder",
        f"[bold green]{os.path.basename(folder_name_1)}",
    )
    table.add_row(
        "[bold white]command file",
        f"[bold green]{os.path.basename(command_file_name_1)}",
    )
    table.add_row(
        "[bold white]velocity file",
        f"[bold green]{os.path.basename(command_1.station_file_name)}",
    )
    table.add_row(
        "[bold white]segment file",
        f"[bold green]{os.path.basename(command_1.segment_file_name)}",
    )
    table.add_row(
        "[bold white]block file",
        f"[bold green]{os.path.basename(command_1.block_file_name)}",
    )
    table.add_row(
        "[bold white]los file",
        f"[bold green]{os.path.basename(command_1.los_file_name)}",
    )
    table.add_row(
        "[bold white]# stations",
        f"[bold green]{len(station_1)}",
    )
    table.add_row(
        "[bold white]# velocities",
        f"[bold green]{2 * len(station_1)}",
    )
    table.add_row(
        "[bold white]MAE",
        f"[bold green]{mae_1:0.2f} (mm/yr) -- unweighted",
    )
    table.add_row(
        "[bold white]MSE",
        f"[bold green]{mse_1:0.2f} (mm/yr)^2 -- unweighted",
    )

    # Weighted residual velocity (this is actually minimized)

    # Number of segments
    # Number of slip rate constraints

    # Number of blocks
    # Number of block motion constraints

    console.print(table)

    # Drop into ipython REPL
    IPython.embed(banner1="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_type",
        type=str,
        default=None,
        help="'report' or 'diff'.  'report' will provide statstics on a single model run while 'diff' will compare two model runs.  If omited 'report' witll be assumed.",
        required=False,
    )
    parser.add_argument(
        "--folder_name_1",
        type=str,
        default=None,
        required=False,
        help="Name of of folder 1.  If omited it will assome the most recent model run folder.",
    )
    parser.add_argument(
        "--folder_name_2",
        type=str,
        default=None,
        required=False,
        help="Name of of folder 2.  Include for 'diff' option",
    )

    args = addict.Dict(vars(parser.parse_args()))
    main(args)
