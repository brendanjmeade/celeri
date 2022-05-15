from typing import Dict
import IPython
import argparse
import numpy as np
import os
import addict
import glob
from prettytable import PrettyTable
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
    station_1 = pd.read_csv(station_file_name_1)
    station_1_vels = np.array([station_1.north_vel, station_1.east_vel]).flatten()
    station_1_model_vels = np.array(
        [station_1.model_north_vel, station_1.model_east_vel]
    ).flatten()
    station_1_residuals = station_1_vels - station_1_model_vels
    mae_1 = np.mean(np.abs(station_1_residuals))
    mse_1 = np.mean(station_1_residuals ** 2.0)

    # Build table for reporting on a single model run
    table = PrettyTable()
    table.field_names = ["property", "value"]
    table.add_row(["command file", command_file_name_1])
    table.add_row(["velocity file", command_1.station_file_name])
    table.add_row(["segment file", command_1.segment_file_name])
    table.add_row(["block file", command_1.block_file_name])
    table.add_row(["los file", command_1.los_file_name])
    table.add_row(["MAE", f"{mae_1:0.2f} (mm/yr) -- unweighted"])
    table.add_row(["MSE", f"{mse_1:0.2f} (mm/yr)^2 -- unweighted"])
    print(table)

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
