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
    # Case 1: No arguments passed.  Assume most recent run folder and report on it.
    if args.folder_name_1 == None:
        print("No folder specified.  Selecting most recent run folder.")
        list_of_folders = filter(os.path.isdir, glob.glob("./../runs/*"))
        list_of_folders = sorted(list_of_folders, key=os.path.getmtime)
        folder_name_1 = list_of_folders[-1]
        print(f"Most recent run folder is: {folder_name_1}.")
    # TODO: If condition when folder_name_1 is specified

    # Case 2: One argument passed.  Report on this folder.
    elif args.folder_name_1 != None:
        print("Reporting on single run.")

    # Case 3: Two arguments passed.  Report on diff between these folders.
    elif (args.folder_name_1 != None) and (args.folder_name_2 != None):
        print("Reporting on difference between two runs.")
        pass

    else:
        print("Invalid comingation of arguments.  See celeri_report --help.")

    # Conditions for a diff report of two runs

    # Find and read the data in folder_1
    command_file_name_1 = glob.glob(os.path.join(folder_name_1, "*_command.json"))[0]
    station_file_name_1 = glob.glob(os.path.join(folder_name_1, "model_station.csv"))[0]
    command_1 = celeri.get_command(command_file_name_1)

    # Check for empty los file.  This is common.
    if command_1.los_file_name == {}:
        command_1.los_file_name = "none"

    # Modify file names to read from `run` rather than `data` folder
    # because they could have changed in `data` folder
    command_1.station_file_name = os.path.join(
        folder_name_1, os.path.basename(command_1.station_file_name)
    )
    command_1.segment_file_name = os.path.join(
        folder_name_1, os.path.basename(command_1.segment_file_name)
    )
    command_1.block_file_name = os.path.join(
        folder_name_1, os.path.basename(command_1.block_file_name)
    )
    command_1.los_file_name = os.path.join(
        folder_name_1, os.path.basename(command_1.los_file_name)
    )

    data_1 = addict.Dict()
    (
        data_1.segment,
        data_1.block,
        data_1.meshes,
        data_1.station,
        data_1.mogi,
        data_1.sar,
    ) = celeri.read_data(command_1)

    n_slip_rate_constraints_1 = (
        np.count_nonzero(data_1.segment.ss_rate_flag)
        + np.count_nonzero(data_1.segment.ds_rate_flag)
        + np.count_nonzero(data_1.segment.ts_rate_flag)
    )
    n_block_constraints_1 = np.count_nonzero(data_1.block.apriori_flag, axis=0)

    # Velocity statistics
    station_1 = pd.read_csv(station_file_name_1)
    station_vel_1 = np.array([station_1.east_vel, station_1.north_vel]).flatten()
    station_sig_1 = np.array([station_1.east_sig, station_1.north_sig]).flatten()
    station_model_vel_1 = np.array(
        [station_1.model_east_vel, station_1.model_north_vel]
    ).flatten()
    station_residual_1 = station_vel_1 - station_model_vel_1
    mae_1 = np.mean(np.abs(station_residual_1))
    mse_1 = np.mean(station_residual_1 ** 2.0)

    # Weighted sum of square residuals.  This is what is really minimized.
    wssr_1 = np.sum((station_residual_1 ** 2.0 / (station_sig_1 ** 2.0)))

    # TODO: Find the names of the 5 stations with largest WSSR
    station_wssr_1 = ((station_1.east_vel - station_1.model_east_vel) ** 2.0) / (
        station_1.east_sig ** 2.0
    ) + ((station_1.north_vel - station_1.model_north_vel) ** 2.0) / (
        station_1.north_sig ** 2.0
    )
    n_largest_contribution_station = 5
    largest_contribution_station_index = (-station_wssr_1).argsort()[
        :n_largest_contribution_station
    ]

    # Reference colors
    color_1 = "cyan"
    color_2 = "yellow"
    color_same = "green"
    color_diff = "red"

    # Build table for reporting on a single model run
    console = Console()
    table = Table(show_header=True, header_style="bold #ffffff")
    table.add_column("property", justify="left")
    table.add_column("value", justify="left")
    table.add_row(
        "[white]run folder",
        f"[{color_1}]{os.path.basename(folder_name_1)}",
    )
    table.add_row(
        "[white]command file",
        f"[{color_1}]{os.path.basename(command_file_name_1)}",
    )
    table.add_row(
        "[white]velocity file",
        f"[{color_1}]{os.path.basename(command_1.station_file_name)}",
    )
    table.add_row(
        "[white]segment file",
        f"[{color_1}]{os.path.basename(command_1.segment_file_name)}",
    )
    table.add_row(
        "[white]block file",
        f"[{color_1}]{os.path.basename(command_1.block_file_name)}",
    )
    table.add_row(
        "[white]los file",
        f"[{color_1}]{os.path.basename(command_1.los_file_name)}",
    )

    # Velocity information
    table.add_row(
        "[white]# stations",
        f"[{color_1}]{len(station_1)}",
    )
    table.add_row(
        "[white]# velocities",
        f"[{color_1}]{2 * len(station_1)}",
    )

    # Block information
    table.add_row(
        "[white]# blocks",
        f"[{color_1}]{len(data_1.block)}",
    )
    table.add_row(
        "[white]# block constraints",
        f"[{color_1}]{n_block_constraints_1}",
    )

    # Segment information
    table.add_row(
        "[white]# segments",
        f"[{color_1}]{len(data_1.segment)}",
    )
    table.add_row(
        "[white]# slip rate constraints",
        f"[{color_1}]{n_slip_rate_constraints_1}",
    )

    # Goodness of fit metrics
    # TODO: Weighted residual velocity (this is actually minimized)
    table.add_row(
        "[white]MAE",
        f"[{color_1}]{mae_1:0.2f} (mm/yr) -- unweighted",
    )
    table.add_row(
        "[white]MSE",
        f"[{color_1}]{mse_1:0.2f} (mm/yr)^2 -- unweighted",
    )
    table.add_row(
        "[white]WSSR",
        f"[{color_1}]{wssr_1:0.2f}",
    )

    for i in range(0, n_largest_contribution_station):
        table.add_row(
            f"[white]#{i + 1} WSSR contributor",
            f"[{color_1}]{station_1.name[largest_contribution_station_index[i]]}",
        )

    console.print(table)

    # Drop into ipython REPL
    IPython.embed(banner1="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        help="Name of of folder 2.",
    )

    args = addict.Dict(vars(parser.parse_args()))
    main(args)
