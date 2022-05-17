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

# Reference colors
COLOR_1 = "cyan"
COLOR_2 = "yellow"
COLOR_SAME = "green"
COLOR_DIFF = "red"


def print_table_single(run_1: Dict):
    # Build table for reporting on a single model run
    console = Console()
    table = Table(show_header=True, header_style="bold #ffffff")
    table.add_column("property", justify="left")
    table.add_column("value", justify="left")
    table.add_row(
        "[white]run folder",
        f"[{COLOR_1}]{os.path.basename(run_1.folder_name)}",
    )
    table.add_row(
        "[white]command file",
        f"[{COLOR_1}]{os.path.basename(run_1.command_file_name)}",
    )
    table.add_row(
        "[white]velocity file",
        f"[{COLOR_1}]{os.path.basename(run_1.command.station_file_name)}",
    )
    table.add_row(
        "[white]segment file",
        f"[{COLOR_1}]{os.path.basename(run_1.command.segment_file_name)}",
    )
    table.add_row(
        "[white]block file",
        f"[{COLOR_1}]{os.path.basename(run_1.command.block_file_name)}",
    )
    table.add_row(
        "[white]los file",
        f"[{COLOR_1}]{os.path.basename(run_1.command.los_file_name)}",
    )

    # Velocity information
    table.add_row(
        "[white]# stations",
        f"[{COLOR_1}]{len(run_1.station)}",
    )
    table.add_row(
        "[white]# velocities",
        f"[{COLOR_1}]{2 * len(run_1.station)}",
    )

    # Block information
    table.add_row(
        "[white]# blocks",
        f"[{COLOR_1}]{len(run_1.block)}",
    )
    table.add_row(
        "[white]# block constraints",
        f"[{COLOR_1}]{run_1.n_block_constraints}",
    )

    # Segment information
    table.add_row(
        "[white]# segments",
        f"[{COLOR_1}]{len(run_1.segment)}",
    )
    table.add_row(
        "[white]# slip rate constraints",
        f"[{COLOR_1}]{run_1.n_slip_rate_constraints}",
    )

    # Goodness of fit metrics
    table.add_row(
        "[white]MAE",
        f"[{COLOR_1}]{run_1.mae:0.2f} (mm/yr) -- unweighted",
    )
    table.add_row(
        "[white]MSE",
        f"[{COLOR_1}]{run_1.mse:0.2f} (mm/yr)^2 -- unweighted",
    )
    table.add_row(
        "[white]WSSR",
        f"[{COLOR_1}]{run_1.wssr:0.2f}",
    )
    for i in range(0, run_1.n_largest_contribution_station):
        table.add_row(
            f"[white]#{i + 1} WSSR contributor",
            f"[{COLOR_1}]{run_1.station.name[run_1.largest_contribution_station_index[i]]}",
        )

    console.print(table)


def read_process_run_folder(folder_name: str):
    # Find and read the data in folder_name
    run = addict.Dict()
    run.folder_name = folder_name
    run.command_file_name = glob.glob(os.path.join(run.folder_name, "*_command.json"))[
        0
    ]
    run.station_file_name = glob.glob(
        os.path.join(run.folder_name, "model_station.csv")
    )[0]
    run.command = celeri.get_command(run.command_file_name)

    # Check for empty los file.  This is common.
    if run.command.los_file_name == {}:
        run.command.los_file_name = "none"

    # Modify file names to read from `run` rather than `data` folder
    # because they could have changed in `data` folder
    run.command.station_file_name = os.path.join(
        run.folder_name, os.path.basename(run.command.station_file_name)
    )
    run.command.segment_file_name = os.path.join(
        run.folder_name, os.path.basename(run.command.segment_file_name)
    )
    run.command.block_file_name = os.path.join(
        run.folder_name, os.path.basename(run.command.block_file_name)
    )
    run.command.los_file_name = os.path.join(
        run.folder_name, os.path.basename(run.command.los_file_name)
    )
    (
        run.segment,
        run.block,
        run.meshes,
        run.station,
        run.mogi,
        run.sar,
    ) = celeri.read_data(run.command)

    run.n_slip_rate_constraints = (
        np.count_nonzero(run.segment.ss_rate_flag)
        + np.count_nonzero(run.segment.ds_rate_flag)
        + np.count_nonzero(run.segment.ts_rate_flag)
    )
    run.n_block_constraints = np.count_nonzero(run.block.apriori_flag, axis=0)

    # Velocity statistics
    run.station = pd.read_csv(run.station_file_name)
    run.station_vel = np.array([run.station.east_vel, run.station.north_vel]).flatten()
    run.station_sig = np.array([run.station.east_sig, run.station.north_sig]).flatten()
    run.station_model_vel = np.array(
        [run.station.model_east_vel, run.station.model_north_vel]
    ).flatten()
    run.station_residual = run.station_vel - run.station_model_vel
    run.mae = np.mean(np.abs(run.station_residual))
    run.mse = np.mean(run.station_residual ** 2.0)

    # Weighted sum of square residuals.  This is what is really minimized.
    run.wssr = np.sum((run.station_residual ** 2.0 / (run.station_sig ** 2.0)))

    # Find the names of the 5 stations with largest WSSR
    run.station_wssr = ((run.station.east_vel - run.station.model_east_vel) ** 2.0) / (
        run.station.east_sig ** 2.0
    ) + ((run.station.north_vel - run.station.model_north_vel) ** 2.0) / (
        run.station.north_sig ** 2.0
    )
    run.n_largest_contribution_station = 5
    run.largest_contribution_station_index = (-run.station_wssr).argsort()[
        : run.n_largest_contribution_station
    ]
    return run


def main(args: Dict):
    # Case 1: No arguments passed.  Assume most recent run folder and report on it.
    if args.folder_name_1 == None:
        print("No folder specified.  Selecting most recent run folder.")
        list_of_folders = filter(os.path.isdir, glob.glob("./../runs/*"))
        list_of_folders = sorted(list_of_folders, key=os.path.getmtime)
        folder_name = list_of_folders[-1]
        print(f"Most recent run folder is: {folder_name}")
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

    # Print report table for single run
    run_1 = read_process_run_folder(folder_name)
    print_table_single(run_1)

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
