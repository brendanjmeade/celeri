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


def print_table_one_run(run: Dict):
    # Build table for reporting on a single model run
    console = Console()
    table = Table(show_header=True, header_style="bold #ffffff")
    table.add_column("property", justify="left")
    table.add_column("value", justify="left")
    table.add_row(
        "[white]run folder",
        f"[{COLOR_1}]{os.path.basename(run.folder_name)}",
    )
    table.add_row(
        "[white]command file",
        f"[{COLOR_1}]{os.path.basename(run.command_file_name)}",
    )
    table.add_row(
        "[white]velocity file",
        f"[{COLOR_1}]{os.path.basename(run.command.station_file_name)}",
    )
    table.add_row(
        "[white]segment file",
        f"[{COLOR_1}]{os.path.basename(run.command.segment_file_name)}",
    )
    table.add_row(
        "[white]block file",
        f"[{COLOR_1}]{os.path.basename(run.command.block_file_name)}",
    )
    table.add_row(
        "[white]los file",
        f"[{COLOR_1}]{os.path.basename(run.command.los_file_name)}",
    )

    # Velocity information
    table.add_row(
        "[white]# stations",
        f"[{COLOR_1}]{len(run.station)}",
    )

    table.add_row(
        "[white]# on stations",
        f"[{COLOR_1}]{run.n_station_flag_on}",
    )

    # Block information
    table.add_row(
        "[white]# blocks",
        f"[{COLOR_1}]{len(run.block)}",
    )
    table.add_row(
        "[white]# block constraints",
        f"[{COLOR_1}]{run.n_block_constraints}",
    )

    # Segment information
    table.add_row(
        "[white]# segments",
        f"[{COLOR_1}]{len(run.segment)}",
    )
    table.add_row(
        "[white]# slip rate constraints",
        f"[{COLOR_1}]{run.n_slip_rate_constraints}",
    )

    # Goodness of fit metrics
    table.add_row(
        "[white]MAE",
        f"[{COLOR_1}]{run.mae:0.2f} (mm/yr) -- unweighted",
    )
    table.add_row(
        "[white]MSE",
        f"[{COLOR_1}]{run.mse:0.2f} (mm/yr)^2 -- unweighted",
    )
    table.add_row(
        "[white]WSSR",
        f"[{COLOR_1}]{run.wssr:0.2f}",
    )
    for i in range(0, run.n_largest_contribution_station):
        table.add_row(
            f"[white]#{i + 1} WSSR contributor",
            f"[{COLOR_1}]{run.station.name[run.largest_contribution_station_index[i]]}",
        )

    console.print(table)


def get_val_text_and_color(eval_value):
    if eval_value == True:
        eval_text = "SAME"
        eval_color = COLOR_SAME
    else:
        eval_text = "DIFF"
        eval_color = COLOR_DIFF
    return eval_text, eval_color


def print_table_two_run(run_1: Dict, run_2: Dict):
    # Build table for reporting on a single model run
    console = Console()
    table = Table(show_header=True, header_style="bold #ffffff")
    table.add_column("property", justify="left")
    table.add_column("eval", justify="left")
    table.add_column("value", justify="left")
    table.add_column("value", justify="left")

    # run folder names
    value_1 = os.path.basename(run_1.folder_name)
    value_2 = os.path.basename(run_2.folder_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]run folder name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # command file names
    value_1 = os.path.basename(run_1.command_file_name)
    value_2 = os.path.basename(run_2.command_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]command file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # station file names
    value_1 = os.path.basename(run_1.command.station_file_name)
    value_2 = os.path.basename(run_2.command.station_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]velocity file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # segment file names
    value_1 = os.path.basename(run_1.command.segment_file_name)
    value_2 = os.path.basename(run_2.command.segment_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]segment file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # block file names
    value_1 = os.path.basename(run_1.command.block_file_name)
    value_2 = os.path.basename(run_2.command.block_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]block file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # los file names
    value_1 = os.path.basename(run_1.command.los_file_name)
    value_2 = os.path.basename(run_2.command.los_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]los file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # number of stations
    value_1 = len(run_1.station)
    value_2 = len(run_2.station)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of stations",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # number of horizontal velocities
    # table.add_row(
    #     "[white]# of velocities",
    #     f"[{COLOR_1}]{2 * len(run.station)}",
    # )

    # number of horizontal velocities
    value_1 = len(run_1.n_station_flag_on)
    value_2 = len(run_2.n_station_flag_on)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of on stations",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # number of blocks
    value_1 = len(run_1.block)
    value_2 = len(run_2.block)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of blocks",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # number of blocks constriants
    value_1 = run_1.n_block_constraints
    value_2 = run_2.n_block_constraints
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of block constraints",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # number of segments
    value_1 = len(run_1.segment)
    value_2 = len(run_2.segment)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of segments",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # number of slip rate constraints
    value_1 = run_1.n_slip_rate_constraints
    value_2 = run_2.n_slip_rate_constraints
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of slip rate constraints",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # MAE
    value_1 = run_1.mae
    value_2 = run_2.mae
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]MAE",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1:0.3f}",
        f"[{COLOR_2}]{value_2:0.3f}",
    )

    # MSE
    value_1 = run_1.mse
    value_2 = run_2.mse
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]MSE",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1:0.3f}",
        f"[{COLOR_2}]{value_2:0.3f}",
    )

    # WSSR
    value_1 = run_1.wssr
    value_2 = run_2.wssr
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]WSSR",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1:0.3f}",
        f"[{COLOR_2}]{value_2:0.3f}",
    )

    # stations that contribute the most to the residual
    for i in range(0, run_1.n_largest_contribution_station):
        value_1 = run_1.station.name[run_1.largest_contribution_station_index[i]]
        value_2 = run_2.station.name[run_2.largest_contribution_station_index[i]]
        eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
        table.add_row(
            f"[white]#{i + 1} WSSR contributor",
            f"[{eval_color}]{eval_text}",
            f"[{COLOR_1}]{value_1}",
            f"[{COLOR_2}]{value_2}",
        )

    console.print(table)


def read_process_run_folder(folder_name: str):
    # Find and read the data in folder_name
    run = addict.Dict()
    run.folder_name = folder_name
    run.command_file_name = glob.glob(
        os.path.join(run.folder_name, "args_*_command.json")
    )[0]
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
    run.n_station_flag_on = np.count_nonzero(run.station.flag)
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
        list_of_folders = filter(os.path.isdir, glob.glob("./../runs/*"))
        list_of_folders = sorted(list_of_folders, key=os.path.getmtime)
        folder_name = list_of_folders[-1]
        print(f"No folder specified.  Selecting most recent run folder: {folder_name}")
        run = read_process_run_folder(folder_name)
        print_table_one_run(run)

    # Case 2: One argument passed.  Report on this folder.
    elif (args.folder_name_1 != None) and (args.folder_name_2 == None):
        folder_name = os.path.join("./../runs/", args.folder_name_1)
        print(f"Reporting on single run {folder_name}")
        run = read_process_run_folder(folder_name)
        print_table_one_run(run)

    # Case 3: Two arguments passed.  Report on diff between these folders.
    elif (args.folder_name_1 != None) and (args.folder_name_2 != None):
        folder_name_1 = os.path.join("./../runs/", args.folder_name_1)
        folder_name_2 = os.path.join("./../runs/", args.folder_name_2)
        print(f"Reporting on runs {folder_name_1} and {folder_name_2}")
        run_1 = read_process_run_folder(folder_name_1)
        run_2 = read_process_run_folder(folder_name_2)
        print_table_two_run(run_1, run_2)
        # print_table_one_run(run_1)
        # print_table_one_run(run_2)

    else:
        print("Invalid comingation of arguments.  See celeri_report --help.")

    # Drop into ipython REPL
    if bool(args.repl):
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
    parser.add_argument(
        "--repl",
        type=int,
        default=0,
        required=False,
        help="Start ipython REPL.",
    )
    args = addict.Dict(vars(parser.parse_args()))
    main(args)
