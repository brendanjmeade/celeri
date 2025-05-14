import argparse
import glob
import os

import addict
import IPython
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import celeri

# Reference colors
COLOR_1 = "cyan"
COLOR_2 = "yellow"
COLOR_SAME = "green"
COLOR_DIFF = "red"


def print_table_one_run(run: dict):
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
        f"[{COLOR_1}]{os.path.basename(run.config_file_name)}",
    )
    table.add_row(
        "[white]velocity file",
        f"[{COLOR_1}]{os.path.basename(run.config.station_file_name)}",
    )
    table.add_row(
        "[white]segment file",
        f"[{COLOR_1}]{os.path.basename(run.config.segment_file_name)}",
    )
    table.add_row(
        "[white]block file",
        f"[{COLOR_1}]{os.path.basename(run.config.block_file_name)}",
    )
    table.add_row(
        "[white]los file",
        f"[{COLOR_1}]{os.path.basename(run.config.los_file_name)}",
    )
    table.add_row(
        "[white]solve type",
        f"[{COLOR_1}]{run.config.solve_type}",
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
            f"[{COLOR_1}]{run.station.name[run.largest_contribution_station_index[i]]} {run.station_wssr_percentage[run.largest_contribution_station_index[i]]:0.3f}%",
        )

    console.print(table)


def get_val_text_and_color(eval_value):
    if eval_value is True:
        eval_text = "SAME"
        eval_color = COLOR_SAME
    else:
        eval_text = "DIFF"
        eval_color = COLOR_DIFF
    return eval_text, eval_color


def print_table_two_run(run_1: dict, run_2: dict):
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
    value_1 = os.path.basename(run_1.config_file_name)
    value_2 = os.path.basename(run_2.config_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]command file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # station file names
    value_1 = os.path.basename(run_1.config.station_file_name)
    value_2 = os.path.basename(run_2.config.station_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]velocity file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # segment file names
    value_1 = os.path.basename(run_1.config.segment_file_name)
    value_2 = os.path.basename(run_2.config.segment_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]segment file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # block file names
    value_1 = os.path.basename(run_1.config.block_file_name)
    value_2 = os.path.basename(run_2.config.block_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]block file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # los file names
    value_1 = os.path.basename(run_1.config.los_file_name)
    value_2 = os.path.basename(run_2.config.los_file_name)
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]los file name",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # Type of solution
    value_1 = run_1.config.solve_type
    value_2 = run_2.config.solve_type
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]solve type",
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
    value_1 = run_1.n_station_flag_on
    value_2 = run_2.n_station_flag_on
    eval_text, eval_color = get_val_text_and_color(value_1 == value_2)
    table.add_row(
        "[white]# of on stations",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]{value_1}",
        f"[{COLOR_2}]{value_2}",
    )

    # Do stations have the same coordinates?
    run_1.station_location_array = np.sort(
        np.array([run_1.station.lon.values, run_1.station.lat.values]).T
    )
    run_2.station_location_array = np.sort(
        np.array([run_2.station.lon.values, run_2.station.lat.values]).T
    )
    eval_text, eval_color = get_val_text_and_color(
        np.allclose(run_1.station_location_array, run_2.station_location_array)
    )
    table.add_row(
        "[white]station locations",
        f"[{eval_color}]{eval_text}",
        f"[{COLOR_1}]---",
        f"[{COLOR_2}]---",
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
        # value_1 = run_1.station.name[run_1.largest_contribution_station_index[i]]
        # value_2 = run_2.station.name[run_2.largest_contribution_station_index[i]]
        value_1 = (
            f"{run_1.station.name[run_1.largest_contribution_station_index[i]]} {run_1.station_wssr_percentage[run_1.largest_contribution_station_index[i]]:0.3f}%",
        )[0]
        value_2 = (
            f"{run_2.station.name[run_2.largest_contribution_station_index[i]]} {run_2.station_wssr_percentage[run_2.largest_contribution_station_index[i]]:0.3f}%",
        )[0]
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
    run.config_file_name = glob.glob(
        os.path.join(run.folder_name, "args_*_command.json")
    )[0]
    run.station_file_name = glob.glob(
        os.path.join(run.folder_name, "model_station.csv")
    )[0]
    run.config = celeri.get_config(run.config_file_name)

    # Check for empty los file.  This is common.
    if run.config.los_file_name == {}:
        run.config.los_file_name = "none"

    # Modify file names to read from `run` rather than `data` folder
    # because they could have changed in `data` folder
    run.config.station_file_name = os.path.join(
        run.folder_name, os.path.basename(run.config.station_file_name)
    )
    run.config.segment_file_name = os.path.join(
        run.folder_name, os.path.basename(run.config.segment_file_name)
    )
    run.config.block_file_name = os.path.join(
        run.folder_name, os.path.basename(run.config.block_file_name)
    )
    run.config.los_file_name = os.path.join(
        run.folder_name, os.path.basename(run.config.los_file_name)
    )
    (
        run.segment,
        run.block,
        run.meshes,
        run.station,
        run.mogi,
        run.sar,
    ) = celeri.read_data(run.config)

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
    run.mse = np.mean(run.station_residual**2.0)

    # Weighted sum of square residuals.  This is what is really minimized.
    run.wssr = np.sum(run.station_residual**2.0 / (run.station_sig**2.0))

    # Find the names of the 5 stations with largest WSSR
    run.station_wssr = ((run.station.east_vel - run.station.model_east_vel) ** 2.0) / (
        run.station.east_sig**2.0
    ) + ((run.station.north_vel - run.station.model_north_vel) ** 2.0) / (
        run.station.north_sig**2.0
    )
    run.station_wssr_percentage = 100 * run.station_wssr / (np.sum(run.station_wssr))

    run.n_largest_contribution_station = 10
    run.largest_contribution_station_index = (-run.station_wssr).argsort()[
        : run.n_largest_contribution_station
    ]
    return run


def main(args: dict):
    # Case 1: No arguments passed.  Report on diff between two most recent folders
    if args.folder_name_1 is None:
        list_of_folders = filter(os.path.isdir, glob.glob("./../runs/*"))
        list_of_folders = sorted(list_of_folders, key=os.path.getmtime)
        folder_name_1 = list_of_folders[-1]
        folder_name_2 = list_of_folders[-2]
        print(
            f"Diffing two most recent most recent run folders: {folder_name_1} and {folder_name_2}"
        )
        run_1 = read_process_run_folder(folder_name_1)
        run_2 = read_process_run_folder(folder_name_2)
        print_table_two_run(run_1, run_2)

    # Case 2: One argument passed.  Report on this folder.
    elif (args.folder_name_1 is not None) and (args.folder_name_2 is None):
        folder_name = os.path.join("./../runs/", args.folder_name_1)
        print(f"Reporting on single run {folder_name}")
        run = read_process_run_folder(folder_name)
        print_table_one_run(run)

    # Case 3: Two arguments passed.  Report on diff between these folders.
    elif (args.folder_name_1 is not None) and (args.folder_name_2 is not None):
        folder_name_1 = os.path.join("./../runs/", args.folder_name_1)
        folder_name_2 = os.path.join("./../runs/", args.folder_name_2)
        print(f"Reporting on runs {folder_name_1} and {folder_name_2}")
        run_1 = read_process_run_folder(folder_name_1)
        run_2 = read_process_run_folder(folder_name_2)
        print_table_two_run(run_1, run_2)

    # Case 4: Report on diff between two most recent folders
    elif (args.diff_back is not None) and (args.folder_name_1 is None):
        list_of_folders = filter(os.path.isdir, glob.glob("./../runs/*"))
        list_of_folders = sorted(list_of_folders, key=os.path.getmtime)
        folder_name_1 = list_of_folders[-1]
        folder_name_2 = list_of_folders[-(args.diff_back + 1)]
        print(f"Diffing most recent run folder ({folder_name_1}) with {folder_name_2}")
        run_1 = read_process_run_folder(folder_name_1)
        run_2 = read_process_run_folder(folder_name_2)
        print_table_two_run(run_1, run_2)

    else:
        print("Invalid combination of arguments.  See celeri_report --help.")

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
        "--diff_back",
        type=int,
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
