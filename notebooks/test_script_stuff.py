from sys import stdout
import addict
import datetime
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from loguru import logger
from tqdm import tqdm
from time import sleep
import IPython

import celeri


def create_output_folder(command):
    # Check to see if "runs" folder exists and if not create it
    if not os.path.exists(command.base_runs_folder):
        os.mkdir(command.base_runs_folder)

    # Make output folder for current run
    os.mkdir(command.output_path)


def get_run_name():
    run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(command.base_runs_folder, command.run_name)
    return run_name, output_path


def get_logger(command):
    # Create logger
    logger.remove()  # Remove any existing loggers includeing default stderr
    logger.add(
        stdout,
        format="<cyan>[{level}]</cyan> <green>{message}</green>",
        colorize=True,
    )
    logger.add(command.run_name + ".log")
    logger.info("RUN_NAME: " + command.run_name)


def test_logger():
    logger.info("AAA slip rate constraints")


@logger.catch
def main():
    # Run with: ipython test_script_stuff.py

    # Read in data
    command_file_name = "../data/command/japan_command.json"

    with open(command_file_name, "r") as f:
        command = json.load(f)
    command = addict.Dict(command)  # Convert to dot notation dictionary
    command.file_name = command_file_name

    # Add run_name and output_path
    command.run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    command.output_path = os.path.join(command.base_runs_folder, command.run_name)

    command, segment, block, meshes, station, mogi, sar = celeri.read_data(
        command_file_name
    )

    get_logger(command)
    test_logger()

    logger.debug(segment.keys())
    for i in tqdm(range(len(segment)), colour="yellow"):
        sleep(0.002)
    logger.success("Looped over segments")

    IPython.embed()


if __name__ == "__main__":
    main()
