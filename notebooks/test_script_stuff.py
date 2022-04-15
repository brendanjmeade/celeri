from sys import stdout
import argparse
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


@logger.catch
def main(args):
    # Run with: python test_script_stuff.py ../data/command/japan_command.json

    # Read in data
    command = celeri.get_command(args.command_file_name)

    # Assign other command line arguments to command
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Start logging
    celeri.get_logger(command)

    # Print command

    command = addict.Dict(sorted(command.items()))
    for key, value in command.items():
        logger.info(f"command.{key}: {value}")

    # Drop into ipython REPL
    if command.repl == "yes":
        IPython.embed(banner1="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command_file_name", type=str, help="name of command file")
    parser.add_argument("--segment_file_name", type=str, default=None, required=False)
    parser.add_argument("--station_file_name", type=str, default=None, required=False)
    parser.add_argument("--block_file_name", type=str, default=None, required=False)
    parser.add_argument("--mesh_file_name", type=str, default=None, required=False)
    parser.add_argument("--los_file_name", type=str, default=None, required=False)
    parser.add_argument("--repl", type=str, default="no", required=False)

    args = parser.parse_args()
    main(args)
