import argparse
from dataclasses import fields

from loguru import logger

from celeri.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command_file_name", type=str, help="Name of *_command.json file"
    )
    parser.add_argument(
        "--segment_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_segment.csv file",
    )
    parser.add_argument(
        "--station_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_station.csv file",
    )
    parser.add_argument(
        "--block_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_block.csv file",
    )
    parser.add_argument(
        "--mesh_parameters_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_mesh_parameters.json file",
    )
    parser.add_argument(
        "--los_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_los.csv file",
    )
    parser.add_argument(
        "--solve_type",
        type=str,
        default=None,
        required=False,
        help="Solution type (dense | hmatrix)",
    )
    parser.add_argument(
        "--repl",
        type=int,
        default=0,
        required=False,
        help="Flag for dropping into REPL (0 | 1)",
    )
    parser.add_argument(
        "--pickle_save",
        type=int,
        default=0,
        required=False,
        help="Flag for saving major data structures in pickle file (0 | 1)",
    )
    parser.add_argument(
        "--plot_input_summary",
        type=int,
        default=0,
        required=False,
        help="Flag for saving summary plot of input data (0 | 1)",
    )
    parser.add_argument(
        "--plot_estimation_summary",
        type=int,
        default=0,
        required=False,
        help="Flag for saving summary plot of model results (0 | 1)",
    )
    parser.add_argument(
        "--save_elastic",
        type=int,
        default=0,
        required=False,
        help="Flag for saving elastic calculations (0 | 1)",
    )
    parser.add_argument(
        "--reuse_elastic",
        type=int,
        default=0,
        required=False,
        help="Flag for reusing elastic calculations (0 | 1)",
    )
    parser.add_argument(
        "--snap_segments",
        type=int,
        default=0,
        required=False,
        help="Flag for snapping segments (0 | 1)",
    )
    parser.add_argument(
        "--atol",
        type=int,
        default=None,
        required=False,
        help="Primary tolerance for H-matrix solve",
    )
    parser.add_argument(
        "--btol",
        type=int,
        default=None,
        required=False,
        help="Secondary tolerance for H-matrix solve",
    )
    parser.add_argument(
        "--iterative_solver",
        type=str,
        default=None,
        required=False,
        help="Interative solver type (lsqr | lsmr)",
    )

    return parser.parse_args()


def process_args(command: Config, args: argparse.Namespace):
    for field in fields(command):
        key = field.name
        if key in args:
            command_val = getattr(args, key)
            if command_val is not None:
                logger.warning(f"ORIGINAL: command.{key}: {command_val}")
                setattr(command, key, getattr(args, key))
                logger.warning(f"REPLACED: command.{key}: {command_val}")
            else:
                logger.info(f"command.{key}: {command_val}")
