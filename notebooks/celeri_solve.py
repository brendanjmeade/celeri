import argparse
import addict
from loguru import logger
from typing import Dict
import IPython
import celeri


@logger.catch
def main(args: Dict):
    # Read in command file and start logging
    command = celeri.get_command(args.command_file_name)
    celeri.create_output_folder(command)
    celeri.get_logger(command)
    celeri.process_args(command, args)

    # Read in and process data files
    data, assembly, operators = celeri.get_processed_data_structures(command)

    # Select either H-matrix sparse interative or full dense solve
    if command.solve_type == "hmatrix":
        logger.info("H-matrix build and solve")
        estimation, operators, index = celeri.build_and_solve_hmatrix(
            command, assembly, operators, data
        )
    elif command.solve_type == "dense":
        logger.info("Dense build and solve")
        estimation, operators, index = celeri.build_and_solve_dense(
            command, assembly, operators, data
        )

    # Copy input files and adata structures to output folder
    celeri.write_output_supplemental(
        args, command, index, data, operators, estimation, assembly
    )

    # Drop into ipython REPL
    if bool(command.repl):
        IPython.embed(banner1="")


if __name__ == "__main__":
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
        "--reuse_elastic",
        type=int,
        default=None,
        required=False,
        help="Flag for reusinging elastic calculations (0 | 1)",
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

    args = addict.Dict(vars(parser.parse_args()))
    main(args)
