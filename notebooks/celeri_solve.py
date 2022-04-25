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
    parser.add_argument("command_file_name", type=str, help="Name of command file")
    parser.add_argument("--segment_file_name", type=str, default=None, required=False)
    parser.add_argument("--station_file_name", type=str, default=None, required=False)
    parser.add_argument("--block_file_name", type=str, default=None, required=False)
    parser.add_argument(
        "--mesh_parameters_file_name", type=str, default=None, required=False
    )
    parser.add_argument("--los_file_name", type=str, default=None, required=False)
    parser.add_argument("--solve_type", type=str, default=None, required=False)
    parser.add_argument("--repl", type=int, default=0, required=False)
    parser.add_argument("--pickle_save", type=int, default=0, required=False)
    parser.add_argument("--plot_input_summary", type=int, default=0, required=False)
    parser.add_argument(
        "--plot_estimation_summary", type=int, default=0, required=False
    )
    parser.add_argument("--reuse_elastic", type=int, default=None, required=False)
    args = addict.Dict(vars(parser.parse_args()))
    main(args)
