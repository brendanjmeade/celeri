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

    # index = celeri.get_index(assembly, data.station, data.block, data.meshes)

    # Quick input plot
    print("**********************", bool(command.plot_input_summary))
    if bool(command.plot_input_summary):
        print("hi")
        celeri.plot_input_summary(
            data.segment,
            data.station,
            data.block,
            data.meshes,
            data.mogi,
            data.sar,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    if command.solve_type == "hmatrix":
        logger.info("H-matrix build and solve")
        # estimation, operators, index = build_and_solve_hmatrix()
    elif command.solve_type == "dense":
        logger.info("Dense build and solve")
        # estimation, operators, index = build_and_solve_dense()

    # Save run to disk
    # celeri.write_output(command, estimation, data.station, data.segment, data.block, data.meshes)

    # # Quick output plot
    # if bool(command.plot_estimation_summary):
    #     celeri.plot_estimation_summary(
    #         segment,
    #         station,
    #         meshes,
    #         estimation,
    #         lon_range=command.lon_range,
    #         lat_range=command.lat_range,
    #         quiver_scale=command.quiver_scale,
    #     )

    # Drop into ipython REPL
    if bool(command.repl):
        IPython.embed(banner1="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command_file_name", type=str, help="name of command file")
    parser.add_argument("--segment_file_name", type=str, default=None, required=False)
    parser.add_argument("--station_file_name", type=str, default=None, required=False)
    parser.add_argument("--block_file_name", type=str, default=None, required=False)
    parser.add_argument("--mesh_file_name", type=str, default=None, required=False)
    parser.add_argument("--los_file_name", type=str, default=None, required=False)
    parser.add_argument("--solve_type", type=str, default=None, required=False)
    parser.add_argument("--repl", type=int, default=0, required=False)
    parser.add_argument("--plot_input_summary", type=int, default=0, required=False)
    parser.add_argument(
        "--plot_estimation_summary", type=int, default=0, required=False
    )

    args = addict.Dict(vars(parser.parse_args()))
    main(args)
