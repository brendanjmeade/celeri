import argparse
import addict
from loguru import logger
from typing import Dict
import IPython
import celeri


def get_common_operators(
    command, segment, station, block, meshes, mogi, operators, assembly
):
    # Get all of the other operators
    celeri.get_all_mesh_smoothing_matrices(meshes, operators)

    operators.rotation_to_velocities = celeri.get_rotation_to_velocities_partials(
        station
    )
    operators.global_float_block_rotation = (
        celeri.get_global_float_block_rotation_partials(station)
    )
    assembly, operators.block_motion_constraints = celeri.get_block_motion_constraints(
        assembly, block, command
    )
    assembly, operators.slip_rate_constraints = celeri.get_slip_rate_constraints(
        assembly, segment, block, command
    )
    operators.rotation_to_slip_rate = celeri.get_rotation_to_slip_rate_partials(
        segment, block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = celeri.get_block_strain_rate_to_velocities_partials(block, station, segment)
    operators.mogi_to_velocities = celeri.get_mogi_to_velocities_partials(
        mogi, station, command
    )
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    celeri.get_tde_slip_rate_constraints(meshes, operators)


@logger.catch
def main(args: Dict):
    # Read in command file and start logging
    command = celeri.get_command(args.command_file_name)
    celeri.create_output_folder(command)
    celeri.get_logger(command)
    celeri.process_args(command, args)

    # Read in and process data files
    segment, block, meshes, station, mogi, sar = celeri.read_data(command)
    station = celeri.process_station(station, command)
    segment = celeri.process_segment(segment, command, meshes)
    sar = celeri.process_sar(sar, command)
    closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)
    assembly = addict.Dict()
    operators = addict.Dict()
    operators.meshes = [addict.Dict()] * len(meshes)

    # Quick input plot

    # Get operators common to both dense and H-matrix
    get_common_operators(
        command, segment, station, block, meshes, mogi, operators, assembly
    )

    # index = celeri.get_index(assembly, station, block, meshes)

    if command.solve_type == "hmatrix":
        estimation, operators, index = build_and_solve_hmatrix()
    elif command.solve_type == "dense":
        estimation, operators, index = build_and_solve_dense()

    # Save run to disk
    celeri.write_output(command, estimation, station, segment, block, meshes)

    # Quick output plot
    if bool(command.plot_estimation_summary):
        celeri.plot_estimation_summary(
            segment,
            station,
            meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

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
    parser.add_argument("--repl", type=int, default="no", required=False)
    args = addict.Dict(vars(parser.parse_args()))
    main(args)
