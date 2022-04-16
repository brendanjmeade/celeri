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

    # Calculate Okada partials for all segments
    celeri.get_elastic_operators_okada(operators, segment, station, command)

    # Get operators common to both dense and H-matrix
    get_common_operators(
        command, segment, station, block, meshes, mogi, operators, assembly
    )

    index = celeri.get_index(assembly, station, block, meshes)

    command.solve_type = "hmatrix"
    if command.solve_type == "hmatrix":
        # Data and data weighting vector
        weighting_vector = celeri.get_weighting_vector(command, station, meshes, index)
        data_vector = celeri.get_data_vector(assembly, index)

        # Apply data weighting
        data_vector = data_vector * np.sqrt(weighting_vector)

        # Cast all block submatrices to sparse
        sparse_block_motion_okada_faults = csr_matrix(
            operators.rotation_to_velocities[index.station_row_keep_index, :]
            - operators.rotation_to_slip_rate_to_okada_to_velocities[
                index.station_row_keep_index, :
            ]
        )
        sparse_block_motion_constraints = csr_matrix(operators.block_motion_constraints)
        sparse_block_slip_rate_constraints = csr_matrix(operators.slip_rate_constraints)

        # Calculate column normalization vector for blocks
        operator_block_only = celeri.get_full_dense_operator_block_only(
            operators, index
        )
        weighting_vector_block_only = weighting_vector[
            0 : operator_block_only.shape[0]
        ][:, None]
        col_norms = np.linalg.norm(
            operator_block_only * np.sqrt(weighting_vector_block_only), axis=0
        )

        # Hmatrix decompositon for each TDE mesh
        H, col_norms = celeri.get_h_matrices_for_tde_meshes(
            command, meshes, station, operators, index, col_norms
        )
    elif command.solve_type == "dense":
        pass

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
