import IPython
from loguru import logger

import celeri


@logger.catch
def main(args: dict):
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
    elif command.solve_type == "dense_no_meshes":
        logger.info("Dense build and solve (no meshes)")
        estimation, operators, index = celeri.build_and_solve_dense_no_meshes(
            command, assembly, operators, data
        )
    elif command.solve_type == "qp_kl":
        logger.info("Quadratic programming with KL modes")
        estimation, operators, index = celeri.build_and_solve_qp_kl(
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
    args = celeri.parse_args()
    main(args)
