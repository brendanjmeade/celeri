from argparse import Namespace

import IPython
from loguru import logger

import celeri


@logger.catch
def main(args: Namespace):
    # Read in command file and start logging
    model = celeri.build_model(args.command_file_name)
    config = model.config
    logger = celeri.get_logger(config)
    celeri.process_args(config, args)

    # Read in and process data files
    data, assembly, operators = celeri.get_processed_data_structures(config)

    # Select either H-matrix sparse interative or full dense solve
    if config.solve_type == "hmatrix":
        logger.info("H-matrix build and solve")
        estimation, operators, index = celeri.build_and_solve_hmatrix(
            config, assembly, operators, data
        )
    elif config.solve_type == "dense":
        logger.info("Dense build and solve")
        estimation, operators, index = celeri.build_and_solve_dense(
            config, assembly, operators, data
        )
    elif config.solve_type == "dense_no_meshes":
        logger.info("Dense build and solve (no meshes)")
        estimation, operators, index = celeri.build_and_solve_dense_no_meshes(
            config, assembly, operators, data
        )
    elif config.solve_type == "qp_kl":
        logger.info("Quadratic programming with KL modes")
        estimation, operators, index = celeri.build_and_solve_qp_kl(
            config, assembly, operators, data
        )

    # Copy input files and adata structures to output folder
    celeri.write_output_supplemental(
        args, config, index, data, operators, estimation, assembly
    )

    # Drop into ipython REPL
    if bool(config.repl):
        IPython.embed(banner1="")


if __name__ == "__main__":
    args = celeri.parse_args()
    main(args)
