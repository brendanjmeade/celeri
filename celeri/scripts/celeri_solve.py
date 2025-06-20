#!/usr/bin/env python3

import IPython
from loguru import logger

import celeri


@logger.catch
def main():
    args = celeri.parse_args()
    # Read in command file and start logging
    config = celeri.get_config(args.command_file_name)
    logger = celeri.get_logger(config)
    celeri.process_args(config, args)
    model = celeri.build_model(config)

    # Select either H-matrix sparse interative or full dense solve
    if config.solve_type == "hmatrix":
        logger.info("H-matrix build and solve")
        raise NotImplementedError("hmatrix is not supported anymore")
    elif config.solve_type == "dense":
        logger.info("Dense build and solve")
        estimation = celeri.build_and_solve_dense(model)
    elif config.solve_type == "dense_no_meshes":
        logger.info("Dense build and solve (no meshes)")
        estimation = celeri.build_and_solve_dense_no_meshes(model)
    elif config.solve_type == "qp_kl":
        logger.info("Quadratic programming with KL modes")
        raise NotImplementedError("qp_kl is not supported yet")
    else:
        raise ValueError(f"Unknown solve type: {config.solve_type}")

    # Copy input files and adata structures to output folder
    celeri.write_output_supplemental(
        args,
        config,
        estimation.operators.index,
        model,
        estimation.operators,
        estimation,
        estimation.operators.assembly,
    )

    # Drop into ipython REPL
    if bool(config.repl):
        IPython.embed(banner1="")


if __name__ == "__main__":
    main()
