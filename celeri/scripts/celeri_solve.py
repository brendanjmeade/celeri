#!/usr/bin/env python3

import IPython
from loguru import logger

import celeri


@logger.catch
def main():
    # Process arguments
    args = celeri.parse_args()

    # Read in command file and start logging
    config = celeri.get_config(args.config_file_name)
    logger = celeri.get_logger(config)
    celeri.process_args(config, args)
    model = celeri.build_model(config)

    if config.solve_type == "dense":
        # Classic dense solve
        logger.info("Dense build and solve")
        estimation = celeri.build_and_solve_dense(model)
    elif config.solve_type == "dense_no_meshes":
        # Classic dense solve with no meshes
        logger.info("Dense build and solve (no meshes)")
        estimation = celeri.build_and_solve_dense_no_meshes(model)
    elif config.solve_type == "qp_kl":
        # Bounded solve
        logger.info("Quadratic programming with KL modes")
        raise NotImplementedError("qp_kl is not supported yet")
    else:
        raise ValueError(f"Unknown solve type: {config.solve_type}")

    # Write output
    celeri.write_output(estimation)

    # Drop into ipython REPL
    if config.repl:
        IPython.embed(banner1="")


if __name__ == "__main__":
    main()
