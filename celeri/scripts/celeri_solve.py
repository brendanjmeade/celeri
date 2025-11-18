#!/usr/bin/env python3

import platform
import re
import subprocess
import warnings

from loguru import logger

import celeri
import celeri.optimize


def is_m4_mac() -> bool:
    """True iff running on macOS and the CPU brand string contains 'M4'."""
    if platform.system() != "Darwin":
        return False
    out = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    )
    return "M4" in out


# One regex to match the three bogus matmul warnings
_MATMUL_MSG = r"(divide by zero|overflow|invalid value) encountered in matmul"


def silence_bogus_matmul_warnings() -> None:
    """Silence NumPy's spurious matmul RuntimeWarnings (see numpy#29820).

    This installs a warnings filter that ignores RuntimeWarnings whose
    message matches the bogus '... encountered in matmul' pattern.
    """
    warnings.filterwarnings(
        action="ignore",
        message=_MATMUL_MSG,
        category=RuntimeWarning,
    )

    # Filter out the dot product warnings from numpy.linalg
    warnings.filterwarnings(
        action="ignore",
        message=r"(divide by zero|overflow|invalid value) encountered in dot",
        category=RuntimeWarning,
    )


@logger.catch(reraise=True)
def main():
    # HACK: Silence rogue M4 numpy warnings
    if is_m4_mac():
        silence_bogus_matmul_warnings()

    # Process arguments
    args = celeri.parse_args()

    # Read in command file and start logging
    config = celeri.get_config(args.config_file_name)
    logger = celeri.get_logger(config)
    celeri.process_args(config, args)
    model = celeri.build_model(config)

    if config.repl:
        import IPython

    if config.solve_type == "dense":
        # Classic dense solve
        logger.info("Dense build and solve")
        estimation = celeri.build_and_solve_dense(model)
    elif config.solve_type == "dense_no_meshes":
        # Classic dense solve with no meshes
        logger.info("Dense build and solve (no meshes)")
        estimation = celeri.build_and_solve_dense_no_meshes(model)
    elif config.solve_type == "qp":
        operators = celeri.build_operators(model, tde=True, eigen=True)
        estimation = celeri.solve_sqp(model, operators)
    elif config.solve_type == "qp2":
        # Bounded solve
        logger.info("Quadratic programming with KL modes")
        estimation = celeri.optimize.solve_sqp2(model)
    elif config.solve_type == "mcmc":
        # MCMC solve
        logger.info("MCMC solve")
        estimation = celeri.solve_mcmc(model)
    else:
        raise ValueError(f"Unknown solve type: {config.solve_type}")

    # Write output
    celeri.write_output(estimation, station=model.station, segment=model.segment, block=model.block, meshes=model.meshes)

    # Drop into ipython REPL
    if config.repl:
        IPython.embed(banner1="")


if __name__ == "__main__":
    main()
