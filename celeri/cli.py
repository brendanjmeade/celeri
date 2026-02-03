import argparse

from loguru import logger

from celeri.config import Config


def str2bool(v):
    """Convert string to boolean for argparse.

    Accepts: 1, 0, true, false, yes, no (case insensitive)
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file_name", type=str, help="Name of *_config.json file")
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
        type=str2bool,
        default=None,
        required=False,
        help="Flag for dropping into REPL (0 | 1)",
    )
    parser.add_argument(
        "--pickle_save",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for saving major data structures in pickle file (0 | 1)",
    )
    parser.add_argument(
        "--save_operators",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for saving full operator arrays (0 | 1). If 0, operators are "
        "loaded from the elastic operator cache, saving several GBs per run.",
    )
    parser.add_argument(
        "--plot_input_summary",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for saving summary plot of input data (0 | 1)",
    )
    parser.add_argument(
        "--plot_estimation_summary",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for saving summary plot of model results (0 | 1)",
    )
    parser.add_argument(
        "--save_elastic",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for saving elastic calculations (0 | 1)",
    )
    parser.add_argument(
        "--reuse_elastic",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for reusing elastic calculations (0 | 1)",
    )
    parser.add_argument(
        "--snap_segments",
        type=str2bool,
        default=None,
        required=False,
        help="Flag for snapping segments (0 | 1)",
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
    parser.add_argument(
        "--mcmc-tune",
        type=int,
        default=None,
        required=False,
        help="Number of MCMC tuning iterations",
    )
    parser.add_argument(
        "--mcmc-draws",
        type=int,
        default=None,
        required=False,
        help="Number of MCMC samples to draw",
    )
    parser.add_argument(
        "--mcmc-seed",
        type=int,
        default=None,
        required=False,
        help="Random seed for MCMC sampling",
    )
    parser.add_argument(
        "--mcmc-chains",
        type=int,
        default=None,
        required=False,
        help="Number of parallel MCMC chains to run",
    )
    parser.add_argument(
        "--mcmc-backend",
        type=str,
        default=None,
        required=False,
        choices=["numba", "jax"],
        help="Backend to use for MCMC computations (numba | jax)",
    )
    parser.add_argument(
        "--mcmc-station-velocity-method",
        type=str,
        default=None,
        required=False,
        choices=["direct", "low_rank", "project_to_eigen"],
        help="Method for computing station velocities in MCMC",
    )
    parser.add_argument(
        "--mcmc-station-weighting",
        type=str,
        default=None,
        required=False,
        help="Station weighting method (voronoi | none)",
    )
    parser.add_argument(
        "--mcmc-station-effective-area",
        type=float,
        default=None,
        required=False,
        help="Effective area (m²) for station likelihood weighting",
    )
    parser.add_argument(
        "--mesh-default-eigenvector-algorithm",
        type=str,
        default=None,
        required=False,
        choices=["eigh", "eigsh"],
        help="Algorithm for mesh eigendecomposition (eigh | eigsh)",
    )

    return parser.parse_args()


def process_args(config: Config, args: argparse.Namespace):
    for key in Config.model_fields:
        if key in args:
            args_val = getattr(args, key)
            if args_val is not None:
                # Handle "none"/"None" string -> None for nullable fields
                if isinstance(args_val, str) and args_val.lower() == "none":
                    args_val = None
                original_val = getattr(config, key)

                # Handle boolean comparison - config may have 1/0 while args has True/False
                if isinstance(args_val, bool) and isinstance(original_val, int | float):
                    original_val = bool(original_val)

                # Only log if the value is actually being changed
                if original_val != args_val:
                    logger.warning(f"ORIGINAL: config.{key}: {original_val}")
                    setattr(config, key, args_val)
                    logger.warning(f"REPLACED: config.{key}: {args_val}")
