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
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


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

    return parser.parse_args()


def process_args(config: Config, args: argparse.Namespace):
    for key in Config.model_fields:
        if key in args:
            args_val = getattr(args, key)
            if args_val is not None:
                original_val = getattr(config, key)
                
                # Handle boolean comparison - config may have 1/0 while args has True/False
                if isinstance(args_val, bool) and isinstance(original_val, (int, float)):
                    original_val = bool(original_val)
                
                # Only log if the value is actually being changed
                if original_val != args_val:
                    logger.warning(f"ORIGINAL: config.{key}: {original_val}")
                    setattr(config, key, args_val)
                    logger.warning(f"REPLACED: config.{key}: {args_val}")
