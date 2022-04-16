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
    celeri.get_logger(command)
    celeri.process_args(command, args)

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
