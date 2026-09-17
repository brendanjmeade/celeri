#!/usr/bin/env python3
"""Run the MCMC solve repeatedly over a range of ``sigmoid_slope`` values.

``sigmoid_slope`` controls how sharply the latent coupling field on each mesh
is squashed into its bounds (see ``Config.mcmc_default_mesh_sigmoid_slope``).
This script runs one full ``celeri-solve``-style MCMC run per value so the
sensitivity of the solution to that choice can be compared across the
regularly numbered run folders it produces.

Usage::

    celeri-sweep-sigmoid-slope <config.json> <lower> <upper> [n_steps=3] [--log]
                               [any celeri-solve flag, e.g. --mcmc-tune 5]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from loguru import logger

import celeri
from celeri.cli import build_parser


def sweep_values(lower: float, upper: float, n_steps: int, log: bool) -> np.ndarray:
    """Return the ``sigmoid_slope`` grid for the sweep.

    Linear spacing by default, geometric with ``log``. A single step uses
    ``lower`` only.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if lower <= 0 or upper <= 0:
        raise ValueError(
            f"sigmoid_slope values must be positive, got lower={lower}, upper={upper}"
        )
    if n_steps == 1:
        return np.array([float(lower)])
    if log:
        return np.geomspace(lower, upper, n_steps)
    return np.linspace(lower, upper, n_steps)


def parse_sweep_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    parser.description = (
        "Run the MCMC solve for several sigmoid_slope values, one numbered "
        "run folder per value. Accepts every celeri-solve flag."
    )
    parser.add_argument("lower", type=float, help="Lowest sigmoid_slope value")
    parser.add_argument("upper", type=float, help="Highest sigmoid_slope value")
    parser.add_argument(
        "n_steps",
        type=int,
        nargs="?",
        default=3,
        help="Number of sigmoid_slope values, inclusive of both ends (default 3)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Space the values geometrically instead of linearly",
    )
    return parser.parse_args(argv)


def prepare_config(
    args: argparse.Namespace, value: float, index: int, n_total: int
) -> celeri.Config:
    """Load the config for one sweep run and set ``sigmoid_slope = value``.

    ``get_config`` allocates the next numbered run folder, so each call here
    lands in its own directory.
    """
    config = celeri.get_config(args.config_file_name)
    celeri.get_logger(config)
    celeri.process_args(config, args)

    if config.solve_type != "mcmc":
        logger.warning(
            f"solve_type is '{config.solve_type}'; the sigmoid_slope sweep "
            "only applies to MCMC, forcing solve_type = 'mcmc'"
        )
        config.solve_type = "mcmc"

    # Set the swept value on the global default and on every mesh. The
    # per-mesh assignment is required: propagate_mesh_defaults only fills
    # entries that are None, so a mesh-params file that already sets
    # sigmoid_slope would otherwise ignore the sweep.
    config.mcmc_default_mesh_sigmoid_slope = value
    for mesh_param in config.mesh_params:
        mesh_param.sigmoid_slope = value

    tag = f"sigmoid_slope sweep {index + 1}/{n_total}: sigmoid_slope={value:g}"
    config.description = f"{config.description} | {tag}" if config.description else tag
    logger.info(tag)
    logger.info(f"Run folder: {config.output_path}")
    return config


def run_one(config: celeri.Config) -> dict:
    """Run a single MCMC solve, mirroring the MCMC path of celeri-solve."""
    start = time.perf_counter()
    model = celeri.build_model(config)
    estimation = celeri.solve_mcmc(model)
    celeri.write_output(estimation)
    if config.plot_estimation_summary:
        celeri.plot_estimation_summary(estimation)
    elapsed = time.perf_counter() - start

    return {
        "sigmoid_slope": float(config.mcmc_default_mesh_sigmoid_slope),
        "run_name": config.run_name,
        "output_path": str(config.output_path),
        "status": "ok",
        "n_divergences": estimation.mcmc_num_divergences,
        "wall_time_s": elapsed,
    }


def _base_runs_folder(config_file: Path) -> Path:
    """Resolve ``base_runs_folder`` relative to the config file (as get_config)."""
    config_data = json.loads(config_file.read_text())
    base = (config_file.parent / Path(config_data["base_runs_folder"])).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_manifest(entries: list[dict], base_runs_folder: Path) -> Path:
    names = [e["run_name"] for e in entries if e.get("run_name")]
    span = f"{names[0]}-{names[-1]}" if names else "empty"
    manifest_path = base_runs_folder / f"sweep_sigmoid_slope_{span}.json"
    manifest_path.write_text(json.dumps(entries, indent=2))
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_sweep_args(argv)
    values = sweep_values(args.lower, args.upper, args.n_steps, args.log)
    logger.info(
        f"sigmoid_slope sweep over {len(values)} value(s): "
        + ", ".join(f"{v:g}" for v in values)
    )

    entries: list[dict] = []
    base_runs_folder: Path | None = None
    for index, value in enumerate(values):
        config = None
        try:
            config = prepare_config(args, float(value), index, len(values))
            entry = run_one(config)
        except Exception:
            logger.error(
                f"Run for sigmoid_slope={value:g} failed:\n{traceback.format_exc()}"
            )
            entry = {
                "sigmoid_slope": float(value),
                "run_name": config.run_name if config is not None else None,
                "output_path": (
                    str(config.output_path) if config is not None else None
                ),
                "status": "failed",
                "n_divergences": None,
                "wall_time_s": None,
            }
        entries.append(entry)
        if base_runs_folder is None and entry["output_path"] is not None:
            base_runs_folder = Path(entry["output_path"]).parent
        logger.info(
            f"sigmoid_slope={value:g} -> {entry['output_path']} ({entry['status']})"
        )

    if base_runs_folder is None:
        # Nothing ran far enough to allocate a folder; resolve the base runs
        # folder the way get_config does, without creating a numbered run.
        base_runs_folder = _base_runs_folder(Path(args.config_file_name))
    manifest_path = write_manifest(entries, base_runs_folder)

    logger.info("sigmoid_slope sweep summary:")
    for entry in entries:
        logger.info(
            f"  sigmoid_slope={entry['sigmoid_slope']:<8g} "
            f"run={entry['run_name']} status={entry['status']} "
            f"divergences={entry['n_divergences']}"
        )
    logger.info(f"Manifest: {manifest_path}")

    n_failed = sum(entry["status"] == "failed" for entry in entries)
    if n_failed:
        logger.error(f"{n_failed} of {len(entries)} sweep run(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
