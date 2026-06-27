#!/usr/bin/env python3
"""Verify an installed ``celeri`` wheel end to end.

``hynek/build-and-inspect-python-package`` builds and inspects the dists but never
installs them, so a malformed entry point (rejected by pip only at install time),
an entry point whose target callable does not exist, or an installable-but-
insufficient dependency set can otherwise ship to PyPI (this is how 0.0.5 shipped
un-installable). The release pipeline runs this against the freshly built wheel,
with its full dependency set installed, as the gate before publishing.

It does two things:

1. Loads every ``celeri`` ``console_scripts`` entry point via the standard
   ``importlib.metadata`` machinery. ``EntryPoint.load()`` imports the target
   module and resolves the attribute, so a missing module or missing ``main``
   raises here -- no AST guesswork.
2. Runs a small real end-to-end MCMC solve (``build_model`` + ``solve_mcmc``),
   which exercises cutde, pymc, pytensor, nutpie, jax, numba, h5py, meshio, scipy
   and the rest of the stack, proving the declared dependencies are sufficient.

Run it against the *installed* package, from a directory that is not the repo root
so the source tree can't shadow the wheel on import::

    python scripts/verify_wheel.py --data-root /path/to/celeri/checkout

In CI ``--data-root`` is the checked-out workspace (``$GITHUB_WORKSPACE``); the
example model data lives in ``data/`` there and is intentionally excluded from the
dists.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import distribution
from pathlib import Path


def check_entry_points() -> None:
    """Load every ``celeri`` console-script entry point, or raise."""
    entry_points = [
        ep
        for ep in distribution("celeri").entry_points
        if ep.group == "console_scripts"
    ]
    if not entry_points:
        raise SystemExit("No celeri console_scripts entry points found")

    print(f"Resolving {len(entry_points)} console_scripts entry points")
    for ep in sorted(entry_points, key=lambda e: e.name):
        ep.load()  # imports the module and resolves the attribute; raises if broken
        print(f"  OK  {ep.name} -> {ep.value}")
    print(f"All {len(entry_points)} entry points resolve to a defined callable")


def run_solve(config_path: Path) -> None:
    """Run a tiny end-to-end MCMC solve against the example data."""
    import celeri

    print(f"Running end-to-end MCMC solve with {config_path}")
    model = celeri.build_model(celeri.get_config(str(config_path)))
    estimation = celeri.solve_mcmc(
        model, sample_kwargs={"tune": 10, "draws": 10, "chains": 2}
    )
    posterior = estimation.mcmc_trace.posterior
    assert len(posterior.coords["draw"]) == 10, posterior.coords["draw"]
    print("MCMC solve OK; posterior sizes:", dict(posterior.sizes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("GITHUB_WORKSPACE"),
        help="celeri checkout holding data/ (default: $GITHUB_WORKSPACE)",
    )
    args = parser.parse_args()

    check_entry_points()

    if not args.data_root:
        raise SystemExit("--data-root (or $GITHUB_WORKSPACE) is required for the solve")
    config_path = Path(args.data_root) / "data" / "config" / "wna_config.json"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    run_solve(config_path)


if __name__ == "__main__":
    main()
