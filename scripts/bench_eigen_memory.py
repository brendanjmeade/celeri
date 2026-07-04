"""Peak-memory benchmark: dense vs batched eigen_to_velocities (issue #485).

Compares the pre-#485 computation (materialize the full (3 * n_stations,
3 * n_tris) TDE matrix, fancy-index copy, one matmul) against the streaming
triangle-chunk accumulation, on the wna test mesh with synthetically inflated
station counts.

Each variant runs in its own subprocess because ru_maxrss is a process-lifetime
high-water mark. The checksums of both variants must agree to ~1e-6 relative.

Usage:
    pixi run python scripts/bench_eigen_memory.py                # both variants
    pixi run python scripts/bench_eigen_memory.py --n-stations 25000
    pixi run python scripts/bench_eigen_memory.py --variant dense
"""

import argparse
import resource
import subprocess
import sys
import time
import tracemalloc

import numpy as np

CONFIG = "tests/configs/test_wna_config.json"


def synthetic_stations(model, n_stations, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(model.station), n_stations)
    station = model.station.iloc[idx].reset_index(drop=True).copy()
    station["lon"] += rng.uniform(-0.05, 0.05, n_stations)
    station["lat"] += rng.uniform(-0.05, 0.05, n_stations)
    return station


def run_variant(variant, n_stations, budget_gb):
    import celeri
    from celeri.celeri_util import get_keep_index_12
    from celeri.operators import (
        _accumulate_eigen_to_velocities_streaming,
        _OperatorBuilder,
        _project_tde_rows_to_eigen,
        _store_eigenvectors_to_tde_slip,
    )
    from celeri.spatial import get_tde_to_velocities_single_mesh

    config = celeri.get_config(CONFIG)
    config.elastic_operator_cache_dir = None
    config.tde_operator_memory_gb = budget_gb
    model = celeri.build_model(config)
    station = synthetic_stations(model, n_stations)

    builder = _OperatorBuilder(model)
    _store_eigenvectors_to_tde_slip(model, builder)
    eigenvectors = builder.eigenvectors_to_tde_slip[0]
    n_tris = model.meshes[0].lon1.size
    dense_gb = (3 * n_stations) * (3 * n_tris) * 8 / 2**30

    start = time.perf_counter()
    tracemalloc.start()
    tracemalloc.reset_peak()
    if variant == "dense":
        tde = get_tde_to_velocities_single_mesh(model.meshes, station, config, 0)
        result = -tde[:, get_keep_index_12(tde.shape[1])] @ eigenvectors
    elif variant == "batched":
        result = _accumulate_eigen_to_velocities_streaming(
            model.meshes,
            station,
            config,
            0,
            eigenvectors,
            target_bytes=int(budget_gb * 2**30),
        )
    elif variant == "rowbatched":
        # The non-streaming/cache-read projection path, fed by a dense matrix
        tde = get_tde_to_velocities_single_mesh(model.meshes, station, config, 0)
        result = _project_tde_rows_to_eigen(tde, eigenvectors, int(budget_gb * 2**30))
    else:
        raise ValueError(variant)
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start

    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        ru_maxrss *= 1024  # linux reports KiB, macOS reports bytes
    print(
        f"{variant:>10s}  n_stations={n_stations}  n_tris={n_tris}  "
        f"dense_equiv={dense_gb:6.2f} GiB  "
        f"tracemalloc_peak={tm_peak / 2**30:6.2f} GiB  "
        f"ru_maxrss={ru_maxrss / 2**30:6.2f} GiB  "
        f"wall={elapsed:7.1f} s  "
        f"checksum={result.sum():.17g}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dense", "batched", "rowbatched"])
    parser.add_argument("--n-stations", type=int, default=25000)
    parser.add_argument("--budget-gb", type=float, default=0.25)
    args = parser.parse_args()

    if args.variant is not None:
        run_variant(args.variant, args.n_stations, args.budget_gb)
        return

    for variant in ("dense", "batched"):
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--variant",
                variant,
                "--n-stations",
                str(args.n_stations),
                "--budget-gb",
                str(args.budget_gb),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
