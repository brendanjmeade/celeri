"""Peak-memory benchmark: dense vs streamed Okada composition.

Compares the pre-refactor computation (materialize the full
(3 * n_stations, 3 * n_segments) okada operator, then one matmul against
rotation_to_slip_rate) with the streamed paths: segment-chunk accumulation
(no cache) and row-slab streaming from an HDF5 cache dataset.

Each variant runs in its own subprocess because ru_maxrss is a
process-lifetime high-water mark. Checksums must agree to ~1e-6 relative.

Usage:
    pixi run python scripts/bench_okada_memory.py                 # all variants
    pixi run python scripts/bench_okada_memory.py --n-stations 25000
    pixi run python scripts/bench_okada_memory.py --variant dense
"""

import argparse
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

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
    import h5py

    import celeri
    from celeri.operators import (
        _compute_okada_composed_and_cache,
        _stream_matmul_rows,
    )
    from celeri.spatial import (
        get_rotation_to_slip_rate_partials,
        get_segment_station_operator_okada,
    )

    config = celeri.get_config(CONFIG)
    config.elastic_operator_cache_dir = None
    config.tde_operator_memory_gb = budget_gb
    model = celeri.build_model(config)
    station = synthetic_stations(model, n_stations)
    rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        model.segment, model.block
    )
    n_segments = len(model.segment)
    dense_gb = (3 * n_stations) * (3 * n_segments) * 8 / 2**30
    target_bytes = int(budget_gb * 2**30)

    start = time.perf_counter()
    tracemalloc.start()
    tracemalloc.reset_peak()
    if variant == "dense":
        okada = get_segment_station_operator_okada(model.segment, station, config)
        result = okada @ rotation_to_slip_rate
    elif variant == "streamed":
        result = _compute_okada_composed_and_cache(
            model.segment, station, config, rotation_to_slip_rate, None, target_bytes
        )
    elif variant == "cachestream":
        # Dense dataset written outside the measured window (setup), then the
        # row-slab streaming path is measured
        tracemalloc.stop()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "okada.hdf5"
            okada = get_segment_station_operator_okada(model.segment, station, config)
            with h5py.File(str(cache), "w") as hdf5_file:
                hdf5_file.create_dataset("slip_rate_to_okada_to_velocities", data=okada)
            del okada
            start = time.perf_counter()
            tracemalloc.start()
            tracemalloc.reset_peak()
            with h5py.File(str(cache), "r") as hdf5_file:
                result = _stream_matmul_rows(
                    hdf5_file["slip_rate_to_okada_to_velocities"],
                    rotation_to_slip_rate,
                    target_bytes,
                )
    else:
        raise ValueError(variant)
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start

    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        ru_maxrss *= 1024  # linux reports KiB, macOS reports bytes
    print(
        f"{variant:>12s}  n_stations={n_stations}  n_segments={n_segments}  "
        f"dense_equiv={dense_gb:6.2f} GiB  "
        f"tracemalloc_peak={tm_peak / 2**30:6.2f} GiB  "
        f"ru_maxrss={ru_maxrss / 2**30:6.2f} GiB  "
        f"wall={elapsed:7.1f} s  "
        # abs-sum: the plain sum of the composed operator nearly cancels, so
        # it cannot serve as a cross-variant comparator
        f"checksum={np.abs(result).sum():.17g}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["dense", "streamed", "cachestream"])
    parser.add_argument("--n-stations", type=int, default=25000)
    parser.add_argument("--budget-gb", type=float, default=0.25)
    args = parser.parse_args()

    if args.variant is not None:
        run_variant(args.variant, args.n_stations, args.budget_gb)
        return

    for variant in ("dense", "streamed", "cachestream"):
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
