#!/usr/bin/env python3
r"""Fit a near-field station covariance to a finished run's velocity residuals.

Writes the JSON file that ``"mcmc_station_weighting": "covariance"`` reads
(``mcmc_station_covariance_file_name``); see docs/station-covariance.md.

For each velocity component, the correlation of the selected stations'
residuals is modelled as a nugget plus a sum of Matérn terms of fixed
smoothness, tapered to zero at the taper radius R:

    rho(h) = nugget * [h == 0]
             + wendland(h; R) * sum_k weight_k * matern(h; length_k, smoothness)

A least-squares fit to the binned residual correlation (the correlogram) is
refined by maximizing the Vecchia Gaussian likelihood of the residuals under
the tapered (and, by default, same-block) model, which is the approximation
the MCMC likelihood uses (``--method correlogram`` keeps the least-squares
fit). The overall scale is profiled out, since the MCMC samples it.

The fit is checked by whitening the same residuals with the fitted
covariance: within R the whitened residuals should be nearly uncorrelated.
The file also records how many independent stations the correlated set is
worth for a common offset and how the approximation converges with the number
of neighbours. A plot of the fit is written next to the file (same name,
.png) unless ``--no-plot`` is given.

Use the same taper radius and same-block setting here as in the MCMC config
(``mcmc_station_covariance_taper_radius``,
``mcmc_station_covariance_same_block``); celeri warns when they differ.

Examples
--------
    # residuals of an earlier (unweighted) MCMC run, correlating only the
    # stations named xu_2021 (a gridded velocity field)
    celeri-fit-station-covariance runs/0000000005 data/covariance/wna_covariance.json \
        --station-names xu_2021

    # correlate every station, 5 km taper, pairs across faults allowed
    celeri-fit-station-covariance runs/0000000005 covariance.json \
        --taper-radius 5 --no-same-block
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from scipy.special import expit, logit

from celeri import station_covariance as sc

# Correlogram bin edges (km): fine near zero, where the nugget is resolved
FINE_EDGES = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
COARSE_EDGES = [12.5, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150]
EDGES = np.array([*FINE_EDGES, *COARSE_EDGES])
# Bins with fewer pairs are not fit
MIN_PAIRS = 200
# Partner points for the correlogram (all points when there are fewer)
MAX_PARTNERS = 200_000
# Smallest nugget the likelihood fit may reach: keeps the kriging systems of
# nearly coincident stations well conditioned
MIN_NUGGET = 1e-3
# Range of the likelihood fit's correlation lengths (km). Within the taper
# radius, a length far below the station spacing acts as extra nugget and one
# far above R as a constant; the likelihood is flat beyond both, so without
# bounds the lengths drift to 0 or infinity.
LENGTH_BOUNDS_KM = (0.01, 1000.0)
DEFAULT_TAPER_RADIUS_KM = 3.0


def correlogram(xyz_km, values, n_anchors, rng, groups=None):
    """Binned correlation of each column of values with distance.

    Pairs are formed between n_anchors random points and every point (or
    MAX_PARTNERS random points) within EDGES[-1]; with groups, only pairs in
    the same group. Returns bin centres, the correlation (bins x columns) and
    the number of pairs per bin.
    """
    n = len(xyz_km)
    values = values - values.mean(axis=0)
    var = (values**2).mean(axis=0)
    anchors = rng.choice(n, size=min(n_anchors, n), replace=False)
    partners = rng.choice(n, size=min(MAX_PARTNERS, n), replace=False)
    tree = cKDTree(xyz_km[partners])
    n_bins = len(EDGES) - 1
    sums = np.zeros((n_bins, values.shape[1]))
    counts = np.zeros(n_bins)
    for i in anchors:
        idx = partners[tree.query_ball_point(xyz_km[i], EDGES[-1])]
        idx = idx[idx != i]
        if groups is not None:
            idx = idx[groups[idx] == groups[i]]
        d = np.linalg.norm(xyz_km[idx] - xyz_km[i], axis=1)
        k = np.searchsorted(EDGES, d, side="right") - 1
        counts += np.bincount(k, minlength=n_bins)
        for c in range(values.shape[1]):
            sums[:, c] += np.bincount(
                k, weights=values[i, c] * values[idx, c], minlength=n_bins
            )
    corr = sums / np.maximum(counts, 1)[:, None] / var
    mid = 0.5 * (EDGES[:-1] + EDGES[1:])
    return mid, corr, counts


def component(theta, smoothness):
    """Correlation parameters from theta = (nugget, splits..., lengths...)."""
    n_terms = (len(theta) + 1) // 2
    nugget = theta[0]
    splits, lengths = theta[1:n_terms], theta[n_terms:]
    # Stick-breaking: each split takes a fraction of the remaining weight
    weights, remaining = [], 1.0 - nugget
    for s in splits:
        weights.append(remaining * s)
        remaining -= weights[-1]
    weights.append(remaining)
    terms = sorted(
        (
            {"weight": float(w), "length_km": float(ell), "smoothness": smoothness}
            for w, ell in zip(weights, lengths, strict=True)
        ),
        key=lambda term: term["length_km"],
    )
    return {"nugget": float(nugget), "terms": terms}


def fit_component(mid, corr, counts, n_terms, smoothness, max_km):
    """Least-squares fit of nugget + n_terms Matérn terms to one correlogram."""
    use = (counts >= MIN_PAIRS) & (mid <= max_km) & (mid > 0)
    if use.sum() < 2 * n_terms:
        raise ValueError(
            f"Only {use.sum()} correlogram bins have {MIN_PAIRS}+ pairs: too few "
            f"stations to fit {n_terms} terms"
        )

    def misfit(theta):
        return sc.correlated_part(mid[use], component(theta, smoothness)) - corr[use]

    lower = [MIN_NUGGET] + [0.0] * (n_terms - 1) + [0.1] * n_terms
    upper = [0.999] + [1.0] * (n_terms - 1) + [500.0] * n_terms
    best = None
    for short in (1.0, 3.0, 8.0):
        for ratio in (3.0, 8.0):
            lengths = short * ratio ** np.arange(n_terms)
            start = [0.15] + [0.5] * (n_terms - 1) + list(np.minimum(lengths, 400.0))
            result = least_squares(misfit, start, bounds=(lower, upper))
            if best is None or result.cost < best.cost:
                best = result
    rms = float(np.sqrt(np.mean(best.fun**2)))
    return component(best.x, smoothness), rms


def profile_log_likelihood(xyz_km, neighbors, used, values, comp, taper_radius):
    """Vecchia Gaussian log likelihood of one residual column, scale profiled out.

    Also returns the ratio of the whitened residuals' sd to the fitted scale.
    """
    factor = sc.vecchia_kriging(xyz_km, neighbors, used, [comp], taper_radius)
    e = factor.whiten(values[:, None])[:, 0]
    mean_square = np.mean(e**2)
    log_likelihood = -0.5 * len(e) * np.log(mean_square) + factor.log_det()[0]
    return log_likelihood, np.sqrt(mean_square)


def mle_component(xyz_km, neighbors, used, values, start, smoothness, taper_radius):
    """Maximum-likelihood correlation parameters, starting from start."""
    terms = start["terms"]
    # Unconstrained coordinates: logit nugget, logit stick-breaking splits, log lengths
    splits, remaining = [], 1.0 - start["nugget"]
    for term in terms[:-1]:
        splits.append(term["weight"] / remaining)
        remaining -= term["weight"]
    nugget = max(start["nugget"], 2 * MIN_NUGGET)
    x0 = np.concatenate(
        [
            [logit((nugget - MIN_NUGGET) / (1 - MIN_NUGGET))],
            logit(np.clip(splits, 1e-3, 1 - 1e-3)),
            np.log([term["length_km"] for term in terms]),
        ]
    )
    n_terms = len(terms)

    def unpack(x):
        nugget = MIN_NUGGET + (1 - MIN_NUGGET) * expit(x[0])
        lengths = np.clip(np.exp(x[n_terms:]), *LENGTH_BOUNDS_KM)
        return component(
            np.concatenate([[nugget], expit(x[1:n_terms]), lengths]),
            smoothness,
        )

    def objective(x):
        try:
            return -profile_log_likelihood(
                xyz_km, neighbors, used, values, unpack(x), taper_radius
            )[0]
        except ValueError:  # non-positive kriging variance
            return np.inf

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 0.5, "maxfev": 600},
    )
    return unpack(result.x), int(result.nfev)


def describe(fit):
    terms = ", ".join(
        f"{t['weight']:.3f} x {t['length_km']:.2f} km" for t in fit["terms"]
    )
    return f"nugget {fit['nugget']:.3f}; {terms}"


def plot(path, names, mid, corr, counts, white_corr, fits, info, taper_radius, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(names), figsize=(6 * len(names), 4.5), squeeze=False
    )
    h = np.geomspace(0.05, EDGES[-1], 400)
    shown = counts >= MIN_PAIRS
    for c, (ax, name) in enumerate(zip(axes[0], names, strict=True)):
        fit = fits[name]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(taper_radius, color="0.6", lw=0.8, ls=":")
        ax.plot(mid[shown], corr[shown, c], "o", color="C0", label="residuals")
        ax.plot(h, sc.correlated_part(h, fit), "--", color="C1", lw=0.8, label="fit")
        ax.plot(
            h,
            sc.correlated_part(h, fit, taper_radius),
            "-",
            color="C1",
            label=f"fit, tapered at {taper_radius:g} km",
        )
        ax.plot(
            mid[shown],
            white_corr[shown, c],
            "x",
            color="C2",
            label="whitened residuals",
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.2, 1.0)
        ax.set_xlabel("distance (km)")
        ax.set_ylabel("correlation")
        ax.set_title(
            f"{name}: {describe(fit)}\n~{info[c]:.0f} independent stations "
            "for a common offset",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="celeri-fit-station-covariance",
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 3)[3],
    )
    parser.add_argument("run_dir", help="finished run directory (model_station.csv)")
    parser.add_argument("output", help="station covariance JSON file to write")
    parser.add_argument(
        "--station-names",
        nargs="+",
        default=None,
        help="names of the stations to correlate (default: every station)",
    )
    parser.add_argument(
        "--taper-radius",
        type=float,
        default=DEFAULT_TAPER_RADIUS_KM,
        help=f"taper radius R in km (default {DEFAULT_TAPER_RADIUS_KM:g})",
    )
    parser.add_argument(
        "--same-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="correlate only stations on the same block (default: yes)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="most earlier neighbours each station is conditioned on (default 30)",
    )
    parser.add_argument(
        "--n-terms", type=int, default=2, help="Matérn terms per component (default 2)"
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.5,
        help="Matérn smoothness (default 0.5: exponential)",
    )
    parser.add_argument(
        "--method",
        choices=["mle", "correlogram"],
        default="mle",
        help="mle (default): least-squares correlogram fit refined by Vecchia "
        "maximum likelihood; correlogram: the least-squares fit only",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=100.0,
        help="largest distance in the least-squares fit (default 100)",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        default=5000,
        help="correlogram anchor stations (default 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="station ordering seed (default 0)"
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="don't check the Vecchia approximation against more neighbours",
    )
    parser.add_argument("--no-plot", action="store_true", help="don't write the plot")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    args = parser.parse_args(argv)
    if not args.taper_radius > 0:
        parser.error("--taper-radius must be positive")
    if args.n_neighbors < 1 or args.n_terms < 1:
        parser.error("--n-neighbors and --n-terms must be at least 1")
    return parser, args


def main(argv=None):
    parser, args = _parse_args(argv)
    run_dir, output = Path(args.run_dir), Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists (pass --overwrite to replace it)")
    station_file = run_dir / "model_station.csv"
    if not station_file.exists():
        parser.error(f"{station_file} not found: is {run_dir} a finished run?")
    run_config_file = run_dir / "config.json"
    run_config = (
        json.loads(run_config_file.read_text()) if run_config_file.exists() else {}
    )
    names = ["east", "north"] + (
        ["up"] if run_config.get("include_vertical_velocity") else []
    )
    residual_columns = [f"model_{c}_vel_residual" for c in names]
    station = pd.read_csv(station_file)
    missing = [
        c
        for c in ["name", "x", "y", "z", "block_label", *residual_columns]
        if c not in station
    ]
    if missing:
        parser.error(f"{station_file} has no {', '.join(missing)} column")
    covariance_stub = {"station_names": args.station_names}
    rows = sc.correlated_rows(station, covariance_stub)
    if rows.size < 10:
        parser.error(
            f"only {rows.size} stations named {args.station_names} in {station_file}"
        )
    station = station.iloc[rows]
    xyz_km = station[["x", "y", "z"]].to_numpy() / 1e3
    groups = station.block_label.to_numpy() if args.same_block else None
    residual = station[residual_columns].to_numpy(dtype=float)
    n = len(station)
    taper = args.taper_radius
    rng = np.random.default_rng(args.seed)
    print(
        f"{run_dir}: {n} stations ({args.station_names or 'all'}); taper radius "
        f"{taper:g} km, same block only {args.same_block}"
    )

    mid, corr, counts = correlogram(xyz_km, residual, args.anchors, rng, groups)
    centered = residual - residual.mean(axis=0)
    sd = centered.std(axis=0)
    neighbors, used = sc.vecchia_neighbors(
        xyz_km, args.n_neighbors, args.seed, taper, groups
    )
    fits, details = {}, {}
    for c, name in enumerate(names):
        fit, rms = fit_component(
            mid, corr[:, c], counts, args.n_terms, args.smoothness, args.max_distance_km
        )
        standardized = centered[:, c] / sd[c]
        ll, ratio = profile_log_likelihood(
            xyz_km, neighbors, used, standardized, fit, taper
        )
        details[name] = {
            "correlogram_fit": fit,
            "correlogram_rms_misfit": round(rms, 4),
            "correlogram_fit_log_likelihood": round(float(ll), 1),
        }
        print(
            f"{name}, correlogram fit: {describe(fit)}; rms misfit {rms:.3f}; "
            f"profile log likelihood {ll:.1f}"
        )
        if args.method == "mle":
            fit, n_eval = mle_component(
                xyz_km, neighbors, used, standardized, fit, args.smoothness, taper
            )
            mle_ll, ratio = profile_log_likelihood(
                xyz_km, neighbors, used, standardized, fit, taper
            )
            details[name]["mle_log_likelihood"] = round(float(mle_ll), 1)
            print(
                f"{name}, maximum likelihood: {describe(fit)}; profile log "
                f"likelihood {mle_ll:.1f} ({mle_ll - ll:+.1f}, {n_eval} evaluations)"
            )
            low, high = LENGTH_BOUNDS_KM
            for term in fit["terms"]:
                if np.isclose(term["length_km"], low):
                    print(
                        f"  the {term['weight']:.3f} term's length is at the fit's "
                        f"lower bound ({low:g} km): it acts as extra nugget"
                    )
                elif np.isclose(term["length_km"], high):
                    print(
                        f"  the {term['weight']:.3f} term's length is at the fit's "
                        f"upper bound ({high:g} km): within {taper:g} km it is "
                        "constant, so its correlation follows the taper's shape"
                    )
        details[name]["whitened_sd_over_fitted_scale"] = round(float(ratio), 4)
        fits[name] = fit

    # Check: within the taper radius the whitened residuals should be uncorrelated
    components = [fits[name] for name in names]
    factor = sc.vecchia_kriging(xyz_km, neighbors, used, components, taper)
    white = factor.whiten(centered / sd)
    _, white_corr, _ = correlogram(
        xyz_km, white, args.anchors, np.random.default_rng(args.seed + 1), groups
    )
    info = sc.mean_information(factor, np.arange(n), len(names))
    inside = (counts >= MIN_PAIRS) & (mid < taper)
    print(
        f"whitened residual correlation within {taper:g} km: "
        f"{np.round(white_corr[inside], 3).tolist()}"
    )
    print(
        "independent stations for a common offset: "
        f"{np.round(info).astype(int).tolist()} of {n}"
    )

    convergence = None
    if not args.skip_convergence:
        convergence = {}
        for m in sorted({10, 20, args.n_neighbors, 2 * args.n_neighbors}):
            f = sc.vecchia_factor(xyz_km, components, m, args.seed, taper, groups)
            e = f.whiten(centered / sd)
            value = -0.5 * (e**2).sum(axis=0) + f.log_det()
            convergence[str(m)] = np.round(value, 2).tolist()
            print(
                f"Vecchia Gaussian log density with {m} neighbours: "
                f"{np.round(value, 1).tolist()}"
            )

    covariance = {
        "description": (
            "Near-field correlation of station velocity residuals for celeri's "
            '"covariance" station weighting: per component, rho(h) = nugget * '
            "[h == 0] + wendland(h; R) * sum_k weight_k * matern(h; length_km_k, "
            "smoothness_k), h the 3-D distance between stations in km and R "
            "mcmc_station_covariance_taper_radius"
        ),
        "station_names": args.station_names,
        "n_neighbors": args.n_neighbors,
        "ordering_seed": args.seed,
        "components": fits,
        "fit": {
            "method": args.method,
            "taper_radius_km": taper,
            "same_block": args.same_block,
            "residual_sd_mm_per_yr": dict(
                zip(names, np.round(sd, 4).tolist(), strict=True)
            ),
            "east_north_correlation": float(
                np.corrcoef(centered[:, 0], centered[:, 1])[0, 1]
            ),
            "components": details,
            "independent_stations_common_offset": dict(
                zip(names, np.round(info, 1).tolist(), strict=True)
            ),
            "vecchia_log_density_by_neighbors": convergence,
            "correlogram": {
                "distance_km": np.round(mid, 4).tolist(),
                "pairs": counts.astype(int).tolist(),
                **{
                    name: np.round(corr[:, c], 4).tolist()
                    for c, name in enumerate(names)
                },
                **{
                    f"{name}_whitened": np.round(white_corr[:, c], 4).tolist()
                    for c, name in enumerate(names)
                },
            },
        },
        "source": {
            "run_dir": str(run_dir.resolve()),
            "station_file_name": run_config.get("station_file_name"),
            "n_stations": n,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "script": "celeri-fit-station-covariance",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(covariance, indent=4) + "\n")
    sc.load_station_covariance(output)
    written = [str(output)]
    if not args.no_plot:
        figure = output.with_suffix(".png")
        plot(
            figure,
            names,
            mid,
            corr,
            counts,
            white_corr,
            fits,
            info,
            taper,
            f"Residual correlation, {run_dir}",
        )
        written.append(str(figure))
    print(f"Wrote {' and '.join(written)}")


if __name__ == "__main__":
    main()
