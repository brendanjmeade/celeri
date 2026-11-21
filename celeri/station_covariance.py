"""Near-field correlated noise for station velocities.

Densely sampled velocity fields (for example InSAR-derived grids subsampled
to 10^5-10^6 points) have residuals that are correlated over kilometres, so
an independent likelihood counts each correlated patch many times over and
its weight grows with the number of points sampled. This module whitens the
residuals with a sparse near-field covariance instead.

For each velocity component the correlation between two stations a distance
h (km) apart is

    rho(h) = nugget * [h == 0]
             + taper(h; R) * sum_k weight_k * matern(h; length_k, smoothness_k)

with nugget + sum(weight_k) = 1. The Wendland taper falls to zero at the
radius R (``mcmc_station_covariance_taper_radius``), so stations farther
apart than R are independent: the covariance only discounts redundancy among
nearby stations and leaves the long-wavelength velocity field anchored by
every independent patch. With ``mcmc_station_covariance_same_block``,
stations on different blocks (opposite sides of a fault) are independent as
well, so a step in velocity across a fault is fit station by station.

The likelihood uses Vecchia's nearest-neighbour approximation. Stations are
put in a fixed random order, and each station's residual is predicted from
the residuals of its nearest earlier stations within R (at most
``n_neighbors``). Only the standardized prediction error enters the
likelihood:

    e_i = (r_i - sum_j b_ij r_j) / sqrt(d_i)

where b_i and d_i are the kriging weights and the kriging variance (as a
fraction of the station's variance). Stations without a covariance (for
example GNSS stations, when only the grid is correlated) keep independent
residuals, e_i = r_i.

The correlation parameters, the stations they apply to, the neighbour count
and the ordering seed come from a JSON file written by
``celeri-fit-station-covariance``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import gamma, kv

COMPONENTS = ("east", "north", "up")

# Matrix entries (rows x neighbours^2) of the batched kriging solves held in
# memory at once
CHUNK_ENTRIES = 20_000_000


@dataclass
class VecchiaFactor:
    """Whitening of residuals r (n, k) at n stations.

    ``e = (r - (coef * r[neighbors]).sum(axis=1)) * inv_sd``

    Attributes:
        neighbors: (n, m) row indices into r; a row's own index in unused slots.
        coef: (n, m, k) kriging weights; 0 in unused slots and for independent
            stations.
        inv_sd: (n, k) inverse kriging standard deviations; 1 for independent
            stations.
    """

    neighbors: np.ndarray
    coef: np.ndarray
    inv_sd: np.ndarray

    def whiten(self, r: np.ndarray) -> np.ndarray:
        """Whitened residuals for r of shape (n, k) or (n, k, p)."""
        r = np.asarray(r, dtype=float)
        prediction = np.einsum("imk,imk...->ik...", self.coef, r[self.neighbors])
        inv_sd = self.inv_sd.reshape(self.inv_sd.shape + (1,) * (r.ndim - 2))
        return (r - prediction) * inv_sd

    def log_det(self) -> np.ndarray:
        """Log absolute determinant of the whitening map, per component (k,)."""
        return np.log(self.inv_sd).sum(axis=0)


def matern(h, length: float, smoothness: float) -> np.ndarray:
    """Matérn correlation at distances h, in the units of length."""
    x = np.asarray(h, dtype=float) / length
    if smoothness == 0.5:
        return np.exp(-x)
    if smoothness == 1.5:
        s = np.sqrt(3) * x
        return (1 + s) * np.exp(-s)
    if smoothness == 2.5:
        s = np.sqrt(5) * x
        return (1 + s + s**2 / 3) * np.exp(-s)
    s = np.sqrt(2 * smoothness) * np.maximum(x, 1e-12)
    return np.where(
        x > 0,
        2 ** (1 - smoothness) / gamma(smoothness) * s**smoothness * kv(smoothness, s),
        1.0,
    )


def wendland(h, radius: float) -> np.ndarray:
    """Wendland taper (1 - h/R)^4 (4 h/R + 1), zero beyond R.

    Compactly supported and positive definite in three dimensions, so the
    product with any valid correlation function is again valid.
    """
    x = np.minimum(np.asarray(h, dtype=float) / radius, 1.0)
    return (1 - x) ** 4 * (4 * x + 1)


def correlated_part(h_km, component: dict, taper_radius: float | None = None):
    """Correlation at distance h_km between two different stations (no nugget).

    Tapered to zero at taper_radius (km) when one is given.
    """
    rho = sum(
        term["weight"] * matern(h_km, term["length_km"], term["smoothness"])
        for term in component["terms"]
    )
    if taper_radius is not None:
        rho = rho * wendland(h_km, taper_radius)
    return rho


def correlation(h_km, component: dict, taper_radius: float | None = None):
    """Correlation function of one component, including the nugget at h = 0."""
    h_km = np.asarray(h_km, dtype=float)
    return np.where(h_km == 0, 1.0, correlated_part(h_km, component, taper_radius))


def load_station_covariance(path: str | Path) -> dict:
    """Read and check a station covariance file.

    Required keys: ``station_names`` (a list of station names, or null for every
    station), ``n_neighbors``, ``ordering_seed`` and ``components``, which maps
    "east", "north" and optionally "up" to ``{"nugget": float, "terms":
    [{"weight": float, "length_km": float, "smoothness": float}, ...]}``.
    """
    covariance = json.loads(Path(path).read_text())
    for key in ("station_names", "n_neighbors", "ordering_seed", "components"):
        if key not in covariance:
            raise ValueError(f"{path}: missing {key!r}")
    names = covariance["station_names"]
    if names is not None and (
        not isinstance(names, list) or not all(isinstance(n, str) for n in names)
    ):
        raise ValueError(f"{path}: station_names must be a list of names or null")
    if int(covariance["n_neighbors"]) < 1:
        raise ValueError(f"{path}: n_neighbors must be at least 1")
    if not covariance["components"]:
        raise ValueError(f"{path}: no components")
    for name, component in covariance["components"].items():
        if name not in COMPONENTS:
            raise ValueError(f"{path}: unknown component {name!r}")
        terms = component.get("terms", [])
        for term in terms:
            if term["weight"] < 0 or term["length_km"] <= 0 or term["smoothness"] <= 0:
                raise ValueError(
                    f"{path}: {name} terms need weight >= 0, length_km > 0 and "
                    "smoothness > 0"
                )
        total = component["nugget"] + sum(t["weight"] for t in terms)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"{path}: {name} nugget and term weights sum to {total}, not 1"
            )
        if component["nugget"] <= 0:
            raise ValueError(f"{path}: {name} nugget must be positive")
    return covariance


def previous_neighbors(points: np.ndarray, m: int) -> np.ndarray:
    """Each point's m nearest earlier points (by row order), nearest first.

    Returns (n, m) row indices, -1 where a point has fewer than m earlier
    points. Exact: the points are processed in blocks of doubling size, each
    searched with a k-d tree over every point up to the block's end.
    """
    n = len(points)
    neighbors = np.full((n, m), -1, dtype=np.int64)
    if m == 0:
        return neighbors
    lo, hi = 1, min(n, 4 * m)
    while lo < n:
        tree = cKDTree(points[:hi])
        todo = np.arange(lo, hi)
        k = min(hi, 3 * m + 1)
        while todo.size:
            _, idx = tree.query(points[todo], k=k)
            idx = idx.reshape(len(todo), k)
            earlier = idx < todo[:, None]
            rank = np.cumsum(earlier, axis=1)
            found = rank[:, -1] >= np.minimum(m, todo)
            rows, cols = np.nonzero(earlier & (rank <= m) & found[:, None])
            neighbors[todo[rows], rank[rows, cols] - 1] = idx[rows, cols]
            todo = todo[~found]
            k = min(hi, 2 * k)
        lo, hi = hi, min(n, 2 * hi)
    return neighbors


def restrict_neighbors(
    xyz_km: np.ndarray,
    neighbors: np.ndarray,
    used: np.ndarray,
    taper_radius: float | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only neighbours closer than taper_radius (km) and in the same group.

    Used slots are moved first and m is cut to the most any row uses (at
    least 1); unused slots hold the row's own index.
    """
    own = np.arange(len(xyz_km))[:, None]
    used = used.copy()
    if taper_radius is not None:
        distance = np.linalg.norm(xyz_km[neighbors] - xyz_km[:, None], axis=-1)
        used &= distance < taper_radius
    if groups is not None:
        groups = np.asarray(groups)
        used &= groups[neighbors] == groups[:, None]
    slots = np.argsort(~used, axis=1, kind="stable")
    used = np.take_along_axis(used, slots, axis=1)
    neighbors = np.where(used, np.take_along_axis(neighbors, slots, axis=1), own)
    m = max(1, int(used.sum(axis=1).max(initial=0)))
    return neighbors[:, :m], used[:, :m]


def vecchia_neighbors(
    xyz_km: np.ndarray,
    n_neighbors: int,
    seed: int,
    taper_radius: float | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditioning sets for vecchia_kriging.

    Each point's n_neighbors nearest points earlier in a random order (seeded),
    restricted by ``restrict_neighbors``. Returns (neighbors, used), both (n, m)
    by row: neighbors are row indices of xyz_km, used marks the real ones.
    """
    n = len(xyz_km)
    m = min(n_neighbors, n - 1)
    order = np.random.default_rng(seed).permutation(n)
    ordered = previous_neighbors(xyz_km[order], m)
    neighbors = np.empty_like(ordered)
    neighbors[order] = np.where(
        ordered >= 0, order[np.maximum(ordered, 0)], order[:, None]
    )
    used = np.empty_like(ordered, dtype=bool)
    used[order] = ordered >= 0
    if m == 0:
        neighbors = np.arange(n)[:, None]
        used = np.zeros((n, 1), dtype=bool)
    return restrict_neighbors(xyz_km, neighbors, used, taper_radius, groups)


def vecchia_kriging(
    xyz_km: np.ndarray,
    neighbors: np.ndarray,
    used: np.ndarray,
    components: list[dict],
    taper_radius: float | None = None,
) -> VecchiaFactor:
    """VecchiaFactor for given conditioning sets.

    One correlation parameter dict per residual column, tapered to zero at
    taper_radius (km) when one is given.
    """
    n, m = neighbors.shape
    coef = np.zeros((n, m, len(components)))
    inv_sd = np.ones((n, len(components)))
    chunk = max(1, CHUNK_ENTRIES // m**2)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(n, start + chunk))
        nb, mask = neighbors[rows], used[rows]
        p = xyz_km[nb]
        d_nn = np.linalg.norm(p[:, :, None] - p[:, None, :], axis=-1)
        d_ni = np.linalg.norm(p - xyz_km[rows][:, None], axis=-1)
        both = mask[:, :, None] & mask[:, None, :]
        for c, component in enumerate(components):
            # Unused slots become identity rows and columns, so their weights are 0
            r_nn = np.where(both, correlated_part(d_nn, component, taper_radius), 0.0)
            r_nn = r_nn + np.eye(m) * np.where(mask, component["nugget"], 1.0)[:, None]
            r_ni = np.where(mask, correlated_part(d_ni, component, taper_radius), 0.0)
            b = np.linalg.solve(r_nn, r_ni[..., None])[..., 0]
            var = 1.0 - np.einsum("ij,ij->i", b, r_ni)
            if np.any(var <= 0):
                raise ValueError(
                    "Non-positive kriging variance: is the nugget too small?"
                )
            coef[rows, :, c] = b
            inv_sd[rows, c] = 1.0 / np.sqrt(var)
    return VecchiaFactor(neighbors, coef, inv_sd)


def vecchia_factor(
    xyz_km: np.ndarray,
    components: list[dict],
    n_neighbors: int,
    seed: int,
    taper_radius: float | None = None,
    groups: np.ndarray | None = None,
) -> VecchiaFactor:
    """Vecchia whitening of correlated residuals at points xyz_km (n, 3).

    Points farther apart than taper_radius (km), or in different groups, are
    independent. Returns the VecchiaFactor in the points' own row order.
    """
    neighbors, used = vecchia_neighbors(xyz_km, n_neighbors, seed, taper_radius, groups)
    return vecchia_kriging(xyz_km, neighbors, used, components, taper_radius)


def correlated_rows(station: pd.DataFrame, covariance: dict) -> np.ndarray:
    """Rows of a station table that the covariance applies to."""
    names = covariance["station_names"]
    if names is None:
        return np.arange(len(station))
    stripped = station["name"].astype(str).str.strip()
    return np.flatnonzero(stripped.isin([name.strip() for name in names]).to_numpy())


def station_factor(
    station: pd.DataFrame,
    covariance: dict,
    n_components: int,
    taper_radius: float | None,
    same_block: bool,
) -> tuple[VecchiaFactor, np.ndarray]:
    """VecchiaFactor over every row of a celeri station table.

    Stations selected by ``covariance["station_names"]`` are correlated with
    each other (within taper_radius km, and on the same block when same_block);
    all others stay independent. Components without parameters (e.g. "up"
    when only east and north were fit) stay independent too.

    Returns the factor and the correlated rows.
    """
    names = list(COMPONENTS[:n_components])
    components = covariance["components"]
    selected = correlated_rows(station, covariance)
    if selected.size == 0:
        raise ValueError(
            f"No stations named {covariance['station_names']} in the station table"
        )
    correlated = [c for c, name in enumerate(names) if name in components]
    xyz_km = station[["x", "y", "z"]].to_numpy()[selected] / 1e3
    groups = station["block_label"].to_numpy()[selected] if same_block else None
    sub = vecchia_factor(
        xyz_km,
        [components[names[c]] for c in correlated],
        int(covariance["n_neighbors"]),
        int(covariance["ordering_seed"]),
        taper_radius,
        groups,
    )
    n = len(station)
    m = sub.neighbors.shape[1]
    neighbors = np.repeat(np.arange(n)[:, None], m, axis=1)
    coef = np.zeros((n, m, n_components))
    inv_sd = np.ones((n, n_components))
    neighbors[selected] = selected[sub.neighbors]
    coef[np.ix_(selected, np.arange(m), correlated)] = sub.coef
    inv_sd[np.ix_(selected, correlated)] = sub.inv_sd
    return VecchiaFactor(neighbors, coef, inv_sd), selected


def mean_information(
    factor: VecchiaFactor, rows: np.ndarray, n_components: int
) -> np.ndarray:
    """Effective number of independent stations among rows for a constant offset.

    ``1' R^-1 1`` per component: the number of independent stations that would
    determine a common offset of the rows' residuals as well as they do.
    """
    ones = np.zeros((len(factor.inv_sd), n_components))
    ones[rows] = 1.0
    return (factor.whiten(ones)[rows] ** 2).sum(axis=0)


def _materialize(x):
    """X, computed in full before the neighbour gather reads it.

    Left alone, XLA (the JAX backend) fuses the computation of the residuals,
    a sum of hundreds of per-mesh velocity predictions, into the gather and
    recomputes that sum for every gathered element: 10x slower gradients at
    100k stations. XLA on CPU drops optimization barriers before fusing, but
    it never fuses a floating-point divide into an op that reads elements
    more than once. Dividing by a shared variable equal to 1 (a runtime input,
    so nothing folds it away) forces x to be computed once.
    """
    import pytensor

    one = pytensor.shared(np.array(1.0, dtype=x.dtype), name="station_covariance_one")
    return x / one


def whiten_tensor(r, factor: VecchiaFactor):
    """PyTensor version of ``VecchiaFactor.whiten`` for r of shape (n, k)."""
    n, m, k = factor.coef.shape
    dtype = r.dtype
    r = _materialize(r)
    gathered = r[factor.neighbors.ravel()].reshape((n, m, k))
    prediction = (factor.coef.astype(dtype) * gathered).sum(axis=1)
    return (r - prediction) * factor.inv_sd.astype(dtype)
