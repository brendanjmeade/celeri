import importlib.util
from typing import Literal

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from loguru import logger
from pymc import Model as PymcModel
from scipy import linalg, spatial

from celeri.celeri_util import get_keep_index_12
from celeri.constants import RADIUS_EARTH
from celeri.model import Model
from celeri.operators import Operators, build_operators
from celeri.solve import Estimation, build_estimation

DIRECTION_IDX = {
    "strike_slip": slice(None, None, 2),
    "dip_slip": slice(1, None, 2),
}


def _constrain_field(
    values,
    lower: float | None,
    upper: float | None,
    softplus_lengthscale: float | None = None,
):
    """Optionally transform values to satisfy bounds, using sigmoid or softplus.

    The transformation applied depends on which bounds are present. All
    transformations are identity-like away from the bounds: values far from
    the constrained region pass through approximately unchanged.

    No bounds: Values are returned unchanged.

    Single bound: A softplus with a given length scale is used.
    For lower bound: values approaching -∞ asymptote to the lower bound,
    while large positive values satisfy output ≈ input.
    For upper bound: values approaching +∞ asymptote to the upper bound,
    while large negative values satisfy output ≈ input.

    Two bounds: A sigmoid scaled to the range [lower, upper] is used.
    The midpoint of the range maps to itself with unit slope (f'(midpoint) = 1).

    Parameters
    ----------
    values : array-like
        The unconstrained values to transform.
    lower : float | None
        Lower bound for the constraint.
    upper : float | None
        Upper bound for the constraint.
    softplus_lengthscale : float | None
        Length scale for softplus operations when only one bound is present.
        Required when exactly one of lower/upper is set.
    """
    import pymc as pm

    if lower is not None and upper is not None:
        scale = upper - lower
        midpoint = (lower + upper) / 2
        return pm.math.sigmoid(4 * (values - midpoint) / scale) * scale + lower  # type: ignore[attr-defined]

    if lower is not None:
        if softplus_lengthscale is None:
            raise ValueError(
                "softplus_lengthscale is required when only lower bound is set"
            )
        return lower + softplus_lengthscale * pt.softplus(  # type: ignore[operator]
            (values - lower) / softplus_lengthscale
        )
    if upper is not None:
        if softplus_lengthscale is None:
            raise ValueError(
                "softplus_lengthscale is required when only upper bound is set"
            )
        return upper - softplus_lengthscale * pt.softplus(  # type: ignore[operator]
            (upper - values) / softplus_lengthscale
        )

    return values


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """log(1 - exp(x)), numerically stable implementation.

    Based on the implementation in PyTensor.

    The domain is x < 0, with _log1mexp(0) = -inf, and _log1mexp(-inf) = 0.

    If x is large negative, then exp(x) is small, so we use the identity:

      log(1 - exp(x)) = log(1 + (-exp(x))) = log1p(-exp(x))

    If x is negative but close to zero, then exp(x) - 1 is small, so we use:

      log(1 - exp(x)) = log(-(exp(x) - 1)) = log(-expm1(x))
    """
    return np.where(x < -0.693, np.log1p(-np.exp(x)), np.log(-np.expm1(x)))


def _softplus_inv(x: np.ndarray) -> np.ndarray:
    """Inverse of softplus: log(exp(x) - 1), numerically stable implementation.

    The domain is x > 0, with _softplus_inv(0) = -inf, and _softplus_inv(inf) = inf.

    For numerical stability, reduce to the case of _log1mexp via the identity:

      log(exp(x) - 1) = log(exp(x) * (1 - exp(-x))) = x + log(1 - exp(-x))
                      = x + _log1mexp(-x)
    """
    return x + _log1mexp(-x)


def _unconstrain_field(
    values: np.ndarray,
    lower: float | None,
    upper: float | None,
    softplus_lengthscale: float | None = None,
) -> np.ndarray:
    """Inverse of _constrain_field: map constrained values to unconstrained space.

    This is the pointwise inverse of _constrain_field, useful for setting
    prior means in the unconstrained parameterization.

    Parameters
    ----------
    values : np.ndarray
        The constrained values to transform back to unconstrained space.
        Must satisfy the bounds (lower < values < upper for two bounds,
        values > lower for lower bound, values < upper for upper bound).
    lower : float | None
        Lower bound for the constraint.
    upper : float | None
        Upper bound for the constraint.
    softplus_lengthscale : float | None
        Length scale for softplus operations when only one bound is present.
        Required when exactly one of lower/upper is set.

    Returns
    -------
    np.ndarray
        The unconstrained values.
    """
    from scipy.special import logit

    if lower is not None and upper is not None:
        scale = upper - lower
        midpoint = (lower + upper) / 2
        # y = sigmoid(4 * (x - midpoint) / scale) * scale + lower
        # => x = midpoint + scale * logit((y - lower) / scale) / 4
        normalized = (values - lower) / scale
        return midpoint + scale * logit(normalized) / 4

    if lower is not None:
        if softplus_lengthscale is None:
            raise ValueError(
                "softplus_lengthscale is required when only lower bound is set"
            )
        # y = lower + L * softplus((x - lower) / L)
        # => x = lower + L * softplus_inv((y - lower) / L)
        return lower + softplus_lengthscale * _softplus_inv(
            (values - lower) / softplus_lengthscale
        )

    if upper is not None:
        if softplus_lengthscale is None:
            raise ValueError(
                "softplus_lengthscale is required when only upper bound is set"
            )
        # y = upper - L * softplus((upper - x) / L)
        # => x = upper - L * softplus_inv((upper - y) / L)
        return upper - softplus_lengthscale * _softplus_inv(
            (upper - values) / softplus_lengthscale
        )

    return values


def _get_unconstrained_mean(
    mean: float,
    mean_parameterization: str,
    lower: float | None,
    upper: float | None,
    softplus_lengthscale: float | None,
    field_name: str,
) -> float:
    """Convert a prior mean to unconstrained space.

    Parameters
    ----------
    mean : float
        The prior mean value.
    mean_parameterization : str
        Either "constrained" (mean is in bounded space) or "unconstrained".
    lower : float | None
        Lower bound for the constraint.
    upper : float | None
        Upper bound for the constraint.
    softplus_lengthscale : float | None
        Length scale for softplus operations when only one bound is present.
    field_name : str
        Name of the field for error messages (e.g., "coupling_ss").

    Returns
    -------
    float
        The unconstrained mean.

    Raises
    ------
    ValueError
        If mean_parameterization is "constrained" and the mean is outside the
        domain of the inverse function (i.e., outside the bounds).
    """
    if mean_parameterization == "unconstrained":
        return mean

    # mean_parameterization == "constrained"
    # Validate that the constrained mean is within bounds
    if lower is not None and mean <= lower:
        raise ValueError(
            f"{field_name}: constrained mean {mean} must be > lower bound {lower}. "
            f"Set a mesh-specific value or use unconstrained parameterization."
        )
    if upper is not None and mean >= upper:
        raise ValueError(
            f"{field_name}: constrained mean {mean} must be < upper bound {upper}. "
            f"Set a mesh-specific value or use unconstrained parameterization."
        )

    return _unconstrain_field(
        np.array([mean]), lower, upper, softplus_lengthscale
    ).item()


def _operator_mult(operator: np.ndarray, vector):
    return operator.astype("f").copy(order="F") @ vector.astype("f")


def _get_eigenmodes(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
) -> np.ndarray:
    """Get the kernel eigenmodes for a mesh and slip type."""
    n_eigs = (
        model.meshes[mesh_idx].config.n_modes_strike_slip
        if kind == "strike_slip"
        else model.meshes[mesh_idx].config.n_modes_dip_slip
    )
    eigenvectors = model.meshes[mesh_idx].eigenvectors
    if eigenvectors is None:
        raise ValueError(
            f"Eigenvectors not computed for mesh {mesh_idx}. "
            "Ensure mesh config includes eigenmode parameters."
        )
    return eigenvectors[:, :n_eigs]


def _get_eigen_to_velocity(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    operators: Operators,
    vel_idx: np.ndarray,
) -> np.ndarray:
    """Get the station velocity operator for a mesh and slip type.

    Args:
        model: The model instance
        mesh_idx: Index of the mesh
        kind: Type of slip ("strike_slip" or "dip_slip")
        operators: Operators containing eigen information
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
    """
    assert operators.eigen is not None

    if kind == "strike_slip":
        n_eigs = model.meshes[mesh_idx].config.n_modes_strike_slip
        start_idx = 0
    else:
        n_eigs = model.meshes[mesh_idx].config.n_modes_dip_slip
        start_idx = model.meshes[mesh_idx].config.n_modes_strike_slip

    to_velocity = operators.eigen.eigen_to_velocities[mesh_idx][vel_idx, :][
        :, start_idx : start_idx + n_eigs
    ]

    return to_velocity


def _get_eigenmode_prior_variances(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenmodes and variances for the normal priors of the GP coefficients.

    The variances are the eigenvalues of the covariance matrix: when we diagonalize,
    the "co-" goes away, so we are left with the "variance" of each eigenmode.

    The mesh kernel eigen-decomposition (in `mesh.py`) is pre-computed with a
    unit amplitude scale parameter (`sigma=1`). We reintroduce the amplitude
    scale parameter here by multiplying the eigenvalues by `sigma**2`.
    """
    eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
    n_eigs = eigenvectors.shape[1]
    mesh = model.meshes[mesh_idx]
    assert mesh.eigenvalues is not None
    unit_amplitude_variances = mesh.eigenvalues[:n_eigs]

    variances = sigma**2 * unit_amplitude_variances
    return eigenvectors, variances


def _station_vel_from_elastic_mesh(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    elastic,
    operators: Operators,
    vel_idx: np.ndarray,
):
    """Compute elastic velocity at stations from slip rates on a mesh.

    Parameters
    ----------
    model : Model
        The model instance
    mesh_idx : int
        Index of the mesh
    kind : Literal["strike_slip", "dip_slip"]
        Type of slip
    elastic : array
        Elastic slip rates on the mesh
    operators : Operators
        Operators containing TDE and eigen information
    vel_idx : np.ndarray
        Row indices to select velocity components (horizontal only or all 3)

    Returns
    -------
    array
        Elastic velocities at station locations (flattened, selected components)
    """
    assert operators.tde is not None
    idx = DIRECTION_IDX[kind]
    method = model.config.mcmc_station_velocity_method

    # Validate that tde_to_velocities is available for methods that need it
    tde_to_velocities = operators.tde.tde_to_velocities
    if method in ("direct", "low_rank") and tde_to_velocities is None:
        raise NotImplementedError(
            f"mcmc_station_velocity_method={method!r} requires tde_to_velocities, "
            "but operators were built with discard_tde_to_velocities=True. "
            "Either use mcmc_station_velocity_method='project_to_eigen' (the default), "
            "or rebuild operators with discard_tde_to_velocities=False."
        )

    if method == "low_rank":
        assert tde_to_velocities is not None
        to_station = tde_to_velocities[mesh_idx][vel_idx, :][:, idx.start : None : 3]
        u, s, vh = linalg.svd(to_station, full_matrices=False)
        threshold = 1e-5
        mask = s > threshold
        s = s[mask].astype("f")
        u = u[:, mask].astype("f")
        vh = vh[mask, :].astype("f")
        elastic_velocity = _operator_mult(-u * s, _operator_mult(vh, elastic))
        return elastic_velocity.astype("d")
    elif method == "project_to_eigen":
        assert operators.eigen is not None
        to_velocity = _get_eigen_to_velocity(
            model,
            mesh_idx,
            kind,
            operators,
            vel_idx,
        )
        eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
        # TODO: This assumes that the eigenvectors are orthogonal
        # with respect to the euclidean inner product. If we change
        # the eigen decomposition to use a different inner product,
        # we will need to change this projection.
        coefs = _operator_mult(eigenvectors.T, elastic)
        # eigen_to_velocities now includes selected velocity components
        elastic_velocity = _operator_mult(to_velocity, coefs)
        return elastic_velocity
    elif method == "direct":
        assert tde_to_velocities is not None
        to_station = tde_to_velocities[mesh_idx][vel_idx, :][:, idx.start : None : 3]
        elastic_velocity = _operator_mult(-to_station, elastic)
        return elastic_velocity
    else:
        raise ValueError(
            f"Unknown mcmc_station_velocity_method: {method}. "
            "Must be one of 'direct', 'low_rank', or 'project_to_eigen'."
        )


def _coupling_component(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    rotation,
    operators: Operators,
    vel_idx: np.ndarray,
    lower: float | None,
    upper: float | None,
    sigma: float,
    mu_unconstrained: float,
):
    """Model elastic slip rate as coupling * kinematic slip rate.

    Returns the estimated elastic slip rates on the TDEs and the
    velocities at the stations due to them.

    Args:
        model: The model instance
        mesh_idx: Index of the mesh
        kind: Type of slip ("strike_slip" or "dip_slip")
        rotation: Rotation parameters
        operators: Operators containing TDE and eigen information
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
        lower: Lower bound for coupling constraint
        upper: Upper bound for coupling constraint
        sigma: Amplitude scale parameter for GP prior
        mu_unconstrained: Prior mean in unconstrained space
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm

    kind_short = {"strike_slip": "ss", "dip_slip": "ds"}[kind]
    idx = DIRECTION_IDX[kind]

    if mesh_idx not in operators.rotation_to_tri_slip_rate:
        raise ValueError(
            f"Mesh {mesh_idx} does not have well defined kinematic slip rates. "
            "Coupling constraints cannot be used."
        )

    operator = operators.rotation_to_tri_slip_rate[mesh_idx][idx, :]
    kinematic = _operator_mult(operator, rotation)
    pm.Deterministic(f"kinematic_{mesh_idx}_{kind_short}", kinematic)

    eigenvectors, variances = _get_eigenmode_prior_variances(
        model, mesh_idx, kind, sigma
    )
    n_eigs = variances.size
    softplus_lengthscale = model.meshes[mesh_idx].config.softplus_lengthscale

    if model.meshes[mesh_idx].config.gp_parameterization == "non_centered":
        # Non-centered: sample white noise, then mollify via eigenvalue scaling
        white_noise = pm.Normal(
            f"coupling_coefs_{mesh_idx}_{kind_short}_white_noise",
            sigma=1,
            shape=n_eigs,
        )
        coefs = pm.Deterministic(
            f"coupling_coefs_{mesh_idx}_{kind_short}",
            white_noise * np.sqrt(variances),
        )
    else:
        # Centered: sample directly with heterogeneous variances
        coefs = pm.Normal(
            f"coupling_coefs_{mesh_idx}_{kind_short}",
            mu=0,
            sigma=np.sqrt(variances),
            shape=n_eigs,
        )

    # Add mean offset in unconstrained space before constraining
    coupling_field = _operator_mult(eigenvectors, coefs) + mu_unconstrained
    coupling_field = _constrain_field(
        coupling_field, lower, upper, softplus_lengthscale
    )
    pm.Deterministic(f"coupling_{mesh_idx}_{kind_short}", coupling_field)
    elastic_tde = kinematic * coupling_field
    pm.Deterministic(f"elastic_{mesh_idx}_{kind_short}", elastic_tde)

    station_vels = _station_vel_from_elastic_mesh(
        model,
        mesh_idx,
        kind,
        elastic_tde,
        operators,
        vel_idx,
    )
    return elastic_tde, station_vels.astype("d")


def _elastic_component(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    operators: Operators,
    vel_idx: np.ndarray,
    lower: float | None,
    upper: float | None,
    sigma: float,
    mu_unconstrained: float,
):
    """Model elastic slip rate as a linear combination of eigenmodes.

    Uses either non-centered parameterization (sample white noise, mollify via
    eigenvalue scaling) or centered parameterization (sample directly with
    heterogeneous variances), controlled by mesh config.

    Returns the estimated elastic slip rates on the TDEs and the
    velocities at the stations due to them.

    Args:
        model: The model instance
        mesh_idx: Index of the mesh
        kind: Type of slip ("strike_slip" or "dip_slip")
        operators: Operators containing TDE and eigen information
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
        lower: Lower bound for elastic constraint
        upper: Upper bound for elastic constraint
        sigma: Amplitude scale parameter for GP prior
        mu_unconstrained: Prior mean in unconstrained space
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm

    kind_short = {"strike_slip": "ss", "dip_slip": "ds"}[kind]

    to_velocity = _get_eigen_to_velocity(
        model,
        mesh_idx,
        kind,
        operators,
        vel_idx,
    )
    eigenvectors, variances = _get_eigenmode_prior_variances(
        model, mesh_idx, kind, sigma
    )
    n_eigs = variances.size
    softplus_lengthscale = model.meshes[mesh_idx].config.softplus_lengthscale

    if model.meshes[mesh_idx].config.gp_parameterization == "non_centered":
        # Non-centered: sample white noise, then mollify via eigenvalue scaling
        white_noise = pm.Normal(
            f"elastic_eigen_{mesh_idx}_{kind_short}_white_noise",
            sigma=1,
            shape=n_eigs,
        )
        param = pm.Deterministic(
            f"elastic_eigen_{mesh_idx}_{kind_short}",
            white_noise * np.sqrt(variances),
        )
    else:
        # Centered: sample directly with heterogeneous variances
        param = pm.Normal(
            f"elastic_eigen_{mesh_idx}_{kind_short}",
            sigma=np.sqrt(variances),
            shape=n_eigs,
        )
    # Add mean offset in unconstrained space before constraining
    elastic_tde = _constrain_field(
        _operator_mult(eigenvectors, param) + mu_unconstrained,
        lower,
        upper,
        softplus_lengthscale,
    )
    pm.Deterministic(f"elastic_{mesh_idx}_{kind_short}", elastic_tde)

    # Compute elastic velocity at stations. The operator already
    # includes a negative sign. eigen_to_velocities uses selected velocity components.
    if lower is None and upper is None:
        station_vels = _operator_mult(to_velocity, param)
    else:
        station_vels = _station_vel_from_elastic_mesh(
            model,
            mesh_idx,
            kind,
            elastic_tde,
            operators,
            vel_idx,
        )

    return elastic_tde, station_vels


def _mesh_component(
    model: Model,
    mesh_idx: int,
    rotation,
    operators: Operators,
    vel_idx: np.ndarray,
):
    """Compute velocity contributions from a mesh.

    Args:
        model: The model instance
        mesh_idx: Index of the mesh
        rotation: Rotation parameters
        operators: Operators containing TDE and eigen information
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
    """
    rates = []

    kinds: tuple[Literal["strike_slip"], Literal["dip_slip"]] = (
        "strike_slip",
        "dip_slip",
    )
    config = model.meshes[mesh_idx].config

    # These should be set by Config.apply_mcmc_prior_defaults
    assert config.coupling_mean_parameterization is not None
    assert config.elastic_mean_parameterization is not None
    assert config.coupling_sigma is not None
    assert config.coupling_mean is not None
    assert config.elastic_sigma is not None
    assert config.elastic_mean is not None

    for kind in kinds:
        kind_short = {"strike_slip": "ss", "dip_slip": "ds"}[kind]
        if kind == "strike_slip":
            coupling_limit = config.coupling_constraints_ss
            rate_limit = config.elastic_constraints_ss
        elif kind == "dip_slip":
            coupling_limit = config.coupling_constraints_ds
            rate_limit = config.elastic_constraints_ds
        else:
            raise ValueError(f"Unknown slip kind: {kind}")

        has_coupling_bound = (
            coupling_limit.lower is not None or coupling_limit.upper is not None
        )

        if has_coupling_bound:
            mu_unconstrained = _get_unconstrained_mean(
                config.coupling_mean,
                config.coupling_mean_parameterization,
                coupling_limit.lower,
                coupling_limit.upper,
                config.softplus_lengthscale,
                f"coupling_{kind_short} mesh {mesh_idx}",
            )
            elastic_tde, station_vels = _coupling_component(
                model,
                mesh_idx,
                kind,
                rotation,
                operators,
                vel_idx,
                lower=coupling_limit.lower,
                upper=coupling_limit.upper,
                sigma=config.coupling_sigma,
                mu_unconstrained=mu_unconstrained,
            )
        else:
            mu_unconstrained = _get_unconstrained_mean(
                config.elastic_mean,
                config.elastic_mean_parameterization,
                rate_limit.lower,
                rate_limit.upper,
                config.softplus_lengthscale,
                f"elastic_{kind_short} mesh {mesh_idx}",
            )
            elastic_tde, station_vels = _elastic_component(
                model,
                mesh_idx,
                kind,
                operators,
                vel_idx,
                lower=rate_limit.lower,
                upper=rate_limit.upper,
                sigma=config.elastic_sigma,
                mu_unconstrained=mu_unconstrained,
            )

        rates.append(station_vels)
        _add_tde_elastic_constraints(model, mesh_idx, elastic_tde, kind)
    return sum(rates)


def _add_tde_elastic_constraints(
    model: Model,
    mesh_idx: int,
    elastic_tde: np.ndarray,
    kind: Literal["strike_slip", "dip_slip"],
):
    """Add TDE elastic constraints to the PyMC model.

    Adds penalty to nonzero elements on top, bottom, and side boundaries of mesh
    with artificial observed 0s.
    """
    import pymc as pm

    mesh = model.meshes[mesh_idx]
    for name, elements, constraint_flag, sigma in [
        (
            "top",
            mesh.top_elements,
            mesh.config.top_slip_rate_constraint,
            mesh.config.top_elastic_constraint_sigma,
        ),
        (
            "bot",
            mesh.bot_elements,
            mesh.config.bot_slip_rate_constraint,
            mesh.config.bot_elastic_constraint_sigma,
        ),
        (
            "side",
            mesh.side_elements,
            mesh.config.side_slip_rate_constraint,
            mesh.config.side_elastic_constraint_sigma,
        ),
    ]:
        if constraint_flag == 1:
            idx = np.where(elements)[0]
            constrained_tde = elastic_tde[idx]

            pm.Normal(
                f"{name}_constraint_{mesh_idx}_{kind}",
                mu=constrained_tde,
                sigma=sigma,
                observed=np.zeros(len(idx)),
            )


def _add_block_strain_rate_component(operators: Operators, vel_idx: np.ndarray):
    """Add block strain rate component to the PyMC model.

    Returns the velocity contribution from block strain rates.

    Args:
        operators: The operators object
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
    """
    import pymc as pm

    raw = pm.Normal("block_strain_rate_raw", sigma=100, dims="block_strain_rate_param")
    op = operators.block_strain_rate_to_velocities[vel_idx, :]
    if op.size == 0:
        scale = 1.0
    else:
        scale = 1 / np.sqrt((op**2).mean())
    block_strain_rate = pm.Deterministic(
        "block_strain_rate", scale * raw, dims="block_strain_rate_param"
    )

    return _operator_mult(op, block_strain_rate)


def _add_rotation_component(operators: Operators, vel_idx: np.ndarray):
    """Add block rotation component to the PyMC model.

    Returns rotation parameters and velocity contributions.

    Args:
        operators: The operators object
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
    """
    import pymc as pm

    rotation_to_vel = operators.rotation_to_velocities[vel_idx, :]
    rotation_to_okada_vel = operators.rotation_to_slip_rate_to_okada_to_velocities[
        vel_idx, :
    ]

    A = rotation_to_vel - rotation_to_okada_vel
    scale = 1e6
    B = A / scale
    _u, _s, vh = linalg.svd(B, full_matrices=False)
    raw = pm.StudentT("rotation_raw", sigma=20, nu=4, dims="rotation_param")

    rotation = pm.Deterministic(
        "rotation", _operator_mult(vh.T, raw / scale), dims="rotation_param"
    )

    rotation_velocity = _operator_mult(rotation_to_vel, rotation)
    rotation_okada_velocity = _operator_mult(-rotation_to_okada_vel, rotation)

    return rotation, rotation_velocity, rotation_okada_velocity


def _add_mogi_component(operators: Operators, vel_idx: np.ndarray):
    """Add Mogi source component to the PyMC model.

    Returns the velocity contribution from Mogi sources.

    Args:
        operators: The operators object
        vel_idx: Row indices to select velocity components (horizontal only or all 3)
    """
    import pymc as pm

    raw = pm.Normal("mogi_raw", dims="mogi_param")
    op = operators.mogi_to_velocities[vel_idx, :]
    if op.size == 0:
        scale = 1.0
    else:
        scale = 1 / np.sqrt((op**2).mean())
    mogi = pm.Deterministic("mogi", scale * raw, dims="mogi_param")

    return _operator_mult(op, mogi)


def _add_station_velocity_likelihood(model: Model, mu):
    """Add station velocity likelihood to the PyMC model.

    Uses area-weighted Student-t likelihood for station observations.
    """
    import pymc as pm

    sigma = pm.HalfNormal("sigma", sigma=2)

    if model.config.include_vertical_velocity:
        data = np.array(
            [model.station.east_vel, model.station.north_vel, model.station.up_vel]
        ).T
    else:
        data = np.array([model.station.east_vel, model.station.north_vel]).T

    lh_dist = pm.StudentT.dist

    def lh(value, weight, mu, sigma):
        dist = lh_dist(nu=6, mu=mu, sigma=sigma)
        return weight * pm.logp(dist, value)

    def random(weight, mu, sigma, rng=None, size=None):
        return lh_dist(nu=6, mu=mu, sigma=sigma, rng=rng, size=size)

    dims = (
        ("station", "xyz")
        if model.config.include_vertical_velocity
        else ("station", "xy")
    )

    if model.config.mcmc_station_weighting is None:
        logger.info(f"Using unweighted station likelihood ({len(data)} stations)")
        pm.StudentT(
            "station_velocity",
            mu=mu,
            sigma=sigma,
            observed=data,
            dims=dims,
            nu=6,
        )
    elif model.config.mcmc_station_weighting == "voronoi":
        effective_area = model.config.mcmc_station_effective_area

        voroni = spatial.SphericalVoronoi(
            model.station[["x", "y", "z"]].values, int(RADIUS_EARTH)
        )
        areas = voroni.calculate_areas()

        areas_clipped = np.minimum(effective_area, areas)
        weight = areas_clipped / effective_area

        # Log diagnostics about the weighting
        effective_n = (weight.sum() ** 2) / (weight**2).sum()
        logger.info("Station weighting diagnostics:")
        logger.info(f"  Number of stations: {len(weight)}")
        logger.info(
            f"  Effective area threshold: {np.sqrt(effective_area) / 1000:.1f} km "
            f"x {np.sqrt(effective_area) / 1000:.1f} km"
        )
        logger.info(f"  Weight range: [{weight.min():.3f}, {weight.max():.3f}]")
        logger.info(
            f"  Effective sample size: {effective_n:.1f} (vs {len(weight)} stations)"
        )
        logger.info(
            "  Stations at full weight (area >= threshold): "
            f"{(areas >= effective_area).sum()}"
        )

        pm.CustomDist(
            "station_velocity",
            weight[:, None],
            mu,
            sigma,
            logp=lh,
            random=random,
            observed=data,
            dims=dims,
        )
    else:
        raise ValueError(
            f"Unknown mcmc_station_weighting: {model.config.mcmc_station_weighting}. "
            "Must be None or 'voronoi'."
        )


def _add_segment_constraints(model: Model, operators: Operators, rotation):
    """Add segment slip rate constraints to the PyMC model.

    Includes regularization, observations, and bounds on slip rates.
    """
    import pymc as pm

    segment_rates = _operator_mult(operators.rotation_to_slip_rate, rotation)
    segment_rates = segment_rates.reshape((-1, 3))

    # Regularization towards zero slip rate
    gamma = model.config.segment_slip_rate_regularization_sigma
    if gamma is not None:
        for i, kind in enumerate(["ss", "ds", "ts"]):
            pm.StudentT(
                f"segment_slip_rate_regularization_{kind}",
                mu=segment_rates[(model.segment[f"{kind}_reg_flag"] == 1).values, i],
                sigma=gamma,
                nu=5,
                observed=np.zeros((model.segment[f"{kind}_reg_flag"] == 1).sum()),
            )

    pm.Deterministic("segment_slip_rate", segment_rates, dims=("segment", "slip_comp"))

    # Slip rate observations
    for comp, flag_attr, rate_attr, sig_attr in [
        ("strike_slip", "ss_rate_flag", "ss_rate", "ss_rate_sig"),
        ("dip_slip", "ds_rate_flag", "ds_rate", "ds_rate_sig"),
        ("tensile_slip", "ts_rate_flag", "ts_rate", "ts_rate_sig"),
    ]:
        flags = getattr(model.segment, flag_attr).values
        if np.any(flags):
            observed_rates = getattr(model.segment, rate_attr).values[flags == 1]
            observed_sigs = getattr(model.segment, sig_attr).values[flags == 1]
            pm.Normal(
                f"segment_{comp}_velocity",
                mu=segment_rates[
                    flags == 1,
                    ["strike_slip", "dip_slip", "tensile_slip"].index(comp),
                ],
                sigma=observed_sigs,
                observed=observed_rates,
            )

    # Slip rate bounds (soft constraints)
    for comp, bound_flag_attr, lower_attr, upper_attr in [
        (
            "strike_slip",
            "ss_rate_bound_flag",
            "ss_rate_bound_min",
            "ss_rate_bound_max",
        ),
        (
            "dip_slip",
            "ds_rate_bound_flag",
            "ds_rate_bound_min",
            "ds_rate_bound_max",
        ),
        (
            "tensile_slip",
            "ts_rate_bound_flag",
            "ts_rate_bound_min",
            "ts_rate_bound_max",
        ),
    ]:
        bound_flags = getattr(model.segment, bound_flag_attr).values
        if np.any(bound_flags):
            lower_bounds = getattr(model.segment, lower_attr).values[bound_flags == 1]
            upper_bounds = getattr(model.segment, upper_attr).values[bound_flags == 1]
            if "slip_rate_bound_sigma" in model.segment.columns:
                bound_sig = model.segment.slip_rate_bound_sigma.values[bound_flags == 1]
            else:
                bound_sig = model.config.segment_slip_rate_bound_sigma
            pm.Censored(
                f"segment_{comp}_rate_lower_bound",
                dist=pm.Normal.dist(
                    mu=segment_rates[
                        bound_flags == 1,
                        ["strike_slip", "dip_slip", "tensile_slip"].index(comp),
                    ],
                    sigma=bound_sig,
                ),
                upper=lower_bounds,
                lower=None,
                observed=lower_bounds,
            )

            pm.Censored(
                f"segment_{comp}_rate_upper_bound",
                dist=pm.Normal.dist(
                    mu=segment_rates[
                        bound_flags == 1,
                        ["strike_slip", "dip_slip", "tensile_slip"].index(comp),
                    ],
                    sigma=bound_sig,
                ),
                upper=None,
                lower=upper_bounds,
                observed=upper_bounds,
            )


def _build_pymc_model(model: Model, operators: Operators) -> PymcModel:
    """Build the complete PyMC model for MCMC inference.

    Combines all velocity components (block strain, rotation, Mogi, elastic)
    and adds likelihoods for station and segment observations.
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm

    # Check if vertical velocities should be included
    # If not, we can exclude vertical components from operators for efficiency
    include_vertical = model.config.include_vertical_velocity
    n_stations = len(model.station)

    if include_vertical:
        # Keep all 3 velocity components per station (east, north, up)
        vel_idx = np.arange(3 * n_stations)
        n_vel_components = 3
    else:
        # Keep only horizontal components (east, north), excluding vertical (up)
        vel_idx = get_keep_index_12(3 * n_stations)
        n_vel_components = 2

    coords = {
        "block_strain_rate_param": pd.RangeIndex(
            operators.block_strain_rate_to_velocities.shape[1]
        ),
        "global_float_block_rotation_param": pd.RangeIndex(
            operators.global_float_block_rotation.shape[1]
        ),
        "mogi_param": pd.RangeIndex(operators.mogi_to_velocities.shape[1]),
        "rotation_param": pd.RangeIndex(operators.rotation_to_velocities.shape[1]),
        "station": model.station.index,
        "segment": model.segment.index,
        "xyz": pd.Index(["x", "y", "z"]),
        "xy": pd.Index(["x", "y"]),
        "slip_comp": pd.Index(["strike_slip", "dip_slip", "tensile_slip"]),
    }

    with pm.Model(coords=coords) as pymc_model:
        block_strain_rate_velocity = _add_block_strain_rate_component(
            operators, vel_idx
        )

        rotation, rotation_velocity, rotation_okada_velocity = _add_rotation_component(
            operators, vel_idx
        )

        mogi_velocity = _add_mogi_component(operators, vel_idx)

        # Add elastic velocity from meshes
        elastic_velocities = []
        for key, _ in enumerate(model.meshes):
            elastic_velocities.append(
                _mesh_component(model, key, rotation, operators, vel_idx)
            )
        elastic_velocity = sum(elastic_velocities)

        mu = (
            block_strain_rate_velocity
            + rotation_velocity
            + rotation_okada_velocity
            + mogi_velocity
            + elastic_velocity
        )
        mu = mu.reshape((n_stations, n_vel_components))

        dims = ("station", "xyz") if include_vertical else ("station", "xy")
        mu_det = pm.Deterministic("mu", mu, dims=dims)

        _add_station_velocity_likelihood(model, mu_det)
        _add_segment_constraints(model, operators, rotation)

    return pymc_model  # type: ignore[return-value]


def solve_mcmc(
    model: Model,
    *,
    operators: Operators | None = None,
    sample_kwargs: dict | None = None,
) -> Estimation:
    if importlib.util.find_spec("nutpie") is None:
        raise ImportError(
            "nutpie is required for MCMC solving. "
            "Please install it with 'pip install nutpie'."
        )
    if importlib.util.find_spec("pymc") is None:
        raise ImportError(
            "pymc is required for MCMC solving. "
            "Please install it with 'pip install pymc'."
        )

    import nutpie

    if model.config.segment_slip_rate_hard_bounds:
        raise ValueError(
            "Hard bounds on segment slip rates are not supported in MCMC solve. "
            "Please use soft bounds with `segment_slip_rate_bound_sigma` instead."
        )

    segment_mesh_indices = set(
        model.segment.mesh_file_index[model.segment.mesh_flag == 1].unique()
    )
    for mesh_idx, mesh in enumerate(model.meshes):
        has_coupling_constraints = (
            mesh.config.coupling_constraints_ss.lower is not None
            or mesh.config.coupling_constraints_ss.upper is not None
            or mesh.config.coupling_constraints_ds.lower is not None
            or mesh.config.coupling_constraints_ds.upper is not None
        )
        if has_coupling_constraints and mesh_idx not in segment_mesh_indices:
            raise ValueError(
                f"Mesh {mesh_idx} ({mesh.file_name}) has coupling constraints but is not "
                f"tied to any segment in the model. "
                "Either remove coupling constraints or ensure the mesh is referenced by "
                "a segment with mesh_flag=1 in the segment file."
            )

    if operators is None:
        # Only use streaming mode (discard_tde_to_velocities=True) when using project_to_eigen.
        # The direct and low_rank methods require the full tde_to_velocities matrices.
        use_streaming = model.config.mcmc_station_velocity_method == "project_to_eigen"
        if use_streaming:
            logger.info(
                "Building operators with streaming mode (discard_tde_to_velocities=True)"
            )
        else:
            logger.info(
                f"Building operators with discard_tde_to_velocities=False "
                f"(required for mcmc_station_velocity_method={model.config.mcmc_station_velocity_method!r})"
            )
        operators = build_operators(
            model, tde=True, eigen=True, discard_tde_to_velocities=use_streaming
        )
    if operators.tde is None or operators.eigen is None:
        raise ValueError(
            "Operators must have both TDE and eigen components for MCMC solve."
        )

    pymc_model = _build_pymc_model(model, operators)

    compiled = nutpie.compile_pymc_model(
        pymc_model,  # type: ignore[arg-type]
        backend=model.config.mcmc_backend,
    )
    kwargs = {
        "low_rank_modified_mass_matrix": True,
        "mass_matrix_eigval_cutoff": 1.5,
        "mass_matrix_gamma": 1e-6,
        "chains": model.config.mcmc_chains,
        "draws": model.config.mcmc_draws,
        "tune": model.config.mcmc_tune,
        "store_unconstrained": True,
        "store_gradient": True,
        "seed": model.config.mcmc_seed,
    }
    kwargs.update(sample_kwargs or {})

    from datetime import UTC, datetime

    mcmc_start_time = datetime.now(UTC)
    trace = nutpie.sample(compiled, **kwargs)
    mcmc_end_time = datetime.now(UTC)
    mcmc_duration = (mcmc_end_time - mcmc_start_time).total_seconds()

    state_vector = _state_vector_from_draw(
        model, operators, trace.mean(["chain", "draw"])
    )
    estimation = build_estimation(model, operators, state_vector)
    estimation.mcmc_trace = trace
    estimation.mcmc_start_time = mcmc_start_time.isoformat()
    estimation.mcmc_end_time = mcmc_end_time.isoformat()
    estimation.mcmc_duration = mcmc_duration
    estimation.mcmc_num_divergences = int(trace.sample_stats.diverging.sum())
    return estimation


def _state_vector_from_draw(
    model: Model,
    operators: Operators,
    trace,
):
    """Build a state vector from MCMC trace using eigen coefficients.

    The state vector uses eigen coefficients (not TDE values) so that
    Estimation.predictions uses eigen_to_velocities for forward predictions.

    For elastic mode, we use the elastic_eigen_* coefficients directly.
    For coupling mode, we project the elastic TDE values onto eigenvectors
    to get equivalent elastic eigen coefficients, since coupling coefficients
    parameterize the coupling field rather than elastic slip directly.
    """
    assert operators.eigen is not None
    assert operators.index.eigen is not None
    n_params = operators.index.n_operator_cols_eigen
    state_vector = np.zeros(n_params)

    start = operators.index.start_block_strain_col
    end = operators.index.end_block_strain_col
    state_vector[start:end] = trace.posterior.block_strain_rate.values

    start = operators.index.start_mogi_col
    end = operators.index.end_mogi_col
    state_vector[start:end] = trace.posterior.mogi.values

    start = operators.index.start_block_col
    end = operators.index.end_block_col
    state_vector[start:end] = trace.posterior.rotation.values

    # Extract eigen coefficients for each mesh
    for mesh_idx in range(len(model.meshes)):
        start = operators.index.eigen.start_col_eigen[mesh_idx]
        end = operators.index.eigen.end_col_eigen[mesh_idx]
        n_modes_ss = model.meshes[mesh_idx].config.n_modes_strike_slip
        n_modes_ds = model.meshes[mesh_idx].config.n_modes_dip_slip

        coefs = np.zeros(n_modes_ss + n_modes_ds)

        kinds: list[tuple[Literal["strike_slip", "dip_slip"], str, int, int]] = [
            ("strike_slip", "ss", n_modes_ss, 0),
            ("dip_slip", "ds", n_modes_ds, n_modes_ss),
        ]
        for kind, kind_short, n_modes, coef_start in kinds:
            elastic_eigen_var = f"elastic_eigen_{mesh_idx}_{kind_short}"
            elastic_tde_var = f"elastic_{mesh_idx}_{kind_short}"

            if elastic_eigen_var in trace.posterior:
                coefs[coef_start : coef_start + n_modes] = trace.posterior[
                    elastic_eigen_var
                ].values
            elif elastic_tde_var in trace.posterior:
                elastic_tde = trace.posterior[elastic_tde_var].values
                eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
                coefs[coef_start : coef_start + n_modes] = eigenvectors.T @ elastic_tde

        state_vector[start:end] = coefs

    return state_vector
