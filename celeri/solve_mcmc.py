import importlib.util
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from celeri.model import Model
from celeri.operators import Operators, build_operators
from celeri.solve import Estimation, build_estimation

if TYPE_CHECKING or importlib.util.find_spec("pymc") is None:
    # Fallback for PyMC if not installed
    # This is a minimal stub for PyMC to allow type checking
    class PymcModel:
        pass
else:
    from pymc import Model as PymcModel


DIRECTION_IDX = {
    "strike_slip": slice(None, None, 2),
    "dip_slip": slice(1, None, 2),
}


def _constrain_field(values, lower: float | None, upper: float | None):
    """Use a sigmoid or softplus to constrain values to a range."""
    import pymc as pm

    if lower is not None and upper is not None:
        scale = upper - lower
        return pm.math.sigmoid(values) * scale + lower
    if lower is not None:
        return pm.math.softplus(values) + lower
    if upper is not None:
        return upper - pm.math.softplus(-values)
    return values


def _clean_operator(operator: np.ndarray):
    return operator.copy(order="F").astype("d")


def _get_eigen_modes(
    model: Model,
    mesh: int,
    kind: Literal["strike_slip", "dip_slip"],
    operators: Operators,
    out_idx: slice,
):
    """Get the eigenmodes and station velocity operator for a mesh and slip type."""
    assert operators.eigen is not None

    if kind == "strike_slip":
        n_eigs = model.meshes[mesh].config.n_modes_strike_slip
        start_idx = 0
    else:
        n_eigs = model.meshes[mesh].config.n_modes_dip_slip
        start_idx = model.meshes[mesh].config.n_modes_strike_slip

    eigenvectors = operators.eigen.eigenvectors_to_tde_slip[mesh][
        out_idx, start_idx : start_idx + n_eigs
    ]
    to_velocity = operators.eigen.eigen_to_velocities[mesh][
        :, start_idx : start_idx + n_eigs
    ]
    return _clean_operator(eigenvectors), _clean_operator(to_velocity)


def _coupling_component(
    model: Model,
    mesh: int,
    kind: Literal["strike_slip", "dip_slip"],
    rotation,
    operators: Operators,
    lower: float | None,
    upper: float | None,
):
    """Model elastic slip rate as coupling * kinematic slip rate.

    Return the resulting elastic velocity at the station locations.
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm

    idx = DIRECTION_IDX[kind]

    if mesh not in operators.rotation_to_tri_slip_rate:
        raise ValueError(
            f"Mesh {mesh} does not have well defined kinematic slip rates. "
            "Coupling constraints cannot be used."
        )

    operator = _clean_operator(operators.rotation_to_tri_slip_rate[mesh][idx, :])
    kinematic = operator @ rotation
    pm.Deterministic(f"kinematic_{mesh}_{kind}", kinematic)

    eigenvectors, _ = _get_eigen_modes(
        model,
        mesh,
        kind,
        operators,
        out_idx=idx,
    )
    n_eigs = eigenvectors.shape[1]
    coefs = pm.Normal(f"coupling_coefs_{mesh}_{kind}", mu=0, sigma=10, shape=n_eigs)

    coupling_field = eigenvectors @ coefs
    coupling_field = _constrain_field(coupling_field, lower, upper)
    pm.Deterministic(f"coupling_{mesh}_{kind}", coupling_field)
    elastic = kinematic * coupling_field
    pm.Deterministic(f"elastic_{mesh}_{kind}", elastic)

    to_station = operators.tde.tde_to_velocities[mesh][:, idx.start : None : 3]
    to_station = _clean_operator(to_station)
    elastic_velocity = -to_station @ elastic
    return elastic_velocity


def _elastic_component(
    model: Model,
    mesh: int,
    kind: Literal["strike_slip", "dip_slip"],
    rotation,
    operators: Operators,
    lower: float | None,
    upper: float | None,
):
    """Model elastic slip rate as a linear combination of eigenmodes.

    Return the resulting elastic velocity at the station locations.
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm
    import pytensor.tensor as pt

    idx = DIRECTION_IDX[kind]

    scale = 0.0
    for op in operators.eigen.eigen_to_velocities.values():
        scale += (op**2).mean()

    scale = scale / len(operators.eigen.eigen_to_velocities)
    scale = 1 / np.sqrt(scale)

    eigenvectors, to_velocity = _get_eigen_modes(
        model,
        mesh,
        kind,
        operators,
        out_idx=idx,
    )
    n_eigs = eigenvectors.shape[1]

    raw = pm.Normal(f"elastic_eigen_raw_{mesh}_{kind}", shape=n_eigs)
    param = pm.Deterministic(f"elastic_eigen_{mesh}_{kind}", scale * raw)
    elastic = _constrain_field(eigenvectors @ param, lower, upper)
    pm.Deterministic(f"elastic_{mesh}_{kind}", elastic)

    # Compute elastic velocity at stations. The operator already
    # includes a negative sign.
    if lower is None and upper is None:
        elastic_velocity = to_velocity @ param
        # We need to return a station velocity for all three components,
        # not just north and east.
        elastic_velocity = pt.concatenate(
            [
                elastic_velocity.reshape((len(model.station), 2)),
                np.zeros((len(model.station), 1)),
            ],
            axis=-1,
        ).ravel()
    else:
        to_station = operators.tde.tde_to_velocities[mesh][:, idx.start : None : 3]
        to_station = _clean_operator(to_station)
        elastic_velocity = -to_station @ elastic

    return elastic_velocity


def _mesh_component(
    model: Model,
    mesh: int,
    rotation,
    operators: Operators,
):
    rates = []

    for kind in ["strike_slip", "dip_slip"]:
        if kind == "strike_slip":
            coupling_limit = model.meshes[mesh].config.coupling_constraints_ss
            rate_limit = model.meshes[mesh].config.elastic_constraints_ss
        else:
            coupling_limit = model.meshes[mesh].config.coupling_constraints_ds
            rate_limit = model.meshes[mesh].config.elastic_constraints_ds

        has_rate_limit = rate_limit.lower is not None or rate_limit.upper is not None
        has_coupling_limit = (
            coupling_limit.lower is not None or coupling_limit.upper is not None
        )

        if has_rate_limit and has_coupling_limit:
            raise ValueError(
                "Cannot have both rate and coupling constraints "
                f"for mesh {mesh} {kind}."
            )

        if has_coupling_limit:
            rates.append(
                _coupling_component(
                    model,
                    mesh,
                    kind,
                    rotation,
                    operators,
                    lower=coupling_limit.lower,
                    upper=coupling_limit.upper,
                )
            )
        else:
            rates.append(
                _elastic_component(
                    model,
                    mesh,
                    kind,
                    rotation,
                    operators,
                    lower=rate_limit.lower,
                    upper=rate_limit.upper,
                )
            )
    return sum(rates)


def _build_pymc_model(model: Model, operators: Operators) -> PymcModel:
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm

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
        "ss_ds_ts": pd.Index(["strike_slip", "dip_slip", "tensile_slip"]),
    }

    with pm.Model(coords=coords) as pymc_model:
        # block strain rate
        raw = pm.Normal("block_strain_rate_raw", dims="block_strain_rate_param")
        if operators.block_strain_rate_to_velocities.size == 0:
            scale = 1.0
        else:
            scale = 1 / np.sqrt((operators.block_strain_rate_to_velocities**2).mean())
        block_strain_rate = pm.Deterministic(
            "block_strain_rate", scale * raw, dims="block_strain_rate_param"
        )

        block_strain_rate_velocity = (
            np.copy(operators.block_strain_rate_to_velocities, order="F")
            @ block_strain_rate
        )

        # block rotation
        raw = pm.Normal("rotation_raw", sigma=10, dims="rotation_param")
        scale = 1 / np.sqrt((operators.rotation_to_velocities**2).mean())
        rotation = pm.Deterministic("rotation", scale * raw, dims="rotation_param")

        rotation_velocity = operators.rotation_to_velocities.copy(order="F") @ rotation
        rotation_okada_velocity = (
            -operators.rotation_to_slip_rate_to_okada_to_velocities.copy(order="F")
            @ rotation
        )

        # mogi
        raw = pm.Normal("mogi_raw", dims="mogi_param")
        if operators.mogi_to_velocities.size == 0:
            scale = 1.0
        else:
            scale = 1 / np.sqrt((operators.mogi_to_velocities**2).mean())
        mogi = pm.Deterministic("mogi", scale * raw, dims="mogi_param")

        mogi_velocity = operators.mogi_to_velocities.copy(order="F") @ mogi

        elastic_velocities = []
        for key, _ in enumerate(model.meshes):
            elastic_velocities.append(_mesh_component(model, key, rotation, operators))

        elastic_velocity = sum(elastic_velocities)

        mu = (
            block_strain_rate_velocity
            + rotation_velocity
            + rotation_okada_velocity
            + mogi_velocity
            + elastic_velocity
        )

        mu = mu.reshape((len(model.station), 3))[:, :2]

        pm.Deterministic("mu", mu, dims=("station", "xy"))

        sigma = pm.HalfNormal("sigma", sigma=2)
        data = np.array([model.station.east_vel, model.station.north_vel]).T
        pm.Normal(
            "station_velocity",
            mu=mu,
            sigma=sigma,
            observed=data,
            dims=("station", "xy"),
        )

        segment_rates = operators.rotation_to_slip_rate.copy(order="F") @ rotation
        segment_rates = segment_rates.reshape((-1, 3))
        gamma = model.config.segment_slip_rate_regularization_sigma
        if gamma != 0.0:
            pm.Normal(
                "segment_slip_rate_regularization",
                mu=segment_rates,
                sigma=gamma,
                observed=np.zeros((len(model.segment), 3)),
            )

        pm.Deterministic(
            "segment_strike_slip", segment_rates, dims=("segment", "ss_ds_ts")
        )

        # model.segment.ss_rate_flag (and ds_rate_flag, ts_rate_flag) indicates
        # if we have a measurement for that compontent. The observed value is
        # stored in model.segment.ss_rate (and ds_rate, ts_rate). The standard
        # deviation of the measurement error is stored in
        # model.segment.ss_rate_sig (and ds_rate_sig, ts_rate_sig). We only want
        # to add likelihood terms for the components that have measurements.
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

        # model.segment.ss_rate_bound_flag (and ds_rate_bound_flag, ts_rate_bound_flag)
        # indicates if we have an interval bound for that component. The lower and upper
        # bounds are stored in model.segment.ss_rate_lower_min and
        # model.segment.ss_rate_upper_max. We use two pm.Bound(pm.Normal) likelihoods
        # to implement the interval constraint. The standard deviation of the bound
        # is set to model.config.segment_slip_rate_bound_sig.
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
                lower_bounds = getattr(model.segment, lower_attr).values[
                    bound_flags == 1
                ]
                upper_bounds = getattr(model.segment, upper_attr).values[
                    bound_flags == 1
                ]
                bound_sig = model.config.segment_slip_rate_bound_sigma
                pm.Censored(
                    f"segment_{comp}_slip_rate_lower_bound",
                    dist=pm.Normal.dist(
                        mu=segment_rates[
                            bound_flags == 1,
                            ["strike_slip", "dip_slip", "tensile_slip"].index(comp),
                        ],
                        sigma=bound_sig,
                    ),
                    lower=lower_bounds,
                    upper=None,
                    observed=lower_bounds,
                )

                pm.Censored(
                    f"segment_{comp}_slip_rate_upper_bound",
                    dist=pm.Normal.dist(
                        mu=segment_rates[
                            bound_flags == 1,
                            ["strike_slip", "dip_slip", "tensile_slip"].index(comp),
                        ],
                        sigma=bound_sig,
                    ),
                    lower=None,
                    upper=upper_bounds,
                    observed=upper_bounds,
                )

    return pymc_model


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

    if operators is None:
        operators = build_operators(model, tde=True, eigen=True)

    if operators.tde is None or operators.eigen is None:
        raise ValueError(
            "Operators must have both TDE and eigen components for MCMC solve."
        )

    pymc_model = _build_pymc_model(model, operators)

    compiled = nutpie.compile_pymc_model(pymc_model, backend="numba")
    kwargs = {
        "low_rank_modified_mass_matrix": True,
        "mass_matrix_eigval_cutoff": 1.5,
        "mass_matrix_gamma": 1e-6,
        "chains": 2,
        "draws": model.config.mcmc_draws,
        "tune": model.config.mcmc_tune,
    }
    kwargs.update(sample_kwargs or {})
    trace = nutpie.sample(compiled, **kwargs)

    operators_tde = build_operators(model, tde=True, eigen=False)
    state_vector = _state_vector_from_draw(
        model, operators_tde, trace.mean(["chain", "draw"])
    )
    estimation = build_estimation(model, operators_tde, state_vector)
    estimation.mcmc_trace = trace
    return estimation


def _state_vector_from_draw(
    model: Model,
    operators_tde: Operators,
    trace,
):
    assert operators_tde.tde is not None
    assert operators_tde.index.tde is not None
    n_params = operators_tde.full_dense_operator.shape[1]
    state_vector = np.zeros(n_params)

    start = operators_tde.index.start_block_strain_col
    end = operators_tde.index.end_block_strain_col
    state_vector[start:end] = trace.posterior.block_strain_rate.values

    start = operators_tde.index.start_mogi_col
    end = operators_tde.index.end_mogi_col
    state_vector[start:end] = trace.posterior.mogi.values

    start = operators_tde.index.start_block_col
    end = operators_tde.index.end_block_col
    state_vector[start:end] = trace.posterior.rotation.values

    for mesh_idx in range(len(model.meshes)):
        indices = {
            "strike_slip": slice(None, None, 2),
            "dip_slip": slice(1, None, 2),
        }
        for name, idx in indices.items():
            start = operators_tde.index.tde.start_tde_col[mesh_idx]
            end = operators_tde.index.tde.end_tde_col[mesh_idx]
            var_name = f"elastic_{mesh_idx}_{name}"

            if var_name in trace.posterior:
                vals = trace.posterior[var_name]
                # if there is only one of strike/dip slip
                if vals.shape == state_vector[start:end].shape:
                    state_vector[start:end] = vals
                else:
                    state_vector[start:end][idx] = trace.posterior[var_name].values
    return state_vector
