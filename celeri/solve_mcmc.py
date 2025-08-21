import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from celeri.mesh import Mesh
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


def _model_mesh(model: Model, key: int, mesh: Mesh, rotation, operators: Operators):
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm
    import pytensor.tensor as pt

    indices = {
        "strike_slip": slice(None, None, 2),
        "dip_slip": slice(1, None, 2),
    }

    elastic_velocities = []

    for name, idx in indices.items():
        if name == "strike_slip":
            coupling_limit = mesh.config.coupling_constraints_ss
        else:
            coupling_limit = mesh.config.coupling_constraints_ds

        if coupling_limit.lower is None:
            scale = 0.0
            for op in operators.eigen.eigen_to_velocities.values():
                scale += (op**2).mean()

            scale = scale / len(operators.eigen.eigen_to_velocities)
            scale = 1 / np.sqrt(scale)

            # TODO split dip slip and strike slip
            if name == "dip_slip":
                continue

            operator = operators.eigen.eigen_to_velocities[key]
            raw = pm.Normal(f"elastic_eigen_raw_{key}_{name}", shape=operator.shape[-1])
            param = pm.Deterministic(f"elastic_eigen_{key}_{name}", scale * raw)
            velocity = operator.copy(order="F") @ param
            elastic_velocities.append(
                pt.concatenate(
                    [
                        velocity.reshape((len(model.station), 2)),
                        np.zeros((len(model.station), 1)),
                    ],
                    axis=-1,
                ).ravel(),
            )
            # TODO handle lower and upper elastic bound
            pm.Deterministic(
                f"elastic_{key}_{name}",
                operators.eigen.eigenvectors_to_tde_slip[key].copy(order="F") @ param,
            )
            continue
        if coupling_limit.upper is None:
            raise ValueError(
                "Must provide either both lower and upper bounds "
                f"for coupling constraints for mesh {key} {name} or none."
            )

        operator = operators.rotation_to_tri_slip_rate[key]
        kinematic = (operator.copy(order="F") @ rotation)[idx]
        pm.Deterministic(f"kinematic_{key}_{name}", kinematic)
        n_coefs = operators.eigen.eigenvectors_to_tde_slip[key][idx].shape[1] // 2
        coefs = pm.Normal(f"coupling_coefs_{key}_{name}", mu=0, sigma=10, shape=n_coefs)
        lower = coupling_limit.lower
        upper = coupling_limit.upper
        coupling_field = (
            pm.math.sigmoid(
                operators.eigen.eigenvectors_to_tde_slip[key][::2, :n_coefs].copy(
                    order="F"
                )
                @ coefs
            )
            * (upper - lower)
            + lower
        )
        pm.Deterministic(f"coupling_{key}_{name}", coupling_field)
        elastic = kinematic * coupling_field
        pm.Deterministic(f"elastic_{key}_{name}", elastic)

        elastic_velocity = (
            operators.tde.tde_to_velocities[key][:, slice(idx.start, None, 3)].copy(
                order="F"
            )
            @ elastic
        )
        elastic_velocities.append(elastic_velocity)
    return sum(elastic_velocities)


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
        "xyz": pd.Index(["x", "y", "z"]),
        "xy": pd.Index(["x", "y"]),
    }

    with pm.Model(coords=coords) as pymc_model:
        # block strain rate
        raw = pm.Normal("block_strain_rate_raw", dims="block_strain_rate_param")
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

        # mogi
        raw = pm.Normal("mogi_raw", dims="mogi_param")
        scale = 1 / np.sqrt((operators.mogi_to_velocities**2).mean())
        mogi = pm.Deterministic("mogi", scale * raw, dims="mogi_param")

        mogi_velocity = operators.mogi_to_velocities.copy(order="F") @ mogi

        elastic_velocities = []
        for key, mesh in enumerate(model.meshes):
            elastic_velocities.append(
                _model_mesh(model, key, mesh, rotation, operators)
            )

        elastic_velocity = sum(elastic_velocities)

        mu = (
            block_strain_rate_velocity
            + rotation_velocity
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
    state_vector = _state_vector_from_draw(model, operators_tde, trace.mean(["chain", "draw"]))
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
