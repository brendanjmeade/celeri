import importlib.util
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy import linalg, spatial

from celeri.constants import RADIUS_EARTH
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

    # Apply numerical stability fix for censored Normal distributions.
    # This is a workaround for https://github.com/pymc-devs/pymc/pull/7996
    # Fixes issue https://github.com/brendanjmeade/celeri/issues/341
    import celeri.censored_distribution_stability_hotfix  # noqa: F401


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
    """Use a sigmoid or softplus to constrain values to a range.

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
        Normally set by MeshConfig validator; falls back to 1.0 if None.
    """
    import pymc as pm

    if lower is not None and upper is not None:
        scale = upper - lower
        return pm.math.sigmoid(values) * scale + lower  # type: ignore[attr-defined]

    if lower is not None:
        return lower + softplus_lengthscale * pm.math.softplus(
            values / softplus_lengthscale
        )  # type: ignore[attr-defined]
    if upper is not None:
        return upper - softplus_lengthscale * pm.math.softplus(
            -values / softplus_lengthscale
        )  # type: ignore[attr-defined]

    return values


def _operator_mult(operator: np.ndarray, vector):
    return operator.astype("f").copy(order="F") @ vector.astype("f")


def _get_eigenmodes(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
) -> np.ndarray:
    """Get the kernel eigenmodes for a mesh and slip type."""
    evecs = model.meshes[mesh_idx].eigenvectors
    assert evecs is not None
    n_eigs = (
        model.meshes[mesh_idx].config.n_modes_strike_slip
        if kind == "strike_slip"
        else model.meshes[mesh_idx].config.n_modes_dip_slip
    )
    return evecs[:, :n_eigs]


def _get_eigen_to_velocity(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    operators: Operators,
) -> np.ndarray:
    """Get the station velocity operator for a mesh and slip type."""
    assert operators.eigen is not None

    if kind == "strike_slip":
        n_eigs = model.meshes[mesh_idx].config.n_modes_strike_slip
        start_idx = 0
    else:
        n_eigs = model.meshes[mesh_idx].config.n_modes_dip_slip
        start_idx = model.meshes[mesh_idx].config.n_modes_strike_slip

    to_velocity = operators.eigen.eigen_to_velocities[mesh_idx][
        :, start_idx : start_idx + n_eigs
    ]

    return to_velocity


def _station_vel_from_elastic_mesh(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    elastic,
    operators: Operators,
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

    Returns
    -------
    array
        Elastic velocities at station locations (flattened, all 3 components)
    """
    import pytensor.tensor as pt

    assert operators.tde is not None
    idx = DIRECTION_IDX[kind]
    method = model.config.mcmc_station_velocity_method

    # Validate that tde_to_velocities is available for methods that need it
    if method in ("direct", "low_rank") and operators.tde.tde_to_velocities is None:
        raise NotImplementedError(
            f"mcmc_station_velocity_method={method!r} requires tde_to_velocities, "
            "but operators were built with discard_tde_to_velocities=True. "
            "Either use mcmc_station_velocity_method='project_to_eigen' (the default), "
            "or rebuild operators with discard_tde_to_velocities=False."
        )

    if method == "low_rank":
        assert operators.tde.tde_to_velocities is not None
        to_station = operators.tde.tde_to_velocities[mesh_idx][:, idx.start : None : 3]
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
        )
        eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
        # TODO: This assumes that the eigenvectors are orthogonal
        # with respect to the euclidean inner product. If we change
        # the eigen decomposition to use a different inner product,
        # we will need to change this projection.
        coefs = _operator_mult(eigenvectors.T, elastic)
        elastic_velocity = _operator_mult(to_velocity, coefs)
        # We need to return a station velocity for all three components,
        # not just north and east.
        elastic_velocity = pt.concatenate(
            [
                elastic_velocity.reshape((len(model.station), 2)),
                np.zeros((len(model.station), 1)),
            ],
            axis=-1,
        ).ravel()  # type: ignore[attr-defined]
        return elastic_velocity
    elif method == "direct":
        assert operators.tde.tde_to_velocities is not None
        to_station = operators.tde.tde_to_velocities[mesh_idx][:, idx.start : None : 3]
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
    lower: float | None,
    upper: float | None,
):
    """Model elastic slip rate as coupling * kinematic slip rate.

    Returns the estimated elastic slip rates on the TDEs and the
    velocities at the stations due to them.
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

    eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
    n_eigs = eigenvectors.shape[1]
    coefs = pm.Normal(
        f"coupling_coefs_{mesh_idx}_{kind_short}", mu=0, sigma=10, shape=n_eigs
    )

    coupling_field = _operator_mult(eigenvectors, coefs)
    softplus_lengthscale = model.meshes[mesh_idx].config.softplus_lengthscale
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
    )
    return elastic_tde, station_vels.astype("d")


def _batched_mesh_components_project_to_eigen(
    model: Model,
    rotation,
    operators: Operators,
):
    """Batched mesh components that minimize graph nodes for many meshes.

    Instead of creating separate pm.Normal/Deterministic for each mesh,
    this creates batched variables and uses block operations.

    Returns (to_velocity_blocks, coef_blocks, elastic_tde_dict) where:
    - to_velocity_blocks: list of numpy arrays for eigen_to_velocity operators
    - coef_blocks: list of pytensor tensors for station eigen coefficients
    - elastic_tde_dict: dict mapping (mesh_idx, kind) -> elastic_tde tensor
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm
    import pytensor.tensor as pt

    logger.info("Building batched mesh components")

    # First pass: collect metadata about all mesh components
    coupling_specs = []  # (mesh_idx, kind, n_eigs, eigenvectors, kinematic_op, lower, upper)
    elastic_specs = []   # (mesh_idx, kind, n_eigs, eigenvectors, lower, upper)

    kinds: tuple[Literal["strike_slip"], Literal["dip_slip"]] = ("strike_slip", "dip_slip")

    for mesh_idx, mesh in enumerate(model.meshes):
        for kind in kinds:
            if kind == "strike_slip":
                coupling_limit = mesh.config.coupling_constraints_ss
                rate_limit = mesh.config.elastic_constraints_ss
            else:
                coupling_limit = mesh.config.coupling_constraints_ds
                rate_limit = mesh.config.elastic_constraints_ds

            has_rate_limit = rate_limit.lower is not None or rate_limit.upper is not None
            has_coupling_limit = (
                coupling_limit.lower is not None or coupling_limit.upper is not None
            )

            if has_rate_limit and has_coupling_limit:
                raise ValueError(
                    f"Mesh {mesh_idx} cannot have both rate and coupling constraints "
                    f"for {kind}."
                )

            eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
            n_eigs = eigenvectors.shape[1]

            if has_coupling_limit:
                if mesh_idx not in operators.rotation_to_tri_slip_rate:
                    raise ValueError(
                        f"Mesh {mesh_idx} does not have well defined kinematic slip rates. "
                        "Coupling constraints cannot be used."
                    )
                idx = DIRECTION_IDX[kind]
                kinematic_op = operators.rotation_to_tri_slip_rate[mesh_idx][idx, :]
                coupling_specs.append((
                    mesh_idx, kind, n_eigs, eigenvectors, kinematic_op,
                    coupling_limit.lower, coupling_limit.upper
                ))
            else:
                elastic_specs.append((
                    mesh_idx, kind, n_eigs, eigenvectors,
                    rate_limit.lower, rate_limit.upper
                ))

    # Compute scale for elastic components (same logic as before)
    scale = 0.0
    for op in operators.eigen.eigen_to_velocities.values():
        scale += (op**2).mean()
    scale = scale / len(operators.eigen.eigen_to_velocities)
    scale = 1 / np.sqrt(scale)

    # Results to collect
    to_velocity_blocks: list[np.ndarray] = []
    coef_blocks = []
    # Dict to store elastic_tde by (mesh_idx, kind) for later deterministic creation
    elastic_tde_dict: dict[tuple[int, Literal["strike_slip", "dip_slip"]], object] = {}

    # --- Process COUPLING components in batch ---
    if coupling_specs:
        total_coupling_coefs = sum(spec[2] for spec in coupling_specs)

        # Single batched Normal for all coupling coefficients
        all_coupling_coefs = pm.Normal(
            "coupling_coefs_all", mu=0, sigma=10, shape=total_coupling_coefs
        )

        # Build block-diagonal eigenvector matrix for coupling -> field projection
        coupling_eigenvectors = [spec[3] for spec in coupling_specs]
        coupling_block_diag = linalg.block_diag(*coupling_eigenvectors).astype("f", order="F")

        # Stack kinematic operators for batched computation
        kinematic_ops = np.vstack([spec[4] for spec in coupling_specs]).astype("f", order="F")
        all_kinematic = _operator_mult(kinematic_ops, rotation)

        # All coupling fields at once
        all_coupling_fields_raw = _operator_mult(coupling_block_diag, all_coupling_coefs)

        # Check if all coupling specs have the same bounds - if so, apply constraint once
        bounds_set = {(spec[5], spec[6]) for spec in coupling_specs}  # (lower, upper)
        if len(bounds_set) == 1:
            # All same bounds - apply constraint to entire array at once
            lower, upper = bounds_set.pop()
            all_coupling_fields = _constrain_field(all_coupling_fields_raw, lower, upper)
            
            # Compute all elastic_tde at once: element-wise kinematic * coupling
            all_coupling_elastic_tde = all_kinematic * all_coupling_fields
            
            # Build eigenvectors_T block diagonal for projection
            coupling_eigenvectors_T = [spec[3].T for spec in coupling_specs]
            coupling_block_diag_T = linalg.block_diag(*coupling_eigenvectors_T).astype("f", order="F")
            
            # Batch projection: all station eigen coefs at once
            all_coupling_station_coefs = _operator_mult(coupling_block_diag_T, all_coupling_elastic_tde)
            
            # Build to_velocity by concatenating numpy arrays (not traced)
            coupling_to_velocity = np.concatenate([
                _get_eigen_to_velocity(model, spec[0], spec[1], operators)
                for spec in coupling_specs
            ], axis=1)
            
            to_velocity_blocks.append(coupling_to_velocity)
            coef_blocks.append(all_coupling_station_coefs.astype("d"))
            
            # Only slice for meshes that have TDE constraints
            field_offset = 0
            for mesh_idx, kind, n_eigs, eigenvectors, kinematic_op, _, _ in coupling_specs:
                n_tde = eigenvectors.shape[0]
                mesh = model.meshes[mesh_idx]
                if (mesh.config.top_slip_rate_constraint == 1 or
                    mesh.config.bot_slip_rate_constraint == 1 or
                    mesh.config.side_slip_rate_constraint == 1):
                    elastic_tde_dict[(mesh_idx, kind)] = all_coupling_elastic_tde[field_offset:field_offset + n_tde]
                field_offset += n_tde
            
            # Store for deterministic
            pm.Deterministic("elastic_tde_coupling_all", all_coupling_elastic_tde)
        else:
            # Different bounds - need per-component constraint application
            logger.warning(
                f"Coupling components have {len(bounds_set)} different bound configurations. "
                "This will create more graph nodes."
            )
            
            # Precompute offsets
            field_offsets = []
            field_offset = 0
            for spec in coupling_specs:
                field_offsets.append(field_offset)
                field_offset += spec[3].shape[0]  # n_tde

            # Apply constraints and compute elastic_tde for each component
            coupling_elastic_tdes = []
            coupling_eigenvectors_T = []

            for i, (mesh_idx, kind, n_eigs, eigenvectors, kinematic_op, lower, upper) in enumerate(coupling_specs):
                n_tde = eigenvectors.shape[0]

                # Extract this component's coupling field
                coupling_field_raw = all_coupling_fields_raw[field_offsets[i]:field_offsets[i] + n_tde]
                coupling_field = _constrain_field(coupling_field_raw, lower, upper)

                # Extract kinematic from batched result
                kinematic = all_kinematic[field_offsets[i]:field_offsets[i] + n_tde]

                # Elastic = kinematic * coupling
                elastic_tde = kinematic * coupling_field

                # Store for later batched deterministic
                elastic_tde_dict[(mesh_idx, kind)] = elastic_tde

                coupling_elastic_tdes.append(elastic_tde)
                coupling_eigenvectors_T.append(eigenvectors.T)

            # Batch the E.T @ elastic_tde projections using block diagonal
            coupling_block_diag_T = linalg.block_diag(*coupling_eigenvectors_T).astype("f", order="F")
            all_coupling_tde = pt.concatenate(coupling_elastic_tdes, axis=0)
            all_coupling_station_coefs = _operator_mult(coupling_block_diag_T, all_coupling_tde)

            # Create deterministic for trace output
            pm.Deterministic("elastic_tde_coupling_all", all_coupling_tde)

            # Build to_velocity by concatenating numpy arrays (not traced)
            coupling_to_velocity = np.concatenate([
                _get_eigen_to_velocity(model, spec[0], spec[1], operators)
                for spec in coupling_specs
            ], axis=1)

            to_velocity_blocks.append(coupling_to_velocity)
            coef_blocks.append(all_coupling_station_coefs.astype("d"))

    # --- Process ELASTIC components in batch ---
    # Track where elastic data lives for deterministic creation
    elastic_tde_tensor = None
    
    if elastic_specs:
        total_elastic_coefs = sum(spec[2] for spec in elastic_specs)

        # Single batched Normal for all elastic coefficients
        all_elastic_raw = pm.Normal("elastic_eigen_raw_all", shape=total_elastic_coefs)
        all_elastic_param = scale * all_elastic_raw

        # Check if all elastic specs are unconstrained
        all_unconstrained = all(
            spec[4] is None and spec[5] is None for spec in elastic_specs
        )

        # Build block diagonal eigenvector matrix
        elastic_eigenvectors = [spec[3] for spec in elastic_specs]
        elastic_block_diag = linalg.block_diag(*elastic_eigenvectors).astype("f", order="F")

        # All elastic fields at once
        all_elastic_fields = _operator_mult(elastic_block_diag, all_elastic_param)

        if all_unconstrained:
            # Fast path: no constraints, station_coefs = param directly
            elastic_tde_tensor = all_elastic_fields
            
            # Station velocity coefs: for unconstrained, E.T @ E @ param = param
            # So we just use all_elastic_param directly
            elastic_station_coefs = all_elastic_param
            
            # Build to_velocity operator by concatenating (numpy, not traced)
            elastic_to_velocity = np.concatenate([
                _get_eigen_to_velocity(model, spec[0], spec[1], operators)
                for spec in elastic_specs
            ], axis=1)
            
            to_velocity_blocks.append(elastic_to_velocity)
            coef_blocks.append(elastic_station_coefs.astype("d"))
            
            # Populate dict for constraints (only slice if there are TDE constraints)
            # Check if any elastic spec mesh has constraints
            needs_constraints = False
            for mesh_idx, kind, _, _, _, _ in elastic_specs:
                mesh = model.meshes[mesh_idx]
                if (mesh.config.top_slip_rate_constraint == 1 or
                    mesh.config.bot_slip_rate_constraint == 1 or
                    mesh.config.side_slip_rate_constraint == 1):
                    needs_constraints = True
                    break
            
            if needs_constraints:
                # Only then do we need to slice for constraints
                field_offset = 0
                for mesh_idx, kind, n_eigs, eigenvectors, lower, upper in elastic_specs:
                    n_tde = eigenvectors.shape[0]
                    elastic_tde_dict[(mesh_idx, kind)] = all_elastic_fields[field_offset:field_offset + n_tde]
                    field_offset += n_tde
        else:
            # Some have constraints - need per-component handling
            # Check if all constrained specs have the same bounds
            constrained_bounds = {(spec[4], spec[5]) for spec in elastic_specs if spec[4] is not None or spec[5] is not None}
            
            if len(constrained_bounds) == 1:
                # All same bounds - apply constraint to entire array
                lower, upper = constrained_bounds.pop()
                all_elastic_fields = _constrain_field(all_elastic_fields, lower, upper)
                elastic_tde_tensor = all_elastic_fields
                
                # Need to project back: E.T @ elastic_fields
                elastic_eigenvectors_T = [spec[3].T for spec in elastic_specs]
                elastic_block_diag_T = linalg.block_diag(*elastic_eigenvectors_T).astype("f", order="F")
                elastic_station_coefs = _operator_mult(elastic_block_diag_T, all_elastic_fields)
                
                elastic_to_velocity = np.concatenate([
                    _get_eigen_to_velocity(model, spec[0], spec[1], operators)
                    for spec in elastic_specs
                ], axis=1)
                
                to_velocity_blocks.append(elastic_to_velocity)
                coef_blocks.append(elastic_station_coefs.astype("d"))
                
                # Populate dict for constraints if needed
                field_offset = 0
                for mesh_idx, kind, n_eigs, eigenvectors, lower, upper in elastic_specs:
                    n_tde = eigenvectors.shape[0]
                    mesh = model.meshes[mesh_idx]
                    if (mesh.config.top_slip_rate_constraint == 1 or
                        mesh.config.bot_slip_rate_constraint == 1 or
                        mesh.config.side_slip_rate_constraint == 1):
                        elastic_tde_dict[(mesh_idx, kind)] = all_elastic_fields[field_offset:field_offset + n_tde]
                    field_offset += n_tde
            else:
                # Different bounds - fall back to per-component (rare case)
                logger.warning(
                    f"Elastic components have {len(constrained_bounds)} different bound configurations."
                )
                field_offset = 0
                param_offset = 0
                elastic_tdes_list = []
                
                for mesh_idx, kind, n_eigs, eigenvectors, lower, upper in elastic_specs:
                    n_tde = eigenvectors.shape[0]
                    elastic_field_raw = all_elastic_fields[field_offset:field_offset + n_tde]
                    
                    if lower is None and upper is None:
                        elastic_tde = elastic_field_raw
                        station_coefs = all_elastic_param[param_offset:param_offset + n_eigs]
                    else:
                        elastic_tde = _constrain_field(elastic_field_raw, lower, upper)
                        station_coefs = _operator_mult(eigenvectors.T, elastic_tde)
                    
                    elastic_tdes_list.append(elastic_tde)
                    elastic_tde_dict[(mesh_idx, kind)] = elastic_tde
                    
                    to_velocity_blocks.append(_get_eigen_to_velocity(model, mesh_idx, kind, operators))
                    coef_blocks.append(station_coefs.astype("d"))
                    
                    field_offset += n_tde
                    param_offset += n_eigs
                
                elastic_tde_tensor = pt.concatenate(elastic_tdes_list, axis=0)

    # Create deterministic for elastic components (coupling deterministic already created above)
    if elastic_tde_tensor is not None:
        pm.Deterministic("elastic_tde_elastic_all", elastic_tde_tensor)
    
    return to_velocity_blocks, coef_blocks, elastic_tde_dict


def _elastic_component(
    model: Model,
    mesh_idx: int,
    kind: Literal["strike_slip", "dip_slip"],
    operators: Operators,
    lower: float | None,
    upper: float | None,
):
    """Model elastic slip rate as a linear combination of eigenmodes.
    Creates parameters for raw elastic eigenmode coefficients, then adds
    scaled elastic eigenmodes as deterministic variables. Also adds a
    deterministic variable for the elastic slip rate field.

    Returns the estimated elastic slip rates on the TDEs and the
    velocities at the stations due to them.
    """
    assert operators.eigen is not None
    assert operators.tde is not None

    import pymc as pm
    import pytensor.tensor as pt

    kind_short = {"strike_slip": "ss", "dip_slip": "ds"}[kind]
    DIRECTION_IDX[kind]

    scale = 0.0
    for op in operators.eigen.eigen_to_velocities.values():
        scale += (op**2).mean()

    scale = scale / len(operators.eigen.eigen_to_velocities)
    scale = 1 / np.sqrt(scale)

    eigenvectors = _get_eigenmodes(model, mesh_idx, kind)
    to_velocity = _get_eigen_to_velocity(
        model,
        mesh_idx,
        kind,
        operators,
    )
    n_eigs = eigenvectors.shape[1]

    raw = pm.Normal(f"elastic_eigen_raw_{mesh_idx}_{kind_short}", shape=n_eigs)
    param = pm.Deterministic(f"elastic_eigen_{mesh_idx}_{kind_short}", scale * raw)
    softplus_lengthscale = model.meshes[mesh_idx].config.softplus_lengthscale
    elastic_tde = _constrain_field(
        _operator_mult(eigenvectors, param), lower, upper, softplus_lengthscale
    )
    pm.Deterministic(f"elastic_{mesh_idx}_{kind_short}", elastic_tde)

    # Compute elastic velocity at stations. The operator already
    # includes a negative sign.
    if lower is None and upper is None:
        station_vels = _operator_mult(to_velocity, param)
        # We need to return a station velocity for all three components,
        # not just north and east.
        station_vels = pt.concatenate(
            [
                station_vels.reshape((len(model.station), 2)),
                np.zeros((len(model.station), 1)),
            ],
            axis=-1,
        ).ravel()  # type: ignore[attr-defined]
    else:
        station_vels = _station_vel_from_elastic_mesh(
            model,
            mesh_idx,
            kind,
            elastic_tde,
            operators,
        )

    return elastic_tde, station_vels


def _mesh_component(
    model: Model,
    mesh_idx: int,
    rotation,
    operators: Operators,
):
    rates = []

    kinds: tuple[Literal["strike_slip"], Literal["dip_slip"]] = (
        "strike_slip",
        "dip_slip",
    )
    for kind in kinds:
        if kind == "strike_slip":
            coupling_limit = model.meshes[mesh_idx].config.coupling_constraints_ss
            rate_limit = model.meshes[mesh_idx].config.elastic_constraints_ss
        else:
            coupling_limit = model.meshes[mesh_idx].config.coupling_constraints_ds
            rate_limit = model.meshes[mesh_idx].config.elastic_constraints_ds

        has_rate_limit = rate_limit.lower is not None or rate_limit.upper is not None
        has_coupling_limit = (
            coupling_limit.lower is not None or coupling_limit.upper is not None
        )

        if has_rate_limit and has_coupling_limit:
            raise ValueError(
                f"Mesh {mesh_idx} cannot have both rate and coupling constraints "
                f"for {kind}."
            )

        if has_coupling_limit:
            elastic_tde, station_vels = _coupling_component(
                model,
                mesh_idx,
                kind,
                rotation,
                operators,
                lower=coupling_limit.lower,
                upper=coupling_limit.upper,
            )
        else:
            elastic_tde, station_vels = _elastic_component(
                model,
                mesh_idx,
                kind,
                operators,
                lower=rate_limit.lower,
                upper=rate_limit.upper,
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


def _add_block_strain_rate_component(operators: Operators):
    """Add block strain rate component to the PyMC model.

    Returns the velocity contribution from block strain rates.
    """
    import pymc as pm

    raw = pm.Normal("block_strain_rate_raw", sigma=100, dims="block_strain_rate_param")
    if operators.block_strain_rate_to_velocities.size == 0:
        scale = 1.0
    else:
        scale = 1 / np.sqrt((operators.block_strain_rate_to_velocities**2).mean())
    block_strain_rate = pm.Deterministic(
        "block_strain_rate", scale * raw, dims="block_strain_rate_param"
    )

    return _operator_mult(operators.block_strain_rate_to_velocities, block_strain_rate)


def _add_rotation_component(operators: Operators):
    """Add block rotation component to the PyMC model.

    Returns rotation parameters and velocity contributions.
    """
    import pymc as pm

    A = (
        operators.rotation_to_velocities
        - operators.rotation_to_slip_rate_to_okada_to_velocities
    )
    scale = 1e6
    B = A / scale
    u, s, vh = linalg.svd(B, full_matrices=False)
    raw = pm.StudentT("rotation_raw", sigma=20, nu=4, dims="rotation_param")

    rotation = pm.Deterministic(
        "rotation", _operator_mult(vh.T, raw / scale), dims="rotation_param"
    )

    rotation_velocity = _operator_mult(operators.rotation_to_velocities, rotation)
    rotation_okada_velocity = _operator_mult(
        -operators.rotation_to_slip_rate_to_okada_to_velocities, rotation
    )

    return rotation, rotation_velocity, rotation_okada_velocity


def _add_mogi_component(operators: Operators):
    """Add Mogi source component to the PyMC model.

    Returns the velocity contribution from Mogi sources.
    """
    import pymc as pm

    raw = pm.Normal("mogi_raw", dims="mogi_param")
    if operators.mogi_to_velocities.size == 0:
        scale = 1.0
    else:
        scale = 1 / np.sqrt((operators.mogi_to_velocities**2).mean())
    mogi = pm.Deterministic("mogi", scale * raw, dims="mogi_param")

    return _operator_mult(operators.mogi_to_velocities, mogi)


def _add_station_velocity_likelihood(model: Model, mu):
    """Add station velocity likelihood to the PyMC model.

    Uses area-weighted Student-t likelihood for station observations.
    """
    import pymc as pm

    sigma = pm.HalfNormal("sigma", sigma=2)
    data = np.array([model.station.east_vel, model.station.north_vel]).T

    lh_dist = pm.StudentT.dist

    def lh(value, weight, mu, sigma):
        dist = lh_dist(nu=6, mu=mu, sigma=sigma)
        return weight * pm.logp(dist, value)

    def random(weight, mu, sigma, rng=None, size=None):
        return lh_dist(nu=6, mu=mu, sigma=sigma, rng=rng, size=size)

    if model.config.mcmc_station_weighting is None:
        logger.info(f"Using unweighted station likelihood ({len(data)} stations)")
        pm.StudentT(
            "station_velocity",
            mu=mu,
            sigma=sigma,
            observed=data,
            dims=("station", "xy"),
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
            dims=("station", "xy"),
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
        block_strain_rate_velocity = _add_block_strain_rate_component(operators)

        rotation, rotation_velocity, rotation_okada_velocity = _add_rotation_component(
            operators
        )

        mogi_velocity = _add_mogi_component(operators)

        # Add elastic velocity from meshes
        if model.config.mcmc_station_velocity_method == "project_to_eigen":
            # Use batched implementation for efficiency with many meshes
            import pytensor.tensor as pt

            to_velocity_blocks, coef_blocks, elastic_tde_dict = (
                _batched_mesh_components_project_to_eigen(model, rotation, operators)
            )

            # Add TDE elastic constraints for all meshes
            for (mesh_idx, kind), elastic_tde in elastic_tde_dict.items():
                _add_tde_elastic_constraints(model, mesh_idx, elastic_tde, kind)  # type: ignore[arg-type]

            if len(to_velocity_blocks) == 0:
                elastic_velocity = pt.zeros((3 * len(model.station),), dtype="d")
            else:
                # Consolidate all mesh contributions into ONE matrix-vector product:
                #   v_xy = [G_1 ... G_M] @ [c_1; ...; c_M]
                G_all = np.concatenate(to_velocity_blocks, axis=1).astype("f", order="F")
                c_all = pt.concatenate(coef_blocks, axis=0)
                vel_xy = _operator_mult(G_all, c_all).astype("d")

                # Expand to 3 components (E,N,0) once at the end.
                elastic_velocity = pt.concatenate(
                    [
                        vel_xy.reshape((len(model.station), 2)),
                        np.zeros((len(model.station), 1)),
                    ],
                    axis=-1,
                ).ravel()  # type: ignore[attr-defined]
        else:
            elastic_velocities = []
            for mesh_idx, _ in enumerate(model.meshes):
                elastic_velocities.append(
                    _mesh_component(model, mesh_idx, rotation, operators)
                )
            elastic_velocity = sum(elastic_velocities)

        mu = (
            block_strain_rate_velocity
            + rotation_velocity
            + rotation_okada_velocity
            + mogi_velocity
            + elastic_velocity
        )
        mu = mu.reshape((len(model.station), 3))[:, :2]
        mu_det = pm.Deterministic("mu", mu, dims=("station", "xy"))
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

    if operators is None:
        # Only use streaming mode (discard_tde_to_velocities=True) when using project_to_eigen.
        # The direct and low_rank methods require the full tde_to_velocities matrices.
        use_streaming = model.config.mcmc_station_velocity_method == "project_to_eigen"
        if use_streaming:
            logger.info("Building operators with streaming mode (discard_tde_to_velocities=True)")
        else:
            logger.info(
                f"Building operators with discard_tde_to_velocities=False "
                f"(required for mcmc_station_velocity_method={model.config.mcmc_station_velocity_method!r})"
            )
        operators = build_operators(model, tde=True, eigen=True, discard_tde_to_velocities=use_streaming)
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

    kinds_map = {
        "strike_slip": ("ss", slice(None, None, 2)),
        "dip_slip": ("ds", slice(1, None, 2)),
    }

    # Check for batched deterministics (new structure)
    has_coupling = "elastic_tde_coupling_all" in trace.posterior
    has_elastic = "elastic_tde_elastic_all" in trace.posterior

    if has_coupling or has_elastic:
        # Batched mode: separate arrays for coupling and elastic components
        # First, determine which mesh/kind combos are coupling vs elastic
        coupling_specs = []
        elastic_specs = []

        for mesh_idx, mesh in enumerate(model.meshes):
            for kind in ["strike_slip", "dip_slip"]:
                if kind == "strike_slip":
                    coupling_limit = mesh.config.coupling_constraints_ss
                else:
                    coupling_limit = mesh.config.coupling_constraints_ds

                has_coupling_limit = (
                    coupling_limit.lower is not None or coupling_limit.upper is not None
                )

                evecs = mesh.eigenvectors
                if evecs is None:
                    continue
                n_tde = evecs.shape[0]

                if has_coupling_limit:
                    coupling_specs.append((mesh_idx, kind, n_tde))
                else:
                    elastic_specs.append((mesh_idx, kind, n_tde))

        # Extract from coupling array
        if has_coupling and coupling_specs:
            coupling_all = trace.posterior.elastic_tde_coupling_all.values
            offset = 0
            for mesh_idx, kind, n_tde in coupling_specs:
                _, state_idx = kinds_map[kind]
                vals = coupling_all[offset:offset + n_tde]
                offset += n_tde

                start = operators_tde.index.tde.start_tde_col[mesh_idx]
                end = operators_tde.index.tde.end_tde_col[mesh_idx]
                state_vector[start:end][state_idx] = vals

        # Extract from elastic array
        if has_elastic and elastic_specs:
            elastic_all = trace.posterior.elastic_tde_elastic_all.values
            offset = 0
            for mesh_idx, kind, n_tde in elastic_specs:
                _, state_idx = kinds_map[kind]
                vals = elastic_all[offset:offset + n_tde]
                offset += n_tde

                start = operators_tde.index.tde.start_tde_col[mesh_idx]
                end = operators_tde.index.tde.end_tde_col[mesh_idx]
                state_vector[start:end][state_idx] = vals

    elif "elastic_tde_all" in trace.posterior:
        # Old batched mode (single concatenated array)
        elastic_tde_all = trace.posterior.elastic_tde_all.values

        tde_offset = 0
        for mesh_idx in range(len(model.meshes)):
            mesh = model.meshes[mesh_idx]
            for kind in ["strike_slip", "dip_slip"]:
                _, state_idx = kinds_map[kind]

                evecs = mesh.eigenvectors
                if evecs is None:
                    continue
                n_tde = evecs.shape[0]

                vals = elastic_tde_all[tde_offset:tde_offset + n_tde]
                tde_offset += n_tde

                start = operators_tde.index.tde.start_tde_col[mesh_idx]
                end = operators_tde.index.tde.end_tde_col[mesh_idx]
                state_vector[start:end][state_idx] = vals
    else:
        # Legacy per-mesh mode
        for mesh_idx in range(len(model.meshes)):
            indices = {
                "ss": slice(None, None, 2),
                "ds": slice(1, None, 2),
            }
            for name, idx in indices.items():
                start = operators_tde.index.tde.start_tde_col[mesh_idx]
                end = operators_tde.index.tde.end_tde_col[mesh_idx]
                var_name = f"elastic_{mesh_idx}_{name}"

                if var_name in trace.posterior:
                    vals = trace.posterior[var_name]
                    if vals.shape == state_vector[start:end].shape:
                        state_vector[start:end] = vals
                    else:
                        state_vector[start:end][idx] = trace.posterior[var_name].values
    return state_vector
