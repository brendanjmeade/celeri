from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from celeri.celeri_util import interleave2
from celeri.mesh import Mesh
from celeri.model import Model
from celeri.operators import Operators, get_qp_all_inequality_operator_and_data_vector
from celeri.output import write_output
from celeri.solve import Estimation, lsqlin_qp


@dataclass
class SqpEstimation(Estimation):
    n_out_of_bounds_trace: np.ndarray
    trace: dict[str, list[np.ndarray]] | None


def _presolve(
    model: Model, operators: Operators, *, show_progress: bool = False
) -> SqpEstimation:
    n_segment_meshes = np.max(model.segment.patch_file_name).astype(int) + 1

    # Get QP bounds as inequality constraints
    qp_inequality_constraints_matrix, qp_inequality_constraints_data_vector = (
        get_qp_all_inequality_operator_and_data_vector(
            model, operators, operators.index
        )
    )

    # QP solve
    opts = {"show_progress": show_progress}

    solution_qp = lsqlin_qp(
        operators.full_dense_operator * np.sqrt(operators.weighting_vector[:, None]),
        operators.data_vector * np.sqrt(operators.weighting_vector),
        0,
        qp_inequality_constraints_matrix,  # Inequality matrix
        qp_inequality_constraints_data_vector,  # Inequality data vector
        None,
        None,
        None,
        None,
        None,
        opts,
    )

    estimation_qp = SqpEstimation(
        data_vector=operators.data_vector,
        weighting_vector=operators.weighting_vector,
        operator=operators.full_dense_operator,
        state_vector=np.array(solution_qp["x"]).flatten(),
        model=model,
        operators=operators,
        state_covariance_matrix=None,
        n_out_of_bounds_trace=np.zeros((n_segment_meshes, 0)),
        trace=None,
    )

    return estimation_qp


def _get_coupling_linear(
    estimated_slip: np.ndarray,
    kinematic_slip: np.ndarray,
    operators: Operators,
    mesh_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert operators.eigen is not None

    # Smooth kinematic slip
    kinematic_slip = (
        operators.eigen.linear_gaussian_smoothing[mesh_idx] @ kinematic_slip
    )

    # Calculate coupling
    coupling = estimated_slip / kinematic_slip
    return coupling, kinematic_slip


@dataclass
class _SlipRateBounds:
    ss_lower: np.ndarray
    ss_upper: np.ndarray
    ds_lower: np.ndarray
    ds_upper: np.ndarray


def _update_slip_rate_bounds(
    meshes: list[Mesh],
    mesh_idx: int,
    tde_coupling_ss: np.ndarray,
    tde_coupling_ds: np.ndarray,
    kinematic_tde_rates_ss: np.ndarray,
    kinematic_tde_rates_ds: np.ndarray,
    current_bounds: _SlipRateBounds,
) -> tuple[int, _SlipRateBounds]:
    """Update slip rate bounds based on coupling constraints.

    For points outside the coupling bounds, adjust the slip rate bounds
    to move toward a midpoint coupling value.
    """
    mesh_config = meshes[mesh_idx].config

    # Find indices where coupling is out of bounds (OOB)
    ss_lower_bound = mesh_config.qp_mesh_tde_slip_rate_lower_bound_ss_coupling
    ss_upper_bound = mesh_config.qp_mesh_tde_slip_rate_upper_bound_ss_coupling
    ds_lower_bound = mesh_config.qp_mesh_tde_slip_rate_lower_bound_ds_coupling
    ds_upper_bound = mesh_config.qp_mesh_tde_slip_rate_upper_bound_ds_coupling

    # Identify out-of-bounds indices
    ss_lower_oob_idx = np.where(tde_coupling_ss < ss_lower_bound)[0]
    ss_upper_oob_idx = np.where(tde_coupling_ss > ss_upper_bound)[0]
    ds_lower_oob_idx = np.where(tde_coupling_ds < ds_lower_bound)[0]
    ds_upper_oob_idx = np.where(tde_coupling_ds > ds_upper_bound)[0]

    # Separate indices by kinematic rate sign
    neg_ss_idx = np.where(kinematic_tde_rates_ss < 0)[0]
    pos_ss_idx = np.where(kinematic_tde_rates_ss >= 0)[0]
    neg_ds_idx = np.where(kinematic_tde_rates_ds < 0)[0]
    pos_ds_idx = np.where(kinematic_tde_rates_ds >= 0)[0]

    # Calculate intersections for both positive and negative kinematic rates
    # Strike-slip intersections
    ss_lower_oob_neg = np.intersect1d(ss_lower_oob_idx, neg_ss_idx)
    ss_upper_oob_neg = np.intersect1d(ss_upper_oob_idx, neg_ss_idx)
    ss_lower_oob_pos = np.intersect1d(ss_lower_oob_idx, pos_ss_idx)
    ss_upper_oob_pos = np.intersect1d(ss_upper_oob_idx, pos_ss_idx)

    # Dip-slip intersections
    ds_lower_oob_neg = np.intersect1d(ds_lower_oob_idx, neg_ds_idx)
    ds_upper_oob_neg = np.intersect1d(ds_upper_oob_idx, neg_ds_idx)
    ds_lower_oob_pos = np.intersect1d(ds_lower_oob_idx, pos_ds_idx)
    ds_upper_oob_pos = np.intersect1d(ds_upper_oob_idx, pos_ds_idx)

    # Calculate total number of OOB coupling constraints
    n_oob = (
        len(ss_lower_oob_idx)
        + len(ss_upper_oob_idx)
        + len(ds_lower_oob_idx)
        + len(ds_upper_oob_idx)
    )

    # Start with copies of current bounds
    updated_ss_bounds_lower = np.copy(current_bounds.ss_lower)
    updated_ss_bounds_upper = np.copy(current_bounds.ss_upper)
    updated_ds_bounds_lower = np.copy(current_bounds.ds_lower)
    updated_ds_bounds_upper = np.copy(current_bounds.ds_upper)

    # Calculate midpoint coupling values
    mid_point_ss_coupling = 0.5 * (ss_lower_bound + ss_upper_bound)
    mid_point_ds_coupling = 0.5 * (ds_lower_bound + ds_upper_bound)

    mid_point_ss_rate = mid_point_ss_coupling * kinematic_tde_rates_ss
    mid_point_ds_rate = mid_point_ds_coupling * kinematic_tde_rates_ds

    reduction_factor = mesh_config.iterative_coupling_linear_slip_rate_reduction_factor

    # Calculate new bounds with linear approach towards midpoint
    new_ss_bounds_lower = current_bounds.ss_lower + reduction_factor * (
        mid_point_ss_rate - current_bounds.ss_lower
    )
    new_ss_bounds_upper = current_bounds.ss_upper + reduction_factor * (
        mid_point_ss_rate - current_bounds.ss_upper
    )
    new_ds_bounds_lower = current_bounds.ds_lower + reduction_factor * (
        mid_point_ds_rate - current_bounds.ds_lower
    )
    new_ds_bounds_upper = current_bounds.ds_upper + reduction_factor * (
        mid_point_ds_rate - current_bounds.ds_upper
    )

    # Update bounds for out-of-bounds points
    # Note: For negative kinematic rates, upper and lower bounds are swapped

    # Negative kinematic case updates
    updated_ss_bounds_lower[ss_upper_oob_neg] = new_ss_bounds_lower[ss_upper_oob_neg]
    updated_ss_bounds_upper[ss_lower_oob_neg] = new_ss_bounds_upper[ss_lower_oob_neg]
    updated_ds_bounds_lower[ds_upper_oob_neg] = new_ds_bounds_lower[ds_upper_oob_neg]
    updated_ds_bounds_upper[ds_lower_oob_neg] = new_ds_bounds_upper[ds_lower_oob_neg]

    # Positive kinematic case updates
    updated_ss_bounds_lower[ss_lower_oob_pos] = new_ss_bounds_lower[ss_lower_oob_pos]
    updated_ss_bounds_upper[ss_upper_oob_pos] = new_ss_bounds_upper[ss_upper_oob_pos]
    updated_ds_bounds_lower[ds_lower_oob_pos] = new_ds_bounds_lower[ds_lower_oob_pos]
    updated_ds_bounds_upper[ds_upper_oob_pos] = new_ds_bounds_upper[ds_upper_oob_pos]

    updated_bounds = _SlipRateBounds(
        ss_lower=updated_ss_bounds_lower,
        ss_upper=updated_ss_bounds_upper,
        ds_lower=updated_ds_bounds_lower,
        ds_upper=updated_ds_bounds_upper,
    )

    return n_oob, updated_bounds


def _check_coupling_bounds_single_mesh(
    operators: Operators,
    block: pd.DataFrame,
    meshes: list[Mesh],
    mesh_idx: int,
    estimation_qp: Estimation,
    current_bounds: _SlipRateBounds,
) -> tuple[
    _SlipRateBounds,  # Updated bounds
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,  # Rates
    int,  # Number of out-of-bounds points
]:
    """Check coupling bounds for a single mesh and update slip rate bounds accordingly.

    Returns updated bounds, kinematic and estimated rates, and count of out-of-bounds points.
    """
    index = operators.index
    assert index.eigen is not None
    assert operators.eigen is not None

    block_size = 3 * len(block)
    eigen_start = index.eigen.start_col_eigen[mesh_idx]
    eigen_end = index.eigen.end_col_eigen[mesh_idx]

    kinematic_tde_rates = (
        operators.rotation_to_tri_slip_rate[mesh_idx]
        @ estimation_qp.state_vector[0:block_size]
    )

    estimated_tde_rates = (
        operators.eigen.eigenvectors_to_tde_slip[mesh_idx]
        @ estimation_qp.state_vector[eigen_start:eigen_end]
    )

    # Separate strike-slip and dip-slip components
    ss_indices = slice(0, None, 2)
    ds_indices = slice(1, None, 2)

    # Calculate coupling and get smoothed kinematic rates
    tde_coupling_ss, kinematic_tde_rates_ss_smooth = _get_coupling_linear(
        estimated_tde_rates[ss_indices],
        kinematic_tde_rates[ss_indices],
        operators,
        mesh_idx,
    )

    tde_coupling_ds, kinematic_tde_rates_ds_smooth = _get_coupling_linear(
        estimated_tde_rates[ds_indices],
        kinematic_tde_rates[ds_indices],
        operators,
        mesh_idx,
    )

    # Update slip rate bounds based on coupling constraints
    n_oob, updated_bounds = _update_slip_rate_bounds(
        meshes,
        mesh_idx,
        tde_coupling_ss,
        tde_coupling_ds,
        kinematic_tde_rates_ss_smooth,
        kinematic_tde_rates_ds_smooth,
        current_bounds,
    )

    return (
        updated_bounds,
        kinematic_tde_rates_ss_smooth,
        kinematic_tde_rates_ds_smooth,
        estimated_tde_rates[ss_indices],
        estimated_tde_rates[ds_indices],
        n_oob,
    )


def solve_sqp(
    model: Model,
    operators: Operators,
    *,
    max_iter: int | None = None,
    percentage_satisfied_target: float | None = None,
) -> SqpEstimation:
    """Solve the sequential quadratic programming problem with coupling constraints.

    Iteratively adjusts slip rate bounds to satisfy coupling constraints.
    """
    # Validate prerequisites
    if operators.eigen is None:
        raise ValueError(
            "Operators must have eigenvectors defined for coupling bounds."
        )

    index = operators.index
    if index.eigen is None:
        raise ValueError("Operators must have eigen index defined.")

    assert index.tde is not None

    # Set default values from model config if not provided
    max_iter = max_iter or model.config.coupling_bounds_max_iter
    percentage_satisfied_target = (
        percentage_satisfied_target
        or model.config.coupling_bounds_total_percentage_satisfied_target
    )

    if max_iter is None:
        raise ValueError("Maximum number of iterations must be defined.")
    if percentage_satisfied_target is None:
        raise ValueError("Percentage satisfied target must be defined.")

    # Initial solve to get starting point
    estimation_qp = _presolve(model, operators)

    # Get QP inequality constraints
    qp_inequality_matrix, qp_inequality_data_vector = (
        get_qp_all_inequality_operator_and_data_vector(
            model, operators, operators.index
        )
    )

    meshes = model.meshes
    config = model.config
    n_segment_meshes = np.max(model.segment.patch_file_name).astype(int) + 1

    # Initialize bounds for each mesh
    slip_rate_bounds = [None] * n_segment_meshes

    # Initialize storage arrays for each mesh
    optimizer_trace = {
        key: [None] * n_segment_meshes
        for key in [
            "ss_lower",
            "ss_upper",
            "ds_lower",
            "ds_upper",
            "ss_kinematic",
            "ss_estimated",
            "ds_kinematic",
            "ds_estimated",
        ]
    }

    # Set initial bounds and create storage arrays
    for i in range(n_segment_meshes):
        mesh_config = meshes[i].config
        n_tde = meshes[i].n_tde

        # Validate mesh configuration
        for bound_type in ["ss", "ds"]:
            for bound_dir in ["lower", "upper"]:
                attr = f"qp_mesh_tde_slip_rate_{bound_dir}_bound_{bound_type}"
                assert getattr(mesh_config, attr) is not None

        assert mesh_config.qp_mesh_tde_slip_rate_lower_bound_ss is not None
        assert mesh_config.qp_mesh_tde_slip_rate_upper_bound_ss is not None
        assert mesh_config.qp_mesh_tde_slip_rate_lower_bound_ds is not None
        assert mesh_config.qp_mesh_tde_slip_rate_upper_bound_ds is not None

        # Set initial bounds using the _SlipRateBounds dataclass
        slip_rate_bounds[i] = _SlipRateBounds(
            ss_lower=mesh_config.qp_mesh_tde_slip_rate_lower_bound_ss * np.ones(n_tde),
            ss_upper=mesh_config.qp_mesh_tde_slip_rate_upper_bound_ss * np.ones(n_tde),
            ds_lower=mesh_config.qp_mesh_tde_slip_rate_lower_bound_ds * np.ones(n_tde),
            ds_upper=mesh_config.qp_mesh_tde_slip_rate_upper_bound_ds * np.ones(n_tde),
        )

        # Initialize storage arrays
        shape = (n_tde, max_iter)

        for key in optimizer_trace:
            optimizer_trace[key][i] = np.zeros(shape)

    # Track out-of-bounds elements
    n_oob_vec = np.zeros((n_segment_meshes, 0))
    tde_total = sum(mesh.n_tde for mesh in meshes)
    total_percentages = []
    iteration = 0

    # Main iteration loop
    for iteration in range(max_iter):
        # Create copy of inequality constraints for this iteration
        updated_inequality_data = np.copy(qp_inequality_data_vector)
        current_noob = np.zeros((n_segment_meshes, 1))

        # Process each mesh
        for mesh_idx in range(n_segment_meshes):
            current_bounds = slip_rate_bounds[mesh_idx]

            # Check coupling bounds and get updated bounds
            (
                updated_bounds,
                kinematic_ss,
                kinematic_ds,
                estimated_ss,
                estimated_ds,
                n_oob,
            ) = _check_coupling_bounds_single_mesh(
                operators,
                model.block,
                meshes,
                mesh_idx,
                estimation_qp,
                current_bounds,
            )

            logger.info(f"Iteration: {iteration}, Mesh: {mesh_idx}, NOOB: {n_oob}")
            current_noob[mesh_idx, 0] = n_oob

            # Update inequality constraints
            updated_lower_bounds = -1.0 * interleave2(
                updated_bounds.ss_lower, updated_bounds.ds_lower
            )
            updated_upper_bounds = interleave2(
                updated_bounds.ss_upper, updated_bounds.ds_upper
            )

            # Lower bounds
            lower_start = index.eigen.qp_constraint_tde_rate_start_row_eigen[mesh_idx]
            lower_end = lower_start + 2 * index.tde.n_tde[mesh_idx]
            updated_inequality_data[lower_start:lower_end] = updated_lower_bounds

            # Upper bounds
            upper_start = lower_end
            upper_end = index.eigen.qp_constraint_tde_rate_end_row_eigen[mesh_idx]
            updated_inequality_data[upper_start:upper_end] = updated_upper_bounds

            # Update current bounds for next iteration
            slip_rate_bounds[mesh_idx] = updated_bounds

            # Store values for visualization and debugging
            optimizer_trace["ss_lower"][mesh_idx][:, iteration] = (
                updated_bounds.ss_lower
            )
            optimizer_trace["ss_upper"][mesh_idx][:, iteration] = (
                updated_bounds.ss_upper
            )
            optimizer_trace["ds_lower"][mesh_idx][:, iteration] = (
                updated_bounds.ds_lower
            )
            optimizer_trace["ds_upper"][mesh_idx][:, iteration] = (
                updated_bounds.ds_upper
            )
            optimizer_trace["ss_estimated"][mesh_idx][:, iteration] = estimated_ss
            optimizer_trace["ds_estimated"][mesh_idx][:, iteration] = estimated_ds
            optimizer_trace["ss_kinematic"][mesh_idx][:, iteration] = kinematic_ss
            optimizer_trace["ds_kinematic"][mesh_idx][:, iteration] = kinematic_ds

        # Update out-of-bounds tracking
        n_oob_vec = np.hstack((n_oob_vec, current_noob))

        # Solve QP with updated constraints
        weighted_operator = operators.full_dense_operator * np.sqrt(
            operators.weighting_vector[:, None]
        )
        weighted_data = operators.data_vector * np.sqrt(operators.weighting_vector)

        solution_qp = lsqlin_qp(
            weighted_operator,
            weighted_data,
            0,
            qp_inequality_matrix,
            updated_inequality_data,
            None,
            None,
            None,
            None,
            None,
            {"show_progress": False},
        )

        if solution_qp["status"] != "optimal":
            logger.error(f"NON OPTIMAL SOLUTION AT: iteration={iteration}")
            raise ValueError("Solver did not converge")

        # Create estimation object with updated solution
        estimation_qp = SqpEstimation(
            data_vector=operators.data_vector,
            weighting_vector=operators.weighting_vector,
            operator=operators.full_dense_operator,
            state_vector=np.array(solution_qp["x"]).flatten(),
            model=model,
            operators=operators,
            state_covariance_matrix=None,
            n_out_of_bounds_trace=n_oob_vec.copy(),
            trace=None,
        )

        # Check convergence
        total_oob = np.sum(current_noob)
        percent_oob = total_oob / (2 * tde_total) * 100
        percent_satisfied = 100 - percent_oob
        total_percentages.append(percent_oob)

        logger.info(
            f"Iteration: {iteration}, Total %TDE inside coupling bounds: {percent_satisfied:0.3f}"
        )

        # Check if we've reached the target percentage of satisfied constraints
        if percent_satisfied >= percentage_satisfied_target:
            break
    else:
        # This block is executed if the loop completes without breaking
        logger.info(
            f"Maximum iterations ({max_iter}) reached without meeting target percentage."
        )

    # Write output
    write_output(
        config, estimation_qp, model.station, model.segment, model.block, meshes
    )

    # Trim storage arrays to actual number of iterations
    for mesh_idx in range(n_segment_meshes):
        for key in optimizer_trace:
            optimizer_trace[key][mesh_idx] = optimizer_trace[key][mesh_idx][
                :, 0 : iteration + 1
            ]

    estimation_qp.trace = optimizer_trace
    return estimation_qp


def plot_iterative_convergence(estimation: SqpEstimation, *, plot_in_bounds: bool = False):
    """Plot convergence of out-of-bounds and in-bounds percentages during SQP iterations."""
    meshes = estimation.model.meshes
    n_oob_vec = estimation.n_out_of_bounds_trace
    n_meshes, n_iter = n_oob_vec.shape

    assert len(meshes) >= n_meshes
    mesh_names = [mesh.name for mesh in meshes[:n_meshes]]

    # Calculate total mesh elements
    tde_total = sum(mesh.n_tde for mesh in meshes[:n_meshes])

    # Calculate percentages
    total_oob = np.sum(n_oob_vec, axis=0)
    total_percentages_oob = total_oob / (2 * tde_total) * 100
    total_percentages_ib = 100 - total_percentages_oob
    iterations = np.arange(len(total_percentages_oob))

    # Common figure settings
    fig_size = (4, 3)
    legend_kwargs = {
        "fancybox": False,
        "framealpha": 1,
        "facecolor": "white",
        "edgecolor": "black",
    }

    plt.figure(figsize=fig_size)

    # Choose appropriate data and labels
    total_percentages = total_percentages_ib if plot_in_bounds else total_percentages_oob
    ylabel = "% IB" if plot_in_bounds else "% OOB"

    # Fill total area
    plt.fill_between(
        iterations,
        total_percentages,
        color="lightgray",
        label=f"total ({total_percentages[-1]:0.2f}%)",
    )

    # Plot individual mesh lines
    for i, mesh_name in enumerate(mesh_names):
        mesh_oob = n_oob_vec[i, :] / (2 * meshes[i].n_tde) * 100
        percentages = 100 - mesh_oob if plot_in_bounds else mesh_oob
        plt.plot(
            iterations,
            percentages,
            linewidth=1.0,
            label=f"{mesh_name} ({percentages[-1]:0.2f}%)",
        )

    # Formatting
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.xlim([0, n_iter - 1])
    plt.ylim([0, 100])
    plt.xticks([0, n_iter - 1])
    plt.yticks([0, 100])
    legend_handle = plt.legend(**legend_kwargs)
    legend_handle.get_frame().set_linewidth(0.5)
