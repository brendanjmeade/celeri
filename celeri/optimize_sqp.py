import numpy as np
from loguru import logger

from celeri.celeri_util import interleave2
from celeri.model import Model
from celeri.operators import Operators, get_qp_all_inequality_operator_and_data_vector
from celeri.output import write_output
from celeri.solve import Estimation, lsqlin_qp


def _presolve(model: Model, operators: Operators, *, show_progress: bool = False):
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

    estimation_qp = Estimation(
        data_vector=operators.data_vector,
        weighting_vector=operators.weighting_vector,
        operator=operators.full_dense_operator,
        state_vector=np.array(solution_qp["x"]).flatten(),
        model=model,
        operators=operators,
        state_covariance_matrix=None,
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


def _update_slip_rate_bounds(
    meshes,
    mesh_idx,
    tde_coupling_ss,
    tde_coupling_ds,
    kinematic_tde_rates_ss,
    kinematic_tde_rates_ds,
    current_ss_bounds_lower,
    current_ss_bounds_upper,
    current_ds_bounds_lower,
    current_ds_bounds_upper,
):
    tde_coupling_ss_lower_oob_idx = np.where(
        tde_coupling_ss
        < meshes[mesh_idx].config.qp_mesh_tde_slip_rate_lower_bound_ss_coupling
    )[0]

    tde_coupling_ss_upper_oob_idx = np.where(
        tde_coupling_ss
        > meshes[mesh_idx].config.qp_mesh_tde_slip_rate_upper_bound_ss_coupling
    )[0]

    tde_coupling_ds_lower_oob_idx = np.where(
        tde_coupling_ds
        < meshes[mesh_idx].config.qp_mesh_tde_slip_rate_lower_bound_ds_coupling
    )[0]

    tde_coupling_ds_upper_oob_idx = np.where(
        tde_coupling_ds
        > meshes[mesh_idx].config.qp_mesh_tde_slip_rate_upper_bound_ds_coupling
    )[0]

    # Find indices of mesh elements with negative kinematic rate
    neg_kinematic_ss_idx = np.where(kinematic_tde_rates_ss < 0)[0]
    neg_kinematic_ds_idx = np.where(kinematic_tde_rates_ds < 0)[0]
    pos_kinematic_ss_idx = np.where(kinematic_tde_rates_ss >= 0)[0]
    pos_kinematic_ds_idx = np.where(kinematic_tde_rates_ds >= 0)[0]

    # NEGATIVE CASE: Find intersection of indices with negative kinematic rates and OOB ss lower bounds
    tde_coupling_ss_lower_oob_and_neg_kinematic_ss = np.intersect1d(
        tde_coupling_ss_lower_oob_idx, neg_kinematic_ss_idx
    )

    # NEGATIVE CASE: Find intersection of indices with negative kinematic rates and OOB ss upper bounds
    tde_coupling_ss_upper_oob_and_neg_kinematic_ss = np.intersect1d(
        tde_coupling_ss_upper_oob_idx, neg_kinematic_ss_idx
    )

    # NEGATIVE CASE: Find intersection of indices with negative kinematic rates and OOB ds lower bounds
    tde_coupling_ds_lower_oob_and_neg_kinematic_ds = np.intersect1d(
        tde_coupling_ds_lower_oob_idx, neg_kinematic_ds_idx
    )

    # NEGATIVE CASE: Find intersection of indices with negative kinematic rates and OOB ds upper bounds
    tde_coupling_ds_upper_oob_and_neg_kinematic_ds = np.intersect1d(
        tde_coupling_ds_upper_oob_idx, neg_kinematic_ds_idx
    )

    # POSITIVE CASE: Find intersection of indices with positive kinematic rates and OOB ss lower bounds
    tde_coupling_ss_lower_oob_and_pos_kinematic_ss = np.intersect1d(
        tde_coupling_ss_lower_oob_idx, pos_kinematic_ss_idx
    )

    # POSITIVE CASE: Find intersection of indices with positive kinematic rates and OOB ss upper bounds
    tde_coupling_ss_upper_oob_and_pos_kinematic_ss = np.intersect1d(
        tde_coupling_ss_upper_oob_idx, pos_kinematic_ss_idx
    )

    # POSITIVE CASE: Find intersection of indices with positive kinematic rates and OOB ds lower bounds
    tde_coupling_ds_lower_oob_and_pos_kinematic_ds = np.intersect1d(
        tde_coupling_ds_lower_oob_idx, pos_kinematic_ds_idx
    )

    # POSITIVE CASE: Find intersection of indices with positive kinematic rates and OOB ds upper bounds
    tde_coupling_ds_upper_oob_and_pos_kinematic_ds = np.intersect1d(
        tde_coupling_ds_upper_oob_idx, pos_kinematic_ds_idx
    )

    # Calculate total number of OOB coupling constraints
    n_oob = (
        len(tde_coupling_ss_lower_oob_idx)
        + len(tde_coupling_ss_upper_oob_idx)
        + len(tde_coupling_ds_lower_oob_idx)
        + len(tde_coupling_ds_upper_oob_idx)
    )

    # Make vectors for update slip rates (not neccesary but useful for debugging)
    updated_ss_bounds_lower = np.copy(current_ss_bounds_lower)
    updated_ss_bounds_upper = np.copy(current_ss_bounds_upper)
    updated_ds_bounds_lower = np.copy(current_ds_bounds_lower)
    updated_ds_bounds_upper = np.copy(current_ds_bounds_upper)

    # Calculate midpoint slip rate assciated with midpoint coupling
    mid_point_ss_coupling = 0.5 * (
        meshes[mesh_idx].config.qp_mesh_tde_slip_rate_lower_bound_ss_coupling
        + meshes[mesh_idx].config.qp_mesh_tde_slip_rate_upper_bound_ss_coupling
    )
    mid_point_ds_coupling = 0.5 * (
        meshes[mesh_idx].config.qp_mesh_tde_slip_rate_lower_bound_ds_coupling
        + meshes[mesh_idx].config.qp_mesh_tde_slip_rate_upper_bound_ds_coupling
    )

    mid_point_ss_rate = mid_point_ss_coupling * kinematic_tde_rates_ss
    mid_point_ds_rate = mid_point_ds_coupling * kinematic_tde_rates_ds

    # Update bounds with a linear approach towards midpoint
    new_ss_bounds_lower = current_ss_bounds_lower + meshes[
        mesh_idx
    ].config.iterative_coupling_linear_slip_rate_reduction_factor * (
        mid_point_ss_rate - current_ss_bounds_lower
    )

    new_ss_bounds_upper = current_ss_bounds_upper + meshes[
        mesh_idx
    ].config.iterative_coupling_linear_slip_rate_reduction_factor * (
        mid_point_ss_rate - current_ss_bounds_upper
    )

    new_ds_bounds_lower = current_ds_bounds_lower + meshes[
        mesh_idx
    ].config.iterative_coupling_linear_slip_rate_reduction_factor * (
        mid_point_ds_rate - current_ds_bounds_lower
    )

    new_ds_bounds_upper = current_ds_bounds_upper + meshes[
        mesh_idx
    ].config.iterative_coupling_linear_slip_rate_reduction_factor * (
        mid_point_ds_rate - current_ds_bounds_upper
    )

    # Update slip rate bounds
    # NOTE: Note upper and lower swap here for negative kinmatic cases (2nd and 3rd quadrants)
    # Negative kinematic case
    updated_ss_bounds_lower[tde_coupling_ss_upper_oob_and_neg_kinematic_ss] = (
        new_ss_bounds_lower[tde_coupling_ss_upper_oob_and_neg_kinematic_ss]
    )
    updated_ss_bounds_upper[tde_coupling_ss_lower_oob_and_neg_kinematic_ss] = (
        new_ss_bounds_upper[tde_coupling_ss_lower_oob_and_neg_kinematic_ss]
    )
    updated_ds_bounds_lower[tde_coupling_ds_upper_oob_and_neg_kinematic_ds] = (
        new_ds_bounds_lower[tde_coupling_ds_upper_oob_and_neg_kinematic_ds]
    )
    updated_ds_bounds_upper[tde_coupling_ds_lower_oob_and_neg_kinematic_ds] = (
        new_ds_bounds_upper[tde_coupling_ds_lower_oob_and_neg_kinematic_ds]
    )

    # Positive kinematic case
    updated_ss_bounds_lower[tde_coupling_ss_lower_oob_and_pos_kinematic_ss] = (
        new_ss_bounds_lower[tde_coupling_ss_lower_oob_and_pos_kinematic_ss]
    )
    updated_ss_bounds_upper[tde_coupling_ss_upper_oob_and_pos_kinematic_ss] = (
        new_ss_bounds_upper[tde_coupling_ss_upper_oob_and_pos_kinematic_ss]
    )
    updated_ds_bounds_lower[tde_coupling_ds_lower_oob_and_pos_kinematic_ds] = (
        new_ds_bounds_lower[tde_coupling_ds_lower_oob_and_pos_kinematic_ds]
    )
    updated_ds_bounds_upper[tde_coupling_ds_upper_oob_and_pos_kinematic_ds] = (
        new_ds_bounds_upper[tde_coupling_ds_upper_oob_and_pos_kinematic_ds]
    )

    return (
        n_oob,
        updated_ss_bounds_lower,
        updated_ss_bounds_upper,
        updated_ds_bounds_lower,
        updated_ds_bounds_upper,
    )


def _check_coupling_bounds_single_mesh(
    operators,
    block,
    index,
    meshes,
    mesh_idx,
    estimation_qp,
    current_ss_bounds_lower,
    current_ss_bounds_upper,
    current_ds_bounds_lower,
    current_ds_bounds_upper,
):
    # Get kinematic rates on mesh elements
    kinematic_tde_rates = (
        operators.rotation_to_tri_slip_rate[mesh_idx]
        @ estimation_qp.state_vector[0 : 3 * len(block)]
    )

    # Get estimated elastic rates on mesh elements
    estimated_tde_rates = (
        operators.eigen.eigenvectors_to_tde_slip[mesh_idx]
        @ estimation_qp.state_vector[
            index.eigen.start_col_eigen[mesh_idx] : index.eigen.end_col_eigen[mesh_idx]
        ]
    )

    # Calculate strike-slip and dip-slip coupling with linear coupling matrix
    tde_coupling_ss, kinematic_tde_rates_ss_smooth = _get_coupling_linear(
        estimated_tde_rates[0::2], kinematic_tde_rates[0::2], operators, mesh_idx
    )

    # Calculate strike-slip and dip-slip coupling with linear coupling matrix
    tde_coupling_ds, kinematic_tde_rates_ds_smooth = _get_coupling_linear(
        estimated_tde_rates[1::2], kinematic_tde_rates[1::2], operators, mesh_idx
    )

    # Update slip rate bounds
    (
        n_oob,
        updated_ss_bounds_lower,
        updated_ss_bounds_upper,
        updated_ds_bounds_lower,
        updated_ds_bounds_upper,
    ) = _update_slip_rate_bounds(
        meshes,
        mesh_idx,
        tde_coupling_ss,
        tde_coupling_ds,
        kinematic_tde_rates_ss_smooth,
        kinematic_tde_rates_ds_smooth,
        current_ss_bounds_lower,
        current_ss_bounds_upper,
        current_ds_bounds_lower,
        current_ds_bounds_upper,
    )

    return (
        updated_ss_bounds_lower,
        updated_ss_bounds_upper,
        updated_ds_bounds_lower,
        updated_ds_bounds_upper,
        kinematic_tde_rates_ss_smooth,
        kinematic_tde_rates_ds_smooth,
        estimated_tde_rates[0::2],
        estimated_tde_rates[1::2],
        n_oob,
    )


def solve_sqp(
    model: Model,
    operators: Operators,
    *,
    max_iter: int | None = None,
    percentage_satisfied_target: float | None = None,
) -> Estimation:
    if operators.eigen is None:
        raise ValueError(
            "Operators must have eigenvectors defined for coupling bounds."
        )

    index = operators.index
    if index.eigen is None:
        raise ValueError("Operators must have eigen index defined.")

    assert index.tde is not None

    if max_iter is None:
        max_iter = model.config.coupling_bounds_max_iter
    if percentage_satisfied_target is None:
        percentage_satisfied_target = (
            model.config.coupling_bounds_total_percentage_satisfied_target
        )

    if max_iter is None:
        raise ValueError("Maximum number of iterations must be defined.")
    if percentage_satisfied_target is None:
        raise ValueError("Percentage satisfied target must be defined.")

    estimation_qp = _presolve(model, operators)

    # Get QP bounds as inequality constraints
    qp_inequality_constraints_matrix, qp_inequality_constraints_data_vector = (
        get_qp_all_inequality_operator_and_data_vector(
            model, operators, operators.index
        )
    )

    data_vector_eigen = operators.data_vector
    weighting_vector_eigen = operators.weighting_vector

    segment = model.segment
    meshes = model.meshes
    config = model.config
    station = model.station
    block = model.block

    # Get total number of segment meshes
    n_segment_meshes = np.max(segment.patch_file_name).astype(int) + 1

    # Count total number of triangles in segment meshes
    n_segment_meshes_tri = 0
    for i in range(n_segment_meshes):
        n_segment_meshes_tri += meshes[i].n_tde

    # Create initial mesh slip rate bound arrays
    current_ss_bounds_lower = [None] * n_segment_meshes
    current_ss_bounds_upper = [None] * n_segment_meshes
    current_ds_bounds_lower = [None] * n_segment_meshes
    current_ds_bounds_upper = [None] * n_segment_meshes
    for i in range(n_segment_meshes):
        mesh_config = meshes[i].config

        assert mesh_config.qp_mesh_tde_slip_rate_lower_bound_ss is not None
        assert mesh_config.qp_mesh_tde_slip_rate_upper_bound_ss is not None
        assert mesh_config.qp_mesh_tde_slip_rate_lower_bound_ds is not None
        assert mesh_config.qp_mesh_tde_slip_rate_upper_bound_ds is not None

        current_ss_bounds_lower[i] = (
            mesh_config.qp_mesh_tde_slip_rate_lower_bound_ss * np.ones(meshes[i].n_tde)
        )
        current_ss_bounds_upper[i] = (
            mesh_config.qp_mesh_tde_slip_rate_upper_bound_ss * np.ones(meshes[i].n_tde)
        )
        current_ds_bounds_lower[i] = (
            mesh_config.qp_mesh_tde_slip_rate_lower_bound_ds * np.ones(meshes[i].n_tde)
        )
        current_ds_bounds_upper[i] = (
            mesh_config.qp_mesh_tde_slip_rate_upper_bound_ds * np.ones(meshes[i].n_tde)
        )

    # Storage for number of OOB coupling values per mesh
    n_oob_vec = np.zeros((n_segment_meshes, 1))

    # Initialize lists and arrays for storing various slip rates
    store_ss_lower = [None] * n_segment_meshes
    store_ss_upper = [None] * n_segment_meshes
    store_ds_lower = [None] * n_segment_meshes
    store_ds_upper = [None] * n_segment_meshes
    store_ss_kinematic = [None] * n_segment_meshes
    store_ss_elcon = [None] * n_segment_meshes
    store_ds_kinematic = [None] * n_segment_meshes
    store_ds_elcon = [None] * n_segment_meshes
    for i in range(n_segment_meshes):
        assert config.coupling_bounds_max_iter is not None
        shape = (meshes[i].n_tde, config.coupling_bounds_max_iter)

        store_ss_lower[i] = np.zeros(shape)
        store_ss_upper[i] = np.zeros(shape)
        store_ds_lower[i] = np.zeros(shape)
        store_ds_upper[i] = np.zeros(shape)
        store_ss_kinematic[i] = np.zeros(shape)
        store_ss_elcon[i] = np.zeros(shape)
        store_ds_kinematic[i] = np.zeros(shape)
        store_ds_elcon[i] = np.zeros(shape)

    # Variables for tracking overall convergence

    tde_total = sum(mesh.n_tde for mesh in meshes)
    total_percentages = list()

    # Coupling bound iteration
    continue_iterating = True
    i = 0
    while continue_iterating:
        # Create storage for updates slip rate constraints
        updated_qp_inequality_constraints_data_vector = np.copy(
            qp_inequality_constraints_data_vector
        )

        # Create storage for n OOB
        current_noob = np.zeros((n_segment_meshes, 1))

        # Loop over meshes
        for j in range(n_segment_meshes):
            (
                updated_ss_bounds_lower,
                updated_ss_bounds_upper,
                updated_ds_bounds_lower,
                updated_ds_bounds_upper,
                kinematic_tde_rates_ss,
                kinematic_tde_rates_ds,
                estimated_tde_rates_ss,
                estimated_tde_rates_ds,
                n_oob,
            ) = _check_coupling_bounds_single_mesh(
                operators,
                block,
                index,
                meshes,
                j,  # This is the mesh index
                estimation_qp,
                current_ss_bounds_lower[j],
                current_ss_bounds_upper[j],
                current_ds_bounds_lower[j],
                current_ds_bounds_upper[j],
            )
            logger.info(f"Iteration: {i}, Mesh: {j}, NOOB: {n_oob}")

            # Store total number of OOB elements at this iteration step
            n_oob_vec[j, i] = n_oob

            # Build and insert update slip rate bounds into QP inequality vector
            updated_lower_bounds = -1.0 * interleave2(
                updated_ss_bounds_lower, updated_ds_bounds_lower
            )
            updated_upper_bounds = interleave2(
                updated_ss_bounds_upper, updated_ds_bounds_upper
            )
            updated_bounds = np.hstack((updated_lower_bounds, updated_upper_bounds))

            # Insert TDE lower bounds into QP constraint data vector
            updated_qp_inequality_constraints_data_vector[
                index.eigen.qp_constraint_tde_rate_start_row_eigen[
                    j
                ] : index.eigen.qp_constraint_tde_rate_start_row_eigen[j]
                + 2 * index.tde.n_tde[j]
            ] = updated_lower_bounds

            # Insert TDE upper bounds into QP constraint data vector
            updated_qp_inequality_constraints_data_vector[
                index.eigen.qp_constraint_tde_rate_start_row_eigen[j]
                + 2
                * index.tde.n_tde[j] : index.eigen.qp_constraint_tde_rate_end_row_eigen[
                    j
                ]
            ] = updated_upper_bounds

            # Set *updated* to *current* for next iteration
            current_ss_bounds_lower[j] = np.copy(updated_ss_bounds_lower)
            current_ss_bounds_upper[j] = np.copy(updated_ss_bounds_upper)
            current_ds_bounds_lower[j] = np.copy(updated_ds_bounds_lower)
            current_ds_bounds_upper[j] = np.copy(updated_ds_bounds_upper)

            # Store values for visualization and debugging
            store_ss_lower[j][:, i] = current_ss_bounds_lower[j]
            store_ss_upper[j][:, i] = current_ss_bounds_upper[j]
            store_ds_lower[j][:, i] = current_ds_bounds_lower[j]
            store_ds_upper[j][:, i] = current_ds_bounds_upper[j]
            store_ss_elcon[j][:, i] = estimated_tde_rates_ss
            store_ds_elcon[j][:, i] = estimated_tde_rates_ds
            store_ss_kinematic[j][:, i] = kinematic_tde_rates_ss
            store_ds_kinematic[j][:, i] = kinematic_tde_rates_ds

        # Store new number of OOB elements permesh
        n_oob_vec = np.hstack((n_oob_vec, current_noob))

        # QP solve with updated TDE slip rate constraints
        solution_qp = lsqlin_qp(
            operators.full_dense_operator * np.sqrt(weighting_vector_eigen[:, None]),
            data_vector_eigen * np.sqrt(weighting_vector_eigen),
            0,
            qp_inequality_constraints_matrix,  # Inequality matrix
            updated_qp_inequality_constraints_data_vector,  # Inequality data vector
            None,
            None,
            None,
            None,
            None,
            {"show_progress": False},
        )

        if solution_qp["status"] != "optimal":
            logger.error(" ")
            logger.error(f"NON OPTIMAL SOLUTION AT: {i=}")
            logger.error(" ")
            raise ValueError("Solver did not converge")

        estimation_qp = Estimation(
            data_vector=operators.data_vector,
            weighting_vector=operators.weighting_vector,
            operator=operators.full_dense_operator,
            state_vector=np.array(solution_qp["x"]).flatten(),
            model=model,
            operators=operators,
            state_covariance_matrix=None,
        )

        # Calculate total percentage of OOB elements to determine if we iterate again
        total_oob = np.sum(n_oob_vec[:, i], axis=0)
        total_percentages.append(total_oob / (2 * tde_total) * 100)
        total_percentage_satisfied = 100 - total_percentages[-1]
        logger.info(
            f"Iteration: {i}, Total %TDE inside coupling bounds: {100 - total_percentages[-1]:0.3f}"
        )

        # Decide if iteration should continue
        if i + 1 < max_iter:
            if (
                total_percentage_satisfied
                <= config.coupling_bounds_total_percentage_satisfied_target
            ):
                continue_iterating = True
                i += 1
            else:
                continue_iterating = False
        else:
            continue_iterating = False
        n_iter = np.copy(i)

    # Write output
    write_output(config, estimation_qp, station, segment, block, meshes)

    # Delete columns less that n_iter
    for j in range(n_segment_meshes):
        store_ss_lower[j] = store_ss_lower[j][:, 0:n_iter]
        store_ss_upper[j] = store_ss_upper[j][:, 0:n_iter]
        store_ds_lower[j] = store_ds_lower[j][:, 0:n_iter]
        store_ds_upper[j] = store_ds_upper[j][:, 0:n_iter]
        store_ss_elcon[j] = store_ss_elcon[j][:, 0:n_iter]
        store_ds_elcon[j] = store_ds_elcon[j][:, 0:n_iter]
        store_ss_kinematic[j] = store_ss_kinematic[j][:, 0:n_iter]
        store_ds_kinematic[j] = store_ds_kinematic[j][:, 0:n_iter]
    return estimation_qp
