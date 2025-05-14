# TODO figure out how to distinguish between different solvers
# Right now we have solve.py and optimize.py. This is a bit
# confusing.

import timeit

import addict
import cvxopt
import numpy as np
import pandas as pd
import scipy
from loguru import logger
from scipy.sparse import csr_matrix

from celeri.celeri_util import get_keep_index_12
from celeri.config import Config
from celeri.model import Model
from celeri.operators import (
    Index,
    Operators,
    _get_index,
    _store_all_mesh_smoothing_matrices,
    _store_block_motion_constraints,
    _store_elastic_operators,
    _store_elastic_operators_okada,
    _store_tde_slip_rate_constraints,
    build_operators,
    get_block_strain_rate_to_velocities_partials,
    get_data_vector,
    get_full_dense_operator,
    get_full_dense_operator_block_only,
    get_global_float_block_rotation_partials,
    get_h_matrices_for_tde_meshes,
    get_index_no_meshes,
    get_mogi_to_velocities_partials,
    get_rotation_to_slip_rate_partials,
    get_rotation_to_velocities_partials,
    get_slip_rate_constraints,
    get_weighting_vector,
    get_weighting_vector_no_meshes,
    rotation_vector_err_to_euler_pole_err,
    rotation_vectors_to_euler_poles,
)
from celeri.output import write_output
from celeri.plot import plot_estimation_summary

type Estimator = addict.Dict


def assemble_and_solve_dense(model: Model) -> tuple[Operators, Estimator]:
    # TODO This should be able to ask for specific subsets of operators?
    operators = build_operators(model, eigen=False)

    estimation = addict.Dict()
    estimation.data_vector = get_data_vector(model, operators.assembly, operators.index)
    estimation.weighting_vector = get_weighting_vector(model, operators.index)
    estimation.operator = get_full_dense_operator(model, operators)

    # DEBUG HERE:
    # IPython.embed(banner1="")

    # Solve the overdetermined linear system using only a weighting vector rather than matrix
    # TODO: Do this with a decompositon
    estimation.state_covariance_matrix = np.linalg.inv(
        estimation.operator.T * estimation.weighting_vector @ estimation.operator
    )
    estimation.state_vector = (
        estimation.state_covariance_matrix
        @ estimation.operator.T
        * estimation.weighting_vector
        @ estimation.data_vector
    )
    return operators, estimation


def post_process_estimation(
    estimation: addict.Dict,
    operators: Operators,
    station: pd.DataFrame,
    index: Index,
):
    """Calculate derived values derived from the block model linear estimate (e.g., velocities, undertainties).

    Args:
        estimation (Dict): Estimated state vector and model covariance
        operators (Dict): All linear operators
        station (pd.DataFrame): GPS station data
        idx (Dict): Indices and counts of data and array sizes
    """
    # TODO We should probably be able to handle this case
    assert index.tde is not None

    # Convert rotation vectors to Euler poles

    estimation.predictions = estimation.operator @ estimation.state_vector
    estimation.vel = estimation.predictions[0 : 2 * index.n_stations]
    estimation.east_vel = estimation.vel[0::2]
    estimation.north_vel = estimation.vel[1::2]

    # Estimate slip rate uncertainties
    estimation.slip_rate_sigma = np.sqrt(
        np.diag(
            operators.rotation_to_slip_rate
            @ estimation.state_covariance_matrix[
                0 : 3 * index.n_blocks, 0 : 3 * index.n_blocks
            ]
            @ operators.rotation_to_slip_rate.T
        )
    )  # I don't think this is correct because for the case when there is a rotation vector a priori
    estimation.strike_slip_rate_sigma = estimation.slip_rate_sigma[0::3]
    estimation.dip_slip_rate_sigma = estimation.slip_rate_sigma[1::3]
    estimation.tensile_slip_rate_sigma = estimation.slip_rate_sigma[2::3]

    # Calculate mean squared residual velocity
    estimation.east_vel_residual = estimation.east_vel - station.east_vel
    estimation.north_vel_residual = estimation.north_vel - station.north_vel

    # Extract TDE slip rates from state vector
    estimation.tde_rates = estimation.state_vector[
        3 * index.n_blocks : 3 * index.n_blocks + 2 * index.n_tde_total
    ]
    estimation.tde_strike_slip_rates = estimation.tde_rates[0::2]
    estimation.tde_dip_slip_rates = estimation.tde_rates[1::2]

    # Extract segment slip rates from state vector
    estimation.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.strike_slip_rates = estimation.slip_rates[0::3]
    estimation.dip_slip_rates = estimation.slip_rates[1::3]
    estimation.tensile_slip_rates = estimation.slip_rates[2::3]

    # Extract estimated block strain rates
    estimation.block_strain_rates = estimation.state_vector[
        3 * index.n_blocks + 2 * index.n_tde_total : 3 * index.n_blocks
        + 2 * index.n_tde_total
        + index.n_block_strain_components
    ]

    # Extract Mogi parameters
    estimation.mogi_volume_change_rates = estimation.state_vector[
        +3 * index.n_blocks
        + 2 * index.n_tde_total
        + index.n_block_strain_components : +3 * index.n_blocks
        + 2 * index.n_tde_total
        + index.n_block_strain_components
        + index.n_mogis
    ]

    # Calculate rotation only velocities
    estimation.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_rotation = estimation.vel_rotation[0::2]
    estimation.north_vel_rotation = estimation.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_elastic_segment = estimation.vel_elastic_segment[0::2]
    estimation.north_vel_elastic_segment = estimation.vel_elastic_segment[1::2]

    # Calculate block strain rate velocities
    estimation.vel_block_strain_rate = (
        operators.block_strain_rate_to_velocities[index.station_row_keep_index, :]
        @ estimation.block_strain_rates
    )
    estimation.east_vel_block_strain_rate = estimation.vel_block_strain_rate[0::2]
    estimation.north_vel_block_strain_rate = estimation.vel_block_strain_rate[1::2]

    # Calculate Euler pole longitudes, latitutudes and rotation rates
    rotation_vector = estimation.state_vector[0 : 3 * index.n_blocks]
    rotation_vector_x = rotation_vector[0::3]
    rotation_vector_y = rotation_vector[1::3]
    rotation_vector_z = rotation_vector[2::3]
    estimation.euler_lon, estimation.euler_lat, estimation.euler_rate = (
        rotation_vectors_to_euler_poles(
            rotation_vector_x, rotation_vector_y, rotation_vector_z
        )
    )

    # Calculate Euler pole uncertainties
    omega_cov = np.zeros((3 * len(rotation_vector_x), 3 * len(rotation_vector_x)))
    estimation.euler_lon_err, estimation.euler_lat_err, estimation.euler_rate_err = (
        rotation_vector_err_to_euler_pole_err(
            rotation_vector_x, rotation_vector_y, rotation_vector_z, omega_cov
        )
    )

    # Calculate Mogi source velocities
    estimation.vel_mogi = (
        operators.mogi_to_velocities[index.station_row_keep_index, :]
        @ estimation.mogi_volume_change_rates
    )
    estimation.east_vel_mogi = estimation.vel_mogi[0::2]
    estimation.north_vel_mogi = estimation.vel_mogi[1::2]

    # Calculate TDE velocities
    estimation.vel_tde = np.zeros(2 * index.n_stations)
    for i in range(len(operators.tde_to_velocities)):
        tde_keep_row_index = get_keep_index_12(operators.tde_to_velocities[i].shape[0])
        tde_keep_col_index = get_keep_index_12(operators.tde_to_velocities[i].shape[1])
        estimation.vel_tde += (
            operators.tde_to_velocities[i][tde_keep_row_index, :][:, tde_keep_col_index]
            @ estimation.state_vector[
                index.tde.start_tde_col[i] : index.tde.end_tde_col[i]
            ]
        )
    estimation.east_vel_tde = estimation.vel_tde[0::2]
    estimation.north_vel_tde = estimation.vel_tde[1::2]

    # Calculate and insert kinematic and coupling triangle rates
    # TODO: This is a placeholder for the real calculation of TDE kinematic
    # and coupling rates
    estimation.tde_tensile_slip_rates = np.zeros_like(estimation.tde_strike_slip_rates)

    estimation.tde_strike_slip_rates_kinematic = np.zeros_like(
        estimation.tde_strike_slip_rates
    )
    estimation.tde_dip_slip_rates_kinematic = np.zeros_like(
        estimation.tde_dip_slip_rates
    )
    estimation.tde_tensile_slip_rates_kinematic = np.zeros_like(
        estimation.tde_tensile_slip_rates
    )
    estimation.tde_strike_slip_rates_coupling = np.zeros_like(
        estimation.tde_strike_slip_rates
    )
    estimation.tde_dip_slip_rates_coupling = np.zeros_like(
        estimation.tde_dip_slip_rates
    )
    estimation.tde_tensile_slip_rates_coupling = np.zeros_like(
        estimation.tde_tensile_slip_rates
    )


def post_process_estimation_eigen(
    model: Model, estimation_eigen: Estimator, operators: Operators
):
    """Calculate derived values derived from the block model linear estimate (e.g., velocities, undertainties).

    Args:
        estimation (Dict): Estimated state vector and model covariance
        operators (Dict): All linear operators
        station (pd.DataFrame): GPS station data
        idx (Dict): Indices and counts of data and array sizes
    """
    index = operators.index
    assert index.tde is not None
    assert index.eigen is not None

    # Isolate eigenvalues (weights for eigenmodes)
    estimation_eigen.eigenvalues = estimation_eigen.state_vector[
        index.eigen.start_col_eigen[0] : index.eigen.end_col_eigen[-1]
    ]

    estimation_eigen.predictions = (
        estimation_eigen.operator @ estimation_eigen.state_vector
    )

    estimation_eigen.vel = estimation_eigen.predictions[0 : 2 * index.n_stations]
    estimation_eigen.east_vel = estimation_eigen.vel[0::2]
    estimation_eigen.north_vel = estimation_eigen.vel[1::2]

    # Estimate slip rate uncertainties
    # Set to ones because we have no meaningful uncertainties for
    # The quadratic programming solve
    estimation_eigen.strike_slip_rate_sigma = np.zeros(index.n_segments)
    estimation_eigen.dip_slip_rate_sigma = np.zeros(index.n_segments)
    estimation_eigen.tensile_slip_rate_sigma = np.zeros(index.n_segments)

    # Calculate mean squared residual velocity
    estimation_eigen.east_vel_residual = (
        estimation_eigen.east_vel - model.station.east_vel
    )
    estimation_eigen.north_vel_residual = (
        estimation_eigen.north_vel - model.station.north_vel
    )

    # Extract TDE slip rates from state vector
    estimation_eigen.tde_rates = np.zeros(2 * index.n_tde_total)
    for i in range(index.n_meshes):
        # Temporary indices for easier reading
        start_idx = index.tde.start_tde_col[i] - 3 * index.n_blocks
        end_idx = index.tde.end_tde_col[i] - 3 * index.n_blocks

        # Calcuate estimated TDE rates for the current mesh
        estimation_eigen.tde_rates[start_idx:end_idx] = (
            operators.eigenvectors_to_tde_slip[i]
            @ estimation_eigen.state_vector[
                index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i]
            ]
        )

    # Isolate strike- and dip-slip rates
    estimation_eigen.tde_strike_slip_rates = estimation_eigen.tde_rates[0::2]
    estimation_eigen.tde_dip_slip_rates = estimation_eigen.tde_rates[1::2]
    estimation_eigen.tde_tensile_slip_rates = np.zeros_like(
        estimation_eigen.tde_dip_slip_rates
    )

    # Create a pseudo state vector that is the length of a TDE state vector
    estimation_eigen.pseudo_tde_state_vector = np.zeros(
        3 * index.n_blocks + 2 * index.n_tde_total
    )
    estimation_eigen.pseudo_tde_state_vector[0 : 3 * index.n_blocks] = (
        estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )

    # Insert estimated TDE rates into pseudo state vector
    estimation_eigen.pseudo_tde_state_vector[
        index.tde.start_tde_col[0] : index.tde.end_tde_col[-1]
    ] = estimation_eigen.tde_rates

    # Calculate and insert kinematic and coupling triangle rates
    # TODO: This is a placeholder for the real calculation of TDE kinematic
    # and coupling rates
    estimation_eigen.tde_strike_slip_rates_kinematic = np.zeros_like(
        estimation_eigen.tde_strike_slip_rates
    )
    estimation_eigen.tde_dip_slip_rates_kinematic = np.zeros_like(
        estimation_eigen.tde_dip_slip_rates
    )
    estimation_eigen.tde_tensile_slip_rates_kinematic = np.zeros_like(
        estimation_eigen.tde_tensile_slip_rates
    )
    estimation_eigen.tde_strike_slip_rates_coupling = np.zeros_like(
        estimation_eigen.tde_strike_slip_rates
    )
    estimation_eigen.tde_dip_slip_rates_coupling = np.zeros_like(
        estimation_eigen.tde_dip_slip_rates
    )
    estimation_eigen.tde_tensile_slip_rates_coupling = np.zeros_like(
        estimation_eigen.tde_tensile_slip_rates
    )

    # Extract segment slip rates from state vector
    estimation_eigen.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_eigen.strike_slip_rates = estimation_eigen.slip_rates[0::3]
    estimation_eigen.dip_slip_rates = estimation_eigen.slip_rates[1::3]
    estimation_eigen.tensile_slip_rates = estimation_eigen.slip_rates[2::3]

    # Calculate rotation only velocities
    estimation_eigen.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_eigen.east_vel_rotation = estimation_eigen.vel_rotation[0::2]
    estimation_eigen.north_vel_rotation = estimation_eigen.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation_eigen.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_eigen.east_vel_elastic_segment = estimation_eigen.vel_elastic_segment[
        0::2
    ]
    estimation_eigen.north_vel_elastic_segment = estimation_eigen.vel_elastic_segment[
        1::2
    ]

    # Extract estimated block strain rates
    estimation_eigen.block_strain_rates = estimation_eigen.state_vector[
        index.eigen.start_block_strain_col_eigen : index.eigen.end_block_strain_col_eigen
    ]

    # Calculate block strain rate velocities
    estimation_eigen.vel_block_strain_rate = (
        operators.block_strain_rate_to_velocities[index.station_row_keep_index, :]
        @ estimation_eigen.block_strain_rates
    )
    estimation_eigen.east_vel_block_strain_rate = (
        estimation_eigen.vel_block_strain_rate[0::2]
    )
    estimation_eigen.north_vel_block_strain_rate = (
        estimation_eigen.vel_block_strain_rate[1::2]
    )

    # Calculate Euler pole longitudes, latitutudes and rotation rates
    rotation_vector = estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    rotation_vector_x = rotation_vector[0::3]
    rotation_vector_y = rotation_vector[1::3]
    rotation_vector_z = rotation_vector[2::3]
    (
        estimation_eigen.euler_lon,
        estimation_eigen.euler_lat,
        estimation_eigen.euler_rate,
    ) = rotation_vectors_to_euler_poles(
        rotation_vector_x, rotation_vector_y, rotation_vector_z
    )

    # Calculate Euler pole uncertainties
    omega_cov = np.zeros((3 * len(rotation_vector_x), 3 * len(rotation_vector_x)))
    (
        estimation_eigen.euler_lon_err,
        estimation_eigen.euler_lat_err,
        estimation_eigen.euler_rate_err,
    ) = rotation_vector_err_to_euler_pole_err(
        rotation_vector_x, rotation_vector_y, rotation_vector_z, omega_cov
    )

    # Extract Mogi parameters
    estimation_eigen.mogi_volume_change_rates = estimation_eigen.state_vector[
        index.eigen.start_mogi_col_eigen : index.eigen.end_mogi_col_eigen
    ]

    # Calculate Mogi source velocities
    estimation_eigen.vel_mogi = (
        operators.mogi_to_velocities[index.station_row_keep_index, :]
        @ estimation_eigen.mogi_volume_change_rates
    )
    estimation_eigen.east_vel_mogi = estimation_eigen.vel_mogi[0::2]
    estimation_eigen.north_vel_mogi = estimation_eigen.vel_mogi[1::2]

    # Calculate elastic TDE velocities from eigenmodes
    estimation_eigen.vel_tde = np.zeros(2 * index.n_stations)
    for i in range(len(operators.tde_to_velocities)):
        estimation_eigen.vel_tde += (
            -operators.eigen_to_velocities[i]
            @ estimation_eigen.state_vector[
                index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i]
            ]
        )

    estimation_eigen.east_vel_tde = estimation_eigen.vel_tde[0::2]
    estimation_eigen.north_vel_tde = estimation_eigen.vel_tde[1::2]


def lsqlin_qp(
    C,
    d,
    reg=0,
    A=None,
    b=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    x0=None,
    opts=None,
):
    """Solve linear constrained l2-regularized least squares. Can
    handle both dense and sparse matrices. Call modeled after Matlab's
    lsqlin. It is actually wrapper around CVXOPT QP solver.

        min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
        s.t.  A * x <= b
              Aeq * x = beq
              lb <= x <= ub

    Input arguments:
        C   is m x n dense or sparse matrix
        d   is n x 1 dense matrix
        reg is regularization parameter
        A   is p x n dense or sparse matrix
        b   is p x 1 dense matrix
        Aeq is q x n dense or sparse matrix
        beq is q x 1 dense matrix
        lb  is n x 1 matrix or scalar
        ub  is n x 1 matrix or scalar

    Output arguments:
        Return dictionary, the output of CVXOPT QP.

    Dont pass matlab-like empty lists to avoid setting parameters,
    just use None:
        lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
        lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!

    Provenance notes:
    Found a few places on Github:
    - https://github.com/KasparP/PSI_simulations/blob/master/Python/SLAPMi/lsqlin.py
    - https://github.com/geospace-code/airtools/blob/main/src/airtools/lsqlin.py

    Some attribution:
    __author__ = "Valeriy Vishnevskiy", "Michael Hirsch"
    __email__ = "valera.vishnevskiy@yandex.ru"
    __version__ = "1.0"
    __date__ = "22.11.2013"
    __license__ = "MIT"
    """

    # Helper functions
    def scipy_sparse_to_spmatrix(A):
        coo = A.tocoo()
        SP = cvxopt.spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
        return SP

    def spmatrix_sparse_to_scipy(A):
        data = np.array(A.V).squeeze()
        rows = np.array(A.I).squeeze()
        cols = np.array(A.J).squeeze()
        return scipy.sparse.coo_matrix((data, (rows, cols)))

    def sparse_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return scipy.sparse.vstack([A1, A2])

    def numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        elif isinstance(A1, np.ndarray):
            return np.vstack([A1, A2])
        elif isinstance(A1, cvxopt.spmatrix):
            return np.vstack([cvxopt_to_numpy_matrix(A1).todense(), A2])

    def numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    def numpy_to_cvxopt_matrix(A):
        if A is None:
            return

        if scipy.sparse.issparse(A):
            if isinstance(A, scipy.sparse.spmatrix):
                return scipy_sparse_to_spmatrix(A)
            else:
                return A
        else:
            if isinstance(A, np.ndarray):
                if A.ndim == 1:
                    return cvxopt.matrix(A, (A.shape[0], 1), "d")
                else:
                    return cvxopt.matrix(A, A.shape, "d")
            else:
                return A

    def cvxopt_to_numpy_matrix(A):
        if A is None:
            return
        if isinstance(A, cvxopt.spmatrix):
            return spmatrix_sparse_to_scipy(A)
        elif isinstance(A, cvxopt.matrix):
            return np.asarray(A).squeeze()
        else:
            return np.asarray(A).squeeze()

    # Main function body
    if scipy.sparse.issparse(A):  # detects both np and cxopt sparse
        sparse_case = True
        # We need A to be scipy sparse, as I couldn't find how
        # CVXOPT spmatrix can be vstacked
        if isinstance(A, cvxopt.spmatrix):
            A = spmatrix_sparse_to_scipy(A)
    else:
        sparse_case = False

    C = numpy_to_cvxopt_matrix(C)
    d = numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = -d.T * C
    nvars = C.size[1]

    if reg > 0:
        if sparse_case:
            i = scipy_sparse_to_spmatrix(scipy.sparse.eye(nvars, nvars, format="coo"))
        else:
            i = cvxopt.matrix(np.eye(nvars), (nvars, nvars), "d")
        Q = Q + reg * i

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b = cvxopt_to_numpy_matrix(b)

    if lb is not None:  # Modify 'A' and 'b' to add lb inequalities
        if lb.size == 1:
            lb = np.repeat(lb, nvars)

        if sparse_case:
            lb_A = -scipy.sparse.eye(nvars, nvars, format="coo")
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:  # Modify 'A' and 'b' to add ub inequalities
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = scipy.sparse.eye(nvars, nvars, format="coo")
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    # Convert data to CVXOPT format
    A = numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b = numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    # Set up options
    if opts is not None:
        for k, v in opts.items():
            cvxopt.solvers.options[k] = v

    # Run CVXOPT.SQP solver
    sol = cvxopt.solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
    return sol


def matvec_wrapper(h_matrix_solve_parameters):
    def matvec_caller(x):
        return matvec(x, h_matrix_solve_parameters)

    return matvec_caller


def rmatvec_wrapper(h_matrix_solve_parameters):
    def rmatvec_caller(x):
        return rmatvec(x, h_matrix_solve_parameters)

    return rmatvec_caller


def matvec(v, h_matrix_solve_parameters):
    """Build matvec (matrix vector product) operator for
    scipy.sparse.linalg.LinearOperator.  This returns A * u.

    Args:
        u (nd.array): Candidate state vector

    Returns:
        out (nd.array): Predicted data vector
    """
    # Unpack parameters
    (
        index,
        meshes,
        H,
        operators,
        weighting_vector,
        col_norms,
        sparse_block_motion_okada_faults,
        sparse_block_motion_constraints,
        sparse_block_slip_rate_constraints,
    ) = h_matrix_solve_parameters

    # Column normalize the state vector
    v_scaled = v / col_norms

    # Make storage for output
    out = np.zeros(index.n_operator_rows)
    block_rotations = v_scaled[index.start_block_col : index.end_block_col]

    # Okada
    out[index.start_station_row : index.end_station_row] += (
        sparse_block_motion_okada_faults.dot(block_rotations)
    )

    # Block motion constraints
    out[index.start_block_constraints_row : index.end_block_constraints_row] += (
        sparse_block_motion_constraints.dot(block_rotations)
    )

    # Slip rate constraints
    out[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] += sparse_block_slip_rate_constraints.dot(block_rotations)

    # Loop over TDE meshes
    # for i in range(len(meshes)):
    for i in range(len(meshes)):
        tde_velocities = v_scaled[index.start_tde_col[i] : index.end_tde_col[i]]

        # Insert TDE to velocity matrix
        out[index.start_station_row : index.end_station_row] += H[i].dot(tde_velocities)

        # TDE smoothing
        out[index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i]] += (
            operators.smoothing_matrix[i].dot(tde_velocities)
        )

        # TDE slip rate constraints
        out[index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i]] += (
            operators.tde_slip_rate_constraints[i].dot(tde_velocities)
        )

    # Weight!
    return out * np.sqrt(weighting_vector)


def rmatvec(u, h_matrix_solve_parameters):
    """Build rmatvec (matrix vector product) operator for
    scipy.sparse.linalg.LinearOperator.  This returns:
    Returns A^H * v, where A^H is the conjugate transpose of A
    for a candidate state vector, u.  We do this because
    with the h-matrix approach we no longer have the full matrix
    so we can't take the transpose all at once.

    Args:
        u (nd.array): Candidate state vector

    Returns:
        out (nd.array): Predicted data vector
    """
    # Unpack parameters
    (
        index,
        meshes,
        H,
        operators,
        weighting_vector,
        col_norms,
        sparse_block_motion_okada_faults,
        sparse_block_motion_constraints,
        sparse_block_slip_rate_constraints,
    ) = h_matrix_solve_parameters

    # Weight the data vector
    u_weighted = u * np.sqrt(weighting_vector)

    # Storage for output
    # out = np.zeros(X.shape[1])
    out = np.zeros(index.n_operator_cols)

    # Select subset of weighted data for the observed velocities
    station_rows = u_weighted[index.start_station_row : index.end_station_row]
    block_constraints = u_weighted[
        index.start_block_constraints_row : index.end_block_constraints_row
    ]

    # Select subset of weighted data for the fault slip rate constraints
    slip_rate_constraints = u_weighted[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ]

    # Okada and block rotation contribution to data vector
    out[index.start_block_col : index.end_block_col] += (
        station_rows @ sparse_block_motion_okada_faults
    )

    # Block motion constraints contribution to data vector
    out[index.start_block_col : index.end_block_col] += (
        block_constraints @ sparse_block_motion_constraints
    )

    # Fault slip rate constraints contribution to data vector
    out[index.start_block_col : index.end_block_col] += (
        slip_rate_constraints @ sparse_block_slip_rate_constraints
    )

    for i in range(len(meshes)):
        # Select subset of weighted data for the TDE smoothing
        tde_smoothing = u_weighted[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i]
        ]

        # Select subset of weighted data for the TDE slip rate constraints
        tde_slip_rate = u_weighted[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i]
        ]

        # Hmatrix (TDEs to velocities)
        out[index.start_tde_col[i] : index.end_tde_col[i]] += H[i].transpose_dot(
            station_rows
        )

        # TDE smoothing contribution to data vector
        out[index.start_tde_col[i] : index.end_tde_col[i]] += (
            tde_smoothing @ operators.smoothing_matrix[i]
        )

        # TDE slip rate constraint contributions to data vector
        out[index.start_tde_col[i] : index.end_tde_col[i]] += (
            tde_slip_rate @ operators.tde_slip_rate_constraints[i]
        )

    # Weight
    return out / col_norms


def build_and_solve_hmatrix(command, assembly, operators, data):
    # TODO: Fix this after dataclass refactor
    # TODO This is almost the same as the initialization of Operator...
    logger.info("build_and_solve_hmatrix")

    # Calculate Okada partials for all segments
    _store_elastic_operators_okada(operators, data.segment, data.station, command)

    # Get TDE smoothing operators
    _store_all_mesh_smoothing_matrices(data.meshes, operators)

    # Get non elastic operators
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        data.station, data.block.shape[0]
    )
    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        data.station
    )
    assembly, operators.block_motion_constraints = _store_block_motion_constraints(
        assembly, data.block, command
    )
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, data.segment, data.block, command
    )
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        data.segment, data.block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        data.block, data.station, data.segment
    )
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        data.mogi, data.station, command
    )
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    _store_tde_slip_rate_constraints(data.meshes, operators)

    index = _get_index(assembly, data.station, data.block, data.meshes)

    # Data and data weighting vector
    weighting_vector = get_weighting_vector(command, data.station, data.meshes, index)
    data_vector = get_data_vector(assembly, index)

    # Apply data weighting
    data_vector = data_vector * np.sqrt(weighting_vector)

    # Cast all block submatrices to sparse
    sparse_block_motion_okada_faults = csr_matrix(
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )
    sparse_block_motion_constraints = csr_matrix(operators.block_motion_constraints)
    sparse_block_slip_rate_constraints = csr_matrix(operators.slip_rate_constraints)

    # Calculate column normalization vector for blocks
    operator_block_only = get_full_dense_operator_block_only(operators, index)
    weighting_vector_block_only = weighting_vector[0 : operator_block_only.shape[0]][
        :, None
    ]
    col_norms = np.linalg.norm(
        operator_block_only * np.sqrt(weighting_vector_block_only), axis=0
    )

    # Hmatrix decompositon for each TDE mesh
    logger.info("Start: H-matrix build")
    H, col_norms = get_h_matrices_for_tde_meshes(
        command, data.meshes, data.station, operators, index, col_norms
    )
    logger.success("Finish: H-matrix build")

    # Package parameters that matvec and rmatvec need for the iterative solve
    h_matrix_solve_parameters = (
        index,
        data.meshes,
        H,
        operators,
        weighting_vector,
        col_norms,
        sparse_block_motion_okada_faults,
        sparse_block_motion_constraints,
        sparse_block_slip_rate_constraints,
    )

    # Instantiate the scipy the linear operator for the iterative solver to use
    operator_hmatrix = scipy.sparse.linalg.LinearOperator(
        (index.n_operator_rows, index.n_operator_cols),
        matvec=matvec_wrapper(h_matrix_solve_parameters),
        rmatvec=rmatvec_wrapper(h_matrix_solve_parameters),
    )

    # Solve the linear system
    logger.info("Start: H-matrix solve")
    start_solve_time = timeit.default_timer()

    if command.iterative_solver == "lsmr":
        logger.info("Using LSMR solver")
        sparse_hmatrix_solution = scipy.sparse.linalg.lsmr(
            operator_hmatrix, data_vector, atol=command.atol, btol=command.btol
        )
    elif command.iterative_solver == "lsqr":
        logger.info("Using LSQR solver")
        sparse_hmatrix_solution = scipy.sparse.linalg.lsqr(
            operator_hmatrix, data_vector, atol=command.atol, btol=command.btol
        )

    end_solve_time = timeit.default_timer()
    logger.success(
        f"Finish: H-matrix solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    )

    # Correct the solution for the col_norms preconditioning.
    estimation = addict.Dict()

    sparse_hmatrix_state_vector = sparse_hmatrix_solution[0] / col_norms
    if command.iterative_solver == "lsqr":
        sparse_hmatrix_state_vector_sigma = (
            np.sqrt(sparse_hmatrix_solution[9]) / col_norms
        )
        estimation.state_vector_sigma = sparse_hmatrix_state_vector_sigma

    estimation.data_vector = data_vector
    estimation.weighting_vector = weighting_vector
    estimation.operator = operator_hmatrix
    estimation.state_vector = sparse_hmatrix_state_vector

    post_process_estimation_hmatrix(
        command,
        data.block,
        estimation,
        operators,
        data.meshes,
        H,
        data.station,
        index,
        col_norms,
        h_matrix_solve_parameters,
    )
    write_output(
        command, estimation, data.station, data.segment, data.block, data.meshes
    )

    if bool(command.plot_estimation_summary):
        plot_estimation_summary(
            command,
            data.segment,
            data.station,
            data.meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    return estimation, operators, index


def post_process_estimation_hmatrix(
    command: Config,
    block: pd.DataFrame,
    estimation_hmatrix: addict.Dict,
    operators: Operators,
    meshes: list,
    H: list,
    station: pd.DataFrame,
    index: addict.Dict,
    col_norms: np.ndarray,
    h_matrix_solve_parameters: tuple,
):
    """Calculate derived values derived from the block model linear estimate (e.g., velocities, undertainties).

    Args:
        estimation (Dict): Estimated state vector and model covariance
        operators (Dict): All linear operators
        meshes (List): Mesh geometries
        H (List): Hmatrix decompositions for each TDE mesh
        station (pd.DataFrame): GPS station data
        index (Dict): Indices and counts of data and array sizes
        col_norms (np.array): Column preconditining vector
        h_matrix_solve_parameters (Tuple): Package of sparse and hmatrix operators
    """
    estimation_hmatrix.predictions = matvec(
        estimation_hmatrix.state_vector * col_norms, h_matrix_solve_parameters
    ) / np.sqrt(estimation_hmatrix.weighting_vector)
    estimation_hmatrix.vel = estimation_hmatrix.predictions[0 : 2 * index.n_stations]
    estimation_hmatrix.east_vel = estimation_hmatrix.vel[0::2]
    estimation_hmatrix.north_vel = estimation_hmatrix.vel[1::2]

    # Calculate mean squared residual velocity
    estimation_hmatrix.east_vel_residual = (
        estimation_hmatrix.east_vel - station.east_vel
    )
    estimation_hmatrix.north_vel_residual = (
        estimation_hmatrix.north_vel - station.north_vel
    )

    # Extract TDE slip rates from state vector
    estimation_hmatrix.tde_rates = estimation_hmatrix.state_vector[
        3 * index.n_blocks : 3 * index.n_blocks + 2 * index.n_tde_total
    ]
    estimation_hmatrix.tde_strike_slip_rates = estimation_hmatrix.tde_rates[0::2]
    estimation_hmatrix.tde_dip_slip_rates = estimation_hmatrix.tde_rates[1::2]

    # Extract segment slip rates from state vector
    estimation_hmatrix.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation_hmatrix.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_hmatrix.strike_slip_rates = estimation_hmatrix.slip_rates[0::3]
    estimation_hmatrix.dip_slip_rates = estimation_hmatrix.slip_rates[1::3]
    estimation_hmatrix.tensile_slip_rates = estimation_hmatrix.slip_rates[2::3]

    if command.iterative_solver == "lsmr":
        # All uncertainties set to 1 because lsmr doesn't calculate variance
        logger.warning(
            "Slip rate uncertainty estimates set to 1 because LSMR doesn't provide variance estimates"
        )
        estimation_hmatrix.strike_slip_rate_sigma = np.ones_like(
            estimation_hmatrix.strike_slip_rates
        )
        estimation_hmatrix.dip_slip_rate_sigma = np.ones_like(
            estimation_hmatrix.dip_slip_rates
        )
        estimation_hmatrix.tensile_slip_rate_sigma = np.ones_like(
            estimation_hmatrix.tensile_slip_rates
        )
    elif command.iterative_solver == "lsqr":
        # TODO: Block motion uncertainties
        estimation_hmatrix.slip_rate_sigma = np.sqrt(
            np.diag(
                operators.rotation_to_slip_rate
                @ np.diag(estimation_hmatrix.state_vector_sigma[0 : 3 * index.n_blocks])
                @ operators.rotation_to_slip_rate.T
            )
        )
        estimation_hmatrix.strike_slip_rate_sigma = estimation_hmatrix.slip_rate_sigma[
            0::3
        ]
        estimation_hmatrix.dip_slip_rate_sigma = estimation_hmatrix.slip_rate_sigma[
            1::3
        ]
        estimation_hmatrix.tensile_slip_rate_sigma = estimation_hmatrix.slip_rate_sigma[
            2::3
        ]

    # Calculate rotation only velocities
    estimation_hmatrix.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation_hmatrix.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_hmatrix.east_vel_rotation = estimation_hmatrix.vel_rotation[0::2]
    estimation_hmatrix.north_vel_rotation = estimation_hmatrix.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation_hmatrix.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation_hmatrix.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_hmatrix.east_vel_elastic_segment = (
        estimation_hmatrix.vel_elastic_segment[0::2]
    )
    estimation_hmatrix.north_vel_elastic_segment = (
        estimation_hmatrix.vel_elastic_segment[1::2]
    )

    # TODO: Calculate block strain rate velocities
    estimation_hmatrix.east_vel_block_strain_rate = np.zeros(len(station))
    estimation_hmatrix.north_vel_block_strain_rate = np.zeros(len(station))

    # Calculate TDE velocities
    estimation_hmatrix.vel_tde = np.zeros(2 * index.n_stations)
    for i in range(len(meshes)):
        estimation_hmatrix.vel_tde += H[i].dot(
            estimation_hmatrix.state_vector[
                index.start_tde_col[i] : index.end_tde_col[i]
            ]
        )
    estimation_hmatrix.east_vel_tde = estimation_hmatrix.vel_tde[0::2]
    estimation_hmatrix.north_vel_tde = estimation_hmatrix.vel_tde[1::2]


def build_and_solve_dense(command, assembly, operators, data):
    # NOTE: Used in celeri_solve.py
    # TODO: Again, very similar to Operator initialization
    logger.info("build_and_solve_dense")

    # Get all elastic operators for segments and TDEs
    _store_elastic_operators(
        operators, data.meshes, data.segment, data.station, command
    )

    # Get TDE smoothing operators
    _store_all_mesh_smoothing_matrices(data.meshes, operators)

    # Get non-elastic operators
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        data.station, data.block.shape[0]
    )
    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        data.station
    )
    assembly, operators.block_motion_constraints = _store_block_motion_constraints(
        assembly, data.block, command
    )
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, data.segment, data.block, command
    )
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        data.segment, data.block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        data.block, data.station, data.segment
    )
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        data.mogi, data.station, command
    )
    _store_tde_slip_rate_constraints(data.meshes, operators)

    # Direct solve dense linear system
    logger.info("Start: Dense assemble and solve")
    start_solve_time = timeit.default_timer()
    index, estimation = assemble_and_solve_dense(
        command, assembly, operators, data.station, data.block, data.meshes, data.mogi
    )
    end_solve_time = timeit.default_timer()
    logger.success(
        f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    )

    post_process_estimation(estimation, operators, data.station, index)

    write_output(
        command, estimation, data.station, data.segment, data.block, data.meshes
    )

    if bool(command.plot_estimation_summary):
        plot_estimation_summary(
            command,
            data.segment,
            data.station,
            data.meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    return estimation, operators, index


def build_and_solve_dense_no_meshes(command, assembly, operators, data):
    # NOTE: Used in celeri_solve.py
    # TODO This is almost the same as the initialization of Operator...
    logger.info("build_and_solve_dense_no_meshes")

    # Get all elastic operators for segments and TDEs
    _store_elastic_operators(
        operators, data.meshes, data.segment, data.station, command
    )

    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        data.station, data.block.shape[0]
    )
    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        data.station
    )
    assembly, operators.block_motion_constraints = _store_block_motion_constraints(
        assembly, data.block, command
    )
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, data.segment, data.block, command
    )
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        data.segment, data.block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        data.block, data.station, data.segment
    )
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        data.mogi, data.station, command
    )

    # Blocks only operator
    index = get_index_no_meshes(assembly, data.station, data.block)

    # TODO: Clean up!
    logger.error(operators.keys())

    operator_block_only = get_full_dense_operator_block_only(operators, index)
    # weighting_vector = get_weighting_vector(command, data.station, data.meshes, index)
    weighting_vector = get_weighting_vector_no_meshes(command, data.station, index)
    data_vector = get_data_vector(assembly, index)
    weighting_vector_block_only = weighting_vector[0 : operator_block_only.shape[0]]

    # Solve the overdetermined linear system using only a weighting vector rather than matrix
    estimation = addict.Dict()
    estimation.operator = operator_block_only
    estimation.weighting_vector = weighting_vector_block_only

    estimation.state_covariance_matrix = np.linalg.inv(
        operator_block_only.T * weighting_vector_block_only @ operator_block_only
    )
    estimation.state_vector = (
        estimation.state_covariance_matrix
        @ operator_block_only.T
        * weighting_vector_block_only
        @ data_vector[0 : weighting_vector_block_only.size]
    )

    # Post-processing

    estimation.predictions = estimation.operator @ estimation.state_vector
    estimation.vel = estimation.predictions[0 : 2 * index.n_stations]
    estimation.east_vel = estimation.vel[0::2]
    estimation.north_vel = estimation.vel[1::2]

    # Estimate slip rate uncertainties
    estimation.slip_rate_sigma = np.sqrt(
        np.diag(
            operators.rotation_to_slip_rate
            @ estimation.state_covariance_matrix[
                0 : 3 * index.n_blocks, 0 : 3 * index.n_blocks
            ]
            @ operators.rotation_to_slip_rate.T
        )
    )  # I don't think this is correct because for the case when there is a rotation vector a priori
    estimation.strike_slip_rate_sigma = estimation.slip_rate_sigma[0::3]
    estimation.dip_slip_rate_sigma = estimation.slip_rate_sigma[1::3]
    estimation.tensile_slip_rate_sigma = estimation.slip_rate_sigma[2::3]

    # Calculate mean squared residual velocity
    estimation.east_vel_residual = estimation.east_vel - data.station.east_vel
    estimation.north_vel_residual = estimation.north_vel - data.station.north_vel

    # Extract segment slip rates from state vector
    estimation.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.strike_slip_rates = estimation.slip_rates[0::3]
    estimation.dip_slip_rates = estimation.slip_rates[1::3]
    estimation.tensile_slip_rates = estimation.slip_rates[2::3]

    # Calculate rotation only velocities
    estimation.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_rotation = estimation.vel_rotation[0::2]
    estimation.north_vel_rotation = estimation.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_elastic_segment = estimation.vel_elastic_segment[0::2]
    estimation.north_vel_elastic_segment = estimation.vel_elastic_segment[1::2]

    # TODO: Calculate block strain rate velocities
    estimation.east_vel_block_strain_rate = np.zeros(len(data.station))
    estimation.north_vel_block_strain_rate = np.zeros(len(data.station))

    # # Get all elastic operators for segments and TDEs
    # get_elastic_operators(operators, data.meshes, data.segment, data.station, command)

    # # Get TDE smoothing operators
    # get_all_mesh_smoothing_matrices(data.meshes, operators)

    # # Get non-elastic operators
    # operators.rotation_to_velocities = get_rotation_to_velocities_partials(data.station, data.block.shape[0])
    # operators.global_float_block_rotation = get_global_float_block_rotation_partials(
    #     data.station
    # )
    # assembly, operators.block_motion_constraints = get_block_motion_constraints(
    #     assembly, data.block, command
    # )
    # assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
    #     assembly, data.segment, data.block, command
    # )
    # operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
    #     data.segment, data.block
    # )
    # (
    #     operators.block_strain_rate_to_velocities,
    #     strain_rate_block_index,
    # ) = get_block_strain_rate_to_velocities_partials(
    #     data.block, data.station, data.segment
    # )
    # operators.mogi_to_velocities = get_mogi_to_velocities_partials(
    #     data.mogi, data.station, command
    # )
    # get_tde_slip_rate_constraints(data.meshes, operators)

    # # Direct solve dense linear system
    # logger.info("Start: Dense assemble and solve")
    # start_solve_time = timeit.default_timer()
    # index, estimation = assemble_and_solve_dense(
    #     command, assembly, operators, data.station, data.block, data.meshes
    # )
    # end_solve_time = timeit.default_timer()
    # logger.success(
    #     f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    # )

    # post_process_estimation(estimation, operators, data.station, index)

    write_output(
        command, estimation, data.station, data.segment, data.block, data.meshes
    )

    if bool(command.plot_estimation_summary):
        plot_estimation_summary(
            command,
            data.segment,
            data.station,
            data.meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    return estimation, operators, index


def build_and_solve_qp_kl(command, assembly, operators, data):
    # NOTE: Used in celeri_solve.py
    logger.info("build_and_solve_qp_kl")
    logger.info("PLACEHOLDER")

    # # Get all elastic operators for segments and TDEs
    # get_elastic_operators(operators, data.meshes, data.segment, data.station, command)

    # # Get TDE smoothing operators
    # get_all_mesh_smoothing_matrices(data.meshes, operators)

    # # Get non-elastic operators
    # operators.rotation_to_velocities = get_rotation_to_velocities_partials(
    #     data.station, data.block.shape[0]
    # )
    # operators.global_float_block_rotation = get_global_float_block_rotation_partials(
    #     data.station
    # )
    # assembly, operators.block_motion_constraints = get_block_motion_constraints(
    #     assembly, data.block, command
    # )
    # assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
    #     assembly, data.segment, data.block, command
    # )
    # operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
    #     data.segment, data.block
    # )
    # (
    #     operators.block_strain_rate_to_velocities,
    #     strain_rate_block_index,
    # ) = get_block_strain_rate_to_velocities_partials(
    #     data.block, data.station, data.segment
    # )
    # operators.mogi_to_velocities = get_mogi_to_velocities_partials(
    #     data.mogi, data.station, command
    # )
    # get_tde_slip_rate_constraints(data.meshes, operators)

    # # Direct solve dense linear system
    # logger.info("Start: Dense assemble and solve")
    # start_solve_time = timeit.default_timer()
    # index, estimation = assemble_and_solve_dense(
    #     command, assembly, operators, data.station, data.block, data.meshes
    # )
    # end_solve_time = timeit.default_timer()
    # logger.success(
    #     f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    # )

    # post_process_estimation(estimation, operators, data.station, index)

    # write_output(
    #     command, estimation, data.station, data.segment, data.block, data.meshes
    # )

    # if bool(command.plot_estimation_summary):
    #     plot_estimation_summary(
    #         command,
    #         data.segment,
    #         data.station,
    #         data.meshes,
    #         estimation,
    #         lon_range=command.lon_range,
    #         lat_range=command.lat_range,
    #         quiver_scale=command.quiver_scale,
    #     )

    # return estimation, operators, index
