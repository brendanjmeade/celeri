from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Literal, cast, Any
import addict
import cvxopt
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy import sparse, spatial
import matplotlib.pyplot as plt

from celeri import (
    post_process_estimation_eigen,
    write_output,
    plot_estimation_summary,
    get_qp_all_inequality_operator_and_data_vector,
    get_data_vector_eigen,
    get_weighting_vector_eigen,
)
from celeri.celeri import (
    assign_block_labels,
    create_output_folder,
    get_all_mesh_smoothing_matrices,
    get_block_motion_constraints,
    get_block_strain_rate_to_velocities_partials,
    get_command,
    get_eigenvectors_to_tde_slip,
    get_elastic_operators,
    get_full_dense_operator_eigen,
    get_index_eigen,
    get_mogi_to_velocities_partials,
    get_rotation_to_slip_rate_partials,
    get_rotation_to_velocities_partials,
    get_slip_rate_constraints,
    get_tde_coupling_constraints,
    get_tde_slip_rate_constraints,
    merge_geodetic_data,
    process_sar,
    process_segment,
    process_station,
    read_data,
)


@dataclass
class CeleriProblem:
    """
    Represents a problem configuration for Celeri fault slip rate modeling.

    Stores indices, meshes, operators, and various data components needed
    for solving interseismic coupling and fault slip rate problems.
    """

    index: dict
    meshes: addict.Dict
    operators: addict.Dict
    segment: pd.DataFrame
    block: pd.DataFrame
    station: pd.DataFrame
    assembly: addict.Dict
    command: dict[str, Any]

    @property
    def segment_mesh_indices(self):
        n_segment_meshes = np.max(self.segment.patch_file_name).astype(int) + 1
        return list(range(n_segment_meshes))

    @property
    def total_mesh_points(self):
        return sum([self.meshes[idx]["n_tde"] for idx in self.segment_mesh_indices])


@dataclass
class CouplingItem:
    """
    Coupling between kinematic and estimated slip rates for a fault segment.

    Stores both raw and smoothed kinematic slip rates along with estimated slip rates.
    """

    kinematic: cp.Expression | np.ndarray
    kinematic_smooth: cp.Expression | np.ndarray
    estimated: cp.Expression | np.ndarray

    def copy_numpy(self) -> "CouplingItem":
        """Copy this item with all values as numpy arrays."""
        return CouplingItem(
            kinematic=self.kinematic_numpy(smooth=False),
            kinematic_smooth=self.kinematic_numpy(smooth=True),
            estimated=self.estimated_numpy(),
        )

    def kinematic_numpy(self, *, smooth: bool) -> np.ndarray:
        """Return kinematic slip rates as a numpy array."""
        if smooth:
            kinematic = self.kinematic_smooth
        else:
            kinematic = self.kinematic

        if isinstance(kinematic, cp.Expression):
            kinematic = kinematic.value

        if kinematic is None:
            raise ValueError("Coupling has not been fit")

        return kinematic

    def estimated_numpy(self) -> np.ndarray:
        """Return estimated slip rates as a numpy array."""
        estimated = self.estimated

        if isinstance(estimated, cp.Expression):
            estimated = estimated.value

        if estimated is None:
            raise ValueError("Coupling has not been fit")

        return estimated

    def out_of_bounds(self, *, smooth_kinematic: bool, tol=1e-8) -> tuple[int, int]:
        """Count slip rates that violate coupling constraints.

        Args:
            smooth_kinematic: Whether to use smoothed kinematic values to
            compute the couplings.
            tol: Tolerance for constraint violation.

        Returns:
            Tuple of (number of out-of-bounds points, total points).
        """
        kinematic = self.kinematic_numpy(smooth=smooth_kinematic)
        estimated = self.estimated_numpy()

        total = len(kinematic)
        is_oob = estimated**2 - estimated * kinematic > tol
        return is_oob.sum(), total

    def constraint_loss(self, *, smooth_kinematic: bool) -> float:
        """Calculate quadratic coupling constraint violation"""
        kinematic = self.kinematic_numpy(smooth=smooth_kinematic)
        estimated = self.estimated_numpy()

        constraint = estimated**2 - estimated * kinematic
        return np.clip(constraint, 0, np.inf).sum()


def _get_gaussian_smoothing_operator(meshes, operators, index):
    for i in range(index.n_meshes):
        points = np.vstack((meshes[i].lon_centroid, meshes[i].lat_centroid)).T

        if "iterative_coupling_smoothing_length_scale" in meshes[i]:
            length_scale = meshes[i].iterative_coupling_smoothing_length_scale
        else:
            length_scale = 0.25

        # Compute pairwise Euclidean distance matrix
        D = spatial.distance_matrix(points, points)

        # Define Gaussian weight function
        # W = np.clip(np.exp(-(D**2) / (2 * length_scale**2)), 1e-6, np.inf)
        W = np.exp(-(D**2) / (2 * length_scale**2))

        # Normalize rows so each row sums to 1
        W /= W.sum(axis=1, keepdims=True)

        operators.linear_guassian_smoothing[i] = W
    return operators


# TODO Move this to a more appropriate location
def build_problem(command_path: str | Path) -> CeleriProblem:
    command = get_command(command_path)
    create_output_folder(command)
    segment, block, meshes, station, mogi, sar = read_data(command)
    station = process_station(station, command)
    segment = process_segment(segment, command, meshes)
    sar = process_sar(sar, command)
    closure, block = assign_block_labels(segment, station, block, mogi, sar)
    assembly = addict.Dict()
    operators = addict.Dict()
    operators.meshes = [addict.Dict()] * len(meshes)
    assembly = merge_geodetic_data(assembly, station, sar)

    # Prepare the operators

    # Get all elastic operators for segments and TDEs
    get_elastic_operators(operators, meshes, segment, station, command)

    # Get TDE smoothing operators
    get_all_mesh_smoothing_matrices(meshes, operators)

    # Block rotation to velocity operator
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        station, len(block)
    )

    # Soft block motion constraints
    assembly, operators.block_motion_constraints = get_block_motion_constraints(
        assembly, block, command
    )

    # Soft slip rate constraints
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, segment, block, command
    )

    # Rotation vectors to slip rate operator
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(segment, block)

    # Internal block strain rate operator
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(block, station, segment)

    # Mogi source operator
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        mogi, station, command
    )

    # Soft TDE boundary condition constraints
    get_tde_slip_rate_constraints(meshes, operators)

    # Get index
    index = get_index_eigen(assembly, segment, station, block, meshes, mogi)

    # Get data vector for KL problem
    get_data_vector_eigen(meshes, assembly, index)

    # Get data vector for KL problem
    get_weighting_vector_eigen(command, station, meshes, index)

    # Get KL modes for each mesh
    get_eigenvectors_to_tde_slip(operators, meshes)

    # Get full operator including all blocks, KL modes, strain blocks, and mogis
    operators.eigen = get_full_dense_operator_eigen(operators, meshes, index)

    # Get rotation to TDE kinematic slip rate operator for all meshes tied to segments
    get_tde_coupling_constraints(meshes, segment, block, operators)

    # Get smoothing operators for post-hoc smoothing of slip
    operators = _get_gaussian_smoothing_operator(meshes, operators, index)

    return CeleriProblem(
        index=index,
        meshes=meshes,
        operators=operators,
        segment=segment,
        block=block,
        station=station,
        assembly=assembly,
        command=command,
    )


@dataclass
class Coupling:
    strike_slip: CouplingItem
    dip_slip: CouplingItem

    def out_of_bounds(
        self, *, smooth_kinematic: bool, tol: float = 1e-8
    ) -> tuple[int, int]:
        oob1, total1 = self.strike_slip.out_of_bounds(
            smooth_kinematic=smooth_kinematic, tol=tol
        )
        oob2, total2 = self.dip_slip.out_of_bounds(
            smooth_kinematic=smooth_kinematic, tol=tol
        )
        return oob1 + oob2, total1 + total2

    def constraint_loss(self, *, smooth_kinematic: bool) -> float:
        loss1 = self.strike_slip.constraint_loss(smooth_kinematic=smooth_kinematic)
        loss2 = self.dip_slip.constraint_loss(smooth_kinematic=smooth_kinematic)
        return loss1 + loss2

    def copy_numpy(self) -> "Coupling":
        return Coupling(
            strike_slip=self.strike_slip.copy_numpy(),
            dip_slip=self.dip_slip.copy_numpy(),
        )


@dataclass
class VelocityLimitItem:
    kinematic_lower: np.ndarray
    kinematic_upper: np.ndarray
    constraints_matrix_kinematic: cp.Parameter | None = None
    constraints_matrix_estimated: cp.Parameter | None = None
    constraints_vector: cp.Parameter | None = None

    def copy(self) -> "VelocityLimitItem":
        return VelocityLimitItem(
            kinematic_lower=self.kinematic_lower.copy(),
            kinematic_upper=self.kinematic_upper.copy(),
            constraints_matrix_kinematic=None,
            constraints_matrix_estimated=None,
            constraints_vector=None,
        )

    @classmethod
    def from_scalar(
        cls, length: int, lower: float, upper: float
    ) -> "VelocityLimitItem":
        return VelocityLimitItem(
            kinematic_lower=np.full((length,), lower),
            kinematic_upper=np.full((length,), upper),
        )

    def update_constraints(self):
        if (
            self.constraints_matrix_kinematic is None
            or self.constraints_matrix_estimated is None
            or self.constraints_vector is None
        ):
            n_dim = len(self.kinematic_lower)
            # dims = (mesh_point, constraint, (kinematic, estimated))
            self.constraints_matrix_kinematic = cp.Parameter(shape=(n_dim, 2))
            self.constraints_matrix_estimated = cp.Parameter(shape=(n_dim, 2))
            self.constraints_vector = cp.Parameter(shape=(n_dim, 2))

        kinematic = np.zeros(self.constraints_matrix_kinematic.shape)
        estimated = np.zeros(self.constraints_matrix_estimated.shape)
        const = np.zeros(self.constraints_vector.shape)

        # kinematic <= kinematic_lower
        # kinematic[:, 2] = np.where(self.kinematic_lower == 0.0, 0.0, -1)
        # const[:, 2] = np.where(self.kinematic_lower == 0.0, 0.0, -self.kinematic_lower)

        # kinematic >= kinematic_upper
        # kinematic[:, 3] = np.where(self.kinematic_upper == 0.0, 0.0, 1)
        # const[:, 3] = np.where(self.kinematic_upper == 0.0, 0.0, self.kinematic_upper)

        # Define boundary points in (kinematic, estimated) space

        # The lower left corner of the subset
        lower_left_kinematic = self.kinematic_lower
        lower_left_estimated = np.minimum(0.0, self.kinematic_lower)

        # The upper left corner of the subset
        upper_left_kinematic = self.kinematic_lower
        upper_left_estimated = np.maximum(0.0, self.kinematic_lower)

        # The lower right corner of the subset
        lower_right_kinematic = self.kinematic_upper
        lower_right_estimated = np.minimum(0.0, self.kinematic_upper)

        # The upper right corner of the subset
        upper_right_kinematic = self.kinematic_upper
        upper_right_estimated = np.maximum(0.0, self.kinematic_upper)

        def bounded_through_points(x1, y1, x2, y2):
            """
            Compute coefficients for a line inequality passing through two points.

            Creates the linear inequality: coef_x * x + coef_y * y <= const
            which corresponds to: (y2-y1)*(x-x1) - (x2-x1)*(y-y1) <= 0

            Args:
                x1, y1: Coordinates of the first point
                x2, y2: Coordinates of the second point

            Returns:
                coef_x, coef_y, const: Coefficients for the inequality
            """
            coef_x = y2 - y1
            coef_y = -(x2 - x1)
            const = x1 * y2 - y1 * x2
            return coef_x, coef_y, const

        coef_x, coef_y, bound_const = bounded_through_points(
            lower_left_kinematic,
            lower_left_estimated,
            lower_right_kinematic,
            lower_right_estimated,
        )
        kinematic[:, 0] = coef_x
        estimated[:, 0] = coef_y
        const[:, 0] = bound_const

        coef_x, coef_y, bound_const = bounded_through_points(
            upper_right_kinematic,
            upper_right_estimated,
            upper_left_kinematic,
            upper_left_estimated,
        )
        kinematic[:, 1] = coef_x
        estimated[:, 1] = coef_y
        const[:, 1] = bound_const

        # Rescale matrices so that the largest coefficient for each inequality is 1
        for i in range(kinematic.shape[1]):
            # Find the maximum absolute value in each inequality constraint
            max_coef = np.maximum(np.abs(kinematic[:, i]), np.abs(estimated[:, i]))

            # Avoid division by zero
            max_coef = np.where(max_coef > 0, max_coef, 1.0)

            # Rescale the coefficients
            kinematic[:, i] /= max_coef
            estimated[:, i] /= max_coef
            const[:, i] /= max_coef

        self.constraints_matrix_kinematic.value = kinematic
        self.constraints_matrix_estimated.value = estimated
        self.constraints_vector.value = const

    def build_constraints(
        self, kinematic, estimated, *, mixed_integer=False
    ) -> list[cp.Constraint]:
        if not mixed_integer:
            self.update_constraints()
            assert self.constraints_vector is not None
            assert self.constraints_matrix_kinematic is not None
            assert self.constraints_matrix_estimated is not None

            return [
                cp.multiply(self.constraints_matrix_kinematic, kinematic[:, None])
                + cp.multiply(self.constraints_matrix_estimated, estimated[:, None])
                <= self.constraints_vector
            ]
        else:
            z = cp.Variable(
                shape=kinematic.shape,
                name="z",
                boolean=True,
            )
            eps = 10.0

            kin_lb = self.kinematic_lower
            kin_ub = self.kinematic_upper

            M = np.maximum(kin_ub, -kin_lb) + eps

            return [
                kinematic <= kin_ub,
                kinematic >= kin_lb,
                estimated <= kin_ub,
                estimated >= kin_lb,
                # check these
                kinematic <= cp.multiply(M, z),
                kinematic >= -cp.multiply(M, 1 - z),
                kinematic - estimated <= cp.multiply(M, z),
                kinematic - estimated >= -cp.multiply(M, 1 - z),
            ]

    def plot_constraint(self, index=0, figsize=(8, 6)):
        """
        Plot the constraint boundaries for a specific mesh point.

        Args:
            index: Index of the mesh point to visualize
            figsize: Size of the figure to create

        Returns:
            matplotlib Figure object
        """
        self.update_constraints()

        assert self.constraints_vector is not None
        assert self.constraints_matrix_kinematic is not None
        assert self.constraints_matrix_estimated is not None
        assert self.constraints_vector.value is not None
        assert self.constraints_matrix_kinematic.value is not None
        assert self.constraints_matrix_estimated.value is not None

        # Extract constraints for the specified index
        kin_coefs = self.constraints_matrix_kinematic.value[index]
        est_coefs = self.constraints_matrix_estimated.value[index]
        const_vals = self.constraints_vector.value[index]

        # Create a figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get the kinematic bounds for this index
        k_lower = self.kinematic_lower[index]
        k_upper = self.kinematic_upper[index]

        # Define the range for visualization
        k_range = np.linspace(min(k_lower, -1) - 10, max(k_upper, 1) + 10, 1000)

        # Plot the constraints
        colors = ["C0", "C1", "C2", "C3"]
        labels = [
            "kinematic ≤ lower",
            "kinematic ≥ upper",
            "lower-bound line",
            "upper-bound line",
        ]

        for i in range(4):
            # Skip if coefficients are zero (no constraint)
            if abs(kin_coefs[i]) < 1e-10 and abs(est_coefs[i]) < 1e-10:
                continue

            # If the coefficient for estimated is not zero, we can express estimated as a function of kinematic
            if abs(est_coefs[i]) > 1e-10:
                e_vals = (const_vals[i] - kin_coefs[i] * k_range) / est_coefs[i]
                ax.plot(k_range, e_vals, colors[i], label=labels[i])
            # Otherwise, it's a vertical line
            else:
                vert_line = const_vals[i] / kin_coefs[i]
                ax.axvline(vert_line, color=colors[i], label=labels[i])

        # Shade the feasible region
        xx, yy = np.meshgrid(
            k_range, np.linspace(min(k_lower, -1), max(k_upper, 1), 200)
        )
        feasible = np.ones_like(xx, dtype=bool)

        for i in range(4):
            if abs(est_coefs[i]) < 1e-10 and abs(kin_coefs[i]) < 1e-10:
                continue

            constraint = kin_coefs[i] * xx + est_coefs[i] * yy <= const_vals[i]
            feasible = feasible & constraint

        # Plot the feasible region
        ax.pcolormesh(
            xx, yy, feasible, alpha=0.2, cmap="Blues", zorder=10, vmin=0, vmax=1
        )

        # Add labels and grid
        ax.set_xlabel("Kinematic Slip Rate")
        ax.set_ylabel("Estimated Slip Rate")
        ax.set_title(f"Constraint Visualization for Mesh Point {index}")
        ax.grid(True)
        ax.legend()

        # Add the line y=x for reference
        ax.plot(k_range, k_range, "k--", alpha=0.5, label="y=x")

        # Add the origin
        ax.plot(0, 0, "ko", markersize=5)

        # Set equal aspect ratio
        ax.set_aspect("equal")

        return fig


@dataclass
class VelocityLimit:
    strike_slip: VelocityLimitItem
    dip_slip: VelocityLimitItem

    @classmethod
    def from_scalar(cls, length: int, lower: float, upper: float) -> "VelocityLimit":
        return VelocityLimit(
            strike_slip=VelocityLimitItem.from_scalar(length, lower, upper),
            dip_slip=VelocityLimitItem.from_scalar(length, lower, upper),
        )

    def apply_with_coupling(
        self,
        item_func: Callable[[VelocityLimitItem, CouplingItem], None],
        coupling: Coupling,
    ):
        item_func(self.strike_slip, coupling.strike_slip)
        item_func(self.dip_slip, coupling.dip_slip)
        self.strike_slip.update_constraints()
        self.dip_slip.update_constraints()

    def copy(self) -> "VelocityLimit":
        return VelocityLimit(
            strike_slip=self.strike_slip.copy(),
            dip_slip=self.dip_slip.copy(),
        )


@dataclass
class Minimizer:
    problem: CeleriProblem
    cp_problem: cp.Problem
    params_raw: cp.Expression
    params: cp.Expression
    params_scale: np.ndarray
    objective_norm2: cp.Expression
    constraint_scale: np.ndarray
    coupling: dict[int, Coupling]
    velocity_limits: dict[int, VelocityLimit] | None
    smooth_kinematic: bool = True

    def plot_coupling(self):
        n_plots = len(self.problem.segment_mesh_indices)
        fig, axes = plt.subplots(n_plots, 4, figsize=(20, 12), sharex=True, sharey=True)

        for idx in self.problem.segment_mesh_indices:
            if idx == 0:
                axes[idx, 0].set_title("strike slip")
                axes[idx, 1].set_title("smoothed strike slip")
                axes[idx, 2].set_title("dip slip")
                axes[idx, 3].set_title("smoothed dip slip")

            if idx == self.problem.segment_mesh_indices[-1]:
                for i in range(4):
                    axes[idx, i].set_xlabel("kinetic")
            axes[idx, 0].set_ylabel("estimated")

            a = self.coupling[idx].strike_slip.kinematic.value
            b = self.coupling[idx].strike_slip.estimated.value

            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 0].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

            a = self.coupling[idx].strike_slip.kinematic_smooth.value
            b = self.coupling[idx].strike_slip.estimated.value
            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 1].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

            a = self.coupling[idx].dip_slip.kinematic.value
            b = self.coupling[idx].dip_slip.estimated.value
            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 2].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

            a = self.coupling[idx].dip_slip.kinematic_smooth.value
            b = self.coupling[idx].dip_slip.estimated.value
            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 3].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

    def plot_estimation_summary(self, command):
        estimation_qp = addict.Dict()
        estimation_qp.state_vector = self.params.value
        estimation_qp.operator = self.problem.operators.eigen
        post_process_estimation_eigen(
            estimation_qp,
            self.problem.operators,
            self.problem.station,
            self.problem.index,
        )
        write_output(
            command,
            estimation_qp,
            self.problem.station,
            self.problem.segment,
            self.problem.block,
            self.problem.meshes,
        )

        plot_estimation_summary(
            command,
            self.problem.segment,
            self.problem.station,
            self.problem.meshes,
            estimation_qp,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    def out_of_bounds(self, *, tol: float = 1e-8) -> tuple[int, int]:
        oob, total = 0, 0
        for idx in self.problem.segment_mesh_indices:
            oob_mesh, total_mesh = self.coupling[idx].out_of_bounds(
                smooth_kinematic=self.smooth_kinematic, tol=tol
            )
            oob += oob_mesh
            total += total_mesh
        return oob, total

    def constraint_loss(self) -> float:
        loss = 0.0
        for idx in self.problem.segment_mesh_indices:
            loss += self.coupling[idx].constraint_loss(
                smooth_kinematic=self.smooth_kinematic
            )
        return loss


Objective = Literal["expanded_norm2", "sum_of_squares", "norm2", "norm1"]


def build_cvxpy_problem(
    problem: CeleriProblem,
    *,
    init_params_raw_value: np.ndarray | None = None,
    init_params_value: np.ndarray | None = None,
    velocity_limits: dict[int, VelocityLimit] | None = None,
    smooth_kinematic: bool = True,
    slip_rate_reduction: float | None = None,
    velocity_as_variable: bool = False,
    objective: Objective = "expanded_norm2",
    rescale_parameters: bool = True,
    rescale_constraints: bool = True,
    mixed_integer: bool = False,
) -> Minimizer:
    qp_inequality_constraints_matrix, qp_inequality_constraints_data_vector = (
        get_qp_all_inequality_operator_and_data_vector(
            problem.index,
            problem.meshes,
            problem.operators,
            problem.segment,
            problem.block,
        )
    )

    # Get data vector for KL problem
    data_vector_eigen = get_data_vector_eigen(
        problem.meshes, problem.assembly, problem.index
    )

    # Get data vector for KL problem
    weighting_vector_eigen = get_weighting_vector_eigen(
        problem.command, problem.station, problem.meshes, problem.index
    )

    C = problem.operators.eigen * np.sqrt(weighting_vector_eigen[:, None])
    d = data_vector_eigen * np.sqrt(weighting_vector_eigen)

    A = qp_inequality_constraints_matrix
    b = qp_inequality_constraints_data_vector

    if rescale_parameters:
        scale = np.abs(C).max(0)
    else:
        scale = np.ones(C.shape[1])

    C_hat = C / scale
    P_hat = C_hat.T @ C_hat

    if init_params_value is not None:
        init_params_raw_value = init_params_value * scale

    if init_params_raw_value is not None:
        params_raw = cp.Variable(
            name="params_raw",
            shape=problem.operators.eigen.shape[1],
            value=init_params_raw_value,
        )
    else:
        params_raw = cp.Variable(
            name="params_raw", shape=problem.operators.eigen.shape[1]
        )

    params = params_raw / scale

    coupling = {}
    constraints: list[cp.Constraint] = []

    def adapt_operator(array: np.ndarray):
        return sparse.csr_array(array)

    for mesh_idx in problem.segment_mesh_indices:
        # Matrix vector components of kinematic velocities
        param_slice = slice(0, 3 * len(problem.block))
        kinematic_params = params_raw[param_slice]
        kinematic_operator = (
            problem.operators.rotation_to_tri_slip_rate[mesh_idx]
            / scale[None, param_slice]
        )

        kinematic_operator = adapt_operator(kinematic_operator)

        # Matrix vector components of estimated velocities
        start = problem.index["start_col_eigen"][mesh_idx]
        end = problem.index["end_col_eigen"][mesh_idx]
        param_slice = slice(start, end)
        estimated_params = params_raw[param_slice]
        estimated_operator = (
            problem.operators.eigenvectors_to_tde_slip[mesh_idx]
            / scale[None, param_slice]
        )

        estimated_operator = adapt_operator(estimated_operator)

        smoothing_operator = problem.operators.linear_guassian_smoothing[mesh_idx]

        smoothing_operator = adapt_operator(smoothing_operator)

        # Extract strike and dip components (even and odd indices)
        indices = {
            "strike_slip": slice(None, None, 2),
            "dip_slip": slice(1, None, 2),
        }

        # Create components dictionary to store both strike and dip slip items
        components = {}

        def replace_with_constrained_var(operator, vector):
            # scale_constraint = np.abs(operator.todense()).max(1)
            if sparse.issparse(operator):
                scale_constraint = np.linalg.norm(operator.toarray(), axis=1)
            else:
                scale_constraint = np.linalg.norm(operator, axis=1)
            variable = cp.Variable(shape=operator.shape[0])
            constraint = (
                operator @ vector
            ) / scale_constraint == variable / scale_constraint
            return variable, constraint

        # Process strike and dip components with the same code
        for name, idx in indices.items():
            # Get smoothed kinematic rates for component
            kinematic_smooth_op = smoothing_operator @ kinematic_operator[idx]

            kinematic_smooth_op = adapt_operator(kinematic_smooth_op)
            kinematic_op = adapt_operator(kinematic_operator[idx])
            estimated_op = adapt_operator(estimated_operator[idx])

            kinematic_smooth = kinematic_smooth_op @ kinematic_params
            kinematic = kinematic_op @ kinematic_params
            estimated = estimated_op @ estimated_params

            if velocity_as_variable and smooth_kinematic:
                kinematic_smooth, constraint = replace_with_constrained_var(
                    kinematic_smooth_op, kinematic_params
                )
                constraints.append(constraint)
            elif velocity_as_variable and not smooth_kinematic:
                kinematic, constraint = replace_with_constrained_var(
                    kinematic_op, kinematic_params
                )
                constraints.append(constraint)

            if velocity_as_variable:
                estimated, constraint = replace_with_constrained_var(
                    estimated_op, estimated_params
                )
                constraints.append(constraint)

            # Create CouplingItem for this component
            components[name] = CouplingItem(
                kinematic=kinematic,
                kinematic_smooth=kinematic_smooth,
                estimated=estimated,
            )

        # Create and store Coupling
        coupling[mesh_idx] = Coupling(
            strike_slip=components["strike_slip"],
            dip_slip=components["dip_slip"],
        )

        # Apply velocity limits if provided
        if velocity_limits is not None:
            for name, item in components.items():
                kinematic = (
                    item.kinematic_smooth if smooth_kinematic else item.kinematic
                )
                limits = getattr(velocity_limits[mesh_idx], name)
                constraints.extend(
                    limits.build_constraints(
                        kinematic, item.estimated, mixed_integer=mixed_integer
                    )
                )

    A_hat = np.array(A / scale)

    if rescale_constraints:
        A_scale = np.linalg.norm(A_hat, axis=1)
    else:
        A_scale = np.ones(A_hat.shape[0])
    A_hat_ = A_hat / A_scale[:, None]
    b_hat = b / A_scale

    objective_norm2 = cp.norm2(C_hat @ params_raw - d)

    match objective:
        case "expanded_norm2":
            objective_val = (
                cp.quad_form(params_raw, 0.5 * P_hat, True)
                - (d.T @ C) @ params
                + 0.5 * d.T @ d
            )
        case "sum_of_squares":
            objective_val = cp.sum_squares(C_hat @ params_raw - d)
        case "norm1":
            objective_val = cp.norm1(C_hat @ params_raw - d)
        case "norm2":
            objective_val = objective_norm2
        case _:
            raise ValueError(f"Unknown objective type: {objective}")

    constraint = adapt_operator(A_hat_) @ params_raw <= b_hat

    constraints.append(constraint)

    cp_problem = cp.Problem(cp.Minimize(objective_val), constraints)

    return Minimizer(
        problem=problem,
        cp_problem=cp_problem,
        params_raw=params_raw,
        params=params,
        params_scale=scale,
        objective_norm2=objective_norm2,
        constraint_scale=A_scale,
        coupling=coupling,
        velocity_limits=velocity_limits,
        smooth_kinematic=smooth_kinematic,
    )


def _tighten_kinematic_bounds(
    minimizer: Minimizer,
    *,
    velocity_upper: float,
    velocity_lower: float,
    tighten_all: bool = True,
    factor: float = 0.5,
):
    assert factor > 0 and factor < 1

    def tighten_item(limits: VelocityLimitItem, coupling: CouplingItem):
        estimated = coupling.estimated_numpy()
        kinematic = coupling.kinematic_numpy(smooth=minimizer.smooth_kinematic)

        upper = limits.kinematic_upper.copy()
        lower = limits.kinematic_lower.copy()

        if tighten_all:
            target = kinematic
            diff = upper - target
            upper = target + factor * diff
            diff = lower - target
            lower = target + factor * diff
        else:
            pos = kinematic > 0
            oob = (
                pos
                & ((estimated > kinematic) | (estimated < 0))
                & (lower < 0)
                & (upper > 0)
            )
            target = kinematic[oob]
            diff = upper[oob] - target
            upper[oob] = target + factor * diff
            diff = lower[oob] - target
            lower[oob] = target + factor * diff

            neg = kinematic <= 0
            oob = (
                neg
                & ((estimated > 0) | (estimated < kinematic))
                & (lower < 0)
                & (upper > 0)
            )
            target = kinematic[oob]
            diff = upper[oob] - target
            upper[oob] = target + factor * diff
            diff = lower[oob] - target
            lower[oob] = target + factor * diff

        # Just fix the sign once the interval only positive or negative
        fixed_sign = lower >= 0
        lower[fixed_sign] = 0.0
        upper[fixed_sign] = velocity_upper

        fixed_sign = upper <= 0
        lower[fixed_sign] = velocity_lower
        upper[fixed_sign] = 0.0

        limits.kinematic_lower = lower
        limits.kinematic_upper = upper

    limits = minimizer.velocity_limits
    if limits is None:
        raise ValueError("Velocity limits have not been set")

    velocity_limits = {}
    for idx in minimizer.problem.segment_mesh_indices:
        velocity_limits[idx] = limits[idx].apply_with_coupling(
            tighten_item, minimizer.coupling[idx]
        )


@dataclass
class MinimizerTrace:
    problem: CeleriProblem
    params: list[np.ndarray]
    params_raw: list[np.ndarray]
    coupling: list[dict[int, Coupling]]
    velocit_limits: list[dict[int, VelocityLimit] | None]
    objective: list[float]
    objective_norm2: list[float]
    nonconvex_constraint_loss: list[float]
    out_of_bounds: list[int]
    iter_time: list[float]
    total_time: float
    start_time: float
    last_update_time: float
    minimizer: Minimizer

    def __init__(self, minimizer: Minimizer):
        self.problem = minimizer.problem
        self.params = []
        self.params_raw = []
        self.coupling = []
        self.velocit_limits = []
        self.objective = []
        self.objective_norm2 = []
        self.nonconvex_constraint_loss = []
        self.out_of_bounds = []
        self.iter_time = []
        self.total_time = 0.0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.out_of_bounds = []
        self.minimizer = minimizer

    def print_last_progress(self):
        total = 2 * self.problem.total_mesh_points
        iter_num = len(self.objective)
        oob = self.out_of_bounds[-1]
        objective = self.objective_norm2[-1]
        nonconvex_loss = self.minimizer.constraint_loss()
        iter_time = self.iter_time[-1]

        print(f"Iteration: {iter_num}")
        print(f"{oob} of {total} velocities are out-of-bounds")
        print(f"Non-convex constraint loss: {nonconvex_loss:.2e}")
        print(f"residual 2-norm: {objective:.5e}")
        print(f"Iteration took {iter_time:.2f}s")
        print()

    def store_current(self):
        assert self.minimizer.params.value is not None
        self.params.append(self.minimizer.params.value)
        assert self.minimizer.params_raw.value is not None
        self.params_raw.append(self.minimizer.params_raw.value)
        self.coupling.append(
            {
                idx: self.minimizer.coupling[idx].copy_numpy()
                for idx in self.problem.segment_mesh_indices
            }
        )
        self.velocit_limits.append(self.minimizer.velocity_limits)
        assert self.minimizer.cp_problem.objective.value is not None
        self.objective.append(cast(float, self.minimizer.cp_problem.objective.value))
        self.objective_norm2.append(cast(float, self.minimizer.objective_norm2.value))

        current_time = time.time()
        self.iter_time.append(current_time - self.last_update_time)
        self.total_time += current_time - self.last_update_time
        self.last_update_time = current_time

        self.out_of_bounds.append(self.minimizer.out_of_bounds()[0])
        self.nonconvex_constraint_loss.append(self.minimizer.constraint_loss())


def _custom_cvxopt_solve(problem: cp.Problem, **kwargs):
    """Solve a cvxpy problem with the clarabel reduction but cvxopt solver."""
    data, chain, inverse_data = problem.get_problem_data(
        cp.CLARABEL,
        ignore_dpp=kwargs.get("ignore_dpp", True),
    )

    # Check that ignore_dpp is the only key
    if len(kwargs) > 1:
        raise ValueError("Only 'ignore_dpp' is allowed as a keyword argument.")
    elif len(kwargs) == 0:
        pass
    elif "ignore_dpp" not in kwargs:
        raise ValueError("Only 'ignore_dpp' is allowed as a keyword argument.")

    P = data["P"].tocsc()
    c = data["c"]
    A = data["A"].tocsc()
    b = data["b"]
    dims = data["dims"]

    cvxopt_result = cvxopt.solvers.coneqp(
        cvxopt.matrix(P.todense()),
        cvxopt.matrix(c),
        cvxopt.matrix(A[dims.zero : dims.zero + dims.nonneg].todense()),
        cvxopt.matrix(b[dims.zero : dims.zero + dims.nonneg]),
        A=cvxopt.matrix(A[: dims.zero].todense()),
        b=cvxopt.matrix(b[: dims.zero]),
    )

    Solution = namedtuple(
        "DefaultSolution",
        [
            "x",
            "s",
            "z",
            "status",
            "obj_val",
            "obj_val_dual",
            "solve_time",
            "iterations",
            "r_prim",
            "r_dual",
        ],
    )

    if cvxopt_result["status"]:
        status = "Solved"
    else:
        status = "NumericalError"

    sol = Solution(
        x=np.array(cvxopt_result["x"]).ravel(),
        s=np.array(cvxopt_result["s"]).ravel(),
        z=np.array(cvxopt_result["z"]).ravel(),
        status=status,
        obj_val=cvxopt_result["primal objective"],
        obj_val_dual=cvxopt_result["dual objective"],
        solve_time=None,
        iterations=cvxopt_result["iterations"],
        r_prim=cvxopt_result["primal slack"],
        r_dual=cvxopt_result["dual slack"],
    )

    problem.unpack_results(sol, chain, inverse_data)


def _custom_solve(problem: cp.Problem, solver: str, objective: Objective, **kwargs):
    if solver == "CUSTOM_CVXOPT":
        if objective != "expanded_norm2":
            raise ValueError(
                "CUSTOM_CVXOPT solver only supports expanded_norm2 objective"
            )
        _custom_cvxopt_solve(problem, **kwargs)
    else:
        problem.solve(solver=solver, **kwargs)


def minimize(
    problem: CeleriProblem,
    *,
    velocity_upper: float,
    velocity_lower: float,
    max_iter: int = 20,
    smooth_kinematic: bool = True,
    solve_kwargs: dict | None = None,
    reduction_factor: float = 0.5,
    verbose: bool = False,
    rescale_parameters: bool = True,
    rescale_constraints: bool = True,
    objective: Literal[
        "expanded_norm2", "sum_of_squares", "norm1", "norm2"
    ] = "expanded_norm2",
) -> MinimizerTrace:
    """
    Iteratively solve a constrained optimization problem for fault slip rates.

    Performs multiple iterations of solving the convex problem, tightening bounds
    after each iteration until all velocities satisfy constraints or max iterations reached.

    Args:
        problem: The Celeri problem definition containing mesh and operator data
        velocity_upper: Maximum allowed velocity value
        velocity_lower: Minimum allowed velocity value
        max_iter: Maximum number of optimization iterations
        smooth_kinematic: Whether to use smoothed kinematic velocities
        solve_kwargs: Additional keyword arguments passed to the solver
        reduction_factor: Factor to reduce bounds by in each iteration (0-1)
        verbose: Whether to print progress information

    Returns:
        A trace object containing the optimization history
    """
    limits = {}
    for idx in problem.segment_mesh_indices:
        length = problem.meshes[idx]["n_tde"]
        limits[idx] = VelocityLimit.from_scalar(length, velocity_lower, velocity_upper)

    minimizer = build_cvxpy_problem(
        problem,
        velocity_limits=limits,
        smooth_kinematic=smooth_kinematic,
        rescale_parameters=rescale_parameters,
        rescale_constraints=rescale_constraints,
        objective=objective,
    )
    trace = MinimizerTrace(minimizer)

    default_solve_kwargs = {
        "solver": "CLARABEL",
        "ignore_dpp": True,
        "warm_start": False,
    }

    if solve_kwargs is not None:
        default_solve_kwargs.update(solve_kwargs)

    solver = default_solve_kwargs.pop("solver")

    for num_iter in range(max_iter):
        _custom_solve(
            minimizer.cp_problem,
            solver=solver,
            objective=objective,
            **default_solve_kwargs,
        )
        trace.store_current()
        if verbose:
            trace.print_last_progress()

        num_oob, total = minimizer.out_of_bounds()
        if num_oob == 0:
            break

        _tighten_kinematic_bounds(
            minimizer,
            velocity_upper=velocity_upper,
            velocity_lower=velocity_lower,
            factor=reduction_factor,
            tighten_all=True,
        )

    return trace


def benchmark_solve(
    problem: CeleriProblem,
    *,
    with_limits: tuple[float, float] | None,
    objective: Objective,
    rescale_parameters: bool,
    rescale_constraints: bool,
    solver: str,
    solve_kwargs: dict | None = None,
):
    """Benchmark the performance of solving a CeleriProblem with different configurations.

    Args:
        problem: The Celeri problem to solve
        with_limits: Optional tuple of (lower, upper) velocity limits
        objective: Type of objective function to use
        rescale_parameters: Whether to rescale parameters
        rescale_constraints: Whether to rescale constraints
        solver: Name of the solver to use
        solver_kwargs: Additional solver-specific parameters

    Returns:
        Dictionary containing benchmark results including timing, success status,
        objective values, parameter values, and any error messages.
    """
    if with_limits is not None:
        limits = {}
        for idx in problem.segment_mesh_indices:
            length = problem.meshes[idx]["n_tde"]
            lower, upper = with_limits
            limits[idx] = VelocityLimit.from_scalar(length, lower, upper)
    else:
        limits = None

    minimizer = build_cvxpy_problem(
        problem,
        velocity_limits=limits,
        objective=objective,
        rescale_parameters=rescale_parameters,
        rescale_constraints=rescale_constraints,
        mixed_integer=False,
    )

    default_solve_kwargs = {
        "ignore_dpp": True,
        "warm_start": False,
    }

    if solve_kwargs is not None:
        default_solve_kwargs.update(solve_kwargs)

    start = time.time()
    try:
        _custom_solve(
            minimizer.cp_problem,
            solver=solver,
            objective=objective,
            **default_solve_kwargs,
        )
    except Exception as e:
        success = False
        error = str(e)
    else:
        success = True
        error = None
    end = time.time()

    return {
        "time": end - start,
        "success": success,
        "objective_value": minimizer.cp_problem.objective.value,
        "objective_norm2": minimizer.objective_norm2.value,
        "params": np.array(minimizer.params.value),
        "params_raw": np.array(minimizer.params_raw.value),
        "error": error,
        "solver": solver,
        "solver_kwargs": solve_kwargs,
        "limits": with_limits,
        "objective": objective,
        "rescale_parameters": rescale_parameters,
        "rescale_constraints": rescale_constraints,
    }
