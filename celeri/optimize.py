from dataclasses import dataclass
from typing import Callable, cast, Any
import addict
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy import sparse
import matplotlib.pyplot as plt

from celeri import (
    post_process_estimation_eigen,
    write_output,
    plot_estimation_summary,
    get_qp_all_inequality_operator_and_data_vector,
    get_data_vector_eigen,
    get_weighting_vector_eigen,
)


@dataclass
class CeleriProblem:
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


@dataclass
class CouplingItem:
    kinematic: cp.Expression
    kinematic_smooth: cp.Expression
    estimated: cp.Expression

    def out_of_bounds(self, *, smooth_kinematic: bool, tol=1e-8) -> tuple[int, int]:
        if smooth_kinematic:
            kinematic = self.kinematic_smooth.value
        else:
            kinematic = self.kinematic.value
        estimated = self.estimated.value

        if kinematic is None or estimated is None:
            raise ValueError("Coupling has not been fit")

        total = len(kinematic)
        is_oob = estimated**2 - estimated * kinematic > tol
        return is_oob.sum(), total

    def constraint_loss(self, *, smooth_kinematic: bool) -> float:
        if smooth_kinematic:
            kinematic = self.kinematic_smooth.value
        else:
            kinematic = self.kinematic.value
        estimated = self.estimated.value

        if kinematic is None:
            raise ValueError("Coupling has not been fit")
        if estimated is None:
            raise ValueError("Coupling has not been fit")

        constraint = estimated**2 - estimated * kinematic
        return np.clip(constraint, 0, np.inf).sum()


@dataclass
class Coupling:
    strike_slip: CouplingItem
    dip_slip: CouplingItem
    combined: CouplingItem

    def out_of_bounds(
        self, *, smooth_kinematic: bool, tol: float = 1e-8
    ) -> tuple[int, int]:
        return self.combined.out_of_bounds(smooth_kinematic=smooth_kinematic, tol=tol)

    def constraint_loss(self, *, smooth_kinematic: bool) -> float:
        return self.combined.constraint_loss(smooth_kinematic=smooth_kinematic)


@dataclass
class VelocityLimitItem:
    kinematic_lower: np.ndarray
    kinematic_upper: np.ndarray
    constraints_matrix_kinematic: cp.Parameter | None = None
    constraints_matrix_estimated: cp.Parameter | None = None
    constraints_vector: cp.Parameter | None = None

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
            self.constraints_matrix_kinematic = cp.Parameter(shape=(n_dim, 4))
            self.constraints_matrix_estimated = cp.Parameter(shape=(n_dim, 4))
            self.constraints_vector = cp.Parameter(shape=(n_dim, 4))

        kinematic = np.zeros(self.constraints_matrix_kinematic.shape)
        estimated = np.zeros(self.constraints_matrix_estimated.shape)
        const = np.zeros(self.constraints_vector.shape)

        # kinematic <= kinematic_lower
        kinematic[:, 0] = np.where(self.kinematic_lower == 0.0, 0.0, -1)
        const[:, 0] = np.where(self.kinematic_lower == 0.0, 0.0, -self.kinematic_lower)

        # kinematic >= kinematic_upper
        kinematic[:, 1] = np.where(self.kinematic_upper == 0.0, 0.0, 1)
        const[:, 1] = np.where(self.kinematic_upper == 0.0, 0.0, self.kinematic_upper)

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
            Computes coefficients for a line inequality passing through two points.

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
        kinematic[:, 2] = coef_x
        estimated[:, 2] = coef_y
        const[:, 2] = bound_const

        coef_x, coef_y, bound_const = bounded_through_points(
            upper_right_kinematic,
            upper_right_estimated,
            upper_left_kinematic,
            upper_left_estimated,
        )
        kinematic[:, 3] = coef_x
        estimated[:, 3] = coef_y
        const[:, 3] = bound_const

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

    def build_constraints(self, kinematic, estimated) -> cp.Constraint:
        self.update_constraints()
        assert self.constraints_vector is not None
        assert self.constraints_matrix_kinematic is not None
        assert self.constraints_matrix_estimated is not None

        return (
            cp.multiply(self.constraints_matrix_kinematic, kinematic[:, None])
            + cp.multiply(self.constraints_matrix_estimated, estimated[:, None])
            <= self.constraints_vector
        )

    def plot_constraint(self, index=0, figsize=(8, 6)):
        """
        Plots the constraint boundaries for a specific mesh point.

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


@dataclass
class Minimizer:
    problem: CeleriProblem
    cp_problem: cp.Problem
    params_raw: cp.Expression
    params: cp.Expression
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


def build_cvxpy_problem(
    problem: CeleriProblem,
    *,
    init_params_raw_value: np.ndarray | None = None,
    init_params_value: np.ndarray | None = None,
    velocity_limits: dict[int, VelocityLimit] | None = None,
    mccormick: bool = False,
    smooth_kinematic: bool = True,
    slip_rate_reduction: float | None = None,
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

    # TODO check
    # minimize norm2(C @ p - d)^2, s.t. A @ p <= b
    # We expand norm2(C @ p - d)^2 = p^T C^T @ C @ p - 2 * d @ C^T @ p + d^T d
    # define P = C^T C
    # dims(C) = (observation, parameter)
    # dims(P) = (parameter, parameter)

    scale = np.abs(C).max(0)

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
    constraints = []

    #to_kinematic_tde_rates = problem.operators.rotation_to_tri_slip_rate[mesh_idx]
    #p = params[0 : 3 * len(problem.block)]

    #scale_ = np.max(np.abs(to_kinematic_tde_rates), axis=1)
    #if p.value is not None:
    #    val = to_kinematic_tde_rates @ p.value
    #else:
    #    val = None

    #kinematic_tde_rates = cp.Variable(shape=(to_kinematic_tde_rates.shape[0],), value=val)
    #constraints.append(kinematic_tde_rates / scale_ == (sparse.csr_array(to_kinematic_tde_rates) @ p) / scale_)


    for mesh_idx in problem.segment_mesh_indices:
        kinematic_tde_rates = (
            sparse.coo_array(problem.operators.rotation_to_tri_slip_rate[mesh_idx])
            @ params[0 : 3 * len(problem.block)]
        )

        # Get estimated elastic rates on mesh elements
        estimated_tde_rates = (
            sparse.coo_array(problem.operators.eigenvectors_to_tde_slip[mesh_idx])
            @ params[
                problem.index["start_col_eigen"][mesh_idx] : problem.index[
                    "end_col_eigen"
                ][mesh_idx]
            ]
        )

        def get_coupling_linear(estimated_slip, kinematic_slip):
            # Smooth kinematic slip
            kinematic_slip = (
                sparse.coo_array(problem.operators.linear_guassian_smoothing[mesh_idx])
                @ kinematic_slip
            )

            coupling = estimated_slip / kinematic_slip
            return coupling, kinematic_slip

        # Calculate strike-slip and dip-slip coupling with linear coupling matrix
        tde_coupling_ss, kinematic_tde_rates_ss_smooth = get_coupling_linear(
            estimated_tde_rates[0::2],
            kinematic_tde_rates[0::2],
        )
        tde_coupling_ds, kinematic_tde_rates_ds_smooth = get_coupling_linear(
            estimated_tde_rates[1::2],
            kinematic_tde_rates[1::2],
        )

        strike_slip = CouplingItem(
            kinematic=kinematic_tde_rates[0::2],
            kinematic_smooth=kinematic_tde_rates_ss_smooth,
            estimated=estimated_tde_rates[0::2],
        )

        dip_slip = CouplingItem(
            kinematic=kinematic_tde_rates[1::2],
            kinematic_smooth=kinematic_tde_rates_ds_smooth,
            estimated=estimated_tde_rates[1::2],
        )

        combined = CouplingItem(
            kinematic=kinematic_tde_rates,
            kinematic_smooth=cast(
                cp.Expression,
                cp.concatenate(
                    [
                        kinematic_tde_rates_ss_smooth[:, None],
                        kinematic_tde_rates_ds_smooth[:, None],
                    ],
                    axis=1,
                ),
            ),
            estimated=estimated_tde_rates,
        )

        mesh_coupling = Coupling(
            strike_slip=strike_slip,
            dip_slip=dip_slip,
            combined=combined,
        )
        coupling[mesh_idx] = mesh_coupling

        for name, item in {
            "strike_slip": mesh_coupling.strike_slip,
            "dip_slip": mesh_coupling.dip_slip,
        }.items():
            estimated = item.estimated
            if smooth_kinematic:
                kinematic = item.kinematic_smooth
            else:
                kinematic = item.kinematic

            if velocity_limits is None:
                continue

            limits = getattr(velocity_limits[mesh_idx], name)
            constraints.append(limits.build_constraints(kinematic, estimated))

    A_hat = np.array(A / scale)
    A_scale = np.abs(A_hat).max(1)
    A_hat_ = A_hat / A_scale[:, None]
    b_hat = b / A_scale

    # TODO add the constant term to the objectivo to make it easier to interpret
    objective = cp.Minimize(
        cp.quad_form(params_raw, 0.5 * P_hat, True) - (d.T @ C) @ params
    )
    constraint = sparse.csr_array(A_hat_) @ params_raw <= b_hat

    constraints.append(constraint)

    cp_problem = cp.Problem(objective, constraints)

    return Minimizer(
        problem=problem,
        cp_problem=cp_problem,
        params_raw=params_raw,
        params=params,
        coupling=coupling,
        velocity_limits=velocity_limits,
        smooth_kinematic=smooth_kinematic,
    )


def tighten_kinematic_bounds(
    minimizer: Minimizer,
    max_limit: float,
    tighten_all: bool = True,
    factor: float = 0.5,
):
    assert factor > 0 and factor < 1

    def tighten_item(limits: VelocityLimitItem, coupling: CouplingItem):
        estimated = coupling.estimated.value
        if minimizer.smooth_kinematic:
            kinematic = coupling.kinematic_smooth.value
        else:
            kinematic = coupling.kinematic.value

        if estimated is None or kinematic is None:
            raise ValueError("Minimizer has not been fit")

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
        upper[fixed_sign] = max_limit

        fixed_sign = upper <= 0
        lower[fixed_sign] = -max_limit
        upper[fixed_sign] = 0.0

        limits.kinematic_lower = lower
        limits.kinematic_upper = upper

    limits = minimizer.velocity_limits
    if limits is None:
        raise ValueError("Velocity limits have not been set")

    velocity_limits = {}
    for idx in minimizer.problem.segment_mesh_indices:
        length = minimizer.problem.meshes[idx]["n_tde"]
        velocity_limits[idx] = limits[idx].apply_with_coupling(
            tighten_item, minimizer.coupling[idx]
        )
