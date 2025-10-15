from __future__ import annotations

import time
import warnings
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import cvxopt
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import linalg, sparse

from celeri.mesh import ScalarBound
from celeri.model import Model
from celeri.operators import (
    Operators,
    build_operators,
    get_qp_all_inequality_operator_and_data_vector,
)
from celeri.plot import (
    plot_estimation_summary,
)
from celeri.solve import Estimation, build_estimation


@dataclass
class SlipRateItem:
    """Kinematic and elastic slip rates on a single mesh, either for strike or dip slip."""

    kinematic: cp.Expression | np.ndarray | None
    kinematic_smooth: cp.Expression | np.ndarray | None
    elastic: cp.Expression | np.ndarray

    def copy_numpy(self) -> SlipRateItem:
        """Copy this item with all values as numpy arrays."""
        return SlipRateItem(
            kinematic=self.kinematic_numpy(smooth=False),
            kinematic_smooth=self.kinematic_numpy(smooth=True),
            elastic=self.elastic_numpy(),
        )

    def kinematic_numpy(self, *, smooth: bool) -> np.ndarray | None:
        """Return kinematic slip rates as a numpy array."""
        if smooth:
            kinematic = self.kinematic_smooth
        else:
            kinematic = self.kinematic

        if isinstance(kinematic, cp.Expression):
            kinematic = kinematic.value

        return kinematic

    def elastic_numpy(self) -> np.ndarray:
        """Return elastic slip rates as a numpy array."""
        elastic = self.elastic

        if isinstance(elastic, cp.Expression):
            elastic = elastic.value

        if elastic is None:
            raise ValueError("Coupling has not been fit")

        return elastic

    def out_of_bounds_coupling(
        self, *, smooth_kinematic: bool, tol=1e-8, coupling_bounds: ScalarBound
    ) -> tuple[int, int]:
        """Count slip rates that violate coupling constraints.

        Args:
            smooth_kinematic: Whether to use smoothed kinematic values to
            compute the couplings.
            tol: Tolerance for constraint violation.
            coupling_bounds: Coupling bounds to apply to the slip rates.

        Returns:
            Tuple of (number of out-of-bounds points, total points).
        """
        if coupling_bounds.lower is None and coupling_bounds.upper is None:
            # No coupling bounds defined, so no out-of-bounds points
            return 0, 0

        kinematic = self.kinematic_numpy(smooth=smooth_kinematic)
        if kinematic is None:
            raise ValueError("Invalid coupling constraint on non-segment mesh")

        elastic = self.elastic_numpy()
        lower = coupling_bounds.lower
        upper = coupling_bounds.upper

        total = len(kinematic)
        is_oob = (elastic - lower * kinematic) * (elastic - upper * kinematic) > tol
        return is_oob.sum(), total

    def constraint_loss(
        self, *, smooth_kinematic: bool, coupling_bounds: ScalarBound
    ) -> float:
        """Calculate quadratic coupling constraint violation"""
        if coupling_bounds.lower is None and coupling_bounds.upper is None:
            # No coupling bounds defined, so no out-of-bounds points
            return 0.0

        kinematic = self.kinematic_numpy(smooth=smooth_kinematic)
        if kinematic is None:
            raise ValueError("Invalid coupling constraint on non-segment mesh")
        elastic = self.elastic_numpy()
        lower = coupling_bounds.lower
        upper = coupling_bounds.upper

        constraint = (elastic - lower * kinematic) * (elastic - upper * kinematic)
        return np.clip(constraint, 0, np.inf).sum()


@dataclass
class SlipRate:
    """Slip rates on a single mesh."""

    strike_slip: SlipRateItem
    dip_slip: SlipRateItem

    def out_of_bounds_coupling(
        self, *, smooth_kinematic: bool, tol: float = 1e-8, limits: SlipRateLimit
    ) -> tuple[int, int]:
        oob1, total1 = self.strike_slip.out_of_bounds_coupling(
            smooth_kinematic=smooth_kinematic,
            tol=tol,
            coupling_bounds=limits.strike_slip.coupling_bounds,
        )
        oob2, total2 = self.dip_slip.out_of_bounds_coupling(
            smooth_kinematic=smooth_kinematic,
            tol=tol,
            coupling_bounds=limits.dip_slip.coupling_bounds,
        )
        return oob1 + oob2, total1 + total2

    def out_of_bounds_detailed(
        self, *, smooth_kinematic: bool, tol: float = 1e-8, limits: SlipRateLimit
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Count slip rates that violate coupling constraints with detailed output."""
        oob1, total1 = self.strike_slip.out_of_bounds_coupling(
            smooth_kinematic=smooth_kinematic,
            tol=tol,
            coupling_bounds=limits.strike_slip.coupling_bounds,
        )
        oob2, total2 = self.dip_slip.out_of_bounds_coupling(
            smooth_kinematic=smooth_kinematic,
            tol=tol,
            coupling_bounds=limits.dip_slip.coupling_bounds,
        )
        return (oob1, total1), (oob2, total2)

    def constraint_loss(
        self, *, smooth_kinematic: bool, limits: SlipRateLimit
    ) -> float:
        loss1 = self.strike_slip.constraint_loss(
            smooth_kinematic=smooth_kinematic,
            coupling_bounds=limits.strike_slip.coupling_bounds,
        )
        loss2 = self.dip_slip.constraint_loss(
            smooth_kinematic=smooth_kinematic,
            coupling_bounds=limits.dip_slip.coupling_bounds,
        )
        return loss1 + loss2

    def copy_numpy(self) -> SlipRate:
        return SlipRate(
            strike_slip=self.strike_slip.copy_numpy(),
            dip_slip=self.dip_slip.copy_numpy(),
        )


@dataclass(kw_only=True)
class SlipRateLimitItem:
    """Slip rate limits for a single mesh, either for strike or dip slip."""

    kinematic_lower: np.ndarray | None
    kinematic_upper: np.ndarray | None
    constraints_matrix_kinematic: cp.Parameter | None = None
    constraints_matrix_elastic: cp.Parameter | None = None
    constraints_vector: cp.Parameter | None = None

    # Global bounds on the elastic slip rates from the config
    elastic_bounds: ScalarBound
    coupling_bounds: ScalarBound

    def copy(self) -> SlipRateLimitItem:
        return SlipRateLimitItem(
            kinematic_lower=self.kinematic_lower.copy()
            if self.kinematic_lower is not None
            else None,
            kinematic_upper=self.kinematic_upper.copy()
            if self.kinematic_upper is not None
            else None,
            constraints_matrix_kinematic=None,
            constraints_matrix_elastic=None,
            constraints_vector=None,
            elastic_bounds=self.elastic_bounds,
            coupling_bounds=self.coupling_bounds,
        )

    @classmethod
    def from_scalar(
        cls,
        length: int,
        *,
        elastic_bounds: ScalarBound,
        coupling_bounds: ScalarBound,
        kinematic_hint: ScalarBound,
    ) -> SlipRateLimitItem:
        if coupling_bounds.lower is not None:
            hint = kinematic_hint.lower
            if hint is None:
                raise ValueError("kinematic slip rate lower hint must be defined.")
            lower = np.full((length,), hint)
        else:
            lower = None
        if coupling_bounds.upper is not None:
            hint = kinematic_hint.upper
            if hint is None:
                raise ValueError("kinematic slip rate upper hint must be defined.")
            upper = np.full((length,), hint)
        else:
            upper = None

        assert not ((lower is None) ^ (upper is None))
        return cls(
            kinematic_lower=lower,
            kinematic_upper=upper,
            elastic_bounds=elastic_bounds,
            coupling_bounds=coupling_bounds,
        )

    def update_constraints(self):
        if self.coupling_bounds.lower is None and self.coupling_bounds.upper is None:
            # No coupling limits defined, so no constraints to update
            self.constraints_matrix_kinematic = None
            self.constraints_matrix_elastic = None
            self.constraints_vector = None
            return
        if self.coupling_bounds.lower is None or self.coupling_bounds.upper is None:
            raise ValueError("Both coupling lower and upper bounds must be defined.")
        coupling_lower = self.coupling_bounds.lower
        coupling_upper = self.coupling_bounds.upper

        assert self.kinematic_lower is not None and self.kinematic_upper is not None

        if (
            self.constraints_matrix_kinematic is None
            or self.constraints_matrix_elastic is None
            or self.constraints_vector is None
        ):
            n_dim = len(self.kinematic_lower)
            # dims = (mesh_point, constraint, (kinematic, elastic))
            self.constraints_matrix_kinematic = cp.Parameter(shape=(n_dim, 2))
            self.constraints_matrix_elastic = cp.Parameter(shape=(n_dim, 2))
            self.constraints_vector = cp.Parameter(shape=(n_dim, 2))

        kinematic = np.zeros(self.constraints_matrix_kinematic.shape)
        elastic = np.zeros(self.constraints_matrix_elastic.shape)
        const = np.zeros(self.constraints_vector.shape)

        # Define boundary points in (kinematic, elastic) space

        # The lower left corner of the subset
        lower_left_kinematic = self.kinematic_lower
        lower_left_elastic = np.where(
            self.kinematic_lower < 0,
            self.kinematic_lower * coupling_upper,
            self.kinematic_lower * coupling_lower,
        )

        # The upper left corner of the subset
        upper_left_kinematic = self.kinematic_lower
        upper_left_elastic = np.where(
            self.kinematic_lower < 0,
            self.kinematic_lower * coupling_lower,
            self.kinematic_lower * coupling_upper,
        )

        # The lower right corner of the subset
        lower_right_kinematic = self.kinematic_upper
        lower_right_elastic = np.where(
            self.kinematic_upper < 0,
            self.kinematic_upper * coupling_upper,
            self.kinematic_upper * coupling_lower,
        )

        # The upper right corner of the subset
        upper_right_kinematic = self.kinematic_upper
        upper_right_elastic = np.where(
            self.kinematic_upper < 0,
            self.kinematic_upper * coupling_lower,
            self.kinematic_upper * coupling_upper,
        )

        def bounded_through_points(
            x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Compute coefficients for a line inequality passing through two points.

            This is vectorized, so all the arrays are expected to have the same shape.

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
            lower_left_elastic,
            lower_right_kinematic,
            lower_right_elastic,
        )
        kinematic[:, 0] = coef_x
        elastic[:, 0] = coef_y
        const[:, 0] = bound_const

        coef_x, coef_y, bound_const = bounded_through_points(
            upper_right_kinematic,
            upper_right_elastic,
            upper_left_kinematic,
            upper_left_elastic,
        )
        kinematic[:, 1] = coef_x
        elastic[:, 1] = coef_y
        const[:, 1] = bound_const

        # Rescale matrices so that the largest coefficient for each inequality is 1
        for i in range(kinematic.shape[1]):
            # Find the maximum absolute value in each inequality constraint
            max_coef = np.maximum(np.abs(kinematic[:, i]), np.abs(elastic[:, i]))

            # Avoid division by zero
            max_coef = np.where(max_coef > 0, max_coef, 1.0)

            # Rescale the coefficients
            kinematic[:, i] /= max_coef
            elastic[:, i] /= max_coef
            const[:, i] /= max_coef

        self.constraints_matrix_kinematic.value = kinematic
        self.constraints_matrix_elastic.value = elastic
        self.constraints_vector.value = const

    def build_constraints(self, kinematic, elastic) -> list[cp.Constraint]:
        constraints = []

        if (
            self.coupling_bounds.lower is not None
            or self.coupling_bounds.upper is not None
        ):
            self.update_constraints()
            assert self.constraints_vector is not None
            assert self.constraints_matrix_kinematic is not None
            assert self.constraints_matrix_elastic is not None

            constraints.append(
                cp.multiply(self.constraints_matrix_kinematic, kinematic[:, None])
                + cp.multiply(self.constraints_matrix_elastic, elastic[:, None])
                <= self.constraints_vector,
            )

        if self.elastic_bounds.lower is not None:
            constraints.append(elastic >= self.elastic_bounds.lower)
        if self.elastic_bounds.upper is not None:
            constraints.append(elastic <= self.elastic_bounds.upper)
        return constraints

    def plot_constraint(self, index=0, figsize=(8, 6)):
        """Plot the constraint boundaries for a specific mesh point.

        Args:
            index: Index of the mesh point to visualize
            figsize: Size of the figure to create

        Returns:
            matplotlib Figure object
        """
        self.update_constraints()

        assert self.constraints_vector is not None
        assert self.constraints_matrix_kinematic is not None
        assert self.constraints_matrix_elastic is not None
        assert self.constraints_vector.value is not None
        assert self.constraints_matrix_kinematic.value is not None
        assert self.constraints_matrix_elastic.value is not None

        # Extract constraints for the specified index
        kin_coefs = self.constraints_matrix_kinematic.value[index]
        est_coefs = self.constraints_matrix_elastic.value[index]
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

            # If the coefficient for elastic is not zero, we can express elastic as a function of kinematic
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
        ax.set_ylabel("Elastic Slip Rate")
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
class SlipRateLimit:
    strike_slip: SlipRateLimitItem
    dip_slip: SlipRateLimitItem

    @classmethod
    def from_scalar(
        cls,
        length: int,
        elastic_bounds: ScalarBound,
        coupling_bounds: ScalarBound,
        kinematic_hint: ScalarBound,
    ) -> SlipRateLimit:
        return SlipRateLimit(
            strike_slip=SlipRateLimitItem.from_scalar(
                length,
                kinematic_hint=kinematic_hint,
                elastic_bounds=elastic_bounds,
                coupling_bounds=coupling_bounds,
            ),
            dip_slip=SlipRateLimitItem.from_scalar(
                length,
                kinematic_hint=kinematic_hint,
                elastic_bounds=elastic_bounds,
                coupling_bounds=coupling_bounds,
            ),
        )

    @classmethod
    def from_model(
        cls,
        model: Model,
    ) -> list[SlipRateLimit]:
        """Create velocity limits from the model configuration."""
        limits = []
        for mesh in model.meshes:
            ss = SlipRateLimitItem.from_scalar(
                mesh.n_tde,
                elastic_bounds=mesh.config.elastic_constraints_ss,
                coupling_bounds=mesh.config.coupling_constraints_ss,
                kinematic_hint=mesh.config.sqp_kinematic_slip_rate_hint_ss,
            )

            ds = SlipRateLimitItem.from_scalar(
                mesh.n_tde,
                elastic_bounds=mesh.config.elastic_constraints_ds,
                coupling_bounds=mesh.config.coupling_constraints_ds,
                kinematic_hint=mesh.config.sqp_kinematic_slip_rate_hint_ds,
            )
            limits.append(SlipRateLimit(strike_slip=ss, dip_slip=ds))
        return limits

    def apply_with_coupling(
        self,
        item_func: Callable[[SlipRateLimitItem, SlipRateItem], None],
        coupling: SlipRate,
    ):
        item_func(self.strike_slip, coupling.strike_slip)
        item_func(self.dip_slip, coupling.dip_slip)
        self.strike_slip.update_constraints()
        self.dip_slip.update_constraints()

    def copy(self) -> SlipRateLimit:
        return SlipRateLimit(
            strike_slip=self.strike_slip.copy(),
            dip_slip=self.dip_slip.copy(),
        )


@dataclass
class Minimizer:
    model: Model
    operators: Operators
    cp_problem: cp.Problem
    params_raw: cp.Expression
    params: cp.Expression
    params_scale: np.ndarray
    objective_norm2: cp.Expression
    constraint_scale: np.ndarray
    slip_rate: list[SlipRate]
    slip_rate_limits: list[SlipRateLimit]
    smooth_kinematic: bool = True

    def plot_coupling(self):
        n_plots = len(self.model.segment_mesh_indices)
        fig, axes = plt.subplots(n_plots, 4, figsize=(20, 12), sharex=True, sharey=True)

        for idx in self.model.segment_mesh_indices:
            if idx == 0:
                axes[idx, 0].set_title("strike slip")
                axes[idx, 1].set_title("smoothed strike slip")
                axes[idx, 2].set_title("dip slip")
                axes[idx, 3].set_title("smoothed dip slip")

            if idx == self.model.segment_mesh_indices[-1]:
                for i in range(4):
                    axes[idx, i].set_xlabel("kinetic")
            axes[idx, 0].set_ylabel("elastic")

            a = self.slip_rate[idx].strike_slip.kinematic.value  # type: ignore
            b = self.slip_rate[idx].strike_slip.elastic.value  # type: ignore

            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 0].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

            a = self.slip_rate[idx].strike_slip.kinematic_smooth.value  # type: ignore
            b = self.slip_rate[idx].strike_slip.elastic.value  # type: ignore
            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 1].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

            a = self.slip_rate[idx].dip_slip.kinematic.value  # type: ignore
            b = self.slip_rate[idx].dip_slip.elastic.value  # type: ignore
            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 2].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

            a = self.slip_rate[idx].dip_slip.kinematic_smooth.value  # type: ignore
            b = self.slip_rate[idx].dip_slip.elastic.value  # type: ignore
            if a is None or b is None:
                raise ValueError("Problem has not been fit")
            axes[idx, 3].scatter(a, b, c=b**2 - a * b < 0, marker=".", vmin=0, vmax=1)

    def to_estimation(self) -> Estimation:
        if self.params.value is None:
            raise ValueError("Problem has not been fit")
        return build_estimation(self.model, self.operators, self.params.value)

    def plot_estimation_summary(self):
        estimation = self.to_estimation()
        plot_estimation_summary(
            self.model,
            estimation,
            quiver_scale=self.model.config.quiver_scale,
        )

    def out_of_bounds(self, *, tol: float = 1e-8) -> tuple[int, int]:
        oob, total = 0, 0
        for idx, _ in enumerate(self.model.meshes):
            oob_mesh, total_mesh = self.slip_rate[idx].out_of_bounds_coupling(
                smooth_kinematic=self.smooth_kinematic,
                tol=tol,
                limits=self.slip_rate_limits[idx],
            )
            oob += oob_mesh
            total += total_mesh
        return oob, total

    def out_of_bounds_detailed(
        self, *, tol: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Count slip rates that violate coupling constraints with detailed output."""
        oob = np.zeros((len(self.model.meshes), 2), dtype=int)
        total = np.zeros((len(self.model.meshes), 2), dtype=int)
        for idx, _ in enumerate(self.model.meshes):
            oob_mesh, total_mesh = self.slip_rate[idx].out_of_bounds_detailed(
                smooth_kinematic=self.smooth_kinematic,
                tol=tol,
                limits=self.slip_rate_limits[idx],
            )
            oob[idx] = oob_mesh
            total[idx] = total_mesh
        return oob, total

    def constraint_loss(self) -> float:
        loss = 0.0
        for idx, _ in enumerate(self.model.meshes):
            loss += self.slip_rate[idx].constraint_loss(
                smooth_kinematic=self.smooth_kinematic,
                limits=self.slip_rate_limits[idx],
            )
        return loss


Objective = Literal[
    "expanded_norm2",
    "sum_of_squares",
    "qr_sum_of_squares",
    "svd_sum_of_squares",
    "norm2",
    "norm1",
]


def build_cvxpy_problem(
    model: Model,
    *,
    init_params_raw_value: np.ndarray | None = None,
    init_params_value: np.ndarray | None = None,
    velocity_limits: list[SlipRateLimit] | None = None,
    smooth_kinematic: bool = True,
    slip_rate_reduction: float | None = None,
    objective: Objective = "qr_sum_of_squares",
    rescale_parameters: bool = True,
    rescale_constraints: bool = True,
    operators: Operators | None = None,
) -> Minimizer:
    if operators is None:
        operators = build_operators(model)

    if velocity_limits is None:
        velocity_limits = SlipRateLimit.from_model(model)

    assert operators.eigen is not None

    data_vector_eigen = operators.data_vector
    weighting_vector_eigen = operators.weighting_vector

    C = operators.full_dense_operator * np.sqrt(weighting_vector_eigen[:, None])
    d = data_vector_eigen * np.sqrt(weighting_vector_eigen)

    if rescale_parameters:
        scale = np.abs(C).max(0)
    else:
        scale = np.ones(C.shape[1])

    C_hat = C / scale
    P_hat = C_hat.T @ C_hat

    if init_params_value is not None:
        init_params_raw_value = init_params_value * scale

    assert operators.eigen is not None

    if init_params_raw_value is not None:
        params_raw = cp.Variable(
            name="params_raw",
            shape=operators.full_dense_operator.shape[1],
            value=init_params_raw_value,
        )
    else:
        params_raw = cp.Variable(
            name="params_raw", shape=operators.full_dense_operator.shape[1]
        )

    params = params_raw / scale

    slip_rate = []
    constraints: list[cp.Constraint] = []

    def adapt_operator(array: np.ndarray):
        return sparse.csr_array(array)

    for mesh_idx, _mesh in enumerate(model.meshes):
        is_segment_mesh = mesh_idx in model.segment_mesh_indices

        # Matrix vector components of kinematic velocities
        param_slice = slice(0, 3 * len(model.block))

        kinematic_params = params_raw[param_slice]
        if is_segment_mesh:
            kinematic_operator = (
                operators.rotation_to_tri_slip_rate[mesh_idx] / scale[None, param_slice]
            )
            kinematic_operator = adapt_operator(kinematic_operator)
        else:
            kinematic_params = None
            kinematic_operator = None

        # Matrix vector components of elastic velocities
        assert operators.index.eigen is not None
        start = operators.index.eigen.start_col_eigen[mesh_idx]
        end = operators.index.eigen.end_col_eigen[mesh_idx]
        param_slice = slice(start, end)
        elastic_params = params_raw[param_slice]
        elastic_operator = (
            operators.eigen.eigenvectors_to_tde_slip[mesh_idx]
            / scale[None, param_slice]
        )

        elastic_operator = adapt_operator(elastic_operator)

        smoothing_operator = operators.eigen.linear_gaussian_smoothing[mesh_idx]
        smoothing_operator = adapt_operator(smoothing_operator)

        # Extract strike and dip components (even and odd indices)
        indices = {
            "strike_slip": slice(None, None, 2),
            "dip_slip": slice(1, None, 2),
        }

        # Create components dictionary to store both strike and dip slip items
        components = {}

        # Process strike and dip components with the same code
        for name, idx in indices.items():
            # Get smoothed kinematic rates for component
            if is_segment_mesh:
                kinematic_smooth_op = smoothing_operator @ kinematic_operator[idx]

                kinematic_smooth_op = adapt_operator(kinematic_smooth_op)
                kinematic_op = adapt_operator(kinematic_operator[idx])
                kinematic_smooth = kinematic_smooth_op @ kinematic_params
                kinematic = kinematic_op @ kinematic_params
            else:
                kinematic_smooth = None
                kinematic = None

            elastic_op = adapt_operator(elastic_operator[idx])
            elastic = elastic_op @ elastic_params

            # Create CouplingItem for this component
            components[name] = SlipRateItem(
                kinematic=kinematic,
                kinematic_smooth=kinematic_smooth,
                elastic=elastic,
            )

        # Create and store Coupling
        slip_rate.append(
            SlipRate(
                strike_slip=components["strike_slip"],
                dip_slip=components["dip_slip"],
            )
        )

        # Apply velocity limits if provided
        if velocity_limits is not None:
            for name, item in components.items():
                kinematic = (
                    item.kinematic_smooth if smooth_kinematic else item.kinematic
                )
                limits = getattr(velocity_limits[mesh_idx], name)
                constraints.extend(limits.build_constraints(kinematic, item.elastic))

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
        case "qr_sum_of_squares":
            # C_hat[:, p] = q @ r
            if False:
                q, r, p = linalg.qr(C_hat, mode="economic", pivoting=True)
            else:
                out = linalg.qr(C_hat, mode="economic", pivoting=False)
                q, r = cast(tuple[np.ndarray, np.ndarray], out)
                p = np.arange(C_hat.shape[1])
            p_inv = np.argsort(p)
            np.testing.assert_allclose(q.T @ q, np.eye(len(q.T)), atol=1e-10, rtol=1e-6)
            # r = C_hat @ params_raw - d = q @ r @ P @ params_raw - d
            # y := q^T r = r @ P @ params_raw - q^T @ d
            y = cp.Variable(name="y", shape=q.shape[1])
            constraints.append(y == r[:, p_inv] @ params_raw - q.T @ d)
            objective_val = cp.sum_squares(y)
        case "svd_sum_of_squares":
            # C_hat = U @ s @ Vh
            u, s, vh = linalg.svd(C_hat, full_matrices=False)
            # r = C_hat @ params_raw - d = u @ s @ vh @ params_raw - d

            t = 0.5
            # y := diag(s) ** (-t) u^T r = diag(s)**(1 - t) @ vh @ params_raw - diag(s) ** (-t) u^T d
            y = cp.Variable(name="y", shape=s.shape[0])
            constraints.append(
                y
                == np.diag(s ** (1 - t)) @ vh @ params_raw
                - np.diag(s ** (-t)) @ u.T @ d
            )
            # objective_val = cp.sum_squares(cp.multiply(y, s ** (t)))
            objective_val = cp.quad_form(y, np.diag(s**t) ** 2)
        case "norm1":
            objective_val = cp.norm1(C_hat @ params_raw - d)
        case "norm2":
            objective_val = objective_norm2
        case _:
            raise ValueError(f"Unknown objective type: {objective}")

    A, b = get_qp_all_inequality_operator_and_data_vector(
        model, operators, operators.index, include_tde=False
    )

    A_hat = np.array(A / scale)

    if rescale_constraints:
        A_scale = np.linalg.norm(A_hat, axis=1)
    else:
        A_scale = np.ones(A_hat.shape[0])
    A_hat_ = A_hat / A_scale[:, None]
    b_hat = b / A_scale

    if len(b) > 0:
        constraint = adapt_operator(A_hat_) @ params_raw <= b_hat
        constraints.append(constraint)

    cp_problem = cp.Problem(cp.Minimize(objective_val), constraints)

    return Minimizer(
        model=model,
        operators=operators,
        cp_problem=cp_problem,
        params_raw=params_raw,
        params=params,
        params_scale=scale,
        objective_norm2=objective_norm2,
        constraint_scale=A_scale,
        slip_rate=slip_rate,
        slip_rate_limits=velocity_limits,
        smooth_kinematic=smooth_kinematic,
    )


def _tighten_kinematic_bounds(
    minimizer: Minimizer,
    *,
    tighten_all: bool = True,
    factor: float = 0.5,
):
    assert factor > 0 and factor < 1

    def tighten_item(limits: SlipRateLimitItem, coupling: SlipRateItem):
        elastic = coupling.elastic_numpy()
        kinematic = coupling.kinematic_numpy(smooth=minimizer.smooth_kinematic)

        if kinematic is None:
            assert limits.kinematic_lower is None and limits.kinematic_upper is None
            return

        # No coupling bounds defined, so no need to tighten
        if (
            limits.coupling_bounds.lower is None
            and limits.coupling_bounds.upper is None
        ):
            return

        if limits.kinematic_lower is None or limits.kinematic_upper is None:
            raise ValueError(
                "Invalid coupling bounds. Must have both lower and upper bounds or neither."
            )

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
                & ((elastic > kinematic) | (elastic < 0))
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
                & ((elastic > 0) | (elastic < kinematic))
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
        # Any positive value will do, because we only care about the line
        # that passes through the point
        upper[fixed_sign] = 1.0

        fixed_sign = upper <= 0
        # Any negative value will do, because we only care about the line
        # that passes through the point
        lower[fixed_sign] = -1.0
        upper[fixed_sign] = 0.0

        limits.kinematic_lower = lower
        limits.kinematic_upper = upper

    limits = minimizer.slip_rate_limits
    if limits is None:
        raise ValueError("Velocity limits have not been set")

    velocity_limits = {}
    for idx in minimizer.model.segment_mesh_indices:
        velocity_limits[idx] = limits[idx].apply_with_coupling(
            tighten_item, minimizer.slip_rate[idx]
        )


@dataclass
class MinimizerTrace:
    model: Model
    params: list[np.ndarray]
    params_raw: list[np.ndarray]
    slip_rates: list[list[SlipRate]]
    slip_rate_limits: list[list[SlipRateLimit]]
    objective: list[float]
    objective_norm2: list[float]
    nonconvex_constraint_loss: list[float]
    out_of_bounds: list[int]
    out_of_bounds_detailed: list[np.ndarray]
    iter_time: list[float]
    total_time: float
    start_time: float
    last_update_time: float
    minimizer: Minimizer

    def __init__(self, minimizer: Minimizer):
        self.model = minimizer.model
        self.params = []
        self.params_raw = []
        self.slip_rates = []
        self.slip_rate_limits = []
        self.objective = []
        self.objective_norm2 = []
        self.nonconvex_constraint_loss = []
        self.out_of_bounds = []
        self.out_of_bounds_detailed = []
        self.iter_time = []
        self.total_time = 0.0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.minimizer = minimizer

    def print_last_progress(self):
        total = 2 * self.model.total_mesh_points
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
        self.slip_rates.append(
            [slip_rates.copy_numpy() for slip_rates in self.minimizer.slip_rate]
        )
        self.slip_rate_limits.append(
            [lim.copy() for lim in self.minimizer.slip_rate_limits]
        )
        assert self.minimizer.cp_problem.objective.value is not None
        self.objective.append(cast(float, self.minimizer.cp_problem.objective.value))
        self.objective_norm2.append(cast(float, self.minimizer.objective_norm2.value))

        current_time = time.time()
        self.iter_time.append(current_time - self.last_update_time)
        self.total_time += current_time - self.last_update_time
        self.last_update_time = current_time

        self.out_of_bounds.append(self.minimizer.out_of_bounds()[0])
        self.out_of_bounds_detailed.append(self.minimizer.out_of_bounds_detailed()[0])
        self.nonconvex_constraint_loss.append(self.minimizer.constraint_loss())

    def to_estimation(self) -> Estimation:
        """Convert the minimizer trace to an estimation object."""
        estimation = self.minimizer.to_estimation()
        estimation.n_out_of_bounds_trace = np.array(self.out_of_bounds_detailed)[
            :, :, 0
        ].T
        estimation.trace = self
        return estimation


def _custom_cvxopt_solve(problem: cp.Problem, **kwargs):
    """Solve a cvxpy problem with the clarabel reduction but cvxopt solver."""
    data, chain, inverse_data = problem.get_problem_data(
        cp.CLARABEL,
        ignore_dpp=kwargs.get("ignore_dpp", True),
    )

    warm_start = kwargs.pop("warm_start", False)
    if warm_start:
        raise NotImplementedError("warm_start with custom_cvxopt is not implemented")

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
        s=np.concatenate(
            [np.array(cvxopt_result["y"]).ravel(), np.array(cvxopt_result["s"]).ravel()]
        ),
        z=np.concatenate(
            [np.array(cvxopt_result["y"]).ravel(), np.array(cvxopt_result["z"]).ravel()]
        ),
        status=status,
        obj_val=cvxopt_result["primal objective"],
        obj_val_dual=cvxopt_result["dual objective"],
        solve_time=None,
        iterations=cvxopt_result["iterations"],
        r_prim=cvxopt_result["primal slack"],
        r_dual=cvxopt_result["dual slack"],
    )

    problem.unpack_results(sol, chain, inverse_data)  # type: ignore


def _custom_solve(problem: cp.Problem, solver: str, objective: Objective, **kwargs):
    if solver == "CUSTOM_CVXOPT":
        if objective not in [
            "expanded_norm2",
            "sum_of_squares",
            "qr_sum_of_squares",
            "svd_sum_of_squares",
        ]:
            raise ValueError(
                f"CUSTOM_CVXOPT solver does not support objective {objective}"
            )
        _custom_cvxopt_solve(problem, **kwargs)
    else:
        result = problem.solve(solver=solver, **kwargs)
        if isinstance(result, str):
            raise ValueError(
                f"Solver {solver} returned an error: {result}. "
                "Check if the problem is well-posed and constraints are feasible."
            )
        if not np.isfinite(result):
            raise ValueError(
                f"Solver {solver} failed to solve the problem. "
                "Check if the problem is well-posed and constraints are feasible."
            )


def solve_sqp2(
    model: Model,
    *,
    max_iter: int = 20,
    smooth_kinematic: bool = True,
    solve_kwargs: dict | None = None,
    reduction_factor: float = 0.5,
    verbose: bool = False,
    rescale_parameters: bool = True,
    rescale_constraints: bool = True,
    objective: Objective = "qr_sum_of_squares",
    operators: Operators | None = None,
) -> Estimation:
    """Iteratively solve a constrained optimization problem for fault slip rates.

    Performs multiple iterations of solving the convex problem, tightening bounds
    after each iteration until all velocities satisfy constraints or max iterations reached.

    Args:
        problem: The Celeri problem definition containing mesh and operator data
        max_iter: Maximum number of optimization iterations
        smooth_kinematic: Whether to use smoothed kinematic velocities
        solve_kwargs: Additional keyword arguments passed to the solver
        reduction_factor: Factor to reduce bounds by in each iteration (0-1)
        verbose: Whether to print progress information

    Returns:
        A trace object containing the optimization history
    """
    limits = SlipRateLimit.from_model(model)

    minimizer = build_cvxpy_problem(
        model,
        velocity_limits=limits,
        smooth_kinematic=smooth_kinematic,
        rescale_parameters=rescale_parameters,
        rescale_constraints=rescale_constraints,
        objective=objective,
        operators=operators,
    )
    trace = MinimizerTrace(minimizer)

    default_solve_kwargs = {
        "solver": "CLARABEL",
        "ignore_dpp": True,
        "warm_start": False,
        "verbose": False,
    }

    if solve_kwargs is not None:
        default_solve_kwargs.update(solve_kwargs)

    solver = default_solve_kwargs.pop("solver")

    # Storage for all warnings across loop iterations
    all_warnings = []

    # Intializing this so that warnings check will run even with no iteration case
    num_iter = 0
    for num_iter in range(max_iter):
        # QP solve in context manager to capture warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _custom_solve(
                minimizer.cp_problem,
                solver=solver,
                objective=objective,
                **default_solve_kwargs,
            )

        # Store warnings from this iteration (with iteration info)
        for warning in caught_warnings:
            all_warnings.append(  # noqa: PERF401
                {
                    "iteration": num_iter,
                    "message": str(warning.message),
                    "category": warning.category.__name__,
                    "filename": warning.filename,
                    "lineno": warning.lineno,
                }
            )

        trace.store_current()
        if verbose:
            trace.print_last_progress()

        num_oob, total = minimizer.out_of_bounds()
        if num_oob == 0:
            break

        _tighten_kinematic_bounds(
            minimizer,
            factor=reduction_factor,
            tighten_all=True,
        )

    # Log warning if the last iteration includes an error
    if all_warnings:
        if all_warnings[-1]["iteration"] == num_iter:
            logger.warning(f"SQP iteration: {all_warnings[-1]['message']}")
        else:
            iterations_with_warnings = [d["iteration"] for d in all_warnings]
            logger.info(
                f"SQP iteration: Warnings in iterations {iterations_with_warnings} of {num_iter + 1} total iterations"
            )
    else:
        logger.info("SQP iteration: Ran with no warnings")

    return trace.to_estimation()


def benchmark_solve(
    model: Model,
    *,
    objective: Objective,
    rescale_parameters: bool,
    rescale_constraints: bool,
    solver: str,
    solve_kwargs: dict | None = None,
    operators: Operators | None = None,
):
    """Benchmark the performance of solving a CeleriProblem with different configurations.

    Args:
        model: The Celeri problem to solve
        objective: Type of objective function to use
        rescale_parameters: Whether to rescale parameters
        rescale_constraints: Whether to rescale constraints
        solver: Name of the solver to use
        solver_kwargs: Additional solver-specific parameters

    Returns:
        Dictionary containing benchmark results including timing, success status,
        objective values, parameter values, and any error messages.
    """
    minimizer = build_cvxpy_problem(
        model,
        objective=objective,
        rescale_parameters=rescale_parameters,
        rescale_constraints=rescale_constraints,
        operators=operators,
    )

    default_solve_kwargs = {
        "ignore_dpp": True,
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
        "limits": minimizer.slip_rate_limits,
        "objective": objective,
        "rescale_parameters": rescale_parameters,
        "rescale_constraints": rescale_constraints,
    }
