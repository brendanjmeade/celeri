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
    coupling: cp.Variable
    kinematic: cp.Variable
    kinematic_smooth: cp.Variable
    estimated: cp.Variable

    def constraint_loss(self):
        kinematic = self.kinematic_smooth.value
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

    def out_of_bounds(self):
        if (
            self.dip_slip.coupling.value is None
            or self.strike_slip.coupling.value is None
        ):
            raise ValueError("Coupling has not been fit")
        total = len(self.dip_slip.coupling.value) + len(self.strike_slip.coupling.value)
        oob1 = (
            (self.dip_slip.coupling.value < 0) | (self.dip_slip.coupling.value > 1)
        ).sum()
        oob2 = (
            (self.strike_slip.coupling.value < 0)
            | (self.strike_slip.coupling.value > 1)
        ).sum()
        return oob1 + oob2, total

    def constraint_loss(self):
        return self.strike_slip.constraint_loss() + self.dip_slip.constraint_loss()


@dataclass
class VelocityLimitItem:
    kinematic_lower: np.ndarray
    kinematic_upper: np.ndarray
    estimated_lower: np.ndarray
    estimated_upper: np.ndarray

    @classmethod
    def from_scalar(cls, lower, upper):
        return VelocityLimitItem(
            kinematic_lower=lower,
            kinematic_upper=upper,
            estimated_lower=lower,
            estimated_upper=upper,
        )


@dataclass
class VelocityLimit:
    strike_slip: VelocityLimitItem
    dip_slip: VelocityLimitItem

    @classmethod
    def from_scalar(cls, lower, upper):
        return VelocityLimit(
            strike_slip=VelocityLimitItem.from_scalar(lower, upper),
            dip_slip=VelocityLimitItem.from_scalar(lower, upper),
        )

    def from_coupling(
        self,
        item_func: Callable[[VelocityLimitItem, CouplingItem], VelocityLimitItem],
        coupling,
    ):
        return VelocityLimit(
            strike_slip=item_func(self.strike_slip, coupling.strike_slip),
            dip_slip=item_func(self.dip_slip, coupling.dip_slip),
        )

    def apply(
        self, item_func: Callable[[VelocityLimitItem], VelocityLimitItem]
    ) -> "VelocityLimit":
        return VelocityLimit(
            strike_slip=item_func(self.strike_slip),
            dip_slip=item_func(self.dip_slip),
        )

    def fix(self) -> "VelocityLimit":
        """Tighten the bounds as much as possible based on the coupling bounds 0 to 1"""

        def item_func(limits: VelocityLimitItem):
            a_lb = limits.estimated_lower
            a_ub = limits.estimated_upper
            b_lb = limits.kinematic_lower
            b_ub = limits.kinematic_upper
            (a_lb, a_ub, b_lb, b_ub) = np.broadcast_arrays(a_lb, a_ub, b_lb, b_ub)
            a_lb = a_lb.copy()
            a_ub = a_ub.copy()
            b_lb = b_lb.copy()
            b_ub = b_ub.copy()

            a_lb[b_lb < 0] = np.maximum(a_lb, b_lb)[b_lb < 0]
            a_lb[b_lb >= 0] = 0.0

            a_ub[b_ub > 0] = np.minimum(a_ub, b_ub)[b_ub > 0]
            a_ub[b_ub <= 0] = 0.0

            return VelocityLimitItem(
                estimated_lower=a_lb,
                estimated_upper=a_ub,
                kinematic_lower=b_lb,
                kinematic_upper=b_ub,
            )

        return self.apply(item_func)


@dataclass
class Minimizer:
    problem: CeleriProblem
    cp_problem: cp.Problem
    params_raw: cp.Variable
    params: cp.Variable
    coupling: dict[int, Coupling]
    velocity_limits: dict[int, VelocityLimit] | None

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

    def out_of_bounds(self):
        oob, total = 0, 0
        for idx in self.problem.segment_mesh_indices:
            oob_mesh, total_mesh = self.coupling[idx].out_of_bounds()
            oob += oob_mesh
            total += total_mesh
        return oob, total


def _make_constraints(a, b, a_lb, a_ub, b_lb, b_ub):
    a_lb = np.broadcast_to(a_lb, a.shape).copy()
    b_lb = np.broadcast_to(b_lb, a.shape).copy()
    a_ub = np.broadcast_to(a_ub, a.shape).copy()
    b_ub = np.broadcast_to(b_ub, a.shape).copy()

    def bound_segment(a, b, x, y):
        """Add a linear bound for points (a_i, b_i) that allows values to the left of the line
        defined by (x_1, x_2) to (y_1, y2)"""
        x1, x2 = x
        y1, y2 = y
        lhs = cp.multiply(y1 - x1, b - x2) - cp.multiply(y2 - x2, a - x1)
        max_x = np.maximum(np.abs(x1), np.abs(x2))
        max_y = np.maximum(np.abs(y1), np.abs(y2))
        scale = np.maximum(max_x, max_y)
        return lhs / scale >= 0

    x = (a_lb, a_lb)
    y = (np.maximum(0.0, a_lb), b_ub)
    bound1 = bound_segment(a, b, y, x)

    x = (a_ub, a_ub)
    y = (np.minimum(0.0, a_ub), b_lb)
    bound2 = bound_segment(a, b, y, x)

    a_scale = np.abs(a.value).max()
    b_scale = np.abs(b.value).max()

    constraints = [
        # Current bounds for a and b.
        a / a_scale >= a_lb / a_scale,
        a / a_scale <= a_ub / a_scale,
        b / b_scale >= b_lb / b_scale,
        b / b_scale <= b_ub / b_scale,
        bound1,
        bound2,
    ]

    return constraints


def build_cvxpy_problem(
    problem: CeleriProblem,
    *,
    init_params_raw_value: np.ndarray | None = None,
    init_params_value: np.ndarray | None = None,
    velocity_limits: dict[int, VelocityLimit] | None = None,
    mccormick: bool = False,
    smooth_kinematic: bool = True,
    slip_rate_reduction: float | None = None,
):
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

    for mesh_idx in problem.segment_mesh_indices:
        kinematic_tde_rates = (
            problem.operators.rotation_to_tri_slip_rate[mesh_idx]
            @ params[0 : 3 * len(problem.block)]
        )

        # Get estimated elastic rates on mesh elements
        estimated_tde_rates = (
            problem.operators.eigenvectors_to_tde_slip[mesh_idx]
            @ params[
                problem.index["start_col_eigen"][mesh_idx] : problem.index[
                    "end_col_eigen"
                ][mesh_idx]
            ]
        )

        def get_coupling_linear(estimated_slip, kinematic_slip):
            # Smooth kinematic slip
            kinematic_slip = (
                problem.operators.linear_guassian_smoothing[mesh_idx] @ kinematic_slip
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
            coupling=tde_coupling_ss,
        )

        dip_slip = CouplingItem(
            kinematic=kinematic_tde_rates[1::2],
            kinematic_smooth=kinematic_tde_rates_ds_smooth,
            estimated=estimated_tde_rates[1::2],
            coupling=tde_coupling_ds,
        )

        combined = CouplingItem(
            kinematic=kinematic_tde_rates,
            kinematic_smooth=cast(
                cp.Variable,
                cp.concatenate(
                    [
                        kinematic_tde_rates_ss_smooth[:, None],
                        kinematic_tde_rates_ds_smooth[:, None],
                    ],
                    axis=1,
                ),
            ),
            estimated=estimated_tde_rates,
            coupling=cast(
                cp.Variable,
                cp.concatenate(
                    [tde_coupling_ss[:, None], tde_coupling_ds[:, None]], axis=1
                ),
            ),
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
            x = item.estimated
            if smooth_kinematic:
                y = item.kinematic_smooth
            else:
                y = item.kinematic

            if velocity_limits is None:
                continue

            x_lb = getattr(velocity_limits[mesh_idx], name).estimated_lower
            x_ub = getattr(velocity_limits[mesh_idx], name).estimated_upper
            y_lb = getattr(velocity_limits[mesh_idx], name).kinematic_lower
            y_ub = getattr(velocity_limits[mesh_idx], name).kinematic_upper
            constraints.extend(_make_constraints(x, y, x_lb, x_ub, y_lb, y_ub))

    A_hat = np.array(A / scale)
    A_scale = np.abs(A_hat).max(1)
    A_hat_ = A_hat / A_scale[:, None]
    b_hat = b / A_scale

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
    )


def tighten_kinematic_bounds(
    minimizer: Minimizer,
    max_limit: float,
    tighten_all: bool = True,
    factor: float = 0.5,
    smoothed_kinematic: bool = True,
):
    assert factor > 0 and factor < 1

    def tighten_item(limits: VelocityLimitItem, coupling: CouplingItem):
        estimated = coupling.estimated.value
        if smoothed_kinematic:
            kinematic = coupling.kinematic_smooth.value
        else:
            kinematic = coupling.kinematic.value

        if estimated is None or kinematic is None:
            raise ValueError("Minimizer has not been fit")

        upper = np.broadcast_to(limits.kinematic_upper, estimated.shape).copy()
        lower = np.broadcast_to(limits.kinematic_lower, estimated.shape).copy()

        a_upper = np.broadcast_to(limits.estimated_upper, estimated.shape).copy()
        a_lower = np.broadcast_to(limits.estimated_lower, estimated.shape).copy()

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
        a_lower[fixed_sign] = 0.0
        a_upper[fixed_sign] = max_limit

        fixed_sign = upper <= 0
        lower[fixed_sign] = -max_limit
        upper[fixed_sign] = 0.0
        a_lower[fixed_sign] = -max_limit
        a_upper[fixed_sign] = 0.0

        return VelocityLimitItem(
            kinematic_lower=lower,
            kinematic_upper=upper,
            estimated_lower=a_lower,
            estimated_upper=a_upper,
        )

    old_limits = minimizer.velocity_limits

    velocity_limits = {}
    for idx in minimizer.problem.segment_mesh_indices:
        if old_limits is None:
            old = VelocityLimit.from_scalar(-max_limit, max_limit)
        else:
            old = old_limits[idx]

        velocity_limits[idx] = old.from_coupling(
            tighten_item, minimizer.coupling[idx]
        ).fix()

    return build_cvxpy_problem(
        minimizer.problem,
        init_params_value=minimizer.params.value,
        velocity_limits=velocity_limits,
    )
