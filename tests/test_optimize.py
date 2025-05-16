import os
from pathlib import Path

import numpy as np
import pytest

from celeri.model import build_model
from celeri.operators import build_operators
from celeri.optimize import (
    Coupling,
    CouplingItem,
    Minimizer,
    Model,
    VelocityLimit,
    VelocityLimitItem,
    benchmark_solve,
    build_cvxpy_problem,
    minimize,
)


@pytest.fixture(scope="module")
def small_test_command_path():
    # Use a simple test command file or mock it
    return Path("tests/test_japan_command.json")


@pytest.fixture(scope="module")
def model(small_test_command_path):
    # Skip this test if the command file doesn't exist
    if not os.path.exists(small_test_command_path):
        pytest.skip(f"Command file not found: {small_test_command_path}")
    return build_model(small_test_command_path)


@pytest.fixture(scope="module")
def operators(model):
    return build_operators(model, eigen=True)


def test_build_problem(model):
    """Test that build_problem returns a CeleriProblem instance with expected properties."""
    assert isinstance(model, Model)
    assert hasattr(model, "meshes")
    assert hasattr(model, "segment")
    assert hasattr(model, "block")
    assert hasattr(model, "station")
    assert hasattr(model, "command")
    assert model.segment_mesh_indices is not None
    assert model.total_mesh_points > 0


@pytest.mark.parametrize(
    "velocity_limits, smooth_kinematic, objective",
    [
        (None, True, "expanded_norm2"),
        (None, False, "expanded_norm2"),
        (None, True, "sum_of_squares"),
        (None, True, "qr_sum_of_squares"),
        (None, True, "svd_sum_of_squares"),
        (None, False, "norm2"),
        (None, False, "norm1"),
        ((-50.0, 50.0), True, "expanded_norm2"),
    ],
)
def test_build_cvxpy_problem(
    model: Model, operators, velocity_limits, smooth_kinematic, objective
):
    """Test that build_cvxpy_problem creates a valid Minimizer object with different parameters."""
    # Adjust velocity_limits length if needed
    if velocity_limits is not None:
        # Create velocity limits as a dict with mesh indices as keys
        lower, upper = velocity_limits
        velocity_limits_dict = {}
        for idx in model.segment_mesh_indices:
            mesh_length = model.meshes[idx].n_tde
            velocity_limits_dict[idx] = VelocityLimit.from_scalar(
                mesh_length, lower, upper
            )
        velocity_limits = velocity_limits_dict

    minimizer = build_cvxpy_problem(
        model,
        velocity_limits=velocity_limits,
        smooth_kinematic=smooth_kinematic,
        objective=objective,
        operators=operators,
    )

    assert isinstance(minimizer, Minimizer)
    assert hasattr(minimizer, "cp_problem")
    assert minimizer.cp_problem is not None
    assert hasattr(minimizer, "params")
    assert hasattr(minimizer, "params_raw")
    assert hasattr(minimizer, "coupling")

    # Ensure the coupling dictionary has entries for each mesh index
    for idx in model.segment_mesh_indices:
        assert idx in minimizer.coupling
        assert isinstance(minimizer.coupling[idx], Coupling)
        assert isinstance(minimizer.coupling[idx].strike_slip, CouplingItem)
        assert isinstance(minimizer.coupling[idx].dip_slip, CouplingItem)


def test_velocity_limit_from_scalar():
    """Test creation of VelocityLimit from scalar values."""
    length = 10
    lower = -50.0
    upper = 50.0

    limits = VelocityLimit.from_scalar(length, lower, upper)

    assert isinstance(limits, VelocityLimit)
    assert isinstance(limits.strike_slip, VelocityLimitItem)
    assert isinstance(limits.dip_slip, VelocityLimitItem)

    # Check dimensions and values
    assert len(limits.strike_slip.kinematic_lower) == length
    assert len(limits.strike_slip.kinematic_upper) == length
    assert len(limits.dip_slip.kinematic_lower) == length
    assert len(limits.dip_slip.kinematic_upper) == length

    # Check that all values are set correctly
    np.testing.assert_array_equal(
        limits.strike_slip.kinematic_lower, np.full(length, lower)
    )
    np.testing.assert_array_equal(
        limits.strike_slip.kinematic_upper, np.full(length, upper)
    )
    np.testing.assert_array_equal(
        limits.dip_slip.kinematic_lower, np.full(length, lower)
    )
    np.testing.assert_array_equal(
        limits.dip_slip.kinematic_upper, np.full(length, upper)
    )


@pytest.mark.parametrize(
    "objective", ["expanded_norm2", "sum_of_squares", "qr_sum_of_squares", "norm2"]
)
def test_benchmark_solve(model, operators, objective):
    """Test benchmark_solve function with different objectives."""
    # Use small limits for faster test
    with_limits = (-10.0, 10.0)

    try:
        benchmark_solve(
            model,
            with_limits=with_limits,
            objective=objective,
            rescale_parameters=True,
            rescale_constraints=True,
            solver="CLARABEL",  # Use ECOS for testing as it's generally available
            operators=operators,
        )

    except Exception as e:
        # If the solve fails due to solver not available, skip the test
        if "solver not available" in str(e).lower():
            pytest.skip(f"Solver not available: {e}")
        else:
            raise


@pytest.mark.slow  # Mark as slow since it may take time
def test_minimize(model):
    """Test the minimize function with a small number of iterations."""
    try:
        trace = minimize(
            model,
            velocity_upper=200.0,
            velocity_lower=-200.0,
            verbose=False,
        )

        # Check that we have trace data
        assert len(trace.params) > 0
        assert len(trace.objective) > 0
        assert len(trace.objective_norm2) > 0
        assert len(trace.iter_time) > 0

    except Exception as e:
        # If the solve fails due to solver not available, skip the test
        if "solver not available" in str(e).lower():
            pytest.skip(f"Solver not available: {e}")
        else:
            raise
