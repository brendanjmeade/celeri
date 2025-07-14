from pathlib import Path

import pytest

from celeri.model import build_model
from celeri.operators import build_operators
from celeri.optimize import (
    Minimizer,
    Model,
    SlipRate,
    SlipRateItem,
    benchmark_solve,
    build_cvxpy_problem,
    solve_sqp2,
)


@pytest.fixture(scope="module")
def small_test_config_path():
    # Use a simple test config file or mock it
    return Path("tests/test_japan_config.json")


@pytest.fixture(scope="module")
def model(small_test_config_path: Path):
    # Skip this test if the config file doesn't exist
    if not small_test_config_path.exists():
        pytest.skip(f"Command file not found: {small_test_config_path}")
    return build_model(small_test_config_path)


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
    assert hasattr(model, "config")
    assert model.segment_mesh_indices is not None
    assert model.total_mesh_points > 0


@pytest.mark.parametrize(
    "smooth_kinematic, objective",
    [
        (True, "expanded_norm2"),
        (False, "expanded_norm2"),
        (True, "sum_of_squares"),
        (True, "qr_sum_of_squares"),
        (True, "svd_sum_of_squares"),
        (False, "norm2"),
        (False, "norm1"),
        (True, "expanded_norm2"),
    ],
)
def test_build_cvxpy_problem(model: Model, operators, smooth_kinematic, objective):
    """Test that build_cvxpy_problem creates a valid Minimizer object with different parameters."""
    minimizer = build_cvxpy_problem(
        model,
        smooth_kinematic=smooth_kinematic,
        objective=objective,
        operators=operators,
    )

    assert isinstance(minimizer, Minimizer)
    assert hasattr(minimizer, "cp_problem")
    assert minimizer.cp_problem is not None
    assert hasattr(minimizer, "params")
    assert hasattr(minimizer, "params_raw")

    # Ensure the coupling dictionary has entries for each mesh index
    for idx, _ in enumerate(model.meshes):
        assert isinstance(minimizer.slip_rate[idx], SlipRate)
        assert isinstance(minimizer.slip_rate[idx].strike_slip, SlipRateItem)
        assert isinstance(minimizer.slip_rate[idx].dip_slip, SlipRateItem)


@pytest.mark.parametrize(
    "objective", ["expanded_norm2", "sum_of_squares", "qr_sum_of_squares", "norm2"]
)
def test_benchmark_solve(model, operators, objective):
    """Test benchmark_solve function with different objectives."""
    try:
        benchmark_solve(
            model,
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
        trace = solve_sqp2(
            model,
            verbose=False,
        ).trace

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
