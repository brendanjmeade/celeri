from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celeri.config import Config
from celeri.model import build_model
from celeri.operators import build_operators
from celeri.optimize import (
    Minimizer,
    MinimizerTrace,
    Model,
    SlipRate,
    SlipRateItem,
    benchmark_solve,
    build_cvxpy_problem,
    solve_sqp2,
)


@pytest.fixture(scope="module")
def small_test_config_path():
    return Path("tests/configs/test_japan_config.json")


@pytest.fixture(scope="module")
def model(small_test_config_path: Path):
    # Skip this test if the config file doesn't exist
    if not small_test_config_path.exists():
        pytest.skip(f"Command file not found: {small_test_config_path}")
    return build_model(small_test_config_path)


@pytest.fixture(scope="module")
def operators(model):
    return build_operators(model, eigen=True)


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
        trace = cast(MinimizerTrace, trace)

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


@pytest.mark.slow  # Mark as slow since it may take time
def test_minimize_coupling():
    """Test the minimize function with coupling [-10, 1]"""
    path = Path("tests/configs/test_japan_config.json")
    config = Config.from_file(path)

    # Disable all coupling constraints
    for mesh_params in config.mesh_params:
        if mesh_params.coupling_constraints_ds.lower is not None:
            mesh_params.coupling_constraints_ds = [-10, 1]  # type: ignore
            mesh_params.coupling_constraints_ss = [-10, 1]  # type: ignore

    model = build_model(config)
    try:
        estimation = solve_sqp2(
            model,
            verbose=False,
        )
        trace = cast(MinimizerTrace, estimation.trace)

        # Check that we have trace data
        assert len(trace.params) > 0
        assert len(trace.objective) > 0
        assert len(trace.objective_norm2) > 0
        assert len(trace.iter_time) > 0

        elastic = trace.minimizer.slip_rate[0].strike_slip.elastic
        kinematic = trace.minimizer.slip_rate[0].strike_slip.kinematic_smooth
        coupling = elastic / kinematic
        if not isinstance(coupling, np.ndarray):
            coupling = coupling.value
            assert coupling is not None
        assert np.all(coupling <= 1.0 + 1e-6)
        assert np.all(coupling >= -10.0 - 1e-6)
        assert np.any(coupling < -1)

    except Exception as e:
        # If the solve fails due to solver not available, skip the test
        if "solver not available" in str(e).lower():
            pytest.skip(f"Solver not available: {e}")
        else:
            raise


def test_minimizer_trace_out_of_bounds_trace_sums_components():
    from types import SimpleNamespace

    from celeri.optimize import MinimizerTrace

    trace = MinimizerTrace.__new__(MinimizerTrace)
    # Two iterations, two meshes, (strike-slip, dip-slip) counts
    trace.out_of_bounds_detailed = [
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 5], [6, 0]]),
    ]
    trace.minimizer = SimpleNamespace(to_estimation=lambda: SimpleNamespace())

    estimation = trace.to_estimation()

    np.testing.assert_array_equal(
        estimation.n_out_of_bounds_trace, np.array([[3, 5], [7, 6]])
    )
    assert estimation.trace is trace


def test_regularized_slip_rate_mask_is_interleaved():
    import pandas as pd

    from celeri.optimize import _regularized_slip_rate_mask

    segment = pd.DataFrame(
        {
            "ss_rate_flag": [0, 2, 0],
            "ds_rate_flag": [0, 0, 0],
            "ts_rate_flag": [0, 0, 2],
        }
    )

    mask = _regularized_slip_rate_mask(segment)

    # Segment 1 strike slip (3*1 + 0) and segment 2 tensile slip (3*2 + 2)
    assert mask.dtype == bool
    assert np.flatnonzero(mask).tolist() == [3, 8]


def test_column_scale_leaves_zero_columns_alone():
    from celeri.optimize import _column_scale

    C = np.array([[1.0, 0.0, -3.0], [-2.0, 0.0, 0.5]])

    np.testing.assert_array_equal(_column_scale(C), [2.0, 1.0, 3.0])
    assert np.all(np.isfinite(C / _column_scale(C)))
