import pytest

import celeri


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
        pytest.param(
            "./tests/configs/test_wna_config.json",
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="qp requires numeric coupling bounds on every segment mesh; the WNA test mesh has null bounds",
            ),
        ),
    ],
)
def test_optimize_sqp(config_file):
    config = celeri.get_config(config_file)
    model = celeri.build_model(config)
    operators = celeri.build_operators(model, eigen=True)
    # Lower percentage target to speed up test
    estimation = celeri.solve_sqp(
        model, operators, percentage_satisfied_target=55, max_iter=200
    )

    estimation.tde_rates  # noqa: B018
    estimation.east_vel_residual  # noqa: B018
    estimation.n_out_of_bounds_trace  # noqa: B018


def test_percent_out_of_bounds_counts_segment_meshes_only():
    import copy
    from types import SimpleNamespace

    import numpy as np

    from celeri.optimize_sqp import _percent_out_of_bounds

    stub = SimpleNamespace(total_mesh_points=10)
    assert _percent_out_of_bounds(np.array([[10.0]]), stub) == 50.0

    config = celeri.get_config("./tests/configs/test_japan_config.json")
    model = celeri.build_model(config)
    n_tied = len(model.segment_mesh_indices)
    tied_elements = sum(model.meshes[i].n_tde for i in model.segment_mesh_indices)

    # An extra mesh that no segment refers to must not dilute the percentage
    extra = celeri.build_model(
        config, override_meshes=[*model.meshes, copy.deepcopy(model.meshes[-1])]
    )
    assert sum(mesh.n_tde for mesh in extra.meshes) > tied_elements
    assert extra.total_mesh_points == tied_elements
    one_per_mesh = np.ones((n_tied, 1))
    assert _percent_out_of_bounds(one_per_mesh, extra) == 100 * n_tied / (
        2 * tied_elements
    )


def test_update_slip_rate_bounds_keeps_intervals_ordered():
    from types import SimpleNamespace

    import numpy as np

    from celeri.optimize_sqp import _SlipRateBounds, _update_slip_rate_bounds

    bounds = SimpleNamespace(lower=0.0, upper=1.0)
    config = SimpleNamespace(
        coupling_constraints_ss=bounds,
        coupling_constraints_ds=bounds,
        iterative_coupling_linear_slip_rate_reduction_factor=1.0,
    )
    meshes = [SimpleNamespace(config=config)]
    current = _SlipRateBounds(
        ss_lower=np.array([3.0]),
        ss_upper=np.array([7.0]),
        ds_lower=np.array([3.0]),
        ds_upper=np.array([7.0]),
    )
    # Coupling below the lower bound while the kinematic rate is now negative
    # moves the upper bound to 0.5 * -10 = -5, below the untouched lower bound
    n_oob, updated = _update_slip_rate_bounds(
        meshes,
        0,
        np.array([-0.5]),
        np.array([-0.5]),
        np.array([-10.0]),
        np.array([-10.0]),
        current,
    )

    assert n_oob == 2
    assert np.all(updated.ss_lower <= updated.ss_upper)
    assert np.all(updated.ds_lower <= updated.ds_upper)
    np.testing.assert_allclose([updated.ss_lower[0], updated.ss_upper[0]], [-5.0, 3.0])


def test_lsqlin_qp_restores_cvxopt_options():
    import cvxopt
    import numpy as np

    from celeri.solve import lsqlin_qp

    before = dict(cvxopt.solvers.options)
    solution = lsqlin_qp(
        np.eye(2),
        np.array([1.0, 2.0]),
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {"show_progress": False},
    )

    assert solution["status"] == "optimal"
    assert dict(cvxopt.solvers.options) == before
