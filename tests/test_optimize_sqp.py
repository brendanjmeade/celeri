import pytest

import celeri


@pytest.mark.parametrize(
    "config_file",
    [
        "./tests/configs/test_japan_config.json",
        pytest.param(
            "./tests/configs/test_wna_config.json",
            marks=pytest.mark.xfail(
                raises=ValueError, reason="Solver did not converge"
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
