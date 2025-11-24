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
