"""celeri-forward on a run's own stations reproduces model_station.csv (issue #508)."""

import numpy as np
import pytest

import celeri
from celeri.scripts.celeri_forward import (
    compute_forward_velocities_batch,
    create_forward_operators_batch,
)

COLUMNS = [
    "model_east_vel",
    "model_north_vel",
    "model_east_vel_rotation",
    "model_north_vel_rotation",
    "model_east_elastic_segment",
    "model_north_elastic_segment",
    "model_east_vel_tde",
    "model_north_vel_tde",
    "model_east_vel_block_strain_rate",
    "model_north_vel_block_strain_rate",
    "model_east_vel_mogi",
    "model_north_vel_mogi",
]


@pytest.mark.parametrize("eigen", [False, True])
def test_forward_reproduces_model_station_columns(eigen):
    config = celeri.get_config("./tests/configs/test_wna_config.json")
    config.repl = False
    model = celeri.build_model(config)
    estimation = celeri.assemble_and_solve_dense(model, eigen=eigen, tde=True)
    station = estimation.station

    batch_operators = create_forward_operators_batch(
        estimation, station.lon.to_numpy(), station.lat.to_numpy()
    )
    forward = compute_forward_velocities_batch(estimation, batch_operators)

    for column in COLUMNS:
        np.testing.assert_allclose(
            forward[column], station[column].to_numpy(), atol=1e-6, err_msg=column
        )
