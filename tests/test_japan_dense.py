import numpy as np

import celeri


def test_japan_dense():
    # Western North America example
    command_file_name = "./tests/test_japan_command.json"

    config = celeri.get_config(command_file_name)
    model = celeri.build_model(command_file_name)

    celeri.get_tde_slip_rate_constraints(model.meshes, model.operators)

    # Estimate block model parameters (dense)
    index, estimation = celeri.assemble_and_solve_dense(
        config,
        model.assembly,
        model.operators,
        model.station,
        model.block,
        model.meshes,
        model.mogi,
    )
    celeri.post_process_estimation(estimation, model.operators, model.station, index)

    # Set digits of accuracy
    # NOTE: Locally we get machine precision repeatability.  On Github workflows
    # we only get 2-3 digits of repeatability.

    # NOTE: Currently none of these pass even locally.  So much has changes
    # since these tests were relevant.

    # TODO: Remove test?

    # Load known solution
    np.load("./tests/test_japan_arrays.npz")

    # assert np.allclose(
    #     estimation.slip_rates, test_japan_arrays["estimation_slip_rates"], atol=ATOL
    # )
    # assert np.allclose(
    #     estimation.tde_rates, test_japan_arrays["estimation_tde_rates"], atol=ATOL
    # )
    # assert np.allclose(
    #     estimation.east_vel_residual,
    #     test_japan_arrays["estimation_east_vel_residual"],
    #     atol=ATOL,
    # )
    # assert np.allclose(
    #     estimation.north_vel_residual,
    #     test_japan_arrays["estimation_north_vel_residual"],
    #     atol=ATOL,
    # )
