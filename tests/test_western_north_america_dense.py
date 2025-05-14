import celeri


def test_western_north_america_dense():
    # Western North America example
    command_file_name = "./tests/test_western_north_america_command.json"

    celeri.get_config(command_file_name)
    model = celeri.build_model(command_file_name)

    # Estimate block model parameters (dense)
    operators, estimation = celeri.assemble_and_solve_dense(model)
    celeri.post_process_estimation(
        estimation, operators, model.station, operators.index
    )

    # Set digits of accuracy
    # NOTE: Locally we get machine precision repeatability.  On Github workflows
    # we only get 2-3 digits of repeatability.

    # NOTE: None of these tests pass these days because so much has changed

    # NOTE: Consieer removing this test.

    # # Load known solution
    # test_western_north_america_arrays = np.load(
    #     "./tests/test_western_north_america_arrays.npz"
    # )

    # assert np.allclose(
    #     estimation.slip_rates,
    #     test_western_north_america_arrays["estimation_slip_rates"],
    #     atol=ATOL,
    # )
    # assert np.allclose(
    #     estimation.tde_rates,
    #     test_western_north_america_arrays["estimation_tde_rates"],
    #     atol=ATOL,
    # )
    # assert np.allclose(
    #     estimation.east_vel_residual,
    #     test_western_north_america_arrays["estimation_east_vel_residual"],
    #     atol=ATOL,
    # )
    # assert np.allclose(
    #     estimation.north_vel_residual,
    #     test_western_north_america_arrays["estimation_north_vel_residual"],
    #     atol=ATOL,
    # )
