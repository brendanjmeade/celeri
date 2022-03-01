import addict
import sys
import warnings
import numpy as np

import okada_wrapper
import cutde.halfspace as cutde_halfspace

import celeri


# def test_end2end():
#     """
#     This does not actually check for correctness much at all,
#     but just tests to make sure that a full block model run executes without errors.
#     """
#     command_file_name = "./data/western_north_america/basic_command.json"
#     command, segment, block, meshes, station, mogi, sar = celeri.read_data(
#         command_file_name
#     )
#     station = celeri.process_station(station, command)
#     segment = celeri.process_segment(segment, command, meshes)
#     sar = celeri.process_sar(sar, command)
#     closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)
#     assert closure.n_polygons() == 31

#     assembly = addict.Dict()
#     operators = addict.Dict()
#     assembly = celeri.merge_geodetic_data(assembly, station, sar)
#     assembly, operators.block_motion_constraints = celeri.get_block_motion_constraints(
#         assembly, block, command
#     )
#     assembly, operators.slip_rate_constraints = celeri.get_slip_rate_constraints(
#         assembly, segment, block, command
#     )

#     # Get all elastic operators for segments and TDEs
#     # Force the calculation of elastic partials rather than reading stored version
#     command.reuse_elastic = "no"
#     celeri.get_elastic_operators(operators, meshes, segment, station, command)
#     assert np.allclose(
#         -1.1692932114810847e-08, operators.meshes[0].tde_to_velocities[0, 0]
#     )

#     # Get TDE smoothing operators
#     celeri.get_all_mesh_smoothing_matrices(meshes, operators)
#     celeri.get_all_mesh_smoothing_matrices_simple(meshes, operators)


def test_global_closure():
    """
    This check to make sure that the closure algorithm returns a known
    (and hopefully correct!) answer for the global closure problem.
    Right now all this does is check for the correct number of blocks and
    against one set of polygon edge indices
    """
    import os

    print(os.getcwd())
    # command_file_name = "./data/global/global_command.json"
    command_file_name = "./data/command/global_command.json"

    command, segment, block, meshes, station, mogi, sar = celeri.read_data(
        command_file_name
    )
    station = celeri.process_station(station, command)
    segment = celeri.process_segment(segment, command, meshes)
    sar = celeri.process_sar(sar, command)
    closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)

    # Compare calculated edge indices with stored edge indices
    all_edge_idxs = np.array([])
    for i in range(closure.n_polygons()):
        all_edge_idxs = np.concatenate(
            (all_edge_idxs, np.array(closure.polygons[i].edge_idxs))
        )

    with open("./tests/global_closure_test_data.npy", "rb") as f:
        all_edge_idxs_stored = np.load(f)

    assert np.allclose(all_edge_idxs, all_edge_idxs_stored)


def test_okada_equals_cutde():
    # Observation coordinates
    x_obs = np.array([2.0])
    y_obs = np.array([1.0])
    z_obs = np.array([0.0])

    nx = ny = 100
    x_obs_vec = np.linspace(-1.0, 1.0, nx)
    y_obs_vec = np.linspace(-1.0, 1.0, ny)

    x_obs_mat, y_obs_mat = np.meshgrid(x_obs_vec, y_obs_vec)
    x_obs = x_obs_mat.flatten()
    y_obs = y_obs_mat.flatten()
    z_obs = np.zeros_like(x_obs)

    # Storage for displacements
    u_x_okada = np.zeros_like(x_obs)
    u_y_okada = np.zeros_like(y_obs)
    u_z_okada = np.zeros_like(z_obs)

    # Material properties
    material_lambda = 3e10
    material_mu = 3e10
    poissons_ratio = material_lambda / (2 * (material_lambda + material_mu))
    alpha = (material_lambda + material_mu) / (material_lambda + 2 * material_mu)

    # Fault slip
    strike_slip = 1
    dip_slip = 0
    tensile_slip = 0

    # Parameters for Okada
    segment_locking_depth = 1.0
    segment_dip = 90
    segment_length = 1.0
    segment_up_dip_width = segment_locking_depth

    # Okada
    for i in range(x_obs.size):
        _, u, _ = okada_wrapper.dc3dwrapper(
            alpha,  # (lambda + mu) / (lambda + 2 * mu)
            [
                x_obs[i],
                y_obs[i],
                z_obs[i],
            ],  # (meters) observation point
            segment_locking_depth,  # (meters) depth of the fault origin
            segment_dip,  # (degrees) the dip-angle of the rectangular dislocation surface
            [
                -segment_length / 2,
                segment_length / 2,
            ],  # (meters) the along-strike range of the surface (al1,al2 in the original)
            [
                0,
                segment_up_dip_width,
            ],  # (meters) along-dip range of the surface (aw1, aw2 in the original)
            [strike_slip, dip_slip, tensile_slip],
        )  # (meters) strike-slip, dip-slip, tensile-slip
        u_x_okada[i] = u[0]
        u_y_okada[i] = u[1]
        u_z_okada[i] = u[2]

    # cutde
    tri_x1 = np.array([-0.5, -0.5])
    tri_y1 = np.array([0, 0])
    tri_z1 = np.array([0, 0])
    tri_x2 = np.array([0.5, 0.5])
    tri_y2 = np.array([0, 0.0])
    tri_z2 = np.array([0, -1])
    tri_x3 = np.array([0.5, -0.5])
    tri_y3 = np.array([0, 0])
    tri_z3 = np.array([-1, -1])

    # Package coordinates for cutde call
    obs_coords = np.vstack((x_obs, y_obs, np.zeros_like(x_obs))).T
    tri_coords = np.array(
        [[tri_x1, tri_y1, tri_z1], [tri_x2, tri_y2, tri_z2], [tri_x3, tri_y3, tri_z3]]
    ).astype(float)
    tri_coords = np.transpose(tri_coords, (2, 0, 1))

    # Call cutde, multiply by displacements, and package for the return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mat = cutde_halfspace.disp_matrix(
            obs_pts=obs_coords, tris=tri_coords, nu=poissons_ratio
        )
    slip = np.array([[strike_slip, dip_slip, tensile_slip]])

    # What to do with array dimensions here? We want to compute the
    # displacements at the observation points due to the slip specified
    # in the slip array. That will involve a dot product of the disp_mat with slip.
    # But, it's not that simple. First, take a look at the array shapes:
    # disp_mat has shape (10000, 3, 2, 3)
    # Why?
    # dimension #1: 10000 is the number of observation points
    # dimension #2: 3 is the number of components of the displacement vector
    # dimension #3: 2 is the number of source triangles
    # dimension #3: 3 is the number of components of the slip vector.
    #
    # Then, slip has shape (1, 3)
    # This is sort of "wrong" in that the first dimension should be the number of triangles.
    # But, since we're applying the same slip to both triangles, it's okay.
    #
    # So, to apply that slip to both triangles, we want to first do a do dot product
    # between disp_mat and slip[0] which will multiply and sum the last axis of both arrays
    #
    # Then, we will have a (10000, 3, 2) shape array.
    # Next, since we want the displacement due to the sum of both arrays, let's sum over that
    # last axis with two elements.
    #
    # The final result "u_cutde" will be a (10000, 3) array with the components of displacement
    # for every observation point.
    u_cutde = np.sum(disp_mat.dot(slip[0]), axis=2)

    # Uncomment to plot.
    # field_cutde = u_cutde[:,0].reshape((nx, ny))
    # field_okada = u_x_okada.reshape((nx, ny))
    # plt.subplot(1,2,1)
    # plt.contourf(x_obs_mat, y_obs_mat, field_cutde)
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.contourf(x_obs_mat, y_obs_mat, field_okada)
    # plt.colorbar()
    # plt.show()

    np.testing.assert_almost_equal(u_cutde[:, 0], u_x_okada)
    np.testing.assert_almost_equal(u_cutde[:, 1], u_y_okada)
    np.testing.assert_almost_equal(u_cutde[:, 2], u_z_okada)
