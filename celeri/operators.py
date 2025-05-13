import copy
import os
import warnings
from dataclasses import dataclass, field

import addict
import cutde.halfspace as cutde_halfspace
import h5py
import numpy as np
import okada_wrapper
import pandas as pd
import scipy
from tqdm import tqdm
from loguru import logger
from scipy import spatial
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from celeri.celeri_util import (
    cartesian_vector_to_spherical_vector,
    euler_pole_covariance_to_rotation_vector_covariance,
    get_2component_index,
    get_cross_partials,
    get_keep_index_12,
    get_segment_oblique_projection,
    get_transverse_projection,
    interleave2,
    interleave3,
    latitude_to_colatitude,
    sph2cart,
)
from celeri.config import Config
from celeri.constants import (
    DEG_PER_MYR_TO_RAD_PER_YR,
    GEOID,
    KM2M,
    N_MESH_DIM,
    RADIUS_EARTH,
)
from celeri.mesh import ByMesh, Mesh
from celeri.model import (
    Model,
    get_block_centroid,
    get_shared_sides,
    get_tri_shared_sides_distances,
    merge_geodetic_data,
)


# TODO maybe this should only contain commonly used operators,
# different solve methods could inherit from it to add additional
# operators.
# TODO Should not have None values in the dataclass, but instead
# build them in the constructor
@dataclass
class Operators:
    index: addict.Dict | None = None
    # TODO I'm not sure if this should be part of the model or the operator,
    # or just a separate object
    assembly: addict.Dict | None = None
    meshes: list[addict.Dict] = field(default_factory=list)
    rotation_to_velocities: np.ndarray | None = None
    block_motion_constraints: np.ndarray | None = None
    slip_rate_constraints: np.ndarray | None = None
    rotation_to_slip_rate: np.ndarray | None = None
    block_strain_rate_to_velocities: np.ndarray | None = None
    mogi_to_velocities: np.ndarray | None = None
    eigen: np.ndarray | None = None
    slip_rate_to_skada_to_velocities: np.ndarray | None = None
    eigenvectors_to_tde_slip: dict[int, np.ndarray] = field(default_factory=dict)
    rotation_to_tri_slip_rate: dict[int, np.ndarray] = field(default_factory=dict)
    linear_guassian_smoothing: dict[int, np.ndarray] = field(default_factory=dict)
    tde_to_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    smoothing_matrix: dict[int, np.ndarray] = field(default_factory=dict)
    tde_slip_rate_constraints: dict[int, np.ndarray] = field(default_factory=dict)
    eigen_to_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    eigen_to_tde_bcs: dict[int, np.ndarray] = field(default_factory=dict)


def build_operators(model: Model):
    assembly = addict.Dict()
    operators = Operators()
    operators.assembly = assembly

    operators.meshes = [addict.Dict() for _ in range(len(model.meshes))]
    assembly = merge_geodetic_data(assembly, model.station, model.sar)

    # Get all elastic operators for segments and TDEs
    get_elastic_operators(
        operators, model.meshes, model.segment, model.station, model.command
    )

    # Get TDE smoothing operators
    get_all_mesh_smoothing_matrices(model.meshes, operators)

    # Block rotation to velocity operator
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        model.station, len(model.block)
    )

    # Soft block motion constraints
    # TODO: Why would this return an assembly?
    assembly, operators.block_motion_constraints = get_block_motion_constraints(
        operators.assembly, model.block, model.command
    )
    # TODO This function shouldn't change the model in any way!
    operators.assembly = assembly

    # Soft slip rate constraints
    # TODO: Why would this return an assembly?
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        operators.assembly, model.segment, model.block, model.command
    )
    # TODO This function shouldn't change the model in any way!
    operators.assembly = assembly

    # Rotation vectors to slip rate operator
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        model.segment, model.block
    )

    # Internal block strain rate operator
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        model.block, model.station, model.segment
    )

    # Mogi source operator
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        model.mogi, model.station, model.command
    )

    # Soft TDE boundary condition constraints
    get_tde_slip_rate_constraints(model.meshes, operators)

    # Get index
    index = get_index_eigen(
        assembly, model.segment, model.station, model.block, model.meshes, model.mogi
    )
    operators.index = index

    # Get data vector for KL problem
    get_data_vector_eigen(model.meshes, assembly, index)

    # Get data vector for KL problem
    get_weighting_vector_eigen(model.command, model.station, model.meshes, index)

    # Get KL modes for each mesh
    get_eigenvectors_to_tde_slip(operators, model.meshes)

    # Get full operator including all blocks, KL modes, strain blocks, and mogis
    operators.eigen = get_full_dense_operator_eigen(operators, model.meshes, index)

    # Get rotation to TDE kinematic slip rate operator for all meshes tied to segments
    get_tde_coupling_constraints(model.meshes, model.segment, model.block, operators)

    # Get smoothing operators for post-hoc smoothing of slip
    operators = _get_gaussian_smoothing_operator(model.meshes, operators, index)
    return operators


def _get_gaussian_smoothing_operator(meshes, operators, index):
    for i in range(index.n_meshes):
        points = np.vstack((meshes[i].lon_centroid, meshes[i].lat_centroid)).T

        length_scale = meshes[i].config.iterative_coupling_smoothing_length_scale

        # TODO this default should be in the config
        if length_scale is None:
            length_scale = 0.25

        # Compute pairwise Euclidean distance matrix
        D = spatial.distance_matrix(points, points)

        # Define Gaussian weight function
        W = np.exp(-(D**2) / (2 * length_scale**2))
        # TODO make this configurable
        W[W < 1e-8] = 0.0

        # Normalize rows so each row sums to 1
        W /= W.sum(axis=1, keepdims=True)

        operators.linear_guassian_smoothing[i] = W
    return operators


def get_rotation_to_velocities_partials(station, n_blocks):
    """Calculate block rotation partials operator for stations in dataframe
    station.
    """
    # n_blocks = (
    #     np.max(station.block_label.values) + 1
    # )  # +1 required so that a single block with index zero still propagates
    block_rotation_operator = np.zeros((3 * len(station), 3 * n_blocks))
    for i in range(n_blocks):
        station_idx = np.where(station.block_label == i)[0]
        (
            vel_east_omega_x,
            vel_north_omega_x,
            vel_up_omega_x,
        ) = get_rotation_displacements(
            station.lon.values[station_idx],
            station.lat.values[station_idx],
            omega_x=1,
            omega_y=0,
            omega_z=0,
        )
        (
            vel_east_omega_y,
            vel_north_omega_y,
            vel_up_omega_y,
        ) = get_rotation_displacements(
            station.lon.values[station_idx],
            station.lat.values[station_idx],
            omega_x=0,
            omega_y=1,
            omega_z=0,
        )
        (
            vel_east_omega_z,
            vel_north_omega_z,
            vel_up_omega_z,
        ) = get_rotation_displacements(
            station.lon.values[station_idx],
            station.lat.values[station_idx],
            omega_x=0,
            omega_y=0,
            omega_z=1,
        )
        block_rotation_operator[3 * station_idx, 3 * i] = vel_east_omega_x
        block_rotation_operator[3 * station_idx, 3 * i + 1] = vel_east_omega_y
        block_rotation_operator[3 * station_idx, 3 * i + 2] = vel_east_omega_z
        block_rotation_operator[3 * station_idx + 1, 3 * i] = vel_north_omega_x
        block_rotation_operator[3 * station_idx + 1, 3 * i + 1] = vel_north_omega_y
        block_rotation_operator[3 * station_idx + 1, 3 * i + 2] = vel_north_omega_z
        block_rotation_operator[3 * station_idx + 2, 3 * i] = vel_up_omega_x
        block_rotation_operator[3 * station_idx + 2, 3 * i + 1] = vel_up_omega_y
        block_rotation_operator[3 * station_idx + 2, 3 * i + 2] = vel_up_omega_z
    return block_rotation_operator


def get_rotation_displacements(lon_obs, lat_obs, omega_x, omega_y, omega_z):
    """# TODO: Consider renaming to: get_rotation_velocities
    Get displacments at at longitude and latitude coordinates given rotation
    vector components (omega_x, omega_y, omega_z).
    """
    vel_east = np.zeros(lon_obs.size)
    vel_north = np.zeros(lon_obs.size)
    vel_up = np.zeros(lon_obs.size)
    x, y, z = sph2cart(lon_obs, lat_obs, RADIUS_EARTH)
    for i in range(lon_obs.size):
        cross_product_operator = get_cross_partials([x[i], y[i], z[i]])
        (
            vel_north_from_omega_x,
            vel_east_from_omega_x,
            vel_up_from_omega_x,
        ) = cartesian_vector_to_spherical_vector(
            cross_product_operator[0, 0],
            cross_product_operator[1, 0],
            cross_product_operator[2, 0],
            lon_obs[i],
            lat_obs[i],
        )
        (
            vel_north_from_omega_y,
            vel_east_from_omega_y,
            vel_up_from_omega_y,
        ) = cartesian_vector_to_spherical_vector(
            cross_product_operator[0, 1],
            cross_product_operator[1, 1],
            cross_product_operator[2, 1],
            lon_obs[i],
            lat_obs[i],
        )
        (
            vel_north_from_omega_z,
            vel_east_from_omega_z,
            vel_up_from_omega_z,
        ) = cartesian_vector_to_spherical_vector(
            cross_product_operator[0, 2],
            cross_product_operator[1, 2],
            cross_product_operator[2, 2],
            lon_obs[i],
            lat_obs[i],
        )
        vel_east[i] = (
            omega_x * vel_east_from_omega_x
            + omega_y * vel_east_from_omega_y
            + omega_z * vel_east_from_omega_z
        )
        vel_north[i] = (
            omega_x * vel_north_from_omega_x
            + omega_y * vel_north_from_omega_y
            + omega_z * vel_north_from_omega_z
        )
    return vel_east, vel_north, vel_up


def get_elastic_operators(
    operators: Operators,
    meshes: ByMesh[Mesh],
    segment: pd.DataFrame,
    station: pd.DataFrame,
    command: Config,
):
    """Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces.

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        meshes (List): Geometries of meshes
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        command (Dict): All command data
    """
    if (
        bool(command.reuse_elastic)
        and command.reuse_elastic_file is not None
        and os.path.exists(command.reuse_elastic_file)
    ):
        logger.info("Using precomputed elastic operators")
        hdf5_file = h5py.File(command.reuse_elastic_file, "r")

        operators.slip_rate_to_okada_to_velocities = np.array(
            hdf5_file.get("slip_rate_to_okada_to_velocities")
        )
        for i in range(len(meshes)):
            operators.tde_to_velocities[i] = np.array(
                hdf5_file.get("tde_to_velocities_" + str(i))
            )
        hdf5_file.close()

    else:
        if command.reuse_elastic_file is None or not os.path.exists(
            command.reuse_elastic_file
        ):
            logger.warning("Precomputed elastic operator file not found")
        logger.info("Computing elastic operators")

        # Calculate Okada partials for all segments
        operators.slip_rate_to_okada_to_velocities = get_segment_station_operator_okada(
            segment, station, command
        )

        for i in range(len(meshes)):
            logger.info(
                f"Start: TDE slip to velocity calculation for mesh: {meshes[i].file_name}"
            )
            operators.tde_to_velocities[i] = get_tde_to_velocities_single_mesh(
                meshes, station, command, mesh_idx=i
            )
            logger.success(
                f"Finish: TDE slip to velocity calculation for mesh: {meshes[i].file_name}"
            )

        # Save elastic to velocity matrices
        if bool(command.save_elastic):
            # Check to see if "data/operators" folder exists and if not create it
            if not os.path.exists(command.operators_folder):
                os.mkdir(command.operators_folder)

            logger.info(
                "Saving elastic to velocity matrices to :" + command.save_elastic_file
            )
            hdf5_file = h5py.File(command.save_elastic_file, "w")

            hdf5_file.create_dataset(
                "slip_rate_to_okada_to_velocities",
                data=operators.slip_rate_to_okada_to_velocities,
            )
            for i in range(len(meshes)):
                hdf5_file.create_dataset(
                    "tde_to_velocities_" + str(i),
                    data=operators.tde_to_velocities[i],
                )
            hdf5_file.close()


def get_elastic_operators_okada(
    operators: dict,
    segment: pd.DataFrame,
    station: pd.DataFrame,
    command: dict,
):
    """NOTE: This is for the case with no TDEs.  May be redundant.  Consider.

    Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        command (Dict): All command data
    """
    if bool(command.reuse_elastic) and os.path.exists(command.reuse_elastic_file):
        logger.info("Using precomputed elastic operators")
        hdf5_file = h5py.File(command.reuse_elastic_file, "r")

        operators.slip_rate_to_okada_to_velocities = np.array(
            hdf5_file.get("slip_rate_to_okada_to_velocities")
        )
        hdf5_file.close()

    else:
        if not os.path.exists(command.reuse_elastic_file):
            logger.warning("Precomputed elastic operator file not found")
        logger.info("Computing elastic operators")

        # Calculate Okada partials for all segments
        operators.slip_rate_to_okada_to_velocities = get_segment_station_operator_okada(
            segment, station, command
        )

        # Save elastic to velocity matrices
        if bool(command.save_elastic):
            # Check to see if "data/operators" folder exists and if not create it
            if not os.path.exists(command.operators_folder):
                os.mkdir(command.operators_folder)

            logger.info(
                "Saving elastic to velocity matrices to :" + command.save_elastic_file
            )
            hdf5_file = h5py.File(command.save_elastic_file, "w")

            hdf5_file.create_dataset(
                "slip_rate_to_okada_to_velocities",
                data=operators.slip_rate_to_okada_to_velocities,
            )
            hdf5_file.close()


def get_segment_station_operator_okada(segment, station, command):
    """Calculates the elastic displacement partial derivatives based on the Okada
    formulation, using the source and receiver geometries defined in
    dicitonaries segment and stations. Before calculating the partials for
    each segment, a local oblique Mercator project is done.

    The linear operator is structured as ():

                ss(segment1)  ds(segment1) ts(segment1) ... ss(segmentN) ds(segmentN) ts(segmentN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if not station.empty:
        okada_segment_operator = np.ones((3 * len(station), 3 * len(segment)))
        # Loop through each segment and calculate displacements for each slip component
        for i in tqdm(
            range(len(segment)),
            desc="Calculating Okada partials for segments",
            colour="cyan",
        ):
            (
                u_east_strike_slip,
                u_north_strike_slip,
                u_up_strike_slip,
            ) = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                command.material_lambda,
                command.material_mu,
                1,
                0,
                0,
                station.lon,
                station.lat,
            )
            (
                u_east_dip_slip,
                u_north_dip_slip,
                u_up_dip_slip,
            ) = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                command.material_lambda,
                command.material_mu,
                0,
                1,
                0,
                station.lon,
                station.lat,
            )
            (
                u_east_tensile_slip,
                u_north_tensile_slip,
                u_up_tensile_slip,
            ) = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                command.material_lambda,
                command.material_mu,
                0,
                0,
                1,
                station.lon,
                station.lat,
            )
            segment_column_start_idx = 3 * i
            okada_segment_operator[0::3, segment_column_start_idx] = np.squeeze(
                u_east_strike_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx] = np.squeeze(
                u_north_strike_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx] = np.squeeze(
                u_up_strike_slip
            )
            okada_segment_operator[0::3, segment_column_start_idx + 1] = np.squeeze(
                u_east_dip_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx + 1] = np.squeeze(
                u_north_dip_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx + 1] = np.squeeze(
                u_up_dip_slip
            )
            okada_segment_operator[0::3, segment_column_start_idx + 2] = np.squeeze(
                u_east_tensile_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx + 2] = np.squeeze(
                u_north_tensile_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx + 2] = np.squeeze(
                u_up_tensile_slip
            )
    else:
        okada_segment_operator = np.empty(1)
    return okada_segment_operator


def get_okada_displacements(
    segment_lon1,
    segment_lat1,
    segment_lon2,
    segment_lat2,
    segment_locking_depth,
    segment_burial_depth,
    segment_dip,
    segment_azimuth,
    material_lambda,
    material_mu,
    strike_slip,
    dip_slip,
    tensile_slip,
    station_lon,
    station_lat,
):
    """Caculate elastic displacements in a homogeneous elastic half-space.
    Inputs are in geographic coordinates and then projected into a local
    xy-plane using a oblique Mercator projection that is tangent and parallel
    to the trace of the fault segment.  The elastic calculation is the
    original Okada 1992 Fortran code acceccesed through T. Ben Thompson's
    okada_wrapper: https://github.com/tbenthompson/okada_wrapper.
    """
    segment_locking_depth *= KM2M
    segment_burial_depth *= KM2M

    # Make sure depths are expressed as positive
    segment_locking_depth = np.abs(segment_locking_depth)
    segment_burial_depth = np.abs(segment_burial_depth)

    # Correct sign of dip-slip based on fault dip, as noted on p. 1023 of Okada (1992)
    dip_slip *= np.sign(90 - segment_dip)

    # Project coordinates to flat space using a local oblique Mercator projection
    projection = get_segment_oblique_projection(
        segment_lon1, segment_lat1, segment_lon2, segment_lat2
    )
    station_x, station_y = projection(station_lon, station_lat)
    segment_x1, segment_y1 = projection(segment_lon1, segment_lat1)
    segment_x2, segment_y2 = projection(segment_lon2, segment_lat2)

    # Calculate geometric fault parameters
    segment_strike = np.arctan2(
        segment_y2 - segment_y1, segment_x2 - segment_x1
    )  # radians
    segment_length = np.sqrt(
        (segment_y2 - segment_y1) ** 2.0 + (segment_x2 - segment_x1) ** 2.0
    )
    segment_up_dip_width = (segment_locking_depth - segment_burial_depth) / np.sin(
        np.deg2rad(segment_dip)
    )

    # Translate stations and segment so that segment mid-point is at the origin
    segment_x_mid = (segment_x1 + segment_x2) / 2.0
    segment_y_mid = (segment_y1 + segment_y2) / 2.0
    station_x -= segment_x_mid
    station_y -= segment_y_mid
    segment_x1 -= segment_x_mid
    segment_x2 -= segment_x_mid
    segment_y1 -= segment_y_mid
    segment_y2 -= segment_y_mid

    # Unrotate coordinates to eliminate strike, segment will lie along y = 0
    rotation_matrix = np.array(
        [
            [np.cos(segment_strike), -np.sin(segment_strike)],
            [np.sin(segment_strike), np.cos(segment_strike)],
        ]
    )
    station_x_rotated, station_y_rotated = np.hsplit(
        np.einsum("ij,kj->ik", np.dstack((station_x, station_y))[0], rotation_matrix.T),
        2,
    )

    # Shift station y coordinates by surface projection of locking depth
    # y_shift will be positive for dips <90 and negative for dips > 90
    y_shift = np.cos(np.deg2rad(segment_dip)) * segment_up_dip_width
    station_y_rotated += y_shift

    # Elastic displacements from Okada 1992
    alpha = (material_lambda + material_mu) / (material_lambda + 2 * material_mu)
    u_x = np.zeros_like(station_x)
    u_y = np.zeros_like(station_x)
    u_up = np.zeros_like(station_x)
    for i in range(len(station_x)):
        _, u, _ = okada_wrapper.dc3dwrapper(
            alpha,  # (lambda + mu) / (lambda + 2 * mu)
            [
                station_x_rotated[i].item(),
                station_y_rotated[i].item(),
                0,
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
        u_x[i], u_y, u_up = u

    # Un-rotate displacement to account for projected fault strike
    # u_east, u_north = np.hsplit(
    #     np.einsum("ij,kj->ik", np.dstack((u_x, u_y))[0], rotation_matrix), 2
    # )

    # Rotate x, y displacements about geographic azimuth to yield east, north displacements
    cosstrike = np.cos(np.radians(90 - segment_azimuth))
    sinstrike = np.sin(np.radians(90 - segment_azimuth))
    u_north = sinstrike * u_x + cosstrike * u_y
    u_east = cosstrike * u_x - sinstrike * u_y
    return u_east, u_north, u_up


def get_rotation_to_slip_rate_partials(segment, block):
    """Calculate partial derivatives relating relative block motion to fault slip rates."""
    n_segments = len(segment)
    n_blocks = len(block)
    fault_slip_rate_partials = np.zeros((3 * n_segments, 3 * n_blocks))
    for i in range(n_segments):
        # Project velocities from Cartesian to spherical coordinates at segment mid-points
        row_idx = 3 * i
        column_idx_east = 3 * segment.east_labels[i]
        column_idx_west = 3 * segment.west_labels[i]
        R = get_cross_partials([segment.mid_x[i], segment.mid_y[i], segment.mid_z[i]])
        (
            vel_north_to_omega_x,
            vel_east_to_omega_x,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 0], R[1, 0], R[2, 0], segment.mid_lon[i], segment.mid_lat[i]
        )
        (
            vel_north_to_omega_y,
            vel_east_to_omega_y,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 1], R[1, 1], R[2, 1], segment.mid_lon[i], segment.mid_lat[i]
        )
        (
            vel_north_to_omega_z,
            vel_east_to_omega_z,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 2], R[1, 2], R[2, 2], segment.mid_lon[i], segment.mid_lat[i]
        )

        # Build unit vector for the fault
        # Fault strike calculated in process_segment
        # TODO: Need to check this vs. matlab azimuth for consistency
        unit_x_parallel = np.cos(np.deg2rad(90 - segment.azimuth[i]))
        unit_y_parallel = np.sin(np.deg2rad(90 - segment.azimuth[i]))
        unit_x_perpendicular = np.sin(np.deg2rad(segment.azimuth[i] - 90))
        unit_y_perpendicular = np.cos(np.deg2rad(segment.azimuth[i] - 90))

        # Projection onto fault dip
        if segment.lat2[i] < segment.lat1[i]:
            unit_x_parallel = -unit_x_parallel
            unit_y_parallel = -unit_y_parallel
            unit_x_perpendicular = -unit_x_perpendicular
            unit_y_perpendicular = -unit_y_perpendicular

        # This is the logic for dipping vs. non-dipping faults
        # If fault is dipping make it so that the dip slip rate has a fault normal
        # component equal to the fault normal differential plate velocity.  This
        # is kinematically consistent in the horizontal but *not* in the vertical.
        if segment.dip[i] != 90:
            scale_factor = 1 / abs(np.cos(np.deg2rad(segment.dip[i])))
            slip_rate_matrix = np.array(
                [
                    [
                        unit_x_parallel * vel_east_to_omega_x
                        + unit_y_parallel * vel_north_to_omega_x,
                        unit_x_parallel * vel_east_to_omega_y
                        + unit_y_parallel * vel_north_to_omega_y,
                        unit_x_parallel * vel_east_to_omega_z
                        + unit_y_parallel * vel_north_to_omega_z,
                    ],
                    [
                        scale_factor
                        * (
                            unit_x_perpendicular * vel_east_to_omega_x
                            + unit_y_perpendicular * vel_north_to_omega_x
                        ),
                        scale_factor
                        * (
                            unit_x_perpendicular * vel_east_to_omega_y
                            + unit_y_perpendicular * vel_north_to_omega_y
                        ),
                        scale_factor
                        * (
                            unit_x_perpendicular * vel_east_to_omega_z
                            + unit_y_perpendicular * vel_north_to_omega_z
                        ),
                    ],
                    [0, 0, 0],
                ]
            )
        else:
            scale_factor = (
                -1
            )  # This is for consistency with the Okada convention for tensile faulting
            slip_rate_matrix = np.array(
                [
                    [
                        unit_x_parallel * vel_east_to_omega_x
                        + unit_y_parallel * vel_north_to_omega_x,
                        unit_x_parallel * vel_east_to_omega_y
                        + unit_y_parallel * vel_north_to_omega_y,
                        unit_x_parallel * vel_east_to_omega_z
                        + unit_y_parallel * vel_north_to_omega_z,
                    ],
                    [0, 0, 0],
                    [
                        scale_factor
                        * (
                            unit_x_perpendicular * vel_east_to_omega_x
                            + unit_y_perpendicular * vel_north_to_omega_x
                        ),
                        scale_factor
                        * (
                            unit_x_perpendicular * vel_east_to_omega_y
                            + unit_y_perpendicular * vel_north_to_omega_y
                        ),
                        scale_factor
                        * (
                            unit_x_perpendicular * vel_east_to_omega_z
                            + unit_y_perpendicular * vel_north_to_omega_z
                        ),
                    ],
                ]
            )

        fault_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_east : column_idx_east + 3
        ] = slip_rate_matrix
        fault_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_west : column_idx_west + 3
        ] = -slip_rate_matrix
    return fault_slip_rate_partials


def get_tde_to_velocities(meshes, station, command):
    """Calculates the elastic displacement partial derivatives based on the
    T. Ben Thompson cutde of the Nikhool and Walters (2015) equations
    for the displacements resulting from slip on a triangular
    dislocation in a homogeneous elastic half space.

    The linear operator is structured as ():

                ss(tri1)  ds(tri1) ts(tri1) ... ss(triN) ds(triN) ts(triN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if len(meshes) > 0:
        n_tris = meshes[0].lon1.size
        if not station.empty:
            tri_operator = np.zeros((3 * len(station), 3 * n_tris))

            # Loop through each segment and calculate displacements for each slip component
            for i in tqdm(
                range(n_tris),
                desc="Calculating cutde partials for triangles",
                colour="green",
            ):
                (
                    vel_east_strike_slip,
                    vel_north_strike_slip,
                    vel_up_strike_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=1,
                    dip_slip=0,
                    tensile_slip=0,
                )
                (
                    vel_east_dip_slip,
                    vel_north_dip_slip,
                    vel_up_dip_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=1,
                    tensile_slip=0,
                )
                (
                    vel_east_tensile_slip,
                    vel_north_tensile_slip,
                    vel_up_tensile_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=0,
                    tensile_slip=1,
                )
                tri_operator[0::3, 3 * i] = np.squeeze(vel_east_strike_slip)
                tri_operator[1::3, 3 * i] = np.squeeze(vel_north_strike_slip)
                tri_operator[2::3, 3 * i] = np.squeeze(vel_up_strike_slip)
                tri_operator[0::3, 3 * i + 1] = np.squeeze(vel_east_dip_slip)
                tri_operator[1::3, 3 * i + 1] = np.squeeze(vel_north_dip_slip)
                tri_operator[2::3, 3 * i + 1] = np.squeeze(vel_up_dip_slip)
                tri_operator[0::3, 3 * i + 2] = np.squeeze(vel_east_tensile_slip)
                tri_operator[1::3, 3 * i + 2] = np.squeeze(vel_north_tensile_slip)
                tri_operator[2::3, 3 * i + 2] = np.squeeze(vel_up_tensile_slip)
        else:
            tri_operator = np.empty(0)
    else:
        tri_operator = np.empty(0)
    return tri_operator


def get_tde_to_velocities_single_mesh(meshes, station, command, mesh_idx):
    """Calculates the elastic displacement partial derivatives based on the
    T. Ben Thompson cutde of the Nikhool and Walters (2015) equations
    for the displacements resulting from slip on a triangular
    dislocation in a homogeneous elastic half space.

    The linear operator is structured as ():

                ss(tri1)  ds(tri1) ts(tri1) ... ss(triN) ds(triN) ts(triN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if len(meshes) > 0:
        n_tris = meshes[mesh_idx].lon1.size
        if not station.empty:
            tri_operator = np.zeros((3 * len(station), 3 * n_tris))

            # Loop through each segment and calculate displacements for each slip component
            for i in tqdm(
                range(n_tris),
                desc="Calculating cutde partials for triangles",
                colour="green",
            ):
                (
                    vel_east_strike_slip,
                    vel_north_strike_slip,
                    vel_up_strike_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=1,
                    dip_slip=0,
                    tensile_slip=0,
                    mesh_idx=mesh_idx,
                )
                (
                    vel_east_dip_slip,
                    vel_north_dip_slip,
                    vel_up_dip_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=1,
                    tensile_slip=0,
                    mesh_idx=mesh_idx,
                )
                (
                    vel_east_tensile_slip,
                    vel_north_tensile_slip,
                    vel_up_tensile_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=0,
                    tensile_slip=1,
                    mesh_idx=mesh_idx,
                )
                tri_operator[0::3, 3 * i] = np.squeeze(vel_east_strike_slip)
                tri_operator[1::3, 3 * i] = np.squeeze(vel_north_strike_slip)
                tri_operator[2::3, 3 * i] = np.squeeze(vel_up_strike_slip)
                tri_operator[0::3, 3 * i + 1] = np.squeeze(vel_east_dip_slip)
                tri_operator[1::3, 3 * i + 1] = np.squeeze(vel_north_dip_slip)
                tri_operator[2::3, 3 * i + 1] = np.squeeze(vel_up_dip_slip)
                tri_operator[0::3, 3 * i + 2] = np.squeeze(vel_east_tensile_slip)
                tri_operator[1::3, 3 * i + 2] = np.squeeze(vel_north_tensile_slip)
                tri_operator[2::3, 3 * i + 2] = np.squeeze(vel_up_tensile_slip)
        else:
            tri_operator = np.empty(0)
    else:
        tri_operator = np.empty(0)
    return tri_operator


def get_tri_displacements(
    obs_lon,
    obs_lat,
    meshes,
    material_lambda,
    material_mu,
    tri_idx,
    strike_slip,
    dip_slip,
    tensile_slip,
):
    """Calculate surface displacments due to slip on a triangular dislocation
    element in a half space.  Includes projection from longitude and
    latitude to locally tangent planar coordinate system.
    """
    poissons_ratio = material_mu / (2 * (material_mu + material_lambda))

    # Project coordinates
    tri_centroid_lon = meshes[0].centroids[tri_idx, 0]
    tri_centroid_lat = meshes[0].centroids[tri_idx, 1]
    projection = get_transverse_projection(tri_centroid_lon, tri_centroid_lat)
    obs_x, obs_y = projection(obs_lon, obs_lat)
    tri_x1, tri_y1 = projection(meshes[0].lon1[tri_idx], meshes[0].lat1[tri_idx])
    tri_x2, tri_y2 = projection(meshes[0].lon2[tri_idx], meshes[0].lat2[tri_idx])
    tri_x3, tri_y3 = projection(meshes[0].lon3[tri_idx], meshes[0].lat3[tri_idx])
    tri_z1 = KM2M * meshes[0].dep1[tri_idx]
    tri_z2 = KM2M * meshes[0].dep2[tri_idx]
    tri_z3 = KM2M * meshes[0].dep3[tri_idx]

    # Package coordinates for cutde call
    obs_coords = np.vstack((obs_x, obs_y, np.zeros_like(obs_x))).T
    tri_coords = np.array(
        [[tri_x1, tri_y1, tri_z1], [tri_x2, tri_y2, tri_z2], [tri_x3, tri_y3, tri_z3]]
    )

    # Call cutde, multiply by displacements, and package for the return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mat = cutde_halfspace.disp_matrix(
            obs_pts=obs_coords, tris=np.array([tri_coords]), nu=poissons_ratio
        )
    slip = np.array([[strike_slip, dip_slip, tensile_slip]])
    disp = disp_mat.reshape((-1, 3)).dot(slip.flatten())
    vel_east = disp[0::3]
    vel_north = disp[1::3]
    vel_up = disp[2::3]
    return vel_east, vel_north, vel_up


def get_tri_displacements_single_mesh(
    obs_lon,
    obs_lat,
    meshes,
    material_lambda,
    material_mu,
    tri_idx,
    strike_slip,
    dip_slip,
    tensile_slip,
    mesh_idx,
):
    """Calculate surface displacments due to slip on a triangular dislocation
    element in a half space.  Includes projection from longitude and
    latitude to locally tangent planar coordinate system.
    """
    poissons_ratio = material_mu / (2 * (material_mu + material_lambda))

    # Project coordinates
    tri_centroid_lon = meshes[mesh_idx].centroids[tri_idx, 0]
    tri_centroid_lat = meshes[mesh_idx].centroids[tri_idx, 1]
    projection = get_transverse_projection(tri_centroid_lon, tri_centroid_lat)
    obs_x, obs_y = projection(obs_lon, obs_lat)
    tri_x1, tri_y1 = projection(
        meshes[mesh_idx].lon1[tri_idx], meshes[mesh_idx].lat1[tri_idx]
    )
    tri_x2, tri_y2 = projection(
        meshes[mesh_idx].lon2[tri_idx], meshes[mesh_idx].lat2[tri_idx]
    )
    tri_x3, tri_y3 = projection(
        meshes[mesh_idx].lon3[tri_idx], meshes[mesh_idx].lat3[tri_idx]
    )
    tri_z1 = KM2M * meshes[mesh_idx].dep1[tri_idx]
    tri_z2 = KM2M * meshes[mesh_idx].dep2[tri_idx]
    tri_z3 = KM2M * meshes[mesh_idx].dep3[tri_idx]

    # Package coordinates for cutde call
    obs_coords = np.vstack((obs_x, obs_y, np.zeros_like(obs_x))).T
    tri_coords = np.array(
        [[tri_x1, tri_y1, tri_z1], [tri_x2, tri_y2, tri_z2], [tri_x3, tri_y3, tri_z3]]
    )

    # Call cutde, multiply by displacements, and package for the return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mat = cutde_halfspace.disp_matrix(
            obs_pts=obs_coords, tris=np.array([tri_coords]), nu=poissons_ratio
        )
    slip = np.array([[strike_slip, dip_slip, tensile_slip]])
    disp = disp_mat.reshape((-1, 3)).dot(slip.flatten())
    vel_east = disp[0::3]
    vel_north = disp[1::3]
    vel_up = disp[2::3]
    return vel_east, vel_north, vel_up


def get_tri_smoothing_matrix(share, tri_shared_sides_distances):
    """Produces a smoothing matrix based on the scale-dependent
    umbrella operator (e.g., Desbrun et al., 1999; Resor, 2004).

    Inputs:
    share: n x 3 array of indices of the up to 3 elements sharing a side
        with each of the n elements
    tri_shared_sides_distances: n x 3 array of distances between each of the
        n elements and its up to 3 neighbors

    Outputs:
    smoothing matrix: 3n x 3n smoothing matrix
    """
    # Allocate sparse matrix for contructing smoothing matrix
    n_shared_tris = share.shape[0]
    smoothing_matrix = scipy.sparse.lil_matrix((3 * n_shared_tris, 3 * n_shared_tris))

    # Create a design matrix for Laplacian construction
    share_copy = copy.deepcopy(share)
    share_copy[np.where(share == -1)] = 0
    share_copy[np.where(share != -1)] = 1

    # Sum the distances between each element and its neighbors
    share_distances = np.sum(tri_shared_sides_distances, axis=1)
    leading_coefficient = 2.0 / share_distances

    # Replace zero distances with 1 to avoid divide by zero
    tri_shared_sides_distances[np.where(tri_shared_sides_distances == 0)] = 1

    # Take the reciprocal of the distances
    inverse_tri_shared_sides_distances = 1.0 / tri_shared_sides_distances

    # Diagonal terms # TODO: Defnitely not sure about his line!!!
    diagonal_terms = -leading_coefficient * np.sum(
        inverse_tri_shared_sides_distances * share_copy,
        axis=1,
    )

    # Off-diagonal terms
    off_diagonal_terms = (
        np.vstack((leading_coefficient, leading_coefficient, leading_coefficient)).T
        * inverse_tri_shared_sides_distances
        * share_copy
    )

    # Place the weights into the smoothing operator
    for j in range(3):
        for i in range(n_shared_tris):
            smoothing_matrix[3 * i + j, 3 * i + j] = diagonal_terms[i]
            if share[i, j] != -1:
                k = 3 * i + np.array([0, 1, 2])
                m = 3 * share[i, j] + np.array([0, 1, 2])
                smoothing_matrix[k, m] = off_diagonal_terms[i, j]
    return smoothing_matrix


def get_all_mesh_smoothing_matrices(meshes: list, operators: dict):
    """Build smoothing matrices for each of the triangular meshes
    stored in meshes.
    """
    for i in range(len(meshes)):
        # Get smoothing operator for a single mesh.
        meshes[i].share = get_shared_sides(meshes[i].verts)
        meshes[i].tri_shared_sides_distances = get_tri_shared_sides_distances(
            meshes[i].share,
            meshes[i].x_centroid,
            meshes[i].y_centroid,
            meshes[i].z_centroid,
        )
        operators.smoothing_matrix[i] = get_tri_smoothing_matrix(
            meshes[i].share, meshes[i].tri_shared_sides_distances
        )


def get_all_mesh_smoothing_matrices_simple(meshes: list, operators: dict):
    """Build smoothing matrices for each of the triangular meshes
    stored in meshes
    These are the simple not distance weighted meshes.
    """
    for i in range(len(meshes)):
        # Get smoothing operator for a single mesh.
        meshes[i].share = get_shared_sides(meshes[i].verts)
        operators.smoothing_matrix[i] = get_tri_smoothing_matrix_simple(
            meshes[i].share, N_MESH_DIM
        )


def get_tri_smoothing_matrix_simple(share, n_dim):
    """Produces a smoothing matrix based without scale-dependent
    weighting.

    Inputs:
    share: n x 3 array of indices of the up to 3 elements sharing a side
        with each of the n elements

    Outputs:
    smoothing matrix: n_dim * n x n_dim * n smoothing matrix
    """
    # Allocate sparse matrix for contructing smoothing matrix
    n_shared_tri = share.shape[0]
    smoothing_matrix = scipy.sparse.lil_matrix(
        (n_dim * n_shared_tri, n_dim * n_shared_tri)
    )

    for j in range(n_dim):
        for i in range(n_shared_tri):
            smoothing_matrix[n_dim * i + j, n_dim * i + j] = 3
            if share[i, j] != -1:
                k = n_dim * i + np.arange(n_dim)
                m = n_dim * share[i, j] + np.arange(n_dim)
                smoothing_matrix[k, m] = -1
    return smoothing_matrix


def get_tde_slip_rate_constraints(meshes: dict, operators: dict):
    """Construct TDE slip rate constraint matrices for each mesh.
    These are identity matrices, used to set TDE slip rates on
    or coupling fractions on elements lining the edges of the mesh,
    as controlled by input parameters
    top_slip_rate_constraint,
    bot_slip_rate_constraint,
    side_slip_rate_constraint,.

    and at other elements with indices specified as
    ss_slip_constraint_idx,
    ds_slip_constraint_idx,
    coupling_constraint_idx

    Args:
        meshes (List): list of mesh dictionaries
        operators (Dict): dictionary of linear operators
    """
    for i in range(len(meshes)):
        # Empty constraint matrix
        tde_slip_rate_constraints = np.zeros((2 * meshes[i].n_tde, 2 * meshes[i].n_tde))
        # Counting index
        start_row = 0
        end_row = 0
        # Top constraints
        # A value of 1 (free slip) or 2 (full coupling) will satisfy the following condition
        if meshes[i].config.top_slip_rate_constraint > 0:
            # Indices of top elements
            top_indices = np.asarray(np.where(meshes[i].top_elements))
            # Indices of top elements' 2 slip components
            meshes[i].top_slip_idx = get_2component_index(top_indices)
            end_row = len(meshes[i].top_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].top_slip_idx] = (
                np.eye(len(meshes[i].top_slip_idx))
            )
        # Bottom constraints
        if meshes[i].config.bot_slip_rate_constraint > 0:
            # Indices of bottom elements
            bot_indices = np.asarray(np.where(meshes[i].bot_elements))
            # Indices of bottom elements' 2 slip components
            meshes[i].bot_slip_idx = get_2component_index(bot_indices)
            start_row = end_row
            end_row = start_row + len(meshes[i].bot_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].bot_slip_idx] = (
                np.eye(len(meshes[i].bot_slip_idx))
            )
        # Side constraints
        if meshes[i].config.side_slip_rate_constraint > 0:
            # Indices of side elements
            side_indices = np.asarray(np.where(meshes[i].side_elements))
            # Indices of side elements' 2 slip components
            meshes[i].side_slip_idx = get_2component_index(side_indices)
            start_row = end_row
            end_row = start_row + len(meshes[i].side_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].side_slip_idx] = (
                np.eye(len(meshes[i].side_slip_idx))
            )
        # Other element indices
        if len(meshes[i].config.coupling_constraint_idx) > 0:
            meshes[i].coup_idx = get_2component_index(
                np.asarray(meshes[i].config.coupling_constraint_idx)
            )
            start_row += end_row
            end_row += 2 * len(meshes[i].config.coupling_constraint_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].coup_idx] = np.eye(
                2 * len(meshes[i].config.coupling_constraint_idx)
            )
        if len(meshes[i].config.ss_slip_constraint_idx) > 0:
            ss_idx = get_2component_index(
                np.asarray(meshes[i].config.ss_slip_constraint_idx)
            )
            meshes[i].ss_slip_idx = ss_idx[0::2]
            start_row += end_row
            end_row += len(meshes[i].ss_slip_idx)
            # Component slip needs identities placed every other row, column
            tde_slip_rate_constraints[start_row:end_row, meshes[i].ss_slip_idx] = (
                np.eye(len(meshes[i].ss_slip_idx))
            )
        if len(meshes[i].config.ds_slip_constraint_idx) > 0:
            ds_idx = get_2component_index(
                np.asarray(meshes[i].config.ds_slip_constraint_idx)
            )
            meshes[i].ds_slip_idx = ds_idx[1::2]
            start_row += end_row
            end_row += len(meshes[i].ds_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].ds_slip_idx] = (
                np.eye(len(meshes[i].ds_slip_idx))
            )
        # Eliminate blank rows
        sum_constraint_columns = np.sum(tde_slip_rate_constraints, 1)
        tde_slip_rate_constraints = tde_slip_rate_constraints[
            sum_constraint_columns > 0, :
        ]
        operators.tde_slip_rate_constraints[i] = tde_slip_rate_constraints
        # Total number of slip constraints:
        # 2 for each element that has coupling constrained (top, bottom, side, specified indices)
        # 1 for each additional slip component that is constrained (specified indices)

        # TODO: Number of total constraints is determined by just finding 1 in the
        # constraint array. This could cause an error when the index Dict is constructed,
        # if an individual element has a constraint imposed, but that element is also
        # a constrained edge element. Need to build in some uniqueness tests.
        meshes[i].n_tde_constraints = np.sum(sum_constraint_columns > 0)


def get_tde_coupling_constraints(meshes, segment, block, operators):
    """Get partials relating block motion to TDE slip rates for coupling constraints."""
    # for mesh_idx in range(len(meshes)):
    # Loop only over meshes that are tied to fault segments.  This *should*
    # eliminate touching CMI meshes which have problems with this function
    # becase it assumes that a mesh is tied to segments.
    for mesh_idx in range(np.max(segment.patch_file_name) + 1):
        operators.rotation_to_tri_slip_rate[mesh_idx] = (
            get_rotation_to_tri_slip_rate_partials(
                meshes[mesh_idx], mesh_idx, segment, block
            )
        )
        # Trim tensile rows
        tri_keep_rows = get_keep_index_12(
            np.shape(operators.rotation_to_tri_slip_rate[mesh_idx])[0]
        )
        operators.rotation_to_tri_slip_rate[mesh_idx] = (
            operators.rotation_to_tri_slip_rate[mesh_idx][tri_keep_rows, :]
        )


def get_rotation_to_tri_slip_rate_partials(meshes, mesh_idx, segment, block):
    """Calculate partial derivatives relating relative block motion to TDE slip rates
    for a single mesh. Called within a loop from get_tde_coupling_constraints().
    """
    n_blocks = len(block)
    tri_slip_rate_partials = np.zeros((3 * meshes.n_tde, 3 * n_blocks))

    # Generate strikes for elements using same sign convention as segments
    np.array(
        meshes.strike + 90
    )  # Dip direction. Not wrapping to 360 so we can use it as a threshold
    # dipdir(dipdir > 360) -= 360
    tristrike = np.array(meshes.strike)
    tristrike[meshes.strike > 180] -= 180
    tristrike2 = np.array(meshes.strike)
    tristrike2[meshes.strike > 180] -= 360
    tridip = np.array(meshes.dip)
    tridip[meshes.strike > 180] = 180 - tridip[meshes.strike > 180]
    # Find subset of segments that are replaced by this mesh
    seg_replace_idx = np.where(
        (segment.patch_flag != 0) & (segment.patch_file_name == mesh_idx)
    )
    # Find closest segment midpoint to each element centroid, using scipy.spatial.cdist
    meshes.closest_segment_idx = seg_replace_idx[0][
        cdist(
            np.array([meshes.lon_centroid, meshes.lat_centroid]).T,
            np.array(
                [
                    segment.mid_lon[seg_replace_idx[0]],
                    segment.mid_lat[seg_replace_idx[0]],
                ]
            ).T,
        ).argmin(axis=1)
    ]
    # Add segment labels to elements
    meshes.east_labels = np.array(segment.east_labels[meshes.closest_segment_idx])
    meshes.west_labels = np.array(segment.west_labels[meshes.closest_segment_idx])

    # Check for switching of block labels
    seg_dip_dir = np.array(segment.azimuth)
    seg_dip_dir = seg_dip_dir + np.sign(np.cos(np.deg2rad(segment.dip))) * 90
    seg_dip_dir_x = np.cos(np.deg2rad(90 - seg_dip_dir))
    seg_dip_dir_y = np.sin(np.deg2rad(90 - seg_dip_dir))
    seg_comps = np.vstack(
        [seg_dip_dir_x[:], seg_dip_dir_y[:], np.zeros_like(seg_dip_dir_x)]
    ).T
    tri_dip_dir = np.array(meshes.strike) + 90
    tri_dip_dir_x = np.cos(np.deg2rad(90 - tri_dip_dir))
    tri_dip_dir_y = np.sin(np.deg2rad(90 - tri_dip_dir))
    tri_comps = np.vstack(
        [tri_dip_dir_x[:], tri_dip_dir_y[:], np.zeros_like(tri_dip_dir_x)]
    ).T
    north_tri_cross = np.cross(
        np.array([0, 1, 0]),
        tri_comps,
    )
    north_seg_cross = np.cross(
        np.array([0, 1, 0]),
        seg_comps,
    )

    # Find rotation partials for each element
    for el_idx in range(meshes.n_tde):
        # Project velocities from Cartesian to spherical coordinates at element centroids
        row_idx = 3 * el_idx
        column_idx_east = 3 * meshes.east_labels[el_idx]
        column_idx_west = 3 * meshes.west_labels[el_idx]
        R = get_cross_partials(
            [
                meshes.x_centroid[el_idx],
                meshes.y_centroid[el_idx],
                meshes.z_centroid[el_idx],
            ]
        )
        (
            vel_north_to_omega_x,
            vel_east_to_omega_x,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 0],
            R[1, 0],
            R[2, 0],
            meshes.lon_centroid[el_idx],
            meshes.lat_centroid[el_idx],
        )
        (
            vel_north_to_omega_y,
            vel_east_to_omega_y,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 1],
            R[1, 1],
            R[2, 1],
            meshes.lon_centroid[el_idx],
            meshes.lat_centroid[el_idx],
        )
        (
            vel_north_to_omega_z,
            vel_east_to_omega_z,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 2],
            R[1, 2],
            R[2, 2],
            meshes.lon_centroid[el_idx],
            meshes.lat_centroid[el_idx],
        )
        # This correction gives -1 for strikes > 90
        # Equivalent to the if statement in get_rotation_to_slip_rate_partials
        sign_corr = -np.sign(tristrike[el_idx] - (90 + 1e-5))
        # sign_corr = np.sign(meshes.strike[el_idx] - (180 + 1e-5))
        # Sign correction based on dip direction. Sign is flipped for south-dipping elements
        # sign_corr = -np.sign(dipdir[el_idx] - (270 + 1e-5))

        # sign_corr = 1
        # Project about fault strike
        unit_x_parallel = sign_corr * np.cos(np.deg2rad(90 - tristrike[el_idx]))
        unit_y_parallel = sign_corr * np.sin(np.deg2rad(90 - tristrike[el_idx]))
        unit_x_perpendicular = sign_corr * np.sin(np.deg2rad(tristrike[el_idx] - 90))
        unit_y_perpendicular = sign_corr * np.cos(np.deg2rad(tristrike[el_idx] - 90))
        # Project by fault dip
        scale_factor = 1.0 / (np.cos(np.deg2rad(meshes.dip[el_idx])))
        slip_rate_matrix = np.array(
            [
                [
                    (
                        unit_x_parallel * vel_east_to_omega_x
                        + unit_y_parallel * vel_north_to_omega_x
                    ),
                    (
                        unit_x_parallel * vel_east_to_omega_y
                        + unit_y_parallel * vel_north_to_omega_y
                    ),
                    (
                        unit_x_parallel * vel_east_to_omega_z
                        + unit_y_parallel * vel_north_to_omega_z
                    ),
                ],
                [
                    scale_factor
                    * (
                        unit_x_perpendicular * vel_east_to_omega_x
                        + unit_y_perpendicular * vel_north_to_omega_x
                    ),
                    scale_factor
                    * (
                        unit_x_perpendicular * vel_east_to_omega_y
                        + unit_y_perpendicular * vel_north_to_omega_y
                    ),
                    scale_factor
                    * (
                        unit_x_perpendicular * vel_east_to_omega_z
                        + unit_y_perpendicular * vel_north_to_omega_z
                    ),
                ],
                [0, 0, 0],
            ]
        )
        # Additional sign correction needs to compare TDE strike and corresponding segment strike
        # If they're on different sides of an E-W line, we need to apply a negative sign
        # This effectively flips the east and west labels
        # Equivalently, this situation is where the dip directions of the segment and triangle are
        # on different sides of a N-S line
        # ew_switch = np.sign(meshes.strike[el_idx] - 270) * np.sign(
        #     segment.azimuth[meshes.closest_segment_idx[el_idx]] - 90
        # )
        # ew_switch = np.sign(90 - tristrike2[el_idx]) * np.sign(
        #     90 - segment.azimuth[meshes.closest_segment_idx[el_idx]]
        # )
        ew_switch = np.sign(north_tri_cross[el_idx, 2]) * np.sign(
            north_seg_cross[meshes.closest_segment_idx[el_idx], 2]
        )
        # ew_switch = 1
        # Insert this element's partials into operator
        tri_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_east : column_idx_east + 3
        ] = ew_switch * slip_rate_matrix
        tri_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_west : column_idx_west + 3
        ] = -ew_switch * slip_rate_matrix
    return tri_slip_rate_partials


def get_block_motion_constraints(assembly: dict, block: pd.DataFrame, command: dict):
    """Applying a priori block motion constraints."""
    block_constraint_partials = get_block_motion_constraint_partials(block)
    assembly.index.block_constraints_idx = np.where(block.rotation_flag == 1)[0]

    assembly.data.n_block_constraints = len(assembly.index.block_constraints_idx)
    assembly.data.block_constraints = np.zeros(block_constraint_partials.shape[0])
    assembly.sigma.block_constraints = np.zeros(block_constraint_partials.shape[0])
    if assembly.data.n_block_constraints > 0:
        (
            assembly.data.block_constraints[0::3],
            assembly.data.block_constraints[1::3],
            assembly.data.block_constraints[2::3],
        ) = sph2cart(
            block.euler_lon[assembly.index.block_constraints_idx],
            block.euler_lat[assembly.index.block_constraints_idx],
            np.deg2rad(block.rotation_rate[assembly.index.block_constraints_idx]),
        )
        euler_pole_covariance_all = np.diag(
            np.concatenate(
                (
                    np.deg2rad(
                        block.euler_lat_sig[assembly.index.block_constraints_idx]
                    ),
                    np.deg2rad(
                        block.euler_lon_sig[assembly.index.block_constraints_idx]
                    ),
                    np.deg2rad(
                        block.rotation_rate_sig[assembly.index.block_constraints_idx]
                    ),
                )
            )
        )
        (
            assembly.sigma.block_constraints[0::3],
            assembly.sigma.block_constraints[1::3],
            assembly.sigma.block_constraints[2::3],
        ) = euler_pole_covariance_to_rotation_vector_covariance(
            assembly.data.block_constraints[0::3],
            assembly.data.block_constraints[1::3],
            assembly.data.block_constraints[2::3],
            euler_pole_covariance_all,
        )
    assembly.sigma.block_constraint_weight = command.block_constraint_weight
    return assembly, block_constraint_partials


def get_block_motion_constraint_partials(block):
    """Partials for a priori block motion constraints.
    Essentially a set of eye(3) matrices.
    """
    # apriori_block_idx = np.where(block.apriori_flag.to_numpy() == 1)[0]
    apriori_rotation_block_idx = np.where(block.rotation_flag.to_numpy() == 1)[0]

    operator = np.zeros((3 * len(apriori_rotation_block_idx), 3 * len(block)))
    for i in range(len(apriori_rotation_block_idx)):
        start_row = 3 * i
        start_column = 3 * apriori_rotation_block_idx[i]
        operator[start_row : start_row + 3, start_column : start_column + 3] = np.eye(3)
    return operator


def get_slip_rate_constraints(assembly, segment, block, command):
    n_total_slip_rate_contraints = (
        np.sum(segment.ss_rate_flag.values)
        + np.sum(segment.ds_rate_flag.values)
        + np.sum(segment.ts_rate_flag.values)
    )
    if n_total_slip_rate_contraints > 0:
        logger.info(f"Found {n_total_slip_rate_contraints} slip rate constraints")
        for i in range(len(segment.lon1)):
            if segment.ss_rate_flag[i] == 1:
                logger.info(
                    "Strike-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + f"{segment.ss_rate[i]:.2f}"
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + f"{segment.ss_rate_sig[i]:.2f}"
                    + " (mm/yr)"
                )
            if segment.ds_rate_flag[i] == 1:
                logger.info(
                    "Dip-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + f"{segment.ds_rate[i]:.2f}"
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + f"{segment.ds_rate_sig[i]:.2f}"
                    + " (mm/yr)"
                )
            if segment.ts_rate_flag[i] == 1:
                logger.info(
                    "Tensile-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + f"{segment.ts_rate[i]:.2f}"
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + f"{segment.ts_rate_sig[i]:.2f}"
                    + " (mm/yr)"
                )
    else:
        logger.info("No slip rate constraints")

    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(segment, block)

    slip_rate_constraint_flag = interleave3(
        segment.ss_rate_flag, segment.ds_rate_flag, segment.ts_rate_flag
    )
    assembly.index.slip_rate_constraints = np.where(slip_rate_constraint_flag == 1)[0]
    assembly.data.n_slip_rate_constraints = len(assembly.index.slip_rate_constraints)

    assembly.data.slip_rate_constraints = interleave3(
        segment.ss_rate, segment.ds_rate, segment.ts_rate
    )

    assembly.data.slip_rate_constraints = assembly.data.slip_rate_constraints[
        assembly.index.slip_rate_constraints
    ]

    assembly.sigma.slip_rate_constraints = interleave3(
        segment.ss_rate_sig, segment.ds_rate_sig, segment.ts_rate_sig
    )

    assembly.sigma.slip_rate_constraints = assembly.sigma.slip_rate_constraints[
        assembly.index.slip_rate_constraints
    ]

    slip_rate_constraint_partials = slip_rate_constraint_partials[
        assembly.index.slip_rate_constraints, :
    ]
    assembly.sigma.slip_rate_constraint_weight = command.slip_constraint_weight
    return assembly, slip_rate_constraint_partials


def get_slip_rake_constraints(assembly, segment, block, command):
    n_total_slip_rake_contraints = np.sum(segment.rake_flag.values)
    if n_total_slip_rake_contraints > 0:
        logger.info(f"Found {n_total_slip_rake_contraints} slip rake constraints")
        for i in range(len(segment.lon1)):
            if segment.rake_flag[i] == 1:
                logger.info(
                    "Rake constraint on "
                    + segment.name[i].strip()
                    + ": rake = "
                    + f"{segment.rake[i]:.2f}"
                    + ", constraint strike = "
                    + f"{segment.rake_strike[i]:.2f}"
                    + ", 1-sigma uncertainty = +/-"
                    + f"{segment.rake_sig[i]:.2f}"
                )
    else:
        logger.info("No slip rake constraints")
    # To keep this a standalone function, let's calculate the full set of slip rate partials
    # TODO: Check how get_slip_rate_constraints is called to see if we need to recalculate the full set of partials, or if we can reuse a previous calculation
    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(segment, block)
    # Figure out effective rake. This is a simple correction of the rake data by the calculated strike of the segment
    # The idea is that the source of the rake constraint will include its own strike (and dip), which may differ from the model segment geometry
    # TODO: Full three-dimensional rotation of rake vector, based on strike and dip of constraint source?
    effective_rakes = segment.rake[segment.rake_flag] + (
        segment.strike[segment.rake_flag] - segment.rake_strike[segment.rake_flag]
    )

    # Find indices of constrained segments
    assembly.index.slip_rake_constraints = np.where(segment.rake_flag == 1)[0]
    assembly.data.n_slip_rake_constraints = len(assembly.index.slip_rake_constraints)

    # Get component indices of slip rate partials
    rake_constraint_component_indices = get_2component_index(
        assembly.index.slip_rake_constraints
    )
    # Rotate slip partials about effective rake. We just want to use the second row (second basis vector) of a full rotation matrix, because we want to set slip in that direction to zero as a constraint
    slip_rake_constraint_partials = (
        np.cos(np.radians(effective_rakes))
        * slip_rate_constraint_partials[rake_constraint_component_indices[0::2]]
        + np.sin(np.radians(effective_rakes))
        * slip_rate_constraint_partials[rake_constraint_component_indices[1::2]]
    )

    # Constraint data is all zeros, because we're setting slip perpendicular to the rake direction equal to zero
    assembly.data.slip_rake_constraints = np.zeros(
        assembly.data.n_total_slip_rake_contraints
    )

    # Insert sigmas into assembly dict
    assembly.sigma.slip_rake_constraints = segment.rake_sig

    # Using the same weighting here as for slip rate constraints.
    assembly.sigma.slip_rake_constraint_weight = command.slip_constraint_weight
    return assembly, slip_rake_constraint_partials


def get_mogi_to_velocities_partials(mogi, station, command):
    """Mogi volume change to station displacement operator."""
    if mogi.empty:
        mogi_operator = np.zeros((3 * len(station), 0))
    else:
        poissons_ratio = command.material_mu / (
            2 * (command.material_lambda + command.material_mu)
        )
        mogi_operator = np.zeros((3 * len(station), len(mogi)))
        for i in range(len(mogi)):
            mogi_depth = KM2M * mogi.depth[i]
            u_east, u_north, u_up = mogi_forward(
                mogi.lon[i],
                mogi.lat[i],
                mogi_depth,
                poissons_ratio,
                station.lon,
                station.lat,
            )

            # Insert components into partials matrix
            mogi_operator[0::3, i] = u_east
            mogi_operator[1::3, i] = u_north
            mogi_operator[2::3, i] = u_up
    return mogi_operator


def mogi_forward(mogi_lon, mogi_lat, mogi_depth, poissons_ratio, obs_lon, obs_lat):
    """Calculate displacements from a single Mogi source using
    equation 7.14 from "Earthquake and Volcano Deformation" by Paul Segall.
    """
    u_east = np.zeros_like(obs_lon)
    u_north = np.zeros_like(obs_lon)
    u_up = np.zeros_like(obs_lon)
    for i in range(obs_lon.size):
        # Find angle between source and observation as well as distance betwen them
        source_to_obs_forward_azimuth, _, source_to_obs_distance = GEOID.inv(
            mogi_lon, mogi_lat, obs_lon[i], obs_lat[i]
        )

        # Mogi displacements in cylindrical coordinates
        u_up[i] = (
            (1 - poissons_ratio)
            / np.pi
            * mogi_depth
            / ((source_to_obs_distance**2.0 + mogi_depth**2) ** 1.5)
        )
        u_radial = (
            (1 - poissons_ratio)
            / np.pi
            * source_to_obs_distance
            / ((source_to_obs_distance**2 + mogi_depth**2.0) ** 1.5)
        )

        # Convert radial displacement to east and north components
        u_east[i] = u_radial * np.sin(np.deg2rad(source_to_obs_forward_azimuth))
        u_north[i] = u_radial * np.cos(np.deg2rad(source_to_obs_forward_azimuth))
    return u_east, u_north, u_up


def get_strain_rate_displacements(
    lon_obs,
    lat_obs,
    centroid_lon,
    centroid_lat,
    strain_rate_lon_lon,
    strain_rate_lat_lat,
    strain_rate_lon_lat,
):
    """Calculate displacements due to three strain rate components.
    Equations are from Savage (2001) and expressed concisely in McCaffrey (2005)
    In McCaffrey (2005) these are the two unnumbered equations at the bottom
    of page 2.
    """
    centroid_lon = np.deg2rad(centroid_lon)
    centroid_lat = latitude_to_colatitude(centroid_lat)
    centroid_lat = np.deg2rad(centroid_lat)
    lon_obs = np.deg2rad(lon_obs)
    lat_obs = latitude_to_colatitude(lat_obs)
    lat_obs = np.deg2rad(lat_obs)
    # Calculate displacements from homogeneous strain
    u_up = np.zeros(
        lon_obs.size
    )  # Always zero here because we're assuming plane strain on the sphere
    u_east = strain_rate_lon_lon * (
        RADIUS_EARTH * (lon_obs - centroid_lon) * np.sin(centroid_lat)
    ) + strain_rate_lon_lat * (RADIUS_EARTH * (lat_obs - centroid_lat))
    u_north = strain_rate_lon_lat * (
        RADIUS_EARTH * (lon_obs - centroid_lon) * np.sin(centroid_lat)
    ) + strain_rate_lat_lat * (RADIUS_EARTH * (lat_obs - centroid_lat))
    return u_east, u_north, u_up


def get_block_strain_rate_to_velocities_partials(block, station, segment):
    """Calculate strain partial derivatives assuming a strain centroid at the center of each block."""
    strain_rate_block_idx = np.where(block.strain_rate_flag.to_numpy() > 0)[0]
    # Allocate space. Zero width, if no blocks should have strain estimated, helps with indexing
    block_strain_rate_operator = np.zeros(
        (3 * len(station), 3 * strain_rate_block_idx.size)
    )
    if strain_rate_block_idx.size > 0:
        for i in range(strain_rate_block_idx.size):
            # Find centroid of current block
            block_centroid_lon, block_centroid_lat = get_block_centroid(
                segment, strain_rate_block_idx[i]
            )
            # Find stations on current block
            station_idx = np.where(station.block_label == strain_rate_block_idx[i])[0]
            stations_block_lon = station.lon[station_idx].to_numpy()
            stations_block_lat = station.lat[station_idx].to_numpy()

            # Calculate partials for each component of strain rate
            (
                vel_east_lon_lon,
                vel_north_lon_lon,
                vel_up_lon_lon,
            ) = get_strain_rate_displacements(
                stations_block_lon,
                stations_block_lat,
                block_centroid_lon,
                block_centroid_lat,
                strain_rate_lon_lon=1,
                strain_rate_lat_lat=0,
                strain_rate_lon_lat=0,
            )
            (
                vel_east_lat_lat,
                vel_north_lat_lat,
                vel_up_lat_lat,
            ) = get_strain_rate_displacements(
                stations_block_lon,
                stations_block_lat,
                block_centroid_lon,
                block_centroid_lat,
                strain_rate_lon_lon=0,
                strain_rate_lat_lat=1,
                strain_rate_lon_lat=0,
            )
            (
                vel_east_lon_lat,
                vel_north_lon_lat,
                vel_up_lon_lat,
            ) = get_strain_rate_displacements(
                stations_block_lon,
                stations_block_lat,
                block_centroid_lon,
                block_centroid_lat,
                strain_rate_lon_lon=0,
                strain_rate_lat_lat=0,
                strain_rate_lon_lat=1,
            )
            # The sign convention established here (with negative signs on the lat_lat components) is that
            # positive longitudinal strain is extensional, and
            # positive shear strain is sinistral (counterclockwise)
            block_strain_rate_operator[3 * station_idx, 3 * i] = vel_east_lon_lon
            block_strain_rate_operator[3 * station_idx, 3 * i + 1] = -vel_east_lat_lat
            block_strain_rate_operator[3 * station_idx, 3 * i + 2] = vel_east_lon_lat
            block_strain_rate_operator[3 * station_idx + 1, 3 * i] = vel_north_lon_lon
            block_strain_rate_operator[
                3 * station_idx + 1, 3 * i + 1
            ] = -vel_north_lat_lat
            block_strain_rate_operator[3 * station_idx + 1, 3 * i + 2] = (
                vel_north_lon_lat
            )
            block_strain_rate_operator[3 * station_idx + 2, 3 * i] = vel_up_lon_lon
            block_strain_rate_operator[3 * station_idx + 2, 3 * i + 1] = -vel_up_lat_lat
            block_strain_rate_operator[3 * station_idx + 2, 3 * i + 2] = vel_up_lon_lat
    return block_strain_rate_operator, strain_rate_block_idx


def get_global_float_block_rotation_partials(station):
    """Return a linear operator for the rotations of all stations assuming they
    are the on the same block (i.e., the globe). The purpose of this is to
    allow all of the stations to "float" in the inverse problem reducing the
    dependence on reference frame specification. This is done by making a
    copy of the station data frame, setting "block_label" for all stations
    equal to zero and then calling the standard block rotation operator
    function. The matrix returned here only has 3 columns.
    """
    station_all_on_one_block = station.copy()
    station_all_on_one_block.block_label.values[:] = (
        0  # Force all stations to be on one block
    )
    global_float_block_rotation_operator = get_rotation_to_velocities_partials(
        station_all_on_one_block, 1
    )
    return global_float_block_rotation_operator


def get_data_vector(assembly, index, meshes):
    data_vector = np.zeros(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )

    # Add GPS stations to data vector
    data_vector[index.start_station_row : index.end_station_row] = interleave2(
        assembly.data.east_vel, assembly.data.north_vel
    )

    # Add block motion constraints to data vector
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * assembly.data.block_constraints
    )

    # Add slip rate constraints to data vector
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = assembly.data.slip_rate_constraints

    # Add TDE slip rate constraints to data vector
    # Coupling fractions remain zero but slip component constraints are as specified in mesh_param
    for i in range(len(meshes)):
        data_vector[
            index.start_tde_ss_slip_constraint_row[
                i
            ] : index.end_tde_ss_slip_constraint_row[i]
        ] = meshes[i].config.ss_slip_constraint_rate

        data_vector[
            index.start_tde_ds_slip_constraint_row[
                i
            ] : index.end_tde_ds_slip_constraint_row[i]
        ] = meshes[i].config.ds_slip_constraint_rate
    return data_vector


def get_weighting_vector(command, station, meshes, index):
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )
    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = command.block_constraint_weight
    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = command.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    for i in range(len(meshes)):
        # Insert smoothing weight into weighting vector
        weighting_vector[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i]
        ] = meshes[i].config.smoothing_weight * np.ones(2 * index.n_tde[i])
        weighting_vector[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i]
        ] = command.tri_con_weight * np.ones(index.n_tde_constraints[i])
    return weighting_vector


def get_weighting_vector_no_meshes(command, station, index):
    # NOTE: Consider combining with above
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )
    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )
    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = 1.0
    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = command.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    return weighting_vector


def get_weighting_vector_single_mesh_for_col_norms(
    command, station, meshes, index, mesh_index: np.int_
):
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 2 * index.n_tde[mesh_index]
        + index.n_tde_constraints[mesh_index]
    )

    weighting_vector[0 : 2 * index.n_stations] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )

    weighting_vector[
        2 * index.n_stations : 2 * index.n_stations + 2 * index.n_tde[mesh_index]
    ] = meshes[mesh_index].config.smoothing_weight * np.ones(
        2 * index.n_tde[mesh_index]
    )

    weighting_vector[2 * index.n_stations + 2 * index.n_tde[mesh_index] : :] = (
        command.tri_con_weight * np.ones(index.n_tde_constraints[mesh_index])
    )

    return weighting_vector


def get_data_vector_eigen(meshes, assembly, index):
    data_vector = np.zeros(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    # Add GPS stations to data vector
    data_vector[index.start_station_row : index.end_station_row] = interleave2(
        assembly.data.east_vel, assembly.data.north_vel
    )

    # Add block motion constraints to data vector
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * assembly.data.block_constraints
    )

    # Add slip rate constraints to data vector
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = assembly.data.slip_rate_constraints

    # Add TDE boundary condtions constraints
    for i in range(index.n_meshes):
        # Place strike-slip TDE BCs
        data_vector[
            index.start_tde_constraint_row_eigen[
                i
            ] : index.end_tde_constraint_row_eigen[i] : 2
        ] = meshes[i].config.ss_slip_constraint_rate

        # Place dip-slip TDE BCs
        data_vector[
            index.start_tde_constraint_row_eigen[i]
            + 1 : index.end_tde_constraint_row_eigen[i] : 2
        ] = meshes[i].config.ds_slip_constraint_rate

    return data_vector


def get_weighting_vector_eigen(command, station, meshes, index):
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )

    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = command.block_constraint_weight

    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = command.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    # TODO: Need to think about constraints weights
    # This is the only place where any individual constraint weights enter
    # I'm only using on of them: meshes[i].bot_slip_rate_weight
    for i in range(len(meshes)):
        weighting_vector[
            index.start_tde_constraint_row_eigen[
                i
            ] : index.end_tde_constraint_row_eigen[i]
        ] = meshes[i].config.bot_slip_rate_weight * np.ones(index.n_tde_constraints[i])

    return weighting_vector


def get_full_dense_operator_block_only(operators, index):
    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints,
            3 * index.n_blocks,
        )
    )

    # Insert block rotations and elastic velocities from fully locked segments
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_col : index.end_block_col,
    ] = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )

    # Insert block motion constraints
    operator[
        index.start_block_constraints_row : index.end_block_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.block_motion_constraints

    # Insert slip rate constraints
    operator[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.slip_rate_constraints
    return operator


def get_full_dense_operator(operators, meshes, index):
    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + 2 * index.n_tde_total
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.n_block_strain_components
            + index.n_mogi
            + 2 * index.n_tde_total,
        )
    )

    # DEBUG:
    # IPython.embed(banner1="")

    # Insert block rotations and elastic velocities from fully locked segments
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_col : index.end_block_col,
    ] = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )

    # Insert block motion constraints
    operator[
        index.start_block_constraints_row : index.end_block_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.block_motion_constraints

    # Insert slip rate constraints
    operator[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.slip_rate_constraints

    # Insert all TDE operators
    for i in range(len(meshes)):
        # Insert TDE to velocity matrix
        tde_keep_row_index = get_keep_index_12(operators.tde_to_velocities[i].shape[0])
        tde_keep_col_index = get_keep_index_12(operators.tde_to_velocities[i].shape[1])
        operator[
            index.start_station_row : index.end_station_row,
            index.start_tde_col[i] : index.end_tde_col[i],
        ] = -operators.tde_to_velocities[i][tde_keep_row_index, :][
            :, tde_keep_col_index
        ]

        # Insert TDE smoothing matrix
        smoothing_keep_index = get_keep_index_12(
            operators.tde_to_velocities[i].shape[1]
        )
        operator[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i],
            index.start_tde_col[i] : index.end_tde_col[i],
        ] = operators.smoothing_matrix[i].toarray()[smoothing_keep_index, :][
            :, smoothing_keep_index
        ]

        # Insert TDE slip rate constraints into estimation operator
        # These are just the identity matrices, and we'll insert any block motion constraints next
        operator[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i],
            index.start_tde_col[i] : index.end_tde_col[i],
        ] = operators.tde_slip_rate_constraints[i]
        # Insert block motion constraints for any coupling-constrained rows
        if meshes[i].config.top_slip_rate_constraint == 2:
            operator[
                index.start_tde_top_constraint_row[
                    i
                ] : index.end_tde_top_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                meshes[i].top_slip_idx,
                index.start_block_col : index.end_block_col,
            ]
        if meshes[i].config.bot_slip_rate_constraint == 2:
            operator[
                index.start_tde_bot_constraint_row[
                    i
                ] : index.end_tde_bot_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                meshes[i].bot_slip_idx,
                :,
            ]
        if meshes[i].config.side_slip_rate_constraint == 2:
            operator[
                index.start_tde_side_constraint_row[
                    i
                ] : index.end_tde_side_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                meshes[i].side_slip_idx,
                :,
            ]
    # Insert block strain operator
    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_strain_col : index.end_block_strain_col,
    ] = operators.block_strain_rate_to_velocities[index.station_row_keep_index, :]

    # Insert Mogi source operators
    operator[
        index.start_station_row : index.end_station_row,
        index.start_mogi_col : index.end_mogi_col,
    ] = operators.mogi_to_velocities[index.station_row_keep_index, :]
    # Insert TDE coupling constraints into estimation operator
    # The identity matrices were already inserted as part of the standard slip constraints,
    # so here we can just insert the rotation-to-slip partials into the block rotation columns
    # operator[
    #     index.start_tde_coup_constraint_row[i] : index.end_tde_coup_constraint_row[
    #         i
    #     ],
    #     index.start_block_col : index.end_block_col,
    # ] = operators.tde_coupling_constraints[i]

    return operator


def get_full_dense_operator_eigen(operators, meshes, index):
    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.n_eigen_total
            + 3 * index.n_strain_blocks
            + index.n_mogis,
        )
    )

    # Insert block rotations and elastic velocities from fully locked segments
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_col : index.end_block_col,
    ] = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )

    # Insert block motion constraints
    operator[
        index.start_block_constraints_row : index.end_block_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.block_motion_constraints

    # Insert slip rate constraints
    operator[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.slip_rate_constraints

    # EIGEN Eigenvector to velocity matrix
    for i in range(index.n_meshes):
        # Eliminate vertical elastic velocities
        tde_keep_row_index = get_keep_index_12(operators.tde_to_velocities[i].shape[0])
        tde_keep_col_index = get_keep_index_12(operators.tde_to_velocities[i].shape[1])

        # Create eigenvector to velocities operator
        operators.eigen_to_velocities[i] = (
            -operators.tde_to_velocities[i][tde_keep_row_index, :][
                :, tde_keep_col_index
            ]
            @ operators.eigenvectors_to_tde_slip[i]
        )

        # Insert eigenvector to velocities operator
        operator[
            index.start_station_row : index.end_station_row,
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = operators.eigen_to_velocities[i]

    # EIGEN Eigenvector to TDE boundary conditions matrix
    for i in range(index.n_meshes):
        # Create eigenvector to TDE boundary conditions matrix
        operators.eigen_to_tde_bcs[i] = (
            meshes[i].config.mesh_tde_modes_bc_weight
            * operators.tde_slip_rate_constraints[i]
            @ operators.eigenvectors_to_tde_slip[i]
        )

        # Insert eigenvector to TDE boundary conditions matrix
        operator[
            index.start_tde_constraint_row_eigen[
                i
            ] : index.end_tde_constraint_row_eigen[i],
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = operators.eigen_to_tde_bcs[i]

    # EIGEN: Block strain operator
    operator[
        0 : 2 * index.n_stations,
        index.start_block_strain_col_eigen : index.end_block_strain_col_eigen,
    ] = operators.block_strain_rate_to_velocities[tde_keep_row_index, :]

    # EIGEN: Mogi operator
    operator[
        0 : 2 * index.n_stations,
        index.start_mogi_col_eigen : index.end_mogi_col_eigen,
    ] = operators.mogi_to_velocities[tde_keep_row_index, :]

    return operator


def get_slip_rate_bounds(segment, block):
    n_total_slip_rate_bounds = (
        np.sum(segment.ss_rate_bound_flag.values)
        + np.sum(segment.ds_rate_bound_flag.values)
        + np.sum(segment.ts_rate_bound_flag.values)
    )
    if n_total_slip_rate_bounds > 0:
        logger.info(f"Found {n_total_slip_rate_bounds} slip rate bounds")
        for i in range(len(segment.lon1)):
            if segment.ss_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP strike-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + f"{segment.ss_rate_bound_min[i]:.2f}"
                    + " (mm/yr), rate (upper bound) = "
                    + f"{segment.ss_rate_bound_max[i]:.2f}"
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert segment.ss_rate_bound_min[i] < segment.ss_rate_bound_max[i], (
                    "Bounds min max error"
                )

            if segment.ds_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP dip-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + f"{segment.ds_rate_bound_min[i]:.2f}"
                    + " (mm/yr), rate (upper bound) = "
                    + f"{segment.ds_rate_bound_max[i]:.2f}"
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert segment.ds_rate_bound_min[i] < segment.ds_rate_bound_max[i], (
                    "Bounds min max error"
                )

            if segment.ts_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP tensile-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + f"{segment.ts_rate_bound_min[i]:.2f}"
                    + " (mm/yr), rate (upper bound) = "
                    + f"{segment.ts_rate_bound_max[i]:.2f}"
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert segment.ts_rate_bound_min[i] < segment.ts_rate_bound_max[i], (
                    "Bounds min max error"
                )

    else:
        logger.info("No hard slip rate bounds")

    # Find 3-strided indices for slip rate bounds
    slip_rate_bounds_idx = np.where(
        interleave3(
            segment.ss_rate_bound_flag,
            segment.ds_rate_bound_flag,
            segment.ts_rate_bound_flag,
        )
        == 1
    )[0]

    # Data vector for minimum slip rate bounds
    slip_rate_bound_min = interleave3(
        segment.ss_rate_bound_min, segment.ds_rate_bound_min, segment.ts_rate_bound_min
    )[slip_rate_bounds_idx]

    # Data vector for maximum slip rate bounds
    slip_rate_bound_max = interleave3(
        segment.ss_rate_bound_max, segment.ds_rate_bound_max, segment.ts_rate_bound_max
    )[slip_rate_bounds_idx]

    # Linear opeartor for slip rate bounds
    slip_rate_bound_partials = get_rotation_to_slip_rate_partials(segment, block)[
        slip_rate_bounds_idx, :
    ]
    return slip_rate_bound_min, slip_rate_bound_max, slip_rate_bound_partials


def get_qp_tde_inequality_operator_and_data_vector(index, meshes, operators):
    qp_constraint_matrix = np.zeros(
        (4 * index.n_tde_total, index.n_operator_cols_eigen)
    )
    qp_constraint_data_vector = np.zeros(4 * index.n_tde_total)

    for i in range(index.n_meshes):
        # TDE strike- and dip-slip lower bounds
        lower_bound_current_mesh = interleave2(
            meshes[i].config.qp_mesh_tde_slip_rate_lower_bound_ss
            * np.ones(index.n_tde[i]),
            meshes[i].config.qp_mesh_tde_slip_rate_lower_bound_ds
            * np.ones(index.n_tde[i]),
        )

        # TDE strike- and dip-slip upper bounds
        upper_bound_current_mesh = interleave2(
            meshes[i].config.qp_mesh_tde_slip_rate_upper_bound_ss
            * np.ones(index.n_tde[i]),
            meshes[i].config.qp_mesh_tde_slip_rate_upper_bound_ds
            * np.ones(index.n_tde[i]),
        )

        # Insert TDE lower bounds into QP constraint data vector (note negative sign)
        qp_constraint_data_vector[
            index.qp_constraint_tde_rate_start_row_eigen[
                i
            ] : index.qp_constraint_tde_rate_start_row_eigen[i] + 2 * index.n_tde[i]
        ] = -lower_bound_current_mesh

        # Insert TDE upper bounds into QP constraint data vector
        qp_constraint_data_vector[
            index.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.n_tde[i] : index.qp_constraint_tde_rate_end_row_eigen[i]
        ] = upper_bound_current_mesh

        # Insert eigenmode to TDE slip operator into QP constraint data vector for lower bounds (note negative sign)
        qp_constraint_matrix[
            index.qp_constraint_tde_rate_start_row_eigen[
                i
            ] : index.qp_constraint_tde_rate_start_row_eigen[i] + 2 * index.n_tde[i],
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = -operators.eigenvectors_to_tde_slip[i]

        # Insert eigenmode to TDE slip operator into QP constraint data vector for lower bounds
        qp_constraint_matrix[
            index.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.n_tde[i] : index.qp_constraint_tde_rate_end_row_eigen[i],
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = operators.eigenvectors_to_tde_slip[i]

    return qp_constraint_matrix, qp_constraint_data_vector


def get_qp_all_inequality_operator_and_data_vector(
    index, meshes, operators, segment, block
):
    # Create arrays and data vector of correct size for linear inequality constraints
    # Stack TDE slip rate bounds on top of slip rate bounds
    #   TDE slip rate bounds
    #   slip rate bounds

    # Get QP TDE bounds
    qp_tde_inequality_matrix, qp_tde_inequality_data_vector = (
        get_qp_tde_inequality_operator_and_data_vector(index, meshes, operators)
    )

    # Get QP slip rate bounds
    qp_slip_rate_inequality_matrix, qp_slip_rate_inequality_data_vector = (
        get_qp_slip_rate_inequality_operator_and_data_vector(index, segment, block)
    )

    # NOTE: This effectively doubles the memory requirements for the problem.
    # I could try creating qp_tde_inequality_matrix as sparse and casting
    # and to full only at the very end
    qp_inequality_constraints_matrix = np.vstack(
        (qp_tde_inequality_matrix, qp_slip_rate_inequality_matrix)
    )

    # Build data vector for QP inequality constraints
    qp_inequality_constraints_data_vector = np.hstack(
        (
            qp_tde_inequality_data_vector,
            qp_slip_rate_inequality_data_vector,
        )
    )

    return qp_inequality_constraints_matrix, qp_inequality_constraints_data_vector


def get_qp_slip_rate_inequality_operator_and_data_vector(index, segment, block):
    # Get slip rate bounds vectors and operators
    slip_rate_bound_min, slip_rate_bound_max, slip_rate_bound_partials = (
        get_slip_rate_bounds(segment, block)
    )

    # Combine minimum and maximum slip rate bound operator
    slip_rate_bound_matrix = np.zeros(
        (2 * index.n_slip_rate_bounds, index.n_operator_cols_eigen)
    )
    slip_rate_bound_matrix[0 : index.n_slip_rate_bounds, 0 : 3 * index.n_blocks] = (
        slip_rate_bound_partials
    )
    slip_rate_bound_matrix[
        index.n_slip_rate_bounds : 2 * index.n_slip_rate_bounds, 0 : 3 * index.n_blocks
    ] = -slip_rate_bound_partials

    slip_rate_bound_data_vector = np.hstack((slip_rate_bound_max, -slip_rate_bound_min))

    return slip_rate_bound_matrix, slip_rate_bound_data_vector


def get_elastic_operator_single_mesh(
    meshes: list, station: pd.DataFrame, command: dict, mesh_index: np.int_
):
    """Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces.

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        meshes (List): Geometries of meshes
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        command (Dict): All command data
    """
    if bool(command.reuse_elastic) and os.path.exists(command.reuse_elastic_file):
        logger.info("Using precomputed elastic operators")
        hdf5_file = h5py.File(command.reuse_elastic_file, "r")
        tde_to_velocities = np.array(
            hdf5_file.get("tde_to_velocities_" + str(mesh_index))
        )
        hdf5_file.close()

    else:
        if not os.path.exists(command.reuse_elastic_file):
            logger.warning("Precomputed elastic operator file not found")
        logger.info("Computing elastic operators")
        logger.info(
            f"Start: TDE slip to velocity calculation for mesh: {meshes[mesh_index].file_name}"
        )
        tde_to_velocities = get_tde_to_velocities_single_mesh(
            meshes, station, command, mesh_idx=mesh_index
        )
        logger.success(
            f"Finish: TDE slip to velocity calculation for mesh: {meshes[mesh_index].file_name}"
        )

    # Save tde to velocity matrix for current mesh
    if bool(command.save_elastic):
        # Check to see if "data/operators" folder exists and if not create it
        if not os.path.exists(command.operators_folder):
            os.mkdir(command.operators_folder)

        logger.info(
            "Saving elastic to velocity matrices to :" + command.save_elastic_file
        )

        # Check if file exists.  If it does append.
        if os.path.exists(command.save_elastic_file):
            hdf5_file = h5py.File(command.save_elastic_file, "a")
            current_mesh_label = "tde_to_velocities_" + str(mesh_index)
            if current_mesh_label in hdf5_file:
                hdf5_file[current_mesh_label][...] = tde_to_velocities
            else:
                hdf5_file.create_dataset(current_mesh_label, data=tde_to_velocities)
        else:
            hdf5_file = h5py.File(command.save_elastic_file, "w")
            hdf5_file.create_dataset(
                "tde_to_velocities_" + str(mesh_index), data=tde_to_velocities
            )
        hdf5_file.close()
    return tde_to_velocities


def get_h_matrices_for_tde_meshes(
    command, meshes, station, operators, index, col_norms
):
    # TODO should this be in hmatrix.py?
    # TODO fix late import
    from celeri.hmatrix import build_hmatrix_from_mesh_tdes

    # Create lists for all TDE matrices per mesh
    H = []
    for i in range(len(meshes)):
        # Get full TDE to velocity matrix for current mesh
        tde_to_velocities = get_elastic_operator_single_mesh(
            meshes, station, command, i
        )

        # H-matrix representation
        H.append(
            build_hmatrix_from_mesh_tdes(
                meshes[i],
                station,
                -tde_to_velocities,
                command.h_matrix_tol,
                command.h_matrix_min_separation,
                command.h_matrix_min_pts_per_box,
            )
        )

        logger.info(
            f"mesh {i} ({meshes[i].file_name}) H-matrix compression ratio: {H[i].report_compression_ratio():0.4f}"
        )

        # Case smoothing matrices and tde slip rate constraints to sparse
        smoothing_keep_index = get_keep_index_12(operators.smoothing_matrix[i].shape[0])
        operators.smoothing_matrix[i] = csr_matrix(
            operators.smoothing_matrix[i][smoothing_keep_index, :][
                :, smoothing_keep_index
            ]
        )
        operators.tde_slip_rate_constraints[i] = csr_matrix(
            operators.tde_slip_rate_constraints[i]
        )

        # Eliminate unused columns and rows of TDE to velocity matrix
        tde_to_velocities = np.delete(
            tde_to_velocities, np.arange(2, tde_to_velocities.shape[0], 3), axis=0
        )
        tde_to_velocities = np.delete(
            tde_to_velocities, np.arange(2, tde_to_velocities.shape[1], 3), axis=1
        )

        # Calculate column normalization vector current TDE mesh
        weighting_vector_no_zero_rows = get_weighting_vector_single_mesh_for_col_norms(
            command, station, meshes, index, i
        )
        current_tde_mesh_columns_full_no_zero_rows = np.vstack(
            (
                -tde_to_velocities,
                operators.smoothing_matrix[i].toarray(),
                operators.tde_slip_rate_constraints[i].toarray(),
            )
        ) * np.sqrt(weighting_vector_no_zero_rows[:, None])

        # Concatenate everthing we need for col_norms
        col_norms_current_tde_mesh = np.linalg.norm(
            current_tde_mesh_columns_full_no_zero_rows, axis=0
        )
        col_norms = np.hstack((col_norms, col_norms_current_tde_mesh))

        # Free memory.  We have the Hmatrix version of this.
        del tde_to_velocities
    return H, col_norms


def get_eigenvalues_and_eigenvectors(n_eigenvalues, x, y, z, distance_exponent):
    n_tde = x.size

    # Calculate Cartesian distances between triangle centroids
    centroid_coordinates = np.array([x, y, z]).T
    distance_matrix = scipy.spatial.distance.cdist(
        centroid_coordinates, centroid_coordinates, "euclidean"
    )

    # Rescale distance matrix to the range 0-1
    distance_matrix = (distance_matrix - np.min(distance_matrix)) / np.ptp(
        distance_matrix
    )

    # Calculate correlation matrix
    correlation_matrix = np.exp(-(distance_matrix**distance_exponent))

    # https://stackoverflow.com/questions/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        correlation_matrix,
        subset_by_index=[n_tde - n_eigenvalues, n_tde - 1],
    )
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    eigenvectors[np.abs(eigenvectors) < 1e-6] = 0.0
    ordered_index = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[ordered_index]
    eigenvectors = eigenvectors[:, ordered_index]
    return eigenvalues, eigenvectors


def get_eigenvectors_to_tde_slip(operators, meshes):
    for i in range(len(meshes)):
        logger.info(f"Start: Eigenvectors to TDE slip for mesh: {meshes[i].file_name}")
        # Get eigenvectors for curren mesh
        _, eigenvectors = get_eigenvalues_and_eigenvectors(
            meshes[i].n_modes,
            meshes[i].x_centroid,
            meshes[i].y_centroid,
            meshes[i].z_centroid,
            distance_exponent=1.0,  # Make this something set in mesh_parameters.json
        )

        # Create eigenvectors to TDE slip matrix
        operators.eigenvectors_to_tde_slip[i] = np.zeros(
            (
                2 * eigenvectors.shape[0],
                meshes[i].config.n_modes_strike_slip
                + meshes[i].config.n_modes_dip_slip,
            )
        )

        # Place strike-slip panel
        operators.eigenvectors_to_tde_slip[i][
            0::2, 0 : meshes[i].config.n_modes_strike_slip
        ] = eigenvectors[:, 0 : meshes[i].config.n_modes_strike_slip]

        # Place dip-slip panel
        operators.eigenvectors_to_tde_slip[i][
            1::2,
            meshes[i].config.n_modes_strike_slip : meshes[i].config.n_modes_strike_slip
            + meshes[i].config.n_modes_dip_slip,
        ] = eigenvectors[:, 0 : meshes[i].config.n_modes_dip_slip]
        logger.success(
            f"Finish: Eigenvectors to TDE slip for mesh: {meshes[i].file_name}"
        )


def rotation_vectors_to_euler_poles(
    rotation_vector_x, rotation_vector_y, rotation_vector_z
):
    def xyz_to_lon_lat(x, y, z):
        # TODO: Should I use proj and proper ellipsoid here?
        lon = np.arctan2(y, x)
        lat = np.arcsin(z)
        return lon, lat

    n_poles = len(rotation_vector_x)

    # Initialize arrays
    euler_lon = np.zeros(n_poles)
    euler_lat = np.zeros(n_poles)
    euler_rate = np.zeros(n_poles)

    # Loop over each pole
    for i in range(n_poles):
        euler_rate[i] = np.sqrt(
            rotation_vector_x[i] ** 2.0
            + rotation_vector_y[i] ** 2.0
            + rotation_vector_z[i] ** 2.0
        )
        unit_vec = (
            np.array([rotation_vector_x[i], rotation_vector_y[i], rotation_vector_z[i]])
            / euler_rate[i]
        )
        tlon, tlat = xyz_to_lon_lat(unit_vec[0], unit_vec[1], unit_vec[2])
        euler_lon[i] = tlon
        euler_lat[i] = tlat

    # Convert longitude and latitude from radians to degrees
    euler_lon = np.rad2deg(euler_lon)
    euler_lat = np.rad2deg(euler_lat)

    # Make sure we have west longitude
    euler_lon = np.where(euler_lon < 0, euler_lon + 360, euler_lon)

    # Convert the rotation rate from rad/yr to degrees per million years
    SCALE_TO_DEG_PER_MILLION_YEARS = 1e3  # TODO: Check this
    euler_rate = SCALE_TO_DEG_PER_MILLION_YEARS * np.rad2deg(euler_rate)

    return euler_lon, euler_lat, euler_rate


def rotation_vector_err_to_euler_pole_err(omega_x, omega_y, omega_z, omega_cov):
    # Linearized propagatin of rotation vector uncertainties to Euler pole uncertainties

    # Declare variables
    n_poles = len(omega_x)
    A = np.zeros((3 * n_poles, 3 * n_poles))

    # Loop over each set of estimates
    for i in range(n_poles):
        idx = 3 * i
        x = omega_x[i]
        y = omega_y[i]
        z = omega_z[i]

        # Calculate the partial derivatives
        dlat_dx = -z / (x**2 + y**2) ** (3 / 2) / (1 + z**2 / (x**2 + y**2)) * x
        dlat_dy = -z / (x**2 + y**2) ** (3 / 2) / (1 + z**2 / (x**2 + y**2)) * y
        dlat_dz = 1 / (x**2 + y**2) ** (1 / 2) / (1 + z**2 / (x**2 + y**2))
        dlon_dx = -y / x**2 / (1 + (y / x) ** 2)
        dlon_dy = 1 / x / (1 + (y / x) ** 2)
        dlon_dz = 0
        dmag_dx = x / np.sqrt(x**2 + y**2 + z**2)
        dmag_dy = y / np.sqrt(x**2 + y**2 + z**2)
        dmag_dz = z / np.sqrt(x**2 + y**2 + z**2)

        # Organize them into a matrix
        A_small = np.array(
            [
                [dlat_dx, dlat_dy, dlat_dz],
                [dlon_dx, dlon_dy, dlon_dz],
                [dmag_dx, dmag_dy, dmag_dz],
            ]
        )
        # Put the small set of partials into the big set
        A[idx : idx + 3, idx : idx + 3] = A_small

    # Propagate the uncertainties and the new covariance matrix
    euler_cov = A @ omega_cov @ A.T

    # Organize data for the return
    diag_vec = np.diag(euler_cov)
    euler_lat_err = np.sqrt(diag_vec[0::3])
    euler_lon_err = np.sqrt(diag_vec[1::3])
    euler_rate_err = np.sqrt(diag_vec[2::3])

    # Convert longitude and latitude from radians to degrees
    euler_lon_err = np.rad2deg(euler_lon_err)
    euler_lat_err = np.rad2deg(euler_lat_err)

    # Convert the rotation rate from rad/yr to degrees per million years
    SCALE_TO_DEG_PER_MILLION_YEARS = 1e3  # TODO: Check this
    euler_rate_err = SCALE_TO_DEG_PER_MILLION_YEARS * np.rad2deg(euler_rate_err)

    return euler_lon_err, euler_lat_err, euler_rate_err


def get_index(assembly, station, block, meshes, mogi):
    # Create dictionary to store indices and sizes for operator building
    index = addict.Dict()
    index.n_stations = assembly.data.n_stations
    # index.n_stations = len(station)
    index.vertical_velocities = np.arange(2, 3 * index.n_stations, 3)
    index.n_blocks = len(block)
    index.n_block_constraints = assembly.data.n_block_constraints
    index.station_row_keep_index = get_keep_index_12(3 * len(station))
    index.start_station_row = 0
    index.end_station_row = 2 * len(station)
    index.start_block_col = 0
    index.end_block_col = 3 * len(block)
    index.start_block_constraints_row = index.end_station_row
    index.end_block_constraints_row = (
        index.start_block_constraints_row + 3 * index.n_block_constraints
    )
    index.n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    index.start_slip_rate_constraints_row = index.end_block_constraints_row
    index.end_slip_rate_constraints_row = (
        index.start_slip_rate_constraints_row + index.n_slip_rate_constraints
    )
    index.n_tde_total = 0
    index.n_tde_constraints_total = 0
    # A bunch of declarations of TDE-related indices as arrays, otherwise they're dicts
    index.n_tde = np.zeros((len(meshes),), dtype=int)
    index.n_tde_constraints = np.zeros((len(meshes),), dtype=int)
    index.start_tde_col = np.zeros((len(meshes),), dtype=int)
    index.end_tde_col = np.zeros((len(meshes),), dtype=int)
    index.start_tde_smoothing_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_smoothing_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_top_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_top_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_bot_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_bot_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_side_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_side_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_coup_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_coup_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_ss_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_ss_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_ds_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_ds_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_col_eigen = np.zeros((len(meshes),), dtype=int)
    index.end_tde_col_eigen = np.zeros((len(meshes),), dtype=int)
    index.start_tde_constraint_row_eigen = np.zeros((len(meshes),), dtype=int)
    index.end_tde_constraint_row_eigen = np.zeros((len(meshes),), dtype=int)

    for i in range(len(meshes)):
        index.n_tde[i] = meshes[i].n_tde
        index.n_tde_total += index.n_tde[i]
        index.n_tde_constraints[i] = meshes[i].n_tde_constraints
        index.n_tde_constraints_total += index.n_tde_constraints[i]

        # Set column indices for current mesh
        index.start_tde_col[i] = (
            index.end_block_col if i == 0 else index.end_tde_col[i - 1]
        )
        index.end_tde_col[i] = index.start_tde_col[i] + 2 * index.n_tde[i]

        # Set smoothing row indices for current mesh
        start_row = (
            index.end_slip_rate_constraints_row
            if i == 0
            else index.end_tde_constraint_row[i - 1]
        )
        index.start_tde_smoothing_row[i] = start_row
        index.end_tde_smoothing_row[i] = (
            index.start_tde_smoothing_row[i] + 2 * index.n_tde[i]
        )

        # Set constraint row indices for current mesh
        index.start_tde_constraint_row[i] = index.end_tde_smoothing_row[i]
        index.end_tde_constraint_row[i] = (
            index.start_tde_constraint_row[i] + index.n_tde_constraints[i]
        )

        # Set top constraint row indices and adjust count based on available data
        index.start_tde_top_constraint_row[i] = index.end_tde_smoothing_row[i]
        count = len(idx) if (idx := meshes[i].top_slip_idx) is not None else 0
        index.end_tde_top_constraint_row[i] = (
            index.start_tde_top_constraint_row[i] + count
        )

        # Set bottom constraint row indices
        index.start_tde_bot_constraint_row[i] = index.end_tde_top_constraint_row[i]
        count = len(idx) if (idx := meshes[i].bot_slip_idx) is not None else 0
        index.end_tde_bot_constraint_row[i] = (
            index.start_tde_bot_constraint_row[i] + count
        )

        # Set side constraint row indices
        index.start_tde_side_constraint_row[i] = index.end_tde_bot_constraint_row[i]
        count = len(idx) if (idx := meshes[i].side_slip_idx) is not None else 0
        index.end_tde_side_constraint_row[i] = (
            index.start_tde_side_constraint_row[i] + count
        )

        # Set coupling constraint row indices
        index.start_tde_coup_constraint_row[i] = index.end_tde_side_constraint_row[i]
        count = len(idx) if (idx := meshes[i].coup_idx) is not None else 0
        index.end_tde_coup_constraint_row[i] = (
            index.start_tde_coup_constraint_row[i] + count
        )

        # Set strike-slip constraint row indices
        index.start_tde_ss_slip_constraint_row[i] = index.end_tde_coup_constraint_row[i]
        count = len(idx) if (idx := meshes[i].ss_slip_idx) is not None else 0
        index.end_tde_ss_slip_constraint_row[i] = (
            index.start_tde_ss_slip_constraint_row[i] + count
        )

        # Set dip-slip constraint row indices
        index.start_tde_ds_slip_constraint_row[i] = (
            index.end_tde_ss_slip_constraint_row[i]
        )
        count = len(idx) if (idx := meshes[i].ds_slip_idx) is not None else 0
        index.end_tde_ds_slip_constraint_row[i] = (
            index.start_tde_ds_slip_constraint_row[i] + count
        )

        # # Add eigen specific entries to index
        # index.start_tde_col_eigen = np.zeros(len(meshes), dtype=int)
        # index.end_tde_col_eigen = np.zeros(len(meshes), dtype=int)
        # index.start_tde_constraint_row_eigen = np.zeros(len(meshes), dtype=int)
        # index.end_tde_constraint_row_eigen = np.zeros(len(meshes), dtype=int)

        # # TODO: Double-check this:
        # # Eigenvalue indexing should be unchanged by the above specification
        # # of indices for individual constraint styles, because n_tde_constraints
        # # is tracked correctly as the total number of constraint rows (2 each for
        # # coupling constraints, one each for slip component constraints)
        # if len(meshes) > 0:
        #     for i in range(len(meshes)):
        #         if i == 0:
        #             index.start_tde_col_eigen[i] = 3 * len(block)
        #             index.end_tde_col_eigen[i] = (
        #                 index.start_tde_col_eigen[i] + 2 * meshes[i].n_eigen
        #             )
        #             if meshes[i].n_tde_constraints > 0:
        #                 index.start_tde_constraint_row_eigen[i] = (
        #                     index.end_slip_rate_constraints_row
        #                 )
        #                 index.end_tde_constraint_row_eigen[i] = (
        #                     index.start_tde_constraint_row_eigen[i]
        #                     + meshes[i].n_tde_constraints
        #                 )
        #             else:
        #                 index.start_tde_constraint_row_eigen[i] = 0
        #                 index.end_tde_constraint_row_eigen[i] = 0
        #         else:
        #             index.start_tde_col_eigen[i] = index.end_tde_col_eigen[i - 1]
        #             index.end_tde_col_eigen[i] = (
        #                 index.start_tde_col_eigen[i] + 2 * meshes[i].n_eigen
        #             )
        #             if meshes[i].n_tde_constraints > 0:
        #                 index.start_tde_constraint_row_eigen[i] = (
        #                     index.end_tde_constraint_row_eigen[i - 1]
        #                 )
        #                 index.end_tde_constraint_row_eigen[i] = (
        #                     index.start_tde_constraint_row_eigen[i]
        #                     + meshes[i].n_tde_constraints
        #                 )
        #             else:
        #                 index.start_tde_constraint_row_eigen[i] = (
        #                     index.end_tde_constraint_row_eigen[i - 1]
        #                 )
        #                 index.end_tde_constraint_row_eigen[i] = (
        #                     index.end_tde_constraint_row_eigen[i - 1]
        #                 )
    # Index for block strain
    index.n_block_strain_components = 3 * np.sum(block.strain_rate_flag)
    index.start_block_strain_col = index.end_tde_col[-1]
    index.end_block_strain_col = (
        index.start_block_strain_col + index.n_block_strain_components
    )
    # Index for Mogi sources
    index.n_mogi = len(mogi)
    index.start_mogi_col = index.end_block_strain_col
    index.end_mogi_col = index.start_mogi_col + index.n_mogi

    index.n_operator_rows = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    index.n_operator_cols = 3 * index.n_blocks + 2 * index.n_tde_total
    return index


def get_index_no_meshes(assembly, station, block):
    # NOTE: Merge with above if possible.
    # Make sure empty meshes work

    # Create dictionary to store indices and sizes for operator building
    index = addict.Dict()
    index.n_stations = assembly.data.n_stations
    # index.n_stations = len(station)
    index.vertical_velocities = np.arange(2, 3 * index.n_stations, 3)
    index.n_blocks = len(block)
    index.n_block_constraints = assembly.data.n_block_constraints
    index.station_row_keep_index = get_keep_index_12(3 * len(station))
    index.start_station_row = 0
    index.end_station_row = 2 * len(station)
    index.start_block_col = 0
    index.end_block_col = 3 * len(block)
    index.start_block_constraints_row = index.end_station_row
    index.end_block_constraints_row = (
        index.start_block_constraints_row + 3 * index.n_block_constraints
    )
    index.n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    index.start_slip_rate_constraints_row = index.end_block_constraints_row
    index.end_slip_rate_constraints_row = (
        index.start_slip_rate_constraints_row + index.n_slip_rate_constraints
    )

    index.n_tde_total = 0
    index.n_tde_constraints_total = 0
    index.n_operator_rows = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )
    index.n_operator_cols = 3 * index.n_blocks
    return index


def get_index_eigen(assembly, segment, station, block, meshes, mogi):
    # Create dictionary to store indices and sizes for operator building
    index = addict.Dict()
    index.n_blocks = len(block)
    index.n_segments = len(segment)
    index.n_stations = assembly.data.n_stations
    index.n_meshes = len(meshes)
    index.n_mogis = len(mogi)
    index.vertical_velocities = np.arange(2, 3 * index.n_stations, 3)
    index.n_block_constraints = assembly.data.n_block_constraints
    index.station_row_keep_index = get_keep_index_12(3 * index.n_stations)
    index.start_station_row = 0
    index.end_station_row = 2 * index.n_stations
    index.start_block_col = 0
    index.end_block_col = 3 * index.n_blocks
    index.start_block_constraints_row = index.end_station_row
    index.end_block_constraints_row = (
        index.start_block_constraints_row + 3 * index.n_block_constraints
    )
    index.n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    index.start_slip_rate_constraints_row = index.end_block_constraints_row
    index.end_slip_rate_constraints_row = (
        index.start_slip_rate_constraints_row + index.n_slip_rate_constraints
    )
    index.n_tde_total = 0
    index.n_tde_constraints_total = 0

    # A bunch of declarations of TDE-related indices as arrays, otherwise they're dicts
    index.n_tde = np.zeros(index.n_meshes, dtype=int)
    index.n_tde_constraints = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_col = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_col = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_smoothing_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_smoothing_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_top_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_top_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_bot_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_bot_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_side_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_side_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_coup_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_coup_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ss_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ss_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ds_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ds_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)

    for i in range(len(meshes)):
        index.n_tde[i] = meshes[i].n_tde
        index.n_tde_total += index.n_tde[i]
        index.n_tde_constraints[i] = meshes[i].n_tde_constraints
        index.n_tde_constraints_total += index.n_tde_constraints[i]

        # Set column indices for current mesh
        index.start_tde_col[i] = (
            index.end_block_col if i == 0 else index.end_tde_col[i - 1]
        )
        index.end_tde_col[i] = index.start_tde_col[i] + 2 * index.n_tde[i]

        # Set smoothing row indices for current mesh
        start_row = (
            index.end_slip_rate_constraints_row
            if i == 0
            else index.end_tde_constraint_row[i - 1]
        )
        index.start_tde_smoothing_row[i] = start_row
        index.end_tde_smoothing_row[i] = (
            index.start_tde_smoothing_row[i] + 2 * index.n_tde[i]
        )

        # Set constraint row indices for current mesh
        index.start_tde_constraint_row[i] = index.end_tde_smoothing_row[i]
        index.end_tde_constraint_row[i] = (
            index.start_tde_constraint_row[i] + index.n_tde_constraints[i]
        )

        # Set top constraint row indices and adjust count based on available data
        index.start_tde_top_constraint_row[i] = index.end_tde_smoothing_row[i]
        count = len(idx) if (idx := meshes[i].top_slip_idx) is not None else 0
        index.end_tde_top_constraint_row[i] = (
            index.start_tde_top_constraint_row[i] + count
        )

        # Set bottom constraint row indices
        index.start_tde_bot_constraint_row[i] = index.end_tde_top_constraint_row[i]
        count = len(idx) if (idx := meshes[i].bot_slip_idx) is not None else 0
        index.end_tde_bot_constraint_row[i] = (
            index.start_tde_bot_constraint_row[i] + count
        )

        # Set side constraint row indices
        index.start_tde_side_constraint_row[i] = index.end_tde_bot_constraint_row[i]
        count = len(idx) if (idx := meshes[i].side_slip_idx) is not None else 0
        index.end_tde_side_constraint_row[i] = (
            index.start_tde_side_constraint_row[i] + count
        )

        # Set coupling constraint row indices
        index.start_tde_coup_constraint_row[i] = index.end_tde_side_constraint_row[i]
        count = len(idx) if (idx := meshes[i].coup_idx) is not None else 0
        index.end_tde_coup_constraint_row[i] = (
            index.start_tde_coup_constraint_row[i] + count
        )

        # Set strike-slip constraint row indices
        index.start_tde_ss_slip_constraint_row[i] = index.end_tde_coup_constraint_row[i]
        count = len(idx) if (idx := meshes[i].ss_slip_idx) is not None else 0
        index.end_tde_ss_slip_constraint_row[i] = (
            index.start_tde_ss_slip_constraint_row[i] + count
        )

        # Set dip-slip constraint row indices
        index.start_tde_ds_slip_constraint_row[i] = (
            index.end_tde_ss_slip_constraint_row[i]
        )
        count = len(idx) if (idx := meshes[i].ds_slip_idx) is not None else 0
        index.end_tde_ds_slip_constraint_row[i] = (
            index.start_tde_ds_slip_constraint_row[i] + count
        )

    # EIGEN: Get total number of eigenmodes
    index.n_eigen_total = 0
    for i in range(index.n_meshes):
        index.n_eigen_total += meshes[i].config.n_modes_strike_slip
        index.n_eigen_total += meshes[i].config.n_modes_dip_slip

    # EIGEN: Count eigenmodes for each mesh
    index.n_modes_mesh = np.zeros(index.n_meshes, dtype=int)
    for i in range(index.n_meshes):
        index.n_modes_mesh[i] = (
            meshes[i].config.n_modes_strike_slip + meshes[i].config.n_modes_dip_slip
        )

    # EIGEN: Column and row indices
    index.start_col_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_col_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ss_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ss_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ds_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ds_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)

    # EIGEN: columns and rows for eigenmodes to velocity
    for i in range(index.n_meshes):
        # First mesh
        if i == 0:
            # Locations for eigenmodes to velocities
            index.start_col_eigen[i] = 3 * index.n_blocks
            index.end_col_eigen[i] = index.start_col_eigen[i] + index.n_modes_mesh[i]
            index.start_tde_row_eigen[i] = 0
            index.end_row_eigen[i] = 2 * index.n_stations

        # Meshes after first mesh
        else:
            # Locations for eigenmodes to velocities
            index.start_col_eigen[i] = index.end_col_eigen[i - 1]
            index.end_col_eigen[i] = index.start_col_eigen[i] + index.n_modes_mesh[i]
            index.start_tde_row_eigen[i] = 0
            index.end_row_eigen[i] = 2 * index.n_stations

    # EIGEN: Set initial values to follow segment slip rate constraints
    index.start_tde_constraint_row_eigen[0] = index.end_slip_rate_constraints_row

    index.end_tde_constraint_row_eigen[0] = (
        index.start_tde_constraint_row_eigen[0] + index.n_tde_constraints[0]
    )

    # EIGEN: Rows for eigen to TDE boundary conditions
    for i in range(1, index.n_meshes):
        # All constraints for eigen to constraint matrix
        index.start_tde_constraint_row_eigen[i] = index.end_tde_constraint_row_eigen[
            i - 1
        ]
        index.end_tde_constraint_row_eigen[i] = (
            index.start_tde_constraint_row_eigen[i] + index.n_tde_constraints[i]
        )

    # EIGEN: Rows for QP bounds
    # Create index components for linear inequality matrix and data vector
    index.qp_constraint_tde_rate_start_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.qp_constraint_slip_rate_end_row_eigen = np.zeros(index.n_meshes, dtype=int)

    # Create index components for slip rate constraints
    index.qp_constraint_slip_rate_start_row_eigen = np.zeros(index.n_meshes, dtype=int)

    index.qp_constraint_tde_rate_start_row_eigen[0] = 0
    index.qp_constraint_tde_rate_end_row_eigen[0] = (
        index.qp_constraint_tde_rate_start_row_eigen[0] + 4 * index.n_tde[0]
    )

    for i in range(1, index.n_meshes):
        # Start row for current mesh
        index.qp_constraint_tde_rate_start_row_eigen[i] = (
            index.qp_constraint_tde_rate_end_row_eigen[i - 1]
        )

        # End row for current mesh
        index.qp_constraint_tde_rate_end_row_eigen[i] = (
            index.qp_constraint_tde_rate_start_row_eigen[i] + 4 * index.n_tde[i]
        )

    # Index for block strain
    index.n_strain_blocks = np.sum(block.strain_rate_flag)
    index.n_block_strain_components = 3 * index.n_strain_blocks
    index.start_block_strain_col = index.end_tde_col[-1]
    index.end_block_strain_col = (
        index.start_block_strain_col + index.n_block_strain_components
    )

    # EIGEN: Index for block strain
    index.start_block_strain_col_eigen = index.end_col_eigen[-1]
    index.end_block_strain_col_eigen = (
        index.start_block_strain_col_eigen + index.n_block_strain_components
    )

    # Index for Mogi sources
    index.start_mogi_col = index.end_block_strain_col
    index.end_mogi_col = index.start_mogi_col + index.n_mogis

    # EIGEN: Index for Mogi sources
    index.start_mogi_col_eigen = index.end_block_strain_col_eigen
    index.end_mogi_col_eigen = index.start_mogi_col_eigen + index.n_mogis

    index.n_operator_rows = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    index.n_operator_cols = (
        3 * index.n_blocks
        + 2 * index.n_tde_total
        + 3 * index.n_strain_blocks
        + index.n_mogis
    )

    # Total number of columns for eigenmode problem
    index.n_operator_cols_eigen = (
        3 * index.n_blocks
        + index.n_eigen_total
        + 3 * index.n_strain_blocks
        + index.n_mogis
    )

    # Total number of rows for eigenmode problem
    index.n_operator_rows_eigen = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    # Indices for QP hard slip rate bounds
    index.slip_rate_bounds = np.where(
        interleave3(
            segment.ss_rate_bound_flag,
            segment.ds_rate_bound_flag,
            segment.ts_rate_bound_flag,
        )
        == 1
    )[0]
    index.n_slip_rate_bounds = len(index.slip_rate_bounds)

    return index
