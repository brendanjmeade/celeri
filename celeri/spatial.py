import warnings

import cutde.halfspace as cutde_halfspace
import numpy as np
import pandas as pd
import scipy
from rich.progress import track
from scipy.sparse import csr_matrix

from celeri.celeri_util import (
    cartesian_vector_to_spherical_vector,
    get_cross_partials,
    get_segment_oblique_projection,
    get_transverse_projection,
    latitude_to_colatitude,
    sph2cart,
)
from celeri.constants import GEOID, KM2M, RADIUS_EARTH
from celeri.okada.cutde_okada import TriangulationTypes, dc3dwrapper_cutde_disp


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


def get_segment_station_operator_okada(segment, station, config, *, progress_bar=True):
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

    Args:
        segment: Segment data
        station: Station data
        config: Configuration object
        progress_bar: If True, show a progress bar

    """
    if station.empty:
        # TODO: Shouldn't this return an array of shape (0, 3 * n_segments)?
        # Currently this returns an array of shape (1,) with an uninitialized value,
        # so this seems wrong.
        return np.empty(1)
    n_segments = len(segment)
    n_stations = len(station)
    okada_segment_operator = np.full((3 * n_stations, 3 * n_segments), np.nan)
    # Loop through each segment and calculate displacements for each slip component
    segment_iter = (
        range(len(segment))
        if not progress_bar
        else track(range(len(segment)), description="Rectangular segment elastic")
    )
    for i in segment_iter:
        for slip_type in ["strike", "dip", "tensile"]:
            # Each `u` has shape (n_stations, )
            u_east, u_north, u_up = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                config.material_lambda,
                config.material_mu,
                1 if slip_type == "strike" else 0,
                1 if slip_type == "dip" else 0,
                1 if slip_type == "tensile" else 0,
                station.lon,
                station.lat,
            )
            if slip_type == "strike":
                col_idx = 3 * i
            elif slip_type == "dip":
                col_idx = 3 * i + 1
            elif slip_type == "tensile":
                col_idx = 3 * i + 2
            else:
                raise ValueError(f"Invalid slip type: {slip_type}")
            # The i::3 notation sets an offset of i and a skip of 3 so that
            # east/north/up displacements are interleaved.
            okada_segment_operator[0::3, col_idx] = np.squeeze(u_east)
            okada_segment_operator[1::3, col_idx] = np.squeeze(u_north)
            okada_segment_operator[2::3, col_idx] = np.squeeze(u_up)
    return okada_segment_operator


def get_okada_displacements(
    segment_lon1,
    segment_lat1,
    segment_lon2,
    segment_lat2,
    segment_locking_depth,
    segment_dip,
    segment_azimuth,
    material_lambda,
    material_mu,
    strike_slip,
    dip_slip,
    tensile_slip,
    station_lon,
    station_lat,
    triangulation: TriangulationTypes = "auto",
):
    """Caculate elastic displacements in a homogeneous elastic half-space.
    Inputs are in geographic coordinates and then projected into a local
    xy-plane using a oblique Mercator projection that is tangent and parallel
    to the trace of the fault segment.  The elastic calculation breaks each
    rectangle into triangles and uses T. Ben Thompson's cutde library for TDEs.
    """
    # TODO(Brendan): Previous version might have changed the value inplace?
    # If segment_locking_depth is a reference to a pandas Series.
    segment_locking_depth = segment_locking_depth * KM2M

    # Make sure depths are expressed as positive
    segment_locking_depth = np.abs(segment_locking_depth)

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
    segment_up_dip_width = segment_locking_depth / np.sin(np.deg2rad(segment_dip))

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
    station_xy_rotated = np.einsum(
        "ij,kj->ik",
        np.dstack((station_x, station_y))[0],
        rotation_matrix.T,
    )
    station_x_rotated, station_y_rotated = np.hsplit(station_xy_rotated, 2)

    # Shift station y coordinates by surface projection of locking depth
    # y_shift will be positive for dips <90 and negative for dips > 90
    y_shift = np.cos(np.deg2rad(segment_dip)) * segment_up_dip_width
    station_y_rotated += y_shift
    station_z_rotated = np.zeros_like(station_x_rotated)
    obs_pts = np.concat(
        [station_x_rotated, station_y_rotated, station_z_rotated], axis=1
    )

    # Elastic displacements from Okada 1992
    alpha = (material_lambda + material_mu) / (material_lambda + 2 * material_mu)

    u = dc3dwrapper_cutde_disp(
        alpha,  # (lambda + mu) / (lambda + 2 * mu)
        obs_pts,  # (meters) observation point
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
        triangulation=triangulation,
    )  # (meters) strike-slip, dip-slip, tensile-slip
    u_x, u_y, u_up = u.T

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


def get_tde_to_velocities(meshes, station, config):
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

            for i in track(range(n_tris), description="Meshed surface elastic     "):
                (
                    vel_east_strike_slip,
                    vel_north_strike_slip,
                    vel_up_strike_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    config.material_lambda,
                    config.material_mu,
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
                    config.material_lambda,
                    config.material_mu,
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
                    config.material_lambda,
                    config.material_mu,
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


def get_tde_to_velocities_single_mesh(meshes, station, config, mesh_idx):
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
            for i in track(range(n_tris), description="Meshed surface elastic     "):
                (
                    vel_east_strike_slip,
                    vel_north_strike_slip,
                    vel_up_strike_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    config.material_lambda,
                    config.material_mu,
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
                    config.material_lambda,
                    config.material_mu,
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
                    config.material_lambda,
                    config.material_mu,
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


def get_tri_smoothing_matrix_simple(share, n_dim) -> csr_matrix:
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
    return smoothing_matrix.tocsr()


def get_tri_smoothing_matrix(share, tri_shared_sides_distances) -> csr_matrix:
    """Produces a smoothing matrix based on the scale-dependent
    umbrella operator (e.g., Desbrun et al., 1999; Resor, 2004).

    Inputs:
    share: n x 3 array of indices of the up to 3 elements sharing a side
        with each of the n elements
    tri_shared_sides_distances: n x 3 array of distances between each of the
        n elements and its up to 3 neighbors (NaN where no neighbor)

    Outputs:
    smoothing matrix: 3n x 3n smoothing matrix
    """
    n = share.shape[0]  # number of triangles

    # Sum distances and compute 1/d (NaN propagates harmlessly for missing neighbors)
    leading_coefficient = 2.0 / np.nansum(tri_shared_sides_distances, axis=1)  # (n,)
    # (n, 3) 1/d per neighbor slot, NaN if missing
    inverse_distances = 1.0 / tri_shared_sides_distances

    # (n,) Diagonal terms: -(2/Σd_k) * Σ(1/d_j) over neighbors j
    diagonal_terms = -leading_coefficient * np.nansum(inverse_distances, axis=1)

    # (n, 3) Off-diagonal terms: (2/Σd_k) * (1/d_j) for each neighbor j
    off_diagonal_terms = leading_coefficient[:, np.newaxis] * inverse_distances

    # Build the n x n Laplacian for one slip component.
    has_neighbor = share != -1  # (n, 3) bool mask
    n_shares = int(np.sum(has_neighbor))  # number of shared neighbor slots
    neighbor_rows = np.nonzero(has_neighbor)[0]  # (n_shares,)
    neighbor_cols = share[has_neighbor]  # (n_shares,)
    neighbor_values = off_diagonal_terms[has_neighbor]  # (n_shares,)

    # Build COO indices/values by concatenating diagonals first, then off-diagonals.
    row_idx = np.empty(n + n_shares, dtype=int)  # (n + n_shares,)
    col_idx = np.empty(n + n_shares, dtype=int)  # (n + n_shares,)
    values = np.empty(n + n_shares, dtype=diagonal_terms.dtype)  # (n + n_shares,)
    row_idx[:n] = np.arange(n)
    col_idx[:n] = np.arange(n)
    values[:n] = diagonal_terms
    row_idx[n:] = neighbor_rows
    col_idx[n:] = neighbor_cols
    values[n:] = neighbor_values
    laplacian = scipy.sparse.coo_matrix(
        (values, (row_idx, col_idx)), shape=(n, n)
    ).tocsr()  # (n, n)

    # Expand to 3 components: kron(L, I_3) -> (3n, 3n)
    return scipy.sparse.kron(laplacian, scipy.sparse.eye(3), format="csr")


def get_mogi_to_velocities_partials(mogi, station, config) -> np.ndarray:
    """Mogi volume change to station displacement operator."""
    if mogi.empty:
        mogi_operator = np.zeros((3 * len(station), 0))
    else:
        poissons_ratio = config.material_mu / (
            2 * (config.material_lambda + config.material_mu)
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


def mogi_forward(
    mogi_lon, mogi_lat, mogi_depth, poissons_ratio, obs_lon, obs_lat
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    centroid_lon: np.ndarray,
    centroid_lat: np.ndarray,
    strain_rate_lon_lon: int,
    strain_rate_lat_lat: int,
    strain_rate_lon_lat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate displacements due to three block strain rate components.
    Equations are from Savage (2001) and expressed concisely in McCaffrey (2005):
    https://www.researchgate.net/publication/251436956_Block_kinematics_of_the_Pacific-North_America_plate_boundary_in_the_southwestern_US_from_inversion_of_GPS_seismological_and_geologic_data
    See the two unnumbered equations at the bottom of page 2:

    u_east = ε_λλ * R_E * (λ_obs - λ_c) * sin(φ_c) + ε_λφ * R_E * (φ_obs - φ_c)
    u_north = ε_λφ * R_E * (λ_obs - λ_c) * sin(φ_c) + ε_φφ * R_E * (φ_obs - φ_c)
    u_up = 0

    u_up is zero, since strain is assumed to be strain on the spherical plane.

    Args:
    lon_obs: Longitude coordinates at the stations
    lat_obs: Latitude coordinates at the stations
    centroid_lon: Longitude of the block centroid
    centroid_lat: Latitude of the block centroid
    strain_rate_lon_lon: Strain rate component ε_λλ
    strain_rate_lat_lat: Strain rate component ε_φφ
    strain_rate_lon_lat: Strain rate component ε_λφ

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: Eastward, northward, and upward velocities
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


def get_block_strain_rate_to_velocities_partials(
    blocks, stations, segment
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate strain partial derivatives w.r.t. the block strain rate components ε_λλ, ε_φφ, and ε_λφ, assuming
    a strain centroid on each block.

    Returns a tuple [block_strain_rate_operator, strain_rate_block_idx], where block_strain_rate_operator is a
    np.ndarray of shape (3 * number_of_stations, 3 * number_of_strain_blocks) and strain_rate_block_idx, with the
    indices of the blocks that have strain rates estimated.
    """
    strain_rate_block_idx = np.where(blocks.strain_rate_flag.to_numpy() > 0)[0]
    # Allocate space. Zero width, if no blocks should have strain estimated, helps with indexing
    block_strain_rate_operator = np.zeros(
        (3 * len(stations), 3 * strain_rate_block_idx.size)
    )
    if strain_rate_block_idx.size > 0:
        for i in range(strain_rate_block_idx.size):
            # Find centroid of current block
            block_centroid_lon, block_centroid_lat = get_block_centroid(
                segment, strain_rate_block_idx[i]
            )

            station_idx = np.where(stations.block_label == strain_rate_block_idx[i])[0]
            stations_block_lon = stations.lon[station_idx].to_numpy()
            stations_block_lat = stations.lat[station_idx].to_numpy()

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


def get_global_float_block_rotation_partials(station) -> np.ndarray:
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


def get_block_motion_constraint_partials(block) -> np.ndarray:
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


def get_block_centroid(
    segment: pd.DataFrame, block_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate centroid of a block based on boundary polygon
    We take all block vertices (including duplicates) and estimate
    the centroid by taking the average of longitude and latitude
    weighted by the length of the segment that each vertex is
    attached to.
    """
    segments_with_block_idx = np.union1d(
        np.where(segment.west_labels == block_idx)[0],
        np.where(segment.east_labels == block_idx)[0],
    )
    lon0 = np.concatenate(
        (segment.lon1[segments_with_block_idx], segment.lon2[segments_with_block_idx])
    )
    lat0 = np.concatenate(
        (segment.lat1[segments_with_block_idx], segment.lat2[segments_with_block_idx])
    )
    lengths = np.concatenate(
        (
            segment.length[segments_with_block_idx],
            segment.length[segments_with_block_idx],
        )
    )
    block_centroid_lon = np.average(lon0, weights=lengths)
    block_centroid_lat = np.average(lat0, weights=lengths)
    return np.array([block_centroid_lon]), np.array([block_centroid_lat])


def get_shared_sides(vertices):
    """Determine the indices of the triangular elements sharing
    one side with a particular element.
    Inputs:
    vertices: n x 3 array containing the 3 vertex indices of the n elements,
        assumes that values increase monotonically from 1:n.

    Outputs:
    share: n x 3 array containing the indices of the m elements sharing a
        side with each of the n elements.  "-1" values in the array
        indicate elements with fewer than m neighbors (i.e., on
        the edge of the geometry).

    In general, elements will have 1 (mesh corners), 2 (mesh edges), or 3
    (mesh interiors) neighbors, but in the case of branching faults that
    have been adjusted with mergepatches, it's for edges and corners to
    also up to 3 neighbors.
    """
    # Make side arrays containing vertex indices of sides
    side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
    side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
    side_3 = np.sort(np.vstack((vertices[:, 0], vertices[:, 2])).T, 1)
    sides_all = np.vstack((side_1, side_2, side_3))

    # Find the unique sides - each side can part of at most 2 elements
    _, first_occurence_idx = np.unique(sides_all, return_index=True, axis=0)
    _, last_occurence_idx = np.unique(np.flipud(sides_all), return_index=True, axis=0)
    last_occurence_idx = sides_all.shape[0] - last_occurence_idx - 1

    # Shared sides are those whose first and last indices are not equal
    shared = np.where((last_occurence_idx - first_occurence_idx) != 0)[0]

    # These are the indices of the shared sides
    sside1 = first_occurence_idx[shared]  # What should I name these variables?
    sside2 = last_occurence_idx[shared]

    el1, sh1 = np.unravel_index(
        sside1, vertices.shape, order="F"
    )  # "F" is for fortran ordering.  What should I call this variables?
    el2, sh2 = np.unravel_index(sside2, vertices.shape, order="F")
    share = -1 * np.ones((vertices.shape[0], 3))
    for i in range(el1.size):
        share[el1[i], sh1[i]] = el2[i]
        share[el2[i], sh2[i]] = el1[i]
    share = share.astype(int)
    return share


def get_tri_shared_sides_distances(
    share: np.ndarray,
    x_centroid: np.ndarray,
    y_centroid: np.ndarray,
    z_centroid: np.ndarray,
) -> np.ndarray:
    """Calculates the distances between the centroids of adjacent triangular
    elements, for use in smoothing algorithms.

    Args:
        share: (n, 3) array from get_shared_sides, containing neighbor indices
            or -1 for missing neighbors. Dimensions: [triangle, neighbor_slot].
        x_centroid: (n,) x coordinates of element centroids
        y_centroid: (n,) y coordinates of element centroids
        z_centroid: (n,) z coordinates of element centroids

    Returns:
        (n, 3) array of distances to each neighbor. Dimensions: [triangle, neighbor_slot].
        NaN where share == -1 (no neighbor).
    """
    has_neighbor = share != -1  # (n, 3) bool mask
    # n_shares = int(np.sum(has_neighbor))  # number of shared neighbor slots

    # Extract valid (triangle_idx, neighbor_idx) pairs only
    tri_idx = np.nonzero(has_neighbor)[0]  # (n_shares,) indices with neighbors
    neighbor_idx = share[has_neighbor]  # (n_shares,) indices of neighbors

    centroids = np.column_stack([x_centroid, y_centroid, z_centroid])  # (n, 3)

    # Compute distances only for valid pairs
    valid_distances = np.linalg.norm(
        centroids[tri_idx] - centroids[neighbor_idx], axis=1
    )  # (n_shares,)

    # Scatter into output array (NaN elsewhere = no neighbor)
    distances = np.full(share.shape, np.nan)  # (n, 3)
    distances[has_neighbor] = valid_distances

    return distances
