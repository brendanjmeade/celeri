"""Moment tensor computation for tectonic blocks on a sphere.

## Computational Approach

1. Project block polygon from sphere to plane using Lambert azimuthal equal-area projection
2. Triangulate the planar polygon using Shapely's constrained Delaunay triangulation
3. Project triangles back to sphere
4. Integrate r⊗r over spherical triangles using 7-point quadrature with Jacobian correction
5. Normalize by total area to get moment tensor M

## Moment Tensor and Root Mean Squared Velocity

The moment tensor M characterizes a block's spatial distribution: M = (1/A) ∫∫ r⊗r dA,
where r is the position vector on the unit sphere and A is the block area.

For a block rotating with angular velocity ω, the root mean squared velocity is:
    v_rms = R * sqrt(ω^T (I - M) ω)

where R is Earth's radius. The matrix (I - M) accounts for block geometry: compact blocks
have larger eigenvalues of (I - M), meaning the same rotation produces higher velocities.
This enables geometry-aware priors in Bayesian inference (e.g., MCMC sampling).
"""

import numpy as np
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon

from celeri.constants import RADIUS_EARTH as EARTH_RADIUS_KM

# Constants
EARTH_RADIUS_MM = EARTH_RADIUS_KM * 1e6

# Quadrature rules for planar triangles (7-point rule)
PLANAR_QUADRATURE_7PT = {
    "bary_coords": np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.797426958353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426958353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426958353087],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
        ]
    ),
    "weights": np.array(
        [
            0.225,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
        ]
    ),
}


def spherical_distance(v1, v2):
    """Compute great circle distance between two points on unit sphere.

    Parameters
    ----------
    v1, v2 : array (3,)
        Points on unit sphere

    Returns
    -------
    float
        Angular distance in radians
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1, v2), -1, 1)
    return np.arccos(cos_angle)


def spherical_area_triangle(v1, v2, v3):
    """Compute area of spherical triangle using solid angle formula.

    Parameters
    ----------
    v1, v2, v3 : array (..., 3)
        Triangle vertices. Can be single triangles (3,) or batched (..., 3)

    Returns
    -------
    area : float or array
        Spherical area(s)
    """
    # Normalize (works for both single and batched)
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    v3 = v3 / np.linalg.norm(v3, axis=-1, keepdims=True)

    # Cross product and dot products (vectorized)
    cross_v2_v3 = np.cross(v2, v3)
    det = np.sum(v1 * cross_v2_v3, axis=-1)

    dot_v1_v2 = np.sum(v1 * v2, axis=-1)
    dot_v2_v3 = np.sum(v2 * v3, axis=-1)
    dot_v3_v1 = np.sum(v3 * v1, axis=-1)

    denom = 1.0 + dot_v1_v2 + dot_v2_v3 + dot_v3_v1
    return np.abs(2.0 * np.arctan2(det, denom))


def lambert_project(vertices_xyz, center_xyz):
    """Project spherical points to plane using Lambert azimuthal equal-area projection."""
    # Normalize
    vertices_xyz = vertices_xyz / np.linalg.norm(vertices_xyz, axis=1, keepdims=True)
    center_xyz = center_xyz / np.linalg.norm(center_xyz)

    # Set up local coordinate system at center
    # z-axis points at center
    z_axis = center_xyz

    # x-axis perpendicular to z
    if abs(z_axis[2]) < 0.9:
        temp = np.array([0, 0, 1])
    else:
        temp = np.array([1, 0, 0])

    x_axis = np.cross(z_axis, temp)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    # Lambert azimuthal equal-area projection
    xy = []
    for v in vertices_xyz:
        # Angular distance from center
        cos_c = np.dot(v, center_xyz)
        c = np.arccos(np.clip(cos_c, -1, 1))

        if c < 1e-10:  # At center
            xy.append([0, 0])
            continue

        # Projection
        k = np.sqrt(2 / (1 + cos_c))

        # Project to local coordinates
        dx = k * np.dot(v, x_axis)
        dy = k * np.dot(v, y_axis)

        xy.append([dx, dy])

    return np.array(xy)


def lambert_inverse(xy, center_xyz):
    """Inverse Lambert projection from plane back to sphere.

    Parameters
    ----------
    xy : array (n, 2) or (2,)
        Planar coordinates
    center_xyz : array (3,)
        Center point of projection

    Returns
    -------
    xyz : array (n, 3) or (3,)
        Points on unit sphere
    """
    xy = np.atleast_2d(xy)
    center_xyz = center_xyz / np.linalg.norm(center_xyz)

    # Set up coordinate system
    z_axis = center_xyz

    if abs(z_axis[2]) < 0.9:
        temp = np.array([0, 0, 1])
    else:
        temp = np.array([1, 0, 0])

    x_axis = np.cross(z_axis, temp)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    # Inverse projection
    xyz = []
    for x, y in xy:
        rho = np.sqrt(x**2 + y**2)

        if rho < 1e-10:  # At center
            xyz.append(center_xyz)
            continue

        c = 2 * np.arcsin(rho / 2)

        # Direction in local coordinates
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        local_dir = np.array([sin_c * x / rho, sin_c * y / rho, cos_c])

        # Convert to global coordinates
        v = local_dir[0] * x_axis + local_dir[1] * y_axis + local_dir[2] * z_axis

        xyz.append(v / np.linalg.norm(v))

    result = np.array(xyz)
    return result[0] if len(xy) == 1 else result


def subdivide_spherical_triangle(v1, v2, v3, max_edge_length):
    """Recursively subdivide a spherical triangle if edges exceed max length.

    Parameters
    ----------
    v1, v2, v3 : array (3,)
        Triangle vertices on unit sphere
    max_edge_length : float
        Maximum edge length in radians. Triangles with any edge longer than
        this will be subdivided into 4 smaller triangles.

    Returns
    -------
    triangles : list of tuples
        List of (v1, v2, v3) tuples for triangles that satisfy the constraint
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # Compute edge lengths
    d12 = spherical_distance(v1, v2)
    d23 = spherical_distance(v2, v3)
    d31 = spherical_distance(v3, v1)

    max_edge = max(d12, d23, d31)

    # If all edges are within limit, return this triangle
    if max_edge <= max_edge_length:
        return [(v1, v2, v3)]

    # Otherwise, subdivide into 4 triangles
    # Compute midpoints on sphere (geodesic midpoints)
    m12 = v1 + v2
    m12 = m12 / np.linalg.norm(m12)

    m23 = v2 + v3
    m23 = m23 / np.linalg.norm(m23)

    m31 = v3 + v1
    m31 = m31 / np.linalg.norm(m31)

    # Create 4 sub-triangles and recursively subdivide each
    triangles = []
    for tri in [(v1, m12, m31), (m12, v2, m23), (m31, m23, v3), (m12, m23, m31)]:
        triangles.extend(
            subdivide_spherical_triangle(tri[0], tri[1], tri[2], max_edge_length)
        )

    return triangles


def integrate_spherical_triangle(v1, v2, v3, method="spherical"):
    """Integrate r⊗r over spherical triangles (vectorized).

    Parameters
    ----------
    v1, v2, v3 : array (n, 3)
        Triangle vertices on unit sphere for n triangles
    method : str, optional
        Integration method:
        - "spherical": Proper spherical integration with Jacobian correction
        - "planar": Legacy planar barycentric (incorrect but fast)
        - "centroid": Simple centroid approximation

    Returns
    -------
    M_triangles : array (n, 3, 3)
        Moment tensor contributions
    areas : array (n,)
        Spherical areas
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v3 = np.asarray(v3)

    n_triangles = v1.shape[0]

    # Normalize vertices
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    v3 = v3 / np.linalg.norm(v3, axis=-1, keepdims=True)

    # Compute spherical areas (vectorized)
    areas = spherical_area_triangle(v1, v2, v3)

    if method == "centroid":
        # Simple approximation: use centroid
        centroids = v1 + v2 + v3
        centroids = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
        # Outer product: (n, 3, 1) * (n, 1, 3) = (n, 3, 3)
        M_triangles = areas[:, np.newaxis, np.newaxis] * (
            centroids[:, :, np.newaxis] * centroids[:, np.newaxis, :]
        )

    elif method == "planar":
        # Legacy method: planar barycentric coordinates (not spherically correct)
        M_triangles = np.zeros((n_triangles, 3, 3))
        bary_coords = PLANAR_QUADRATURE_7PT["bary_coords"]
        weights = PLANAR_QUADRATURE_7PT["weights"]

        for w, (b1, b2, b3) in zip(weights, bary_coords, strict=True):
            # Vectorized barycentric interpolation
            p_euclidean = b1 * v1 + b2 * v2 + b3 * v3  # (n, 3)
            norms = np.linalg.norm(p_euclidean, axis=-1, keepdims=True)
            p = p_euclidean / norms  # (n, 3)

            # Vectorized outer product and accumulation
            M_triangles += (
                w
                * areas[:, np.newaxis, np.newaxis]
                * (p[:, :, np.newaxis] * p[:, np.newaxis, :])
            )

    elif method == "spherical":
        # Proper spherical integration with Jacobian correction
        M_triangles = np.zeros((n_triangles, 3, 3))
        bary_coords = PLANAR_QUADRATURE_7PT["bary_coords"]
        weights = PLANAR_QUADRATURE_7PT["weights"]

        # Precompute unnormalized partial derivatives
        dp_db1_unnorm = v1 - v3  # (n, 3)
        dp_db2_unnorm = v2 - v3  # (n, 3)

        # Accumulate contributions from each quadrature point
        total_weights = np.zeros(n_triangles)
        contributions = []

        for w, (b1, b2, b3) in zip(weights, bary_coords, strict=True):
            # Vectorized barycentric interpolation
            p_euclidean = b1 * v1 + b2 * v2 + b3 * v3  # (n, 3)
            r = np.linalg.norm(p_euclidean, axis=-1, keepdims=True)  # (n, 1)
            p_sphere = p_euclidean / r  # (n, 3)

            # Compute Jacobian correction (vectorized)
            # Project tangent vectors onto tangent plane
            dot1 = np.sum(dp_db1_unnorm * p_sphere, axis=-1, keepdims=True)  # (n, 1)
            dot2 = np.sum(dp_db2_unnorm * p_sphere, axis=-1, keepdims=True)  # (n, 1)

            dp_db1 = (dp_db1_unnorm - dot1 * p_sphere) / r  # (n, 3)
            dp_db2 = (dp_db2_unnorm - dot2 * p_sphere) / r  # (n, 3)

            # Cross product and magnitude
            cross = np.cross(dp_db1, dp_db2)  # (n, 3)
            jacobians = np.linalg.norm(cross, axis=-1)  # (n,)

            contributions.append((w, jacobians, p_sphere))
            total_weights += w * jacobians

        # Normalize and integrate
        for w, jacobians, p_sphere in contributions:
            normalized_weights = (w * jacobians) / total_weights  # (n,)
            M_triangles += (normalized_weights * areas)[:, np.newaxis, np.newaxis] * (
                p_sphere[:, :, np.newaxis] * p_sphere[:, np.newaxis, :]
            )

    else:
        raise ValueError(f"Unknown integration method: {method}")

    return M_triangles, areas


def compute_moment_tensor_lambert(
    vertices_xyz,
    interior_point_xyz,
    integration_method="spherical",
    max_triangle_edge_length=None,
):
    """Compute moment tensor using Lambert projection + shapely triangulation.

    Parameters
    ----------
    vertices_xyz : array (n, 3)
        Polygon vertices in Cartesian coordinates on unit sphere
    interior_point_xyz : array (3,)
        A point inside the polygon (used as projection center)
    integration_method : str, optional
        Method for integrating over spherical triangles:
        - "spherical": Proper spherical integration with Jacobian (recommended)
        - "planar": Legacy planar barycentric (faster but less accurate)
        - "centroid": Simple centroid approximation (fastest, least accurate)
    max_triangle_edge_length : float, optional
        Maximum edge length for triangles in radians. If specified, triangles
        larger than this will be recursively subdivided. This improves accuracy
        for large plates. Typical values: 0.1-0.5 radians (5-30 degrees).
        None (default) means no subdivision.

    Returns
    -------
    M : array (3, 3)
        Moment tensor (dimensionless, on unit sphere)
    area : float
        Polygon area (dimensionless, on unit sphere)
    """
    vertices_xyz = np.asarray(vertices_xyz, dtype=float)
    vertices_xyz = vertices_xyz / np.linalg.norm(vertices_xyz, axis=1, keepdims=True)

    # Use provided interior point as center
    center_xyz = np.asarray(interior_point_xyz, dtype=float)
    center_xyz = center_xyz / np.linalg.norm(center_xyz)

    # Project to plane
    xy = lambert_project(vertices_xyz, center_xyz)

    # Create shapely polygon (need to close it)
    poly_coords = [*list(xy), xy[0]]
    polygon = Polygon(poly_coords)

    # Triangulate
    triangles = constrained_delaunay_triangles(polygon)

    # Collect all triangles (with subdivision if requested)
    all_triangles = []

    for triangle in triangles.geoms:
        # Get triangle vertices in plane
        tri_xy = np.array(
            triangle.exterior.coords[:3]
        )  # First 3 points (4th is duplicate)

        # Project back to sphere
        tri_xyz = lambert_inverse(tri_xy, center_xyz)

        # Get triangle vertices
        v1, v2, v3 = tri_xyz

        # Subdivide if requested
        if max_triangle_edge_length is not None:
            sub_triangles = subdivide_spherical_triangle(
                v1, v2, v3, max_triangle_edge_length
            )
            all_triangles.extend(sub_triangles)
        else:
            all_triangles.append((v1, v2, v3))

    # Convert to array format for vectorized processing
    # Stack all triangles into (n_tri, 3) arrays
    triangles_array = np.array(all_triangles)  # (n_tri, 3, 3)
    v1_array = triangles_array[:, 0, :]
    v2_array = triangles_array[:, 1, :]
    v3_array = triangles_array[:, 2, :]

    # Vectorized integration over all triangles at once
    M_triangles, areas = integrate_spherical_triangle(
        v1_array, v2_array, v3_array, method=integration_method
    )

    # Sum contributions
    M_total = np.sum(M_triangles, axis=0)
    total_area = np.sum(areas)

    M = M_total / total_area if total_area > 0 else M_total

    return M, total_area


def euler_rates_to_omega(euler):
    """Convert Euler rates (deg/yr x 10^-6) to angular velocity (rad/yr)

    Parameters
    ----------
    euler : array (3,)
        Euler rates [euler_x, euler_y, euler_z] in deg/yr x 10^-6

    Returns
    -------
    omega : array (3,)
        Angular velocity in rad/yr
    """
    euler = np.asarray(euler)
    return euler * 1e-6 * np.pi / 180


def rms_velocity_mm_per_yr(euler, M):
    """Compute root mean squared velocity in mm/yr from Euler rates and moment tensor

    Parameters
    ----------
    euler : array (3,)
        Euler rates [euler_x, euler_y, euler_z] in deg/yr x 10^-6
    M : array (3, 3)
        Moment tensor

    Returns
    -------
    float
        Root mean squared velocity in mm/yr
    """
    omega = euler_rates_to_omega(euler)
    identity = np.eye(3)
    msv_unit_sphere = omega @ (identity - M) @ omega
    return EARTH_RADIUS_MM * np.sqrt(msv_unit_sphere)


def latlon_to_cartesian(lat_deg, lon_deg):
    """Convert lat/lon (degrees) to Cartesian coordinates on unit sphere"""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return x, y, z
