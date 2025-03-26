import warnings

import numpy as np
from cutde.halfspace import disp_matrix


def sph2cart(lon, lat, radius):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def dc3dwrapper_cutde_disp(alpha, xo, depth, dip, strike_width, dip_width, dislocation):
    """Compute the Okada displacement vector using the cutde library.

    The following should be equivalent up to floating point errors:

    ```python
    _, u, _ = okada_wrapper.dc3dwrapper(
        alpha, xo, depth, dip, strike_width, dip_width, dislocation
    )
    u = dc3dwrapper_cutde_disp(
        alpha, xo, depth, dip, strike_width, dip_width, dislocation
    )
    ```
    """
    # alpha = (lambda + mu) / (lambda + 2 mu)
    # poissons_ratio = lambda / 2(lambda + mu)
    poissons_ratio = 1 - 1 / (2 * alpha)

    # This is slightly more robust than:
    # # obs_pt = np.asarray(xo)
    # because it works with inhomogeneous types, e.g.
    # # xo = [np.array([1]), np.array([2]), 0]
    obs_pt = np.concatenate([np.atleast_1d(x) for x in xo])
    # assert obs_pt.shape == (3,)

    obs_pts = np.array([obs_pt])
    # assert obs_pts.shape == (1, 3)

    # Vertices of the rectangle
    dip_rad = np.deg2rad(dip)
    bl = [
        strike_width[0],
        dip_width[0] * np.cos(dip_rad),
        dip_width[0] * np.sin(dip_rad) - depth,
    ]
    br = [
        strike_width[1],
        dip_width[0] * np.cos(dip_rad),
        dip_width[0] * np.sin(dip_rad) - depth,
    ]
    tr = [
        strike_width[1],
        dip_width[1] * np.cos(dip_rad),
        dip_width[1] * np.sin(dip_rad) - depth,
    ]
    tl = [
        strike_width[0],
        dip_width[1] * np.cos(dip_rad),
        dip_width[1] * np.sin(dip_rad) - depth,
    ]
    tris = np.array(
        [[tr, bl, br], [bl, tr, tl]]
    )  # Oriented counterclockwise looking down

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The obs_pts input array has Fortran ordering. "
            "Converting to C ordering. This may be expensive.",
            category=UserWarning,
        )
        # This warning is false. The array is C-contiguous. It is also F-contiguous.
        # But instead of correctly checking for C-contiguity, it checks incorrectly
        # for Fortran-noncontiguity:
        # # assert not obs_pts.flags.f_contiguous
        assert obs_pts.flags.c_contiguous
        assert tris.flags.c_contiguous
        disp_mat = disp_matrix(obs_pts=obs_pts, tris=tris, nu=poissons_ratio)

    # (obs, space, triangle, slip)
    assert disp_mat.shape == (1, 3, 2, 3)

    slip_vec = np.array(dislocation)  # (strike, dip, open)
    assert slip_vec.shape == (3,)

    # Sum over the triangle axis and contract the slip axis with the slip vector.
    disp_vec = np.einsum("ijkl,l->ij", disp_mat, slip_vec)
    assert disp_vec.shape == (1, 3)  # (obs, space)

    disp_vec0 = disp_vec[0]
    assert disp_vec0.shape == (3,)  # (space,)
    return disp_vec0
