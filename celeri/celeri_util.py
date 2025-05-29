import warnings
from typing import Literal

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


def dc3dwrapper_cutde_disp(
    alpha,
    xo,
    depth,
    dip,
    strike_width,
    dip_width,
    dislocation,
    triangulation: Literal["/", "\\", "V"] = "/",  # Note: "\\" is a single backslash.
):
    r"""Compute the Okada displacement vector using the cutde library.

    The following should be equivalent up to floating point errors:

    ```python
    _, u, _ = okada_wrapper.dc3dwrapper(
        alpha, xo, depth, dip, strike_width, dip_width, dislocation
    )
    u = dc3dwrapper_cutde_disp(
        alpha, xo, depth, dip, strike_width, dip_width, dislocation
    )
    ```

    The `triangulation` argument is used to specify the triangulation of the
    rectangle. The default is to use two triangles along the diagonal /,
    but you can also specify a triangulation \ with the opposite diagonal,
    or a V-shaped triangulation consisting of three triangles, connecting
    the midpoint of the bottom edge to the top two vertices.

    The V triangulation should be advantageous when the observation point
    is near to the centerpoint of the rectangle, because either the / or \
    triangulations have two opposing edges through the centerpoint, and
    hence there is a risk of catastrophic cancellation in the calculation,
    whereas the V triangulation has some distance between the centerpoint
    and the other edges.

    An illustration of the three triangulations, looking down from above:

        /         \            V

    +------+  +------+  +-------------+
    |     /|  |\     |  |\           /|
    |    / |  | \    |  | \         / |
    |   /  |  |  \   |  |  \       /  |
    |  /   |  |   \  |  |   \     /   |
    | /    |  |    \ |  |    \   /    |
    |/     |  |     \|  |     \ /     |
    +------+  +------+  +------+------+

    All triangles are oriented counterclockwise when looking down.
    """
    if dip_width[0] == dip_width[1]:
        # cutde returns nan when two vertices coincide, but the
        # correct answer is obviously zero.
        return np.zeros(3)

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
    if triangulation == "/":
        tris = np.array(
            [[tr, bl, br], [bl, tr, tl]]
        )  # 
    elif triangulation == "\\":
        tris = np.array(
            [[br, tl, bl], [tl, br, tr]]
        )  # Oriented counterclockwise looking down
    elif triangulation == "V":
        bm = [(bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2, (bl[2] + br[2]) / 2]
        tris = np.array(
            [[tr, bm, br], [bm, tl, bl], [tl, bm, tr]]
        )  # Oriented counterclockwise looking down
    else:
        raise ValueError(f"Invalid triangulation parameter: {triangulation}")

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
    if triangulation != "V":
        assert disp_mat.shape == (1, 3, 2, 3)
    else:
        assert disp_mat.shape == (1, 3, 3, 3)

    if dip_width[1] < dip_width[0]:
        slip_vec = np.array([-dislocation[0], dislocation[1], -dislocation[2]])
    else:
        slip_vec = np.array(dislocation)
    assert slip_vec.shape == (3,)  # (strike, dip, tensile)

    # Sum over the triangle axis and contract the slip axis with the slip vector.
    disp_vec = np.einsum("ijkl,l->ij", disp_mat, slip_vec)
    assert disp_vec.shape == (1, 3)  # (obs, space)

    disp_vec0 = disp_vec[0]
    assert disp_vec0.shape == (3,)  # (space,)
    return disp_vec0
