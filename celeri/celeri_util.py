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
    but you can also specify a triangulation \\ with the opposite diagonal,
    or a V-shaped triangulation consisting of three triangles, connecting
    the midpoint of the bottom edge to the top two vertices.

    The V triangulation should be advantageous when the observation point
    is near to the centerpoint of the rectangle, because either the / or \\
    triangulations have two opposing edges through the centerpoint, and
    hence there is a risk of catastrophic cancellation in the calculation,
    whereas the V triangulation has some distance between the centerpoint
    and the other edges.

    An illustration of the three triangulations, looking down from above:

    ```text
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
    ```

    Parameters
    ----------
    alpha : float
        Material parameter (lambda + mu) / (lambda + 2 * mu)
    xo : array_like
        Observation point(s). Can be:
        - Single point: [x, y, z] or (3,) array
        - Multiple points: (N, 3) array or list of N points
    depth : float
        Depth of the fault origin
    dip : float
        Dip angle in degrees
    strike_width : array_like
        [min_strike, max_strike] along-strike extent
    dip_width : array_like
        [min_dip, max_dip] down-dip extent
    dislocation : array_like
        [strike_slip, dip_slip, tensile_slip] slip vector
    triangulation : str, optional
        Triangulation pattern: "/", "\\", or "V"

    Returns
    -------
    displacement : numpy.ndarray
        Displacement vector(s). Shape depends on input format:
        - 1D-like input ([x,y,z], np.array([x,y,z])): (3,) array
        - 2D-like input ([[x,y,z]], np.array([[x,y,z]])): (1,3) array
        - Multiple points ([[x,y,z], [x,y,z], ...]): (N,3) array
    """
    originally_1d, obs_pts = _preprocess_obs_pts(xo)
    n_obs = obs_pts.shape[0]

    # Early return for degenerate case
    if dip_width[0] == dip_width[1]:
        # cutde returns nan when two vertices coincide, but the
        # correct answer is obviously zero.
        if originally_1d:
            return np.zeros(3)
        else:
            return np.zeros((n_obs, 3))

    # alpha = (lambda + mu) / (lambda + 2 mu)
    # poissons_ratio = lambda / 2(lambda + mu)
    poissons_ratio = 1 - 1 / (2 * alpha)

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
        )  # Oriented counterclockwise looking down
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

    n_triangles = 3 if triangulation == "V" else 2
    # (n_obs, dim(halfspace) = 3, n_triangles, dim(slip-vector) = 3)
    assert disp_mat.shape == (n_obs, 3, n_triangles, 3)

    if dip_width[1] < dip_width[0]:
        slip_vec = np.array([-dislocation[0], dislocation[1], -dislocation[2]])
    else:
        slip_vec = np.array(dislocation)
    assert slip_vec.shape == (3,)  # (strike, dip, tensile)

    # Sum over the triangle axis and contract the slip axis with the slip vector.
    disp_vec = np.einsum("ijkl,l->ij", disp_mat, slip_vec)
    assert disp_vec.shape == (n_obs, 3)  # 3 = dim(halfspace)

    # Return format depends on original input format
    if originally_1d:
        return disp_vec[0]  # Shape: (3,)
    else:
        return disp_vec  # Shape: (N, 3)


def _preprocess_obs_pts(xo: list | np.ndarray) -> tuple[bool, np.ndarray]:
    """Preprocess the observation points.

    Parameters
    ----------
    xo : list | np.ndarray
        Observation point(s). Can be:
        - Single point: [x, y, z] or (3,) array
        - Multiple points: (N, 3) array or list of N points

    Returns
    -------
    originally_1d : bool
        Whether the input was 1D-like.
    obs_pts : np.ndarray
        Observation points. Shape: (N, 3)
    """
    originally_1d: bool
    obs_pts: np.ndarray
    if isinstance(xo, list):
        if len(xo) == 0:
            # Empty list of vectors
            originally_1d = False
            # Empty array of shape (0, 3)
            obs_pts = np.array([]).reshape(0, 3)
            return originally_1d, obs_pts
    try:
        obs_pts = np.atleast_2d(xo)
    except ValueError as e:
        warnings.warn(
            "Heterogeneous input types detected. Consider using "
            "consistent numpy arrays or lists. This will cause "
            "an error in future versions.",
            UserWarning,
            stacklevel=2,
        )
        # We only support single heterogeneous input vectors for
        # backwards compatibility with celeri, e.g.
        # # xo = [np.array([1]), np.array([2]), 0]
        originally_1d = True
        if len(xo) != 3:
            raise ValueError(
                f"Heterogeneous input must have exactly 3 elements, got {len(xo)}"
            ) from e
        obs_pt = np.concatenate([np.atleast_1d(x) for x in xo])
        assert obs_pt.shape == (3,)
        obs_pts = obs_pt.reshape(1, 3)
        return originally_1d, obs_pts

    assert obs_pts.ndim >= 2
    if len(obs_pts.shape) > 2:
        raise ValueError(
            f"Got {len(obs_pts.shape)} dimensions, expected 2. Shape: {obs_pts.shape}."
        )
    assert obs_pts.ndim == 2
    if obs_pts.shape[1] != 3:
        error_message = (
            f"Got {obs_pts.shape[1]} elements per vector, expected 3. "
            f"Shape: {obs_pts.shape}."
        )
        if obs_pts.shape[0] == 3:
            error_message += " You probably just need to transpose the input."
        raise ValueError(error_message)
    assert obs_pts.shape[1] == 3
    if obs_pts.shape[0] == 1:
        # We received a single vector as an input, and now need
        # to distinguish between 1d and 2d input.
        first_element_shape = np.atleast_1d(xo[0]).shape
        if first_element_shape == (1,):
            originally_1d = True
        elif first_element_shape == (3,):
            originally_1d = False
        else:
            # This should never happen
            raise ValueError(  # pragma: no cover
                f"Got {first_element_shape} shape for first element, "
                f"expected (1,) or (3,)."
            )
    else:
        originally_1d = False
    return originally_1d, obs_pts
