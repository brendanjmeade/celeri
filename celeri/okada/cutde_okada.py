import warnings
from typing import Literal, TypeAlias

import numpy as np
from cutde.halfspace import disp as disp_cutde

TriangulationTypes: TypeAlias = Literal["/", "\\", "V", "L", "okada", "auto"]


def _determine_auto_triangulation(
    obs_pts, depth, dip, strike_width, dip_width
) -> list[Literal["/", "\\", "V"]]:
    r"""Determine the appropriate triangulation for observation points.

    Assumes inputs have already been preprocessed by _preprocess_obs_pts.

    For a detailed description of this algorithm, see `auto-triangulation.md`.

    Parameters
    ----------
    obs_pts : ndarray, shape (n_obs, 3)
        Observation points [x, y, z]
    depth : ndarray, shape (n_obs,)
        Depth of fault origin
    dip : ndarray, shape (n_obs,)
        Dip angle in degrees
    strike_width : ndarray, shape (n_obs, 2)
        [min_strike, max_strike] extent
    dip_width : ndarray, shape (n_obs, 2)
        [min_dip, max_dip] extent

    Returns
    -------
    triangulations : list of Literal["/", "\\", "V"]
        Triangulation for each observation point: "/", "\\", or "V"
    """
    n_obs = obs_pts.shape[0]

    # Calculate rectangle dimensions
    width = strike_width[:, 1] - strike_width[:, 0]  # Shape: (n_obs,)
    height = dip_width[:, 1] - dip_width[:, 0]  # Shape: (n_obs,)

    # Characteristic length is minimum of width and height
    char_length = np.minimum(np.abs(width), np.abs(height))  # Shape: (n_obs,)

    # Initialize result array with default "/"
    triangulations = np.full(n_obs, "/", dtype="<U1")

    # Calculate rectangle midpoints in 3D space for valid points
    dip_rad = np.deg2rad(dip)  # Shape: (n_obs,)
    strike_mid = (strike_width[:, 0] + strike_width[:, 1]) / 2  # Shape: (n_obs,)
    dip_mid = (dip_width[:, 0] + dip_width[:, 1]) / 2  # Shape: (n_obs,)

    rect_midpoints = np.column_stack(
        [
            strike_mid,
            dip_mid * np.cos(dip_rad),
            dip_mid * np.sin(dip_rad) - depth,
        ]
    )  # Shape: (n_obs, 3)

    # Vector from rectangle midpoint to observation point
    rel_vec = obs_pts - rect_midpoints  # Shape: (n_obs, 3)

    # Normal vectors to the rectangle planes
    normals = np.column_stack(
        [np.zeros(n_obs), -np.sin(dip_rad), np.cos(dip_rad)]
    )  # Shape: (n_obs, 3)

    # Offset in normal direction (vectorized dot product)
    normal_offset = np.sum(rel_vec * normals, axis=1)  # Shape: (n_obs,)

    # Check normal offset condition: un less < 10% of characteristic length, use "/"
    close_mask = np.abs(normal_offset) < 0.1 * char_length  # Shape: (n_obs,)
    close_indices = np.where(close_mask)[0]

    # For points not close to the plane, continue processing
    n_close = len(close_indices)
    if n_close == 0:
        return triangulations.tolist()

    # Project observation points onto the rectangle plane
    # Strike direction unit vectors (along x-axis)
    strike_dirs = np.tile([1, 0, 0], (n_close, 1))  # Shape: (n_close, 3)

    # Dip direction unit vectors (perpendicular to strike, in the plane)
    dip_dirs = np.column_stack(
        [
            np.zeros(n_close),
            np.cos(dip_rad[close_indices]),
            np.sin(dip_rad[close_indices]),
        ]
    )  # Shape: (n_close, 3)

    # Project relative vectors onto strike and dip directions
    strike_dot = np.sum(rel_vec[close_mask] * strike_dirs, axis=1)  # Shape: (n_close,)
    dip_dot = np.sum(rel_vec[close_mask] * dip_dirs, axis=1)  # Shape: (n_close,)

    # Check if the projected point is close to the interior of the rectangle.
    # The threshold for the interior is 0.5, so setting 0.6 gives a small margin.
    close_mask2 = np.abs(strike_dot) < 0.6 * width[close_indices]
    close_mask2 &= np.abs(dip_dot) < 0.6 * height[close_indices]
    # Shape: (n_close,)
    close_indices2 = np.where(close_mask2)[0]
    n_close2 = len(close_indices2)

    # Check central region condition
    width_close = width[close_indices2]
    height_close = height[close_indices2]

    if n_close2 > 0:
        central_mask = np.zeros(n_close2, dtype=bool)
        # central_mask = (
        #     (strike_dot[close_indices2] / width_close) ** 2
        #     + (dip_dot[close_indices2] / height_close) ** 2
        # ) < 0.1**2

        # Equivalent to above commented out, but without the division
        central_mask = (
            (strike_dot[close_indices2] * height_close) ** 2
            + (dip_dot[close_indices2] * width_close) ** 2
        ) < (0.1 * width_close * height_close) ** 2

        # Set central region points to "V"
        central_global_indices = close_indices[close_indices2[central_mask]]
        triangulations[central_global_indices] = "V"

        # Update masks for remaining processing
        remaining_mask = ~central_mask
        if not np.any(remaining_mask):
            return triangulations.tolist()

        # For remaining points, apply XOR logic
        remaining_indices2 = close_indices2[remaining_mask]
        remaining_global_indices = close_indices[remaining_indices2]

        if len(remaining_global_indices) > 0:
            strike_dot_remaining = strike_dot[remaining_indices2]
            dip_dot_remaining = dip_dot[remaining_indices2]

            # XOR logic: use "/" if (strike_dot > 0) XOR (dip_dot > 0), else "\"
            xor_mask = (strike_dot_remaining > 0) != (dip_dot_remaining > 0)
            triangulations[remaining_global_indices[xor_mask]] = "/"
            triangulations[remaining_global_indices[~xor_mask]] = "\\"
    else:
        # No points in bounds, process all close points
        if len(close_indices) > 0:
            strike_dot_remaining = strike_dot
            dip_dot_remaining = dip_dot

            # XOR logic: use "/" if (strike_dot > 0) XOR (dip_dot > 0), else "\"
            xor_mask = (strike_dot_remaining > 0) != (dip_dot_remaining > 0)
            triangulations[close_indices[xor_mask]] = "/"
            triangulations[close_indices[~xor_mask]] = "\\"

    return triangulations.tolist()


def dc3dwrapper_cutde_disp(
    alpha,
    xo,
    depth,
    dip,
    strike_width,
    dip_width,
    dislocation,
    triangulation: TriangulationTypes = "auto",  # Note: "\\" is a single backslash.
):
    r"""Compute displacement using cutde, with an Okada option.

    This function calculates the displacement at observation points due to
    dislocation on one or more rectangular faults in an elastic half-space.
    It can use either a triangulation-based method via the `cutde` library
    or the analytical solution from `okada_wrapper`.

    The `triangulation` argument controls how each rectangle is discretized
    into triangles when using `cutde`. When set to "auto", the triangulation
    is automatically selected for each observation point based on its position
    relative to the fault rectangle.

    All input parameters are broadcastable to a common `n_obs` dimension.

    An illustration of the triangulations, looking down from above:

    ```text
        /         \            V                L

    +------+  +------+  +-------------+  +------+------+
    |     /|  |\     |  |\           /|  |     / \     |
    |    / |  | \    |  | \         / |  |    /   \    |
    |   /  |  |  \   |  |  \       /  |  |   /     \   |
    |  /   |  |   \  |  |   \     /   |  |  /       \  |
    | /    |  |    \ |  |    \   /    |  | /         \ |
    |/     |  |     \|  |     \ /     |  |/           \|
    +------+  +------+  +------+------+  +-------------+

    All triangles are oriented counterclockwise when looking down.
    ```

    Parameters
    ----------
    alpha : float
        Elastic parameter: (lambda + mu) / (lambda + 2 * mu)
    xo : array_like
        Observation point(s). Shape: (3,) or (n_obs, 3 coordinates)
    depth : float or array_like
        Depth of fault origin. Shape: () or (n_obs,)
    dip : float or array_like
        Dip angle in degrees. Shape: () or (n_obs,)
    strike_width : array_like
        [min_strike, max_strike] extent. Shape: (2,) or (n_obs, 2)
    dip_width : array_like
        [min_dip, max_dip] extent. Shape: (2,) or (n_obs, 2)
    dislocation : array_like
        [strike_slip, dip_slip, tensile_slip]. Shape: (3,) or (n_obs, 3 slips)
    triangulation : {'/', '\\', 'V', 'L', 'okada', 'auto'}, optional
        The triangulation method for `cutde` or 'okada' to use the
        analytical solution. When 'auto' is selected, the triangulation
        is automatically chosen based on the observation point's position
        relative to each fault rectangle.

    Returns
    -------
    displacement : numpy.ndarray
        Displacement vector(s).
        - If input is 1D-like: (3 displacements,)
        - If input is 2D-like: (n_obs, 3 displacements)
    """
    (
        obs_pts,  # (n_obs, 3 coordinates)
        depth,  # (n_obs,)
        dip,  # (n_obs,)
        strike_width,  # (n_obs, 2)
        dip_width,  # (n_obs, 2)
        dislocation,  # (n_obs, 3 slips)
        originally_1d,
        n_obs,
    ) = _preprocess_obs_pts(
        xo=xo,
        depth=depth,
        dip=dip,
        strike_width=strike_width,
        dip_width=dip_width,
        dislocation=dislocation,
    )

    if n_obs == 0:
        return np.zeros(3) if originally_1d else np.zeros((0, 3))

    if triangulation == "okada":
        from okada_wrapper import dc3dwrapper

        raw_result = np.array(
            [
                dc3dwrapper(
                    alpha,
                    obs_pts[i],
                    depth[i],
                    dip[i],
                    strike_width[i],
                    dip_width[i],
                    dislocation[i],
                )[1]
                for i in range(n_obs)
            ]
        )  # (n_obs, 3 displacements)
        return raw_result[0] if originally_1d else raw_result

    # Handle auto triangulation mode
    if triangulation == "auto":
        # Determine triangulation for all observation points at once (vectorized)
        auto_triangulations = _determine_auto_triangulation(
            obs_pts, depth, dip, strike_width, dip_width
        )

        # Process each observation point with its determined triangulation
        disp_vec = np.zeros((n_obs, 3))
        for i in range(n_obs):
            # Call recursively with the determined triangulation for this single point
            single_disp = dc3dwrapper_cutde_disp(
                alpha,
                obs_pts[i],
                depth[i],
                dip[i],
                strike_width[i],
                dip_width[i],
                dislocation[i],
                triangulation=auto_triangulations[i],  # type: ignore
            )
            disp_vec[i] = single_disp

        return disp_vec[0] if originally_1d else disp_vec

    if n_obs > 0:
        # Make copies to avoid mutating input parameters
        strike_width = np.copy(strike_width)
        dip_width = np.copy(dip_width)
        orientation_correction = np.ones(n_obs)

        # Only perform swapping operations if arrays are non-empty
        if strike_width.size > 0 and dip_width.size > 0:
            dip_swap_mask = dip_width[:, 0] > dip_width[:, 1]
            if np.any(dip_swap_mask):
                dip_width[dip_swap_mask] = dip_width[dip_swap_mask, [1, 0]]
                orientation_correction[dip_swap_mask] *= -1

            strike_swap_mask = strike_width[:, 0] > strike_width[:, 1]
            if np.any(strike_swap_mask):
                strike_width[strike_swap_mask] = strike_width[strike_swap_mask, [1, 0]]
                orientation_correction[strike_swap_mask] *= -1

    poissons_ratio = 1 - 1 / (2 * alpha)

    # Calculate vertices for all rectangles
    dip_rad = np.deg2rad(dip)
    bl = np.c_[
        strike_width[:, 0],
        dip_width[:, 0] * np.cos(dip_rad),
        dip_width[:, 0] * np.sin(dip_rad) - depth,
    ]  # (n_obs, 3 coordinates)
    br = np.c_[
        strike_width[:, 1],
        dip_width[:, 0] * np.cos(dip_rad),
        dip_width[:, 0] * np.sin(dip_rad) - depth,
    ]  # (n_obs, 3 coordinates)
    tr = np.c_[
        strike_width[:, 1],
        dip_width[:, 1] * np.cos(dip_rad),
        dip_width[:, 1] * np.sin(dip_rad) - depth,
    ]  # (n_obs, 3 coordinates)
    tl = np.c_[
        strike_width[:, 0],
        dip_width[:, 1] * np.cos(dip_rad),
        dip_width[:, 1] * np.sin(dip_rad) - depth,
    ]  # (n_obs, 3 coordinates)

    # Create triangles for each rectangle based on the chosen triangulation
    if triangulation == "/":
        tris = np.stack(
            [
                np.stack([tr, bl, br], axis=1),
                np.stack([bl, tr, tl], axis=1),
                np.stack([tr, tr, tr], axis=1),  # Trivial padding triangle
            ],
            axis=1,
        )
    elif triangulation == "\\":
        tris = np.stack(
            [
                np.stack([br, tl, bl], axis=1),
                np.stack([tl, br, tr], axis=1),
                np.stack([tr, tr, tr], axis=1),
            ],
            axis=1,
        )
    elif triangulation == "V":
        bm = (bl + br) / 2
        tris = np.stack(
            [
                np.stack([tr, bm, br], axis=1),
                np.stack([bm, tl, bl], axis=1),
                np.stack([tl, bm, tr], axis=1),
            ],
            axis=1,
        )
    elif triangulation == "L":
        tm = (tl + tr) / 2
        tris = np.stack(
            [
                np.stack([bl, tm, tl], axis=1),
                np.stack([tm, br, tr], axis=1),
                np.stack([br, tm, bl], axis=1),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Invalid triangulation parameter: {triangulation}")
    # tris shape: (n_obs, 3 triangles, 3 vertices, 3 coordinates)

    # Tile inputs for cutde: one observation point per source triangle
    n_total_tris = n_obs * 3
    obs_pts_tiled = np.repeat(obs_pts, 3, axis=0)  # (n_total_tris, 3 coordinates)
    slips_tiled = np.repeat(dislocation, 3, axis=0)  # (n_total_tris, 3 slips)
    tris_reshaped = tris.reshape(
        n_total_tris, 3, 3
    )  # (n_total_tris, 3 vertices, 3 coordinates)
    orientation_tiled = np.repeat(orientation_correction, 3)  # (n_total_tris,)

    disp_val = (
        disp_cutde(
            obs_pts=obs_pts_tiled,
            tris=tris_reshaped,
            slips=slips_tiled,
            nu=poissons_ratio,
        )
        * orientation_tiled[:, np.newaxis]
    )
    # disp_val shape: (n_total_tris, 3 displacements)

    # Sum contributions from each set of 3 triangles for each observation point
    disp_vec = disp_val.reshape((n_obs, 3, 3)).sum(axis=1)  # (n_obs, 3 displacements)

    return disp_vec[0] if originally_1d else disp_vec


def _preprocess_obs_pts(
    *,
    xo: list | tuple | np.ndarray,
    depth: float | np.ndarray,
    dip: float | np.ndarray,
    strike_width: list | tuple | np.ndarray,
    dip_width: list | tuple | np.ndarray,
    dislocation: list | tuple | np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    int,
]:
    """Preprocess and broadcast fault parameters to a consistent shape.

    This function validates the shapes of all input parameters, determines the
    number of observations (n_obs), and broadcasts all inputs to match this
    dimension, which is always at axis=0. It provides detailed error
    messages for shape mismatches, including hints for suspected transpositions.

    Parameters
    ----------
    xo : list | tuple | np.ndarray
        Observation point(s). Shape: (3,) or (n_obs, 3 coordinates)
    depth : float | np.ndarray
        Depth of the fault origin. Shape: () or (n_obs,)
    dip : float | np.ndarray
        Dip angle in degrees. Shape: () or (n_obs,)
    strike_width : list | tuple | np.ndarray
        [min_strike, max_strike] extent. Shape: (2,) or (n_obs, 2)
    dip_width : list | tuple | np.ndarray
        [min_dip, max_dip] extent. Shape: (2,) or (n_obs, 2)
    dislocation : list | tuple | np.ndarray
        [strike_slip, dip_slip, tensile_slip] slip vector.
        Shape: (3 slips,) or (n_obs, 3 slips)

    Returns
    -------
    tuple
        A tuple containing:
        - xo_b (np.ndarray): Broadcasted observation points. Shape: (n_obs, 3 coordinates)
        - depth_b (np.ndarray): Broadcasted depths. Shape: (n_obs,)
        - dip_b (np.ndarray): Broadcasted dip angles. Shape: (n_obs,)
        - strike_width_b (np.ndarray): Broadcasted strike widths. Shape: (n_obs, 2)
        - dip_width_b (np.ndarray): Broadcasted dip widths. Shape: (n_obs, 2)
        - dislocation_b (np.ndarray): Broadcasted dislocations. Shape: (n_obs, 3 slips)
        - originally_1d (bool): True if all inputs were 1D-like.
        - n_obs (int): The number of observations.
    """

    def validate_and_get_n_obs(arr, name, expected_core_shape):
        """Helper to validate input shapes and determine n_obs."""
        try:
            arr = np.array(arr)
        except ValueError as e:
            if "inhomogeneous" in str(e) and isinstance(arr, list | tuple):
                # Handle heterogeneous inputs like [np.array([1]), np.array([2]), 3.0]
                warnings.warn(
                    "Heterogeneous input types detected. Consider using "
                    "consistent numpy arrays or lists. This will cause "
                    "an error in future versions.",
                    UserWarning,
                    stacklevel=2,
                )

                # Check if it's the expected length for a single observation
                if len(arr) != expected_core_shape[0]:
                    raise ValueError(
                        f"Heterogeneous input must have exactly {expected_core_shape[0]} elements, got {len(arr)}"
                    ) from e

                # Convert heterogeneous input to homogeneous array
                homogeneous_data = np.concatenate([np.atleast_1d(x) for x in arr])
                assert homogeneous_data.shape == (expected_core_shape[0],)
                arr = homogeneous_data.reshape(
                    1, expected_core_shape[0]
                )  # Single observation as 2D array
            else:
                raise  # Re-raise if not a heterogeneous array issue

        if arr.size == 0:
            return 0  # Empty input means 0 observations

        if arr.ndim == 0:
            return 1  # Scalar case
        if arr.ndim == 1:
            if arr.shape[0] == expected_core_shape[0]:
                return 1
            elif expected_core_shape[0] == 1:  # e.g., depth or dip array
                return arr.shape[0]
            else:
                raise ValueError(
                    f"Invalid shape for {name}: expected ({expected_core_shape[0]},) "
                    f"for a single observation, but got {arr.shape}."
                )
        if arr.ndim == 2:
            if arr.shape[1] == expected_core_shape[0]:
                return arr.shape[0]
            elif arr.shape[0] == expected_core_shape[0]:
                raise ValueError(
                    f"Invalid shape for {name}: got {arr.shape}. "
                    f"Did you mean to transpose it to have shape ({arr.shape[1]}, {arr.shape[0]})?"
                )
            else:
                raise ValueError(
                    f"Invalid shape for {name}: expected (n_obs, {expected_core_shape[0]}) "
                    f"but got {arr.shape}."
                )
        raise ValueError(
            f"Invalid number of dimensions for {name}: expected 1 or 2, but got {arr.ndim}."
        )

    inputs = {
        "xo": (xo, (3,)),
        "depth": (np.atleast_1d(depth), (1,)),
        "dip": (np.atleast_1d(dip), (1,)),
        "strike_width": (strike_width, (2,)),
        "dip_width": (dip_width, (2,)),
        "dislocation": (dislocation, (3,)),
    }

    # First pass: validate inputs and store converted arrays
    converted_arrays = {}
    n_obs_values = {}
    for name, (arr, shape) in inputs.items():
        n_obs_val = validate_and_get_n_obs(arr, name, shape)
        n_obs_values[name] = n_obs_val
        # Store the converted array from validate_and_get_n_obs
        try:
            converted_arrays[name] = np.array(arr)
        except ValueError as e:
            if "inhomogeneous" in str(e) and isinstance(arr, list | tuple):
                # Use the same conversion logic as in validate_and_get_n_obs
                homogeneous_data = np.concatenate([np.atleast_1d(x) for x in arr])
                converted_arrays[name] = homogeneous_data.reshape(1, shape[0])
            else:
                raise

    # Determine the number of observations.
    # If any input implies 0 observations, the result is 0 observations.
    # If there are conflicting observation counts (e.g., 2 and 3), it's an error.
    if 0 in n_obs_values.values():
        non_empty_counts = {n for n in n_obs_values.values() if n > 1}
        if non_empty_counts:
            raise ValueError(
                f"Inconsistent number of observations: found empty and non-empty inputs: {n_obs_values}"
            )
        n_obs = 0
    else:
        non_singleton_counts = {n for n in n_obs_values.values() if n > 1}
        if len(non_singleton_counts) > 1:
            raise ValueError(
                f"Inconsistent number of observations: {n_obs_values}. "
                "All array inputs must have the same length in the first dimension."
            )
        n_obs = non_singleton_counts.pop() if non_singleton_counts else 1

    # Determine originally_1d based on the original format of xo before any processing
    try:
        xo_array = np.array(xo) if not isinstance(xo, np.ndarray) else xo
    except ValueError as e:
        if "inhomogeneous" in str(e) and isinstance(xo, list | tuple):
            # For heterogeneous inputs, treat as 1D-like since they represent a single point
            originally_1d = True
        else:
            raise
    else:
        if xo_array.size == 0:
            originally_1d = False  # Empty inputs are considered 2D-like
        elif xo_array.ndim == 1:
            originally_1d = True  # 1D inputs like [1, 2, 3]
        elif xo_array.ndim == 2 and xo_array.shape[0] == 1:
            originally_1d = False  # 2D inputs like [[1, 2, 3]] even if single point
        elif xo_array.ndim == 2 and xo_array.shape[0] > 1:
            originally_1d = False  # 2D inputs with multiple points
        else:
            originally_1d = False  # Default to False for other cases

    if n_obs == 0:
        # Return explicitly shaped empty arrays
        xo_b = np.empty((0, 3))
        depth_b = np.empty((0,))
        dip_b = np.empty((0,))
        strike_width_b = np.empty((0, 2))
        dip_width_b = np.empty((0, 2))
        dislocation_b = np.empty((0, 3))
        return (
            xo_b,
            depth_b,
            dip_b,
            strike_width_b,
            dip_width_b,
            dislocation_b,
            originally_1d,
            n_obs,
        )

    def broadcast(arr, name, expected_core_shape):
        # arr is already converted, no need to call np.array(arr) again
        if arr.ndim == 0:  # scalar depth/dip
            return np.broadcast_to(arr, (n_obs,))
        if arr.ndim == 1:
            if arr.shape[0] == expected_core_shape[0]:
                if expected_core_shape[0] > 1:
                    return np.broadcast_to(arr, (n_obs, expected_core_shape[0]))
                else:  # it's a scalar-like array([5])
                    return np.broadcast_to(arr[0], (n_obs,))
            elif expected_core_shape[0] == 1:  # depth/dip array
                return arr  # it's already (n_obs,)
        return arr

    xo_b = broadcast(converted_arrays["xo"], "xo", (3,))
    depth_b = broadcast(converted_arrays["depth"], "depth", (1,))
    dip_b = broadcast(converted_arrays["dip"], "dip", (1,))
    strike_width_b = broadcast(converted_arrays["strike_width"], "strike_width", (2,))
    dip_width_b = broadcast(converted_arrays["dip_width"], "dip_width", (2,))
    dislocation_b = broadcast(converted_arrays["dislocation"], "dislocation", (3,))

    # Final shape assertions
    # Note: np.atleast_2d behavior means we can't do a simple check for n_obs=0
    if n_obs > 0:
        assert xo_b.shape == (n_obs, 3), f"xo shape: {xo_b.shape}"
        assert depth_b.shape == (n_obs,), f"depth shape: {depth_b.shape}"
        assert dip_b.shape == (n_obs,), f"dip shape: {dip_b.shape}"
        assert strike_width_b.shape == (n_obs, 2), f"sw shape: {strike_width_b.shape}"
        assert dip_width_b.shape == (n_obs, 2), f"dw shape: {dip_width_b.shape}"
        assert dislocation_b.shape == (n_obs, 3), f"disloc shape: {dislocation_b.shape}"

    return (
        xo_b,
        depth_b,
        dip_b,
        strike_width_b,
        dip_width_b,
        dislocation_b,
        originally_1d,
        n_obs,
    )
