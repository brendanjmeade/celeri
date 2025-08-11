import warnings

import numpy as np
import pytest

from celeri.okada.cutde_okada import (
    _determine_auto_triangulation,
    _preprocess_obs_pts,
    dc3dwrapper_cutde_disp,
)


# Fixtures for commonly reused test parameters
@pytest.fixture
def standard_params():
    """Standard parameters used across multiple dc3dwrapper_cutde_disp tests.

    These parameters are chosen to be generic to test the standard case.

    The observation point should have integer coordinates to facilitate
    comparison between the integer and float observation point cases.
    """
    return {
        "alpha": 2 / 3,
        "depth": 4.0,
        "dip": 30.0,
        "strike_width": [3.0, 8.0],
        "dip_width": [2.0, 6.0],
        "obs_point": [5.0, 3.0, -1.0],
    }


SLIP_TEST_CASES = [
    pytest.param(
        [1, 0, 0], np.array([0.31607264, 0.01463053, -0.03332328]), id="strike_slip"
    ),
    pytest.param(
        [0, 1, 0], np.array([-0.00298606, 0.24149895, 0.20563953]), id="dip_slip"
    ),
    pytest.param(
        [0, 0, 1], np.array([-0.00491912, -0.26690619, 0.66723296]), id="tensile_slip"
    ),
]

TRIANGULATION_TEST_CASES = [
    "/",
    "\\",
    "V",
    "L",
    "okada",
]

TRIANGULATION_TEST_IDS = [
    "forward_slash",
    "backslash",
    "V_shape",
    "Λ_shape",
    "okada",
]

# Separate test cases that include auto mode
TRIANGULATION_TEST_CASES_WITH_AUTO = [
    "/",
    "\\",
    "V",
    "L",
    "okada",
    "auto",
]

TRIANGULATION_TEST_IDS_WITH_AUTO = [
    "forward_slash",
    "backslash",
    "V_shape",
    "Λ_shape",
    "okada",
    "auto",
]


class TestDc3dWrapperCutdeDisp:
    """Test suite for the dc3dwrapper_cutde_disp function."""

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    @pytest.mark.parametrize(
        "triangulation",
        TRIANGULATION_TEST_CASES,
        ids=TRIANGULATION_TEST_IDS,
    )
    def test_slip_types_and_triangulations(
        self, standard_params, dislocation, expected_u, triangulation
    ):
        """Test dc3dwrapper_cutde_disp with different slip components and triangulation patterns."""
        u = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation=triangulation,
        )

        # Test against expected values (all triangulations should give equivalent results)
        np.testing.assert_allclose(u, expected_u, rtol=1e-6)

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    @pytest.mark.parametrize(
        "triangulation",
        TRIANGULATION_TEST_CASES,
        ids=TRIANGULATION_TEST_IDS,
    )
    def test_integer_observation_points(
        self, standard_params, dislocation, expected_u, triangulation
    ):
        """Test dc3dwrapper_cutde_disp with different slip components and triangulation patterns."""
        # The observation points should be floats
        assert all(isinstance(xi, float) for xi in standard_params["obs_point"])
        # Convert the observation points to integers
        int_obs_point = [round(xi) for xi in standard_params["obs_point"]]
        assert all(isinstance(xi, int) for xi in int_obs_point)

        # We expect that the standard observation point should have been chosen to have
        # integer coordinates. This allows us to directly compare the results between
        # the integer and float observation point cases.
        assert int_obs_point == standard_params["obs_point"]

        u_from_float = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation=triangulation,
        )
        # Suppress specific cutde warning about int64 to float64 conversion
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The obs_pts input array has type int64 but needs to be converted to dtype float64.*",
                category=UserWarning,
            )
            u_from_int = dc3dwrapper_cutde_disp(
                standard_params["alpha"],
                int_obs_point,
                standard_params["depth"],
                standard_params["dip"],
                standard_params["strike_width"],
                standard_params["dip_width"],
                dislocation,
                triangulation=triangulation,
            )

        # Ensure that the results are identical
        np.testing.assert_equal(u_from_float, u_from_int)

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    @pytest.mark.parametrize(
        "triangulation",
        TRIANGULATION_TEST_CASES,
        ids=TRIANGULATION_TEST_IDS,
    )
    def test_single_vs_multiple_point_consistency(
        self, standard_params, dislocation, expected_u, triangulation
    ):
        """Test that providing [xo, xo] gives the same result repeated twice as providing just xo."""
        obs_point = standard_params["obs_point"]

        # Test single point
        u_single = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            obs_point,
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation=triangulation,
        )

        # Test multiple points (same point repeated)
        u_multiple = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            [obs_point, obs_point],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation=triangulation,
        )

        # Verify shapes
        assert u_single.shape == (3,), (
            f"Single point should return (3,) shape, got {u_single.shape}"
        )
        assert u_multiple.shape == (2, 3), (
            f"Multiple points should return (2, 3) shape, got {u_multiple.shape}"
        )

        # Verify that the multiple point result is exactly the single point result repeated twice
        expected_multiple = np.array([u_single, u_single])
        np.testing.assert_array_equal(
            u_multiple,
            expected_multiple,
            err_msg=f"Multiple point result should be single point repeated with {triangulation} triangulation",
        )

    # Parameter combinations for width testing
    STRIKE_WIDTH_CASES = (
        pytest.param((3.0, 8.0), id="normal_strike"),
        pytest.param((3.0, 3.0), id="degenerate_strike"),
        pytest.param((8.0, 3.0), id="reversed_strike"),
    )

    DIP_WIDTH_CASES = (
        pytest.param((2.0, 6.0), id="normal_dip"),
        pytest.param((2.0, 2.0), id="degenerate_dip"),
        pytest.param((6.0, 2.0), id="reversed_dip"),
    )

    @pytest.mark.parametrize("dislocation, standard_u", SLIP_TEST_CASES)
    @pytest.mark.parametrize(
        "triangulation",
        TRIANGULATION_TEST_CASES,
        ids=TRIANGULATION_TEST_IDS,
    )
    @pytest.mark.parametrize("strike_width", STRIKE_WIDTH_CASES)
    @pytest.mark.parametrize("dip_width", DIP_WIDTH_CASES)
    @pytest.mark.parametrize(
        "use_multiple_points", [False, True], ids=["single_point", "multiple_points"]
    )
    def test_width_combinations(
        self,
        standard_params,
        dislocation,
        standard_u,
        triangulation,
        strike_width,
        dip_width,
        use_multiple_points,
    ):
        """Test all combinations of normal, degenerate, and reversed strike/dip widths."""
        # Set up observation points
        if use_multiple_points:
            obs_point = [standard_params["obs_point"], standard_params["obs_point"]]
            expected_shape = (2, 3)
        else:
            obs_point = standard_params["obs_point"]
            expected_shape = (3,)

        # Calculate displacement
        u = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            obs_point,
            standard_params["depth"],
            standard_params["dip"],
            strike_width,
            dip_width,
            dislocation,
            triangulation=triangulation,
        )

        # Verify shape
        assert u.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {u.shape}"
        )

        # This is the expected sign discrepancy relative to the case of
        # positive strike and dip widths.
        expected_multiplier = np.sign(strike_width[1] - strike_width[0]) * np.sign(
            dip_width[1] - dip_width[0]
        )

        # Calculate expected result
        if use_multiple_points:
            expected_result = np.array(
                [expected_multiplier * standard_u, expected_multiplier * standard_u]
            )
        else:
            expected_result = expected_multiplier * standard_u

        if expected_multiplier == 0.0:
            # The answer is exactly zero, so we can increase the tolerance.
            if triangulation != "okada":
                # cutde returns exactly zero.
                np.testing.assert_array_equal(u, expected_result)
            else:
                # Okada doesn't return exactly zero, but very close.
                np.testing.assert_allclose(u, expected_result, atol=1e-16)
        else:
            # The generic case
            np.testing.assert_allclose(u, expected_result, rtol=1e-6)

    def test_invalid_triangulation(self, standard_params):
        """Test that an exception is raised for invalid triangulation parameter."""
        dislocation = [1.0, 0.0, 0.0]

        with pytest.raises(
            ValueError, match="Invalid triangulation parameter: invalid"
        ):
            dc3dwrapper_cutde_disp(
                standard_params["alpha"],
                standard_params["obs_point"],
                standard_params["depth"],
                standard_params["dip"],
                standard_params["strike_width"],
                standard_params["dip_width"],
                dislocation,
                triangulation="invalid",  # type: ignore
            )

    def test_auto_triangulation_basic(self, standard_params):
        """Test that auto triangulation mode works and produces reasonable results."""
        dislocation = [1.0, 0.0, 0.0]

        # Test that auto triangulation doesn't crash and returns expected shape
        u_auto = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation="auto",
        )

        assert u_auto.shape == (3,), f"Expected shape (3,), got {u_auto.shape}"
        assert np.all(np.isfinite(u_auto)), (
            "Auto triangulation produced non-finite values"
        )

    def test_auto_triangulation_multiple_points(self, standard_params):
        """Test auto triangulation with multiple observation points."""
        dislocation = [1.0, 0.0, 0.0]

        # Create multiple observation points at different positions
        obs_points = [
            [5.0, 3.0, -1.0],  # Standard point
            [5.5, 1.0, -1.0],  # Different position
            [4.0, 4.0, -2.0],  # Another position
        ]

        u_auto = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            obs_points,
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation="auto",
        )

        assert u_auto.shape == (3, 3), f"Expected shape (3, 3), got {u_auto.shape}"
        assert np.all(np.isfinite(u_auto)), (
            "Auto triangulation produced non-finite values"
        )

    def test_auto_triangulation_consistency(self, standard_params):
        """Test that auto triangulation is deterministic and consistent."""
        dislocation = [1.0, 0.0, 0.0]

        # Run auto triangulation twice with the same parameters
        u_auto_1 = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation="auto",
        )

        u_auto_2 = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            standard_params["dip_width"],
            dislocation,
            triangulation="auto",
        )

        np.testing.assert_array_equal(
            u_auto_1, u_auto_2, err_msg="Auto triangulation should be deterministic"
        )


class TestAutoTriangulation:
    """Test suite for the auto triangulation helper function."""

    def test_far_from_plane_uses_forward_slash(self):
        """Test that points far from the plane (normal offset >= 10% char length) use '/'."""
        # Rectangle with char_length = min(5, 4) = 4
        strike_width = np.array([[3.0, 8.0]])  # width = 5
        dip_width = np.array([[2.0, 6.0]])  # height = 4
        depth = np.array([4.0])
        dip = np.array([30.0])

        # Point far above the plane (normal offset > 0.1 * 4 = 0.4)
        obs_pt = np.array([[5.5, 4.0, 0.0]])  # Far above surface

        triangulations = _determine_auto_triangulation(
            obs_pt, depth, dip, strike_width, dip_width
        )
        assert triangulations[0] == "/", f"Expected '/', got '{triangulations[0]}'"

    def test_zero_characteristic_length_uses_forward_slash(self):
        """Test that zero width or height (char_length = 0) uses '/'."""
        # Degenerate rectangle with zero width
        strike_width = np.array([[3.0, 3.0]])  # width = 0
        dip_width = np.array([[2.0, 6.0]])  # height = 4
        depth = np.array([4.0])
        dip = np.array([30.0])
        obs_pt = np.array([[3.0, 4.0, -4.0]])

        triangulations = _determine_auto_triangulation(
            obs_pt, depth, dip, strike_width, dip_width
        )
        assert triangulations[0] == "/", f"Expected '/', got '{triangulations[0]}'"

    def test_central_region_uses_v(self):
        """Test that points in the central region use 'V'."""
        strike_width = np.array([[3.0, 8.0]])  # width = 5
        dip_width = np.array([[2.0, 6.0]])  # height = 4
        depth = np.array([4.0])
        dip = np.array([30.0])

        # Point very close to rectangle center
        dip_rad = np.deg2rad(dip[0])
        strike_mid = (strike_width[0, 0] + strike_width[0, 1]) / 2  # 5.5
        dip_mid = (dip_width[0, 0] + dip_width[0, 1]) / 2  # 4.0

        obs_pt = np.array(
            [
                [
                    strike_mid,  # Exactly at strike center
                    dip_mid * np.cos(dip_rad),  # At dip center projected
                    dip_mid * np.sin(dip_rad) - depth[0],  # At dip center depth
                ]
            ]
        )

        triangulations = _determine_auto_triangulation(
            obs_pt, depth, dip, strike_width, dip_width
        )
        assert triangulations[0] == "V", f"Expected 'V', got '{triangulations[0]}'"

    def test_xor_logic_for_quadrants(self):
        r"""Test the XOR logic for selecting '/' vs '\\' in different quadrants."""
        strike_width = np.array([[3.0, 8.0]])  # width = 5
        dip_width = np.array([[2.0, 6.0]])  # height = 4
        depth = np.array([4.0])
        dip = np.array([30.0])

        # Calculate rectangle center
        dip_rad = np.deg2rad(dip[0])
        strike_mid = (strike_width[0, 0] + strike_width[0, 1]) / 2  # 5.5
        dip_mid = (dip_width[0, 0] + dip_width[0, 1]) / 2  # 4.0

        # Test point in quadrant where strike_dot > 0 and dip_dot > 0 (both positive)
        # XOR: True XOR True = False, so should use "\"
        obs_pt_both_pos = np.array(
            [
                [
                    strike_mid + 1.0,  # strike_dot > 0
                    (dip_mid + 1.0) * np.cos(dip_rad),  # dip_dot > 0
                    (dip_mid + 1.0) * np.sin(dip_rad) - depth[0],
                ]
            ]
        )
        triangulations = _determine_auto_triangulation(
            obs_pt_both_pos, depth, dip, strike_width, dip_width
        )
        assert triangulations[0] == "\\", (
            f"Expected '\\', got '{triangulations[0]}' for both positive quadrant"
        )

        # Test point in quadrant where strike_dot > 0 and dip_dot < 0 (different signs)
        # XOR: True XOR False = True, so should use "/"
        obs_pt_mixed = np.array(
            [
                [
                    strike_mid + 1.0,  # strike_dot > 0
                    (dip_mid - 1.0) * np.cos(dip_rad),  # dip_dot < 0
                    (dip_mid - 1.0) * np.sin(dip_rad) - depth[0],
                ]
            ]
        )
        triangulations = _determine_auto_triangulation(
            obs_pt_mixed, depth, dip, strike_width, dip_width
        )
        assert triangulations[0] == "/", (
            f"Expected '/', got '{triangulations[0]}' for mixed signs quadrant"
        )


class TestPreprocessObsPts:
    """Test suite for the _preprocess_obs_pts utility function."""

    def test_empty_list(self, standard_params):
        """Test empty list input, which is a special case."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = []
        params["dislocation"] = []
        params.pop("obs_point")
        _, _, _, _, _, _, originally_1d, n_obs = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 0

    @pytest.mark.parametrize(
        "xo_1d",
        [
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            (1.0, 2.0, 3.0),
        ],
        ids=["list", "numpy", "tuple"],
    )
    def test_1d_single_point(self, standard_params, xo_1d):
        """Test various 1D-like inputs for a single 3D point."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_1d
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is True
        assert n_obs == 1
        assert xo_b.shape == (1, 3)
        np.testing.assert_array_equal(xo_b, [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        "xo_2d",
        [
            [[1.0, 2.0, 3.0]],
            np.array([[1.0, 2.0, 3.0]]),
            ((1.0, 2.0, 3.0),),
        ],
        ids=["list_of_lists", "2d_numpy", "tuple_of_tuples"],
    )
    def test_2d_single_point(self, standard_params, xo_2d):
        """Test various 2D-like inputs for a single 3D point."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_2d
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 1
        assert xo_b.shape == (1, 3)
        np.testing.assert_array_equal(xo_b, [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        "xo_multi",
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ],
        ids=["list_of_lists", "2d_numpy"],
    )
    def test_multiple_points(self, standard_params, xo_multi):
        """Test inputs with multiple points."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_multi
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 2
        assert xo_b.shape == (2, 3)
        np.testing.assert_array_equal(xo_b, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @pytest.mark.parametrize(
        "xo_hetero",
        [
            [np.array([1.0]), np.array([2.0]), 3.0],
            [np.array([1]), 2, 3.0],
        ],
        ids=["numpy_and_float", "numpy_and_int_float"],
    )
    def test_heterogeneous_input_success(self, standard_params, xo_hetero):
        """Test successful handling of heterogeneous inputs."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_hetero
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (
                xo_b,
                _,
                _,
                _,
                _,
                _,
                originally_1d,
                n_obs,
            ) = _preprocess_obs_pts(**params)

            assert len(w) == 1
            assert "Heterogeneous input types detected" in str(w[0].message)
            assert w[0].category is UserWarning

        assert originally_1d is True
        assert n_obs == 1
        assert xo_b.shape == (1, 3)
        np.testing.assert_array_equal(xo_b, [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        "xo_invalid_len",
        [
            [np.array([1.0]), 2.0],
            [np.array([1.0]), 2.0, 3.0, 4.0],
        ],
        ids=["too_few", "too_many"],
    )
    def test_heterogeneous_input_wrong_length(self, standard_params, xo_invalid_len):
        """Test heterogeneous inputs with incorrect length."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_invalid_len
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        expected_len = len(xo_invalid_len)
        with pytest.raises(
            ValueError,
            match=f"Heterogeneous input must have exactly 3 elements, got {expected_len}",
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Heterogeneous input types detected.*",
                    category=UserWarning,
                )
                _preprocess_obs_pts(**params)

    @pytest.mark.parametrize(
        "xo_invalid_dim, dims",
        [
            (np.array([[[1.0, 2.0, 3.0]]]), 3),
            (np.array([[[[1.0, 2.0, 3.0]]]]), 4),
        ],
        ids=["3d_input", "4d_input"],
    )
    def test_invalid_dimensions(self, standard_params, xo_invalid_dim, dims):
        """Test inputs with too many dimensions."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_invalid_dim
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        with pytest.raises(
            ValueError,
            match=f"Invalid number of dimensions for xo: expected 1 or 2, but got {dims}",
        ):
            _preprocess_obs_pts(**params)

    @pytest.mark.parametrize(
        "xo_invalid_width, width",
        [
            (np.array([[1.0, 2.0]]), 2),
            (np.array([[1.0, 2.0, 3.0, 4.0]]), 4),
        ],
        ids=["too_few", "too_many"],
    )
    def test_invalid_width(self, standard_params, xo_invalid_width, width):
        """Test inputs with incorrect vector width."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo_invalid_width
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        with pytest.raises(
            ValueError,
            match=f"Invalid shape for xo: expected \\(n_obs, 3\\) but got \\(1, {width}\\)",
        ):
            _preprocess_obs_pts(**params)

    def test_wrong_width_with_transpose_hint(self, standard_params):
        """Test if the transpose hint is provided for (3, N) arrays."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        with pytest.raises(
            ValueError,
            match="Invalid shape for xo: got \\(3, 2\\)\\. Did you mean to transpose it to have shape \\(2, 3\\)\\?",
        ):
            _preprocess_obs_pts(**params)

    def test_many_points(self, standard_params):
        """Test many points to ensure originally_1d=False path."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 4
        assert xo_b.shape == (4, 3)

    def test_wrong_width_without_transpose_hint(self, standard_params):
        """Test wrong width case without transpose hint."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        with pytest.raises(ValueError) as exc_info:
            _preprocess_obs_pts(**params)
        assert "transpose" not in str(exc_info.value)
        assert "Invalid shape for xo: expected (n_obs, 3) but got (2, 2)" in str(
            exc_info.value
        )

    def test_single_point_1d_detection_list(self, standard_params):
        """Test 1D detection for list input."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = [1, 2, 3]
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is True
        assert n_obs == 1
        assert xo_b.shape == (1, 3)

    def test_single_point_1d_detection_numpy(self, standard_params):
        """Test 1D detection for numpy array input."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = np.array([1, 2, 3])
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is True
        assert n_obs == 1
        assert xo_b.shape == (1, 3)

    def test_single_point_2d_detection_list(self, standard_params):
        """Test 2D detection for nested list input."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = [[1, 2, 3]]
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 1
        assert xo_b.shape == (1, 3)

    def test_single_point_2d_detection_numpy(self, standard_params):
        """Test 2D detection for numpy array input."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = np.array([[1, 2, 3]])
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 1
        assert xo_b.shape == (1, 3)

    @pytest.mark.parametrize(
        "xo",
        [
            [1, 2, 3],
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            [1, 2.0, 3],
            [1e10, 2e10, 3e10],
            [1e-10, 2e-10, 3e-10],
            [0, 0, 0],
            [-1.0, -2.0, -3.0],
        ],
        ids=[
            "integer_list",
            "int32",
            "float32",
            "mixed_int_float",
            "large_numbers",
            "small_numbers",
            "zeros",
            "negatives",
        ],
    )
    def test_various_numeric_types_and_values(self, standard_params, xo):
        """Test with various numeric types and value ranges."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = xo
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is True
        assert n_obs == 1
        assert xo_b.shape == (1, 3)
        np.testing.assert_allclose(xo_b, [xo])

    @pytest.mark.parametrize(
        "dtype",
        [np.int32, np.int64, np.float32, np.float64],
        ids=["int32", "int64", "float32", "float64"],
    )
    def test_numpy_array_different_dtypes(self, standard_params, dtype):
        """Test various numpy array dtypes."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = np.array([1, 2, 3], dtype=dtype)
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is True
        assert n_obs == 1
        assert xo_b.shape == (1, 3)

    def test_all_valid_single_point_formats(self, standard_params):
        """Test all valid ways to represent a single point."""
        point = [1.0, 2.0, 3.0]
        test_cases = [
            (point, True, "list"),
            (np.array(point), True, "numpy 1D"),
            (tuple(point), True, "tuple"),
            ([point], False, "nested list"),
            (np.array([point]), False, "numpy 2D"),
            ((point,), False, "nested tuple"),
        ]

        for xo, expected_1d, desc in test_cases:
            params = standard_params.copy()
            params.pop("alpha")
            params["xo"] = xo
            params["dislocation"] = [1.0, 0.0, 0.0]
            params.pop("obs_point")
            (
                xo_b,
                _,
                _,
                _,
                _,
                _,
                originally_1d,
                n_obs,
            ) = _preprocess_obs_pts(**params)
            assert originally_1d == expected_1d, f"Failed for {desc}"
            assert n_obs == 1, f"Failed for {desc}"
            assert xo_b.shape == (1, 3), f"Failed for {desc}"
            np.testing.assert_array_equal(xo_b, [point], err_msg=f"Failed for {desc}")

    def test_return_types(self, standard_params):
        """Test that return types are correct."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = [1.0, 2.0, 3.0]
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert isinstance(originally_1d, bool)
        assert isinstance(xo_b, np.ndarray)
        assert xo_b.ndim == 2
        assert xo_b.shape[1] == 3

    @pytest.mark.parametrize(
        "order",
        ["C", "F"],
    )
    def test_array_memory_layout(self, standard_params, order):
        """Test C-contiguous and Fortran-contiguous arrays work correctly."""
        params = standard_params.copy()
        params.pop("alpha")
        params["xo"] = np.array([[1, 2, 3], [4, 5, 6]], order=order)
        params["dislocation"] = [1.0, 0.0, 0.0]
        params.pop("obs_point")
        (
            xo_b,
            _,
            _,
            _,
            _,
            _,
            originally_1d,
            n_obs,
        ) = _preprocess_obs_pts(**params)
        assert originally_1d is False
        assert n_obs == 2
        assert xo_b.shape == (2, 3)
        np.testing.assert_array_equal(xo_b, [[1, 2, 3], [4, 5, 6]])
