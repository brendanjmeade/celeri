import warnings

import numpy as np
import pytest

from celeri.celeri_util import (
    _preprocess_obs_pts,
    cart2sph,
    dc3dwrapper_cutde_disp,
    sph2cart,
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


class TestDc3dWrapperCutdeDisp:
    """Test suite for the dc3dwrapper_cutde_disp function."""

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    @pytest.mark.parametrize(
        "triangulation", ["/", "\\", "V"], ids=["forward_slash", "backslash", "v_shape"]
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
        "triangulation", ["/", "\\", "V"], ids=["forward_slash", "backslash", "v_shape"]
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
        "triangulation", ["/", "\\", "V"], ids=["forward_slash", "backslash", "v_shape"]
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

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    def test_degenerate_dip_width(self, standard_params, dislocation, expected_u):
        """Test degenerate case where dip_width[0] == dip_width[1]."""
        dip_width = [2.0, 2.0]  # Same values - degenerate case

        u = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            dip_width,
            dislocation,
        )

        # Should return zeros for degenerate case
        np.testing.assert_array_equal(u, np.zeros(3))

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    def test_degenerate_dip_width_multiple_points(
        self, standard_params, dislocation, expected_u
    ):
        """Test degenerate case with multiple observation points."""
        dip_width = [2.0, 2.0]  # Same values - degenerate case

        xo = [standard_params["obs_point"], standard_params["obs_point"]]
        u = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            xo,
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            dip_width,
            dislocation,
        )

        # Should return zeros for degenerate case
        np.testing.assert_array_equal(u, np.zeros((2, 3)))

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

    @pytest.mark.parametrize("dislocation, expected_u", SLIP_TEST_CASES)
    def test_reversed_dip_width(self, standard_params, dislocation, expected_u):
        """Test case where dip_width[1] < dip_width[0].

        The result should be equivalent to the negative of the expected displacement.
        """
        dip_width = list(reversed(standard_params["dip_width"]))

        u = dc3dwrapper_cutde_disp(
            standard_params["alpha"],
            standard_params["obs_point"],
            standard_params["depth"],
            standard_params["dip"],
            standard_params["strike_width"],
            dip_width,
            dislocation,
        )

        np.testing.assert_allclose(u, -expected_u, rtol=1e-6)


class TestCoordinateConversions:
    """Test suite for coordinate conversion functions."""

    def test_sph2cart_single_point(self):
        """Test spherical to cartesian conversion for a single point."""
        lon, lat, radius = 45.0, 30.0, 1.0
        x, y, z = sph2cart(lon, lat, radius)

        # Expected values calculated manually
        expected_x = np.cos(np.deg2rad(30)) * np.cos(np.deg2rad(45))
        expected_y = np.cos(np.deg2rad(30)) * np.sin(np.deg2rad(45))
        expected_z = np.sin(np.deg2rad(30))

        np.testing.assert_allclose([x, y, z], [expected_x, expected_y, expected_z])

    def test_cart2sph_single_point(self):
        """Test cartesian to spherical conversion for a single point."""
        x, y, z = 1.0, 1.0, 1.0
        azimuth, elevation, r = cart2sph(x, y, z)

        # Expected values
        expected_azimuth = np.arctan2(1.0, 1.0)  # 45 degrees
        expected_elevation = np.arctan2(1.0, np.sqrt(2.0))
        expected_r = np.sqrt(3.0)

        np.testing.assert_allclose(
            [azimuth, elevation, r], [expected_azimuth, expected_elevation, expected_r]
        )

    def test_sph2cart_arrays(self):
        """Test spherical to cartesian conversion with arrays."""
        lon = np.array([0.0, 90.0, 180.0])
        lat = np.array([0.0, 0.0, 0.0])
        radius = np.array([1.0, 1.0, 1.0])

        x, y, z = sph2cart(lon, lat, radius)

        # Expected: points on equator at different longitudes
        expected_x = np.array([1.0, 0.0, -1.0])
        expected_y = np.array([0.0, 1.0, 0.0])
        expected_z = np.array([0.0, 0.0, 0.0])

        np.testing.assert_allclose(x, expected_x, atol=1e-15)
        np.testing.assert_allclose(y, expected_y, atol=1e-15)
        np.testing.assert_allclose(z, expected_z, atol=1e-15)

    def test_cart2sph_arrays(self):
        """Test cartesian to spherical conversion with arrays."""
        x = np.array([1.0, 0.0, -1.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 0.0])

        azimuth, elevation, r = cart2sph(x, y, z)

        # Expected values
        expected_azimuth = np.array([0.0, np.pi / 2, np.pi])
        expected_elevation = np.array([0.0, 0.0, 0.0])
        expected_r = np.array([1.0, 1.0, 1.0])

        np.testing.assert_allclose(azimuth, expected_azimuth, atol=1e-15)
        np.testing.assert_allclose(elevation, expected_elevation, atol=1e-15)
        np.testing.assert_allclose(r, expected_r, atol=1e-15)

    def test_round_trip_conversion(self):
        """Test that sph2cart and cart2sph are inverse operations."""
        # Test with various points
        test_points = [
            (0.0, 0.0, 1.0),  # Equator, prime meridian
            (90.0, 0.0, 1.0),  # Equator, 90 degrees
            (180.0, 0.0, 1.0),  # Equator, 180 degrees
            (0.0, 90.0, 1.0),  # North pole
            (0.0, -90.0, 1.0),  # South pole
            (45.0, 45.0, 2.0),  # Arbitrary point
        ]

        for lon, lat, radius in test_points:
            # Convert to cartesian and back
            x, y, z = sph2cart(lon, lat, radius)
            azimuth, elevation, r = cart2sph(x, y, z)

            # Convert back to degrees for comparison
            lon_back = np.rad2deg(azimuth)
            lat_back = np.rad2deg(elevation)

            # Handle longitude wrapping (azimuth can be negative)
            if lon_back < 0:
                lon_back += 360

            np.testing.assert_allclose(
                [lon_back, lat_back, r], [lon, lat, radius], atol=1e-12
            )

    def test_round_trip_conversion_reverse(self):
        """Test round trip conversion in the opposite direction: cartesian → spherical → cartesian."""
        # Test with various cartesian points
        test_points = [
            (1.0, 0.0, 0.0),  # Unit vector along x-axis
            (0.0, 1.0, 0.0),  # Unit vector along y-axis
            (0.0, 0.0, 1.0),  # Unit vector along z-axis
            (-1.0, 0.0, 0.0),  # Negative x-axis
            (0.0, -1.0, 0.0),  # Negative y-axis
            (0.0, 0.0, -1.0),  # Negative z-axis
            (1.0, 1.0, 1.0),  # Arbitrary point
            (2.0, -3.0, 4.0),  # Another arbitrary point
            (0.5, 0.866, 0.0),  # Point on xy-plane
        ]

        for x, y, z in test_points:
            # Convert to spherical and back to cartesian
            azimuth, elevation, r = cart2sph(x, y, z)
            x_back, y_back, z_back = sph2cart(
                np.rad2deg(azimuth), np.rad2deg(elevation), r
            )

            np.testing.assert_allclose([x_back, y_back, z_back], [x, y, z], atol=1e-12)

    def test_edge_cases(self):
        """Test edge cases for coordinate conversions."""
        # Test origin
        azimuth, elevation, r = cart2sph(0.0, 0.0, 0.0)
        assert r == 0.0
        # azimuth and elevation are undefined at origin, but function should not crash

        # Test poles
        x, y, z = sph2cart(0.0, 90.0, 1.0)  # North pole
        np.testing.assert_allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-15)

        x, y, z = sph2cart(0.0, -90.0, 1.0)  # South pole
        np.testing.assert_allclose([x, y, z], [0.0, 0.0, -1.0], atol=1e-15)


class TestPreprocessObsPts:
    """Test suite for the _preprocess_obs_pts utility function."""

    def test_empty_list(self):
        """Test empty list input, which is a special case."""
        originally_1d, obs_pts = _preprocess_obs_pts([])
        assert originally_1d is False
        assert obs_pts.shape == (0, 3)
        assert obs_pts.dtype == np.float64

    @pytest.mark.parametrize(
        "xo_1d",
        [
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            (1.0, 2.0, 3.0),
        ],
        ids=["list", "numpy", "tuple"],
    )
    def test_1d_single_point(self, xo_1d):
        """Test various 1D-like inputs for a single 3D point."""
        originally_1d, obs_pts = _preprocess_obs_pts(xo_1d)
        assert originally_1d is True
        assert obs_pts.shape == (1, 3)
        np.testing.assert_array_equal(obs_pts, [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        "xo_2d",
        [
            [[1.0, 2.0, 3.0]],
            np.array([[1.0, 2.0, 3.0]]),
            ((1.0, 2.0, 3.0),),
        ],
        ids=["list_of_lists", "2d_numpy", "tuple_of_tuples"],
    )
    def test_2d_single_point(self, xo_2d):
        """Test various 2D-like inputs for a single 3D point."""
        originally_1d, obs_pts = _preprocess_obs_pts(xo_2d)
        assert originally_1d is False
        assert obs_pts.shape == (1, 3)
        np.testing.assert_array_equal(obs_pts, [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        "xo_multi",
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ],
        ids=["list_of_lists", "2d_numpy"],
    )
    def test_multiple_points(self, xo_multi):
        """Test inputs with multiple points."""
        originally_1d, obs_pts = _preprocess_obs_pts(xo_multi)
        assert originally_1d is False
        assert obs_pts.shape == (2, 3)
        np.testing.assert_array_equal(obs_pts, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @pytest.mark.parametrize(
        "xo_hetero",
        [
            [np.array([1.0]), np.array([2.0]), 3.0],
            [np.array([1]), 2, 3.0],
        ],
        ids=["numpy_and_float", "numpy_and_int_float"],
    )
    def test_heterogeneous_input_success(self, xo_hetero):
        """Test successful handling of heterogeneous inputs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            originally_1d, obs_pts = _preprocess_obs_pts(xo_hetero)

            assert len(w) == 1
            assert "Heterogeneous input types detected" in str(w[0].message)
            assert w[0].category is UserWarning

        assert originally_1d is True
        assert obs_pts.shape == (1, 3)
        np.testing.assert_array_equal(obs_pts, [[1.0, 2.0, 3.0]])

    @pytest.mark.parametrize(
        "xo_invalid_len",
        [
            [np.array([1.0]), 2.0],
            [np.array([1.0]), 2.0, 3.0, 4.0],
        ],
        ids=["too_few", "too_many"],
    )
    def test_heterogeneous_input_wrong_length(self, xo_invalid_len):
        """Test heterogeneous inputs with incorrect length."""
        expected_len = len(xo_invalid_len)
        with pytest.raises(
            ValueError,
            match=f"Heterogeneous input must have exactly 3 elements, got {expected_len}",
        ):
            # Suppress the specific heterogeneous input warning since we're testing error handling
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Heterogeneous input types detected.*",
                    category=UserWarning,
                )
                _preprocess_obs_pts(xo_invalid_len)

    @pytest.mark.parametrize(
        "xo_invalid_dim, dims",
        [
            (np.array([[[1.0, 2.0, 3.0]]]), 3),
            (np.array([[[[1.0, 2.0, 3.0]]]]), 4),
        ],
        ids=["3d_input", "4d_input"],
    )
    def test_invalid_dimensions(self, xo_invalid_dim, dims):
        """Test inputs with too many dimensions."""
        with pytest.raises(ValueError, match=f"Got {dims} dimensions, expected 2"):
            _preprocess_obs_pts(xo_invalid_dim)

    @pytest.mark.parametrize(
        "xo_invalid_width, width",
        [
            (np.array([[1.0, 2.0]]), 2),
            (np.array([[1.0, 2.0, 3.0, 4.0]]), 4),
        ],
        ids=["too_few", "too_many"],
    )
    def test_invalid_width(self, xo_invalid_width, width):
        """Test inputs with incorrect vector width."""
        with pytest.raises(
            ValueError, match=f"Got {width} elements per vector, expected 3"
        ):
            _preprocess_obs_pts(xo_invalid_width)

    def test_wrong_width_with_transpose_hint(self):
        """Test if the transpose hint is provided for (3, N) arrays."""
        xo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape (3, 2)
        with pytest.raises(
            ValueError, match="You probably just need to transpose the input"
        ):
            _preprocess_obs_pts(xo)

    def test_many_points(self):
        """Test many points to ensure originally_1d=False path."""
        xo = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is False
        assert obs_pts.shape == (4, 3)

    def test_wrong_width_without_transpose_hint(self):
        """Test wrong width case without transpose hint."""
        xo = np.array([[1.0, 2.0], [3.0, 4.0]])  # Shape (2, 2) - no transpose hint
        with pytest.raises(ValueError) as exc_info:
            _preprocess_obs_pts(xo)
        # Should NOT contain transpose hint
        assert "transpose" not in str(exc_info.value)
        assert "Got 2 elements per vector, expected 3" in str(exc_info.value)

    def test_single_point_1d_detection_list(self):
        """Test 1D detection for list input."""
        xo = [1, 2, 3]  # xo[0] = 1, np.atleast_1d(1).shape = (1,)
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is True
        assert obs_pts.shape == (1, 3)

    def test_single_point_1d_detection_numpy(self):
        """Test 1D detection for numpy array input."""
        xo = np.array([1, 2, 3])  # xo[0] = 1, np.atleast_1d(1).shape = (1,)
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is True
        assert obs_pts.shape == (1, 3)

    def test_single_point_2d_detection_list(self):
        """Test 2D detection for nested list input."""
        xo = [[1, 2, 3]]  # xo[0] = [1, 2, 3], np.atleast_1d([1, 2, 3]).shape = (3,)
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is False
        assert obs_pts.shape == (1, 3)

    def test_single_point_2d_detection_numpy(self):
        """Test 2D detection for numpy array input."""
        xo = np.array([[1, 2, 3]])  # xo[0] = np.array([1, 2, 3]), shape = (3,)
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is False
        assert obs_pts.shape == (1, 3)

    @pytest.mark.parametrize(
        "xo",
        [
            [1, 2, 3],  # Integer input
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
    def test_various_numeric_types_and_values(self, xo):
        """Test with various numeric types and value ranges."""
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is True
        assert obs_pts.shape == (1, 3)
        np.testing.assert_allclose(obs_pts, [xo])

    @pytest.mark.parametrize(
        "dtype",
        [np.int32, np.int64, np.float32, np.float64],
        ids=["int32", "int64", "float32", "float64"],
    )
    def test_numpy_array_different_dtypes(self, dtype):
        """Test various numpy array dtypes."""
        xo = np.array([1, 2, 3], dtype=dtype)
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is True
        assert obs_pts.shape == (1, 3)

    def test_all_valid_single_point_formats(self):
        """Test all valid ways to represent a single point."""
        point = [1.0, 2.0, 3.0]
        test_cases = [
            # 1D-like inputs
            (point, True, "list"),
            (np.array(point), True, "numpy 1D"),
            (tuple(point), True, "tuple"),
            # 2D-like inputs
            ([point], False, "nested list"),
            (np.array([point]), False, "numpy 2D"),
            ((point,), False, "nested tuple"),
        ]

        for xo, expected_1d, desc in test_cases:
            originally_1d, obs_pts = _preprocess_obs_pts(xo)
            assert originally_1d == expected_1d, f"Failed for {desc}"
            assert obs_pts.shape == (1, 3), f"Failed for {desc}"
            np.testing.assert_array_equal(
                obs_pts, [point], err_msg=f"Failed for {desc}"
            )

    def test_return_types(self):
        """Test that return types are correct."""
        xo = [1.0, 2.0, 3.0]
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert isinstance(originally_1d, bool)
        assert isinstance(obs_pts, np.ndarray)
        assert obs_pts.ndim == 2
        assert obs_pts.shape[1] == 3

    @pytest.mark.parametrize(
        "order",
        ["C", "F"],
    )
    def test_array_memory_layout(self, order):
        """Test C-contiguous and Fortran-contiguous arrays work correctly."""
        xo = np.array([[1, 2, 3], [4, 5, 6]], order=order)
        originally_1d, obs_pts = _preprocess_obs_pts(xo)
        assert originally_1d is False
        assert obs_pts.shape == (2, 3)
        np.testing.assert_array_equal(obs_pts, [[1, 2, 3], [4, 5, 6]])
