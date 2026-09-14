import numpy as np
import pandas as pd
import pytest

from celeri.constants import RADIUS_EARTH
from celeri.spatial import get_block_centroid, get_strain_rate_displacements


@pytest.mark.parametrize("centroid_lat", [35.0, -35.0])
def test_strain_rate_displacements_are_a_symmetric_strain(centroid_lat):
    """Unit strain-rate components produce the matching velocity gradients.

    Finite differences of the velocity field around the centroid must give
    du_e/dx = eps_ll, du_n/dy = eps_pp and du_e/dy = du_n/dx = eps_lp, with no
    rigid rotation, in both hemispheres.
    """
    centroid_lon = 140.0
    step_m = 1_000.0
    dlat = np.degrees(step_m / RADIUS_EARTH)
    dlon = np.degrees(step_m / (RADIUS_EARTH * np.cos(np.radians(centroid_lat))))
    lon = np.array([centroid_lon + dlon, centroid_lon, centroid_lon, centroid_lon])
    lat = np.array([centroid_lat, centroid_lat + dlat, centroid_lat, centroid_lat])

    for eps_ll, eps_pp, eps_lp in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        u_east, u_north, u_up = get_strain_rate_displacements(
            lon, lat, centroid_lon, centroid_lat, eps_ll, eps_pp, eps_lp
        )
        du_e_dx = (u_east[0] - u_east[2]) / step_m
        du_n_dx = (u_north[0] - u_north[2]) / step_m
        du_e_dy = (u_east[1] - u_east[2]) / step_m
        du_n_dy = (u_north[1] - u_north[2]) / step_m
        np.testing.assert_allclose(du_e_dx, eps_ll, atol=1e-6)
        np.testing.assert_allclose(du_n_dy, eps_pp, atol=1e-6)
        np.testing.assert_allclose(du_e_dy, eps_lp, atol=1e-6)
        np.testing.assert_allclose(du_n_dx, eps_lp, atol=1e-6)
        assert np.all(u_up == 0.0)
        # No displacement at the centroid itself
        assert abs(u_east[2]) < 1e-9 and abs(u_north[2]) < 1e-9


def test_block_centroid_across_the_prime_meridian():
    segment = pd.DataFrame(
        {
            "lon1": [359.0, 1.0, 1.0, 359.0],
            "lat1": [10.0, 10.0, 12.0, 12.0],
            "lon2": [1.0, 1.0, 359.0, 359.0],
            "lat2": [10.0, 12.0, 12.0, 10.0],
            "length": [1.0, 1.0, 1.0, 1.0],
            "west_labels": [0, 0, 0, 0],
            "east_labels": [1, 1, 1, 1],
        }
    )

    lon, lat = get_block_centroid(segment, 0)

    assert lon[0] < 1e-6 or lon[0] > 359.999
    np.testing.assert_allclose(lat[0], 11.0, atol=1e-2)


def test_smoothing_matrix_ignores_isolated_elements():
    from celeri.spatial import get_tri_smoothing_matrix

    # Elements 0 and 1 share a side; element 2 shares nothing
    share = np.array([[1, -1, -1], [0, -1, -1], [-1, -1, -1]])
    distances = np.array(
        [[100.0, np.nan, np.nan], [100.0, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )

    smoothing = get_tri_smoothing_matrix(share, distances).toarray()

    assert np.all(np.isfinite(smoothing))
    assert not np.any(smoothing[6:9])
    np.testing.assert_allclose(smoothing[:6].sum(axis=1), 0.0, atol=1e-12)
