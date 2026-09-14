import numpy as np
import pandas as pd

from celeri.celeri_util import sph2cart
from celeri.constants import RADIUS_EARTH
from celeri.model import process_sar


def test_process_sar_non_empty():
    sar = pd.DataFrame({"lon": [140.0, 141.5], "lat": [35.0, 36.0]})

    sar = process_sar(sar, config=None)

    assert len(sar) == 2
    x, y, z = sph2cart(np.array([140.0, 141.5]), np.array([35.0, 36.0]), RADIUS_EARTH)
    np.testing.assert_allclose(sar.x.to_numpy(), x)
    np.testing.assert_allclose(sar.y.to_numpy(), y)
    np.testing.assert_allclose(sar.z.to_numpy(), z)
    np.testing.assert_array_equal(sar.depth.to_numpy(), 0.0)
    np.testing.assert_array_equal(sar.block_label.to_numpy(), -1)


def test_process_sar_empty():
    sar = pd.DataFrame(columns=["lon", "lat"])

    sar = process_sar(sar, config=None)

    assert len(sar) == 0
    assert {"depth", "x", "y", "z", "block_label"} <= set(sar.columns)


def test_zero_mesh_segment_locking_depth_bounds():
    from celeri.model import zero_mesh_segment_locking_depth

    segment = pd.DataFrame(
        {
            "mesh_flag": [1, 1, 1, 0, 1],
            "mesh_file_index": [0, 2, 1, 0, -1],
            "locking_depth": [15.0, 15.0, 15.0, 15.0, 15.0],
        }
    )
    meshes = [object(), object()]

    zeroed = zero_mesh_segment_locking_depth(segment, meshes)

    # Only segments tied to an existing mesh (index < number of meshes) are zeroed
    assert zeroed.locking_depth.tolist() == [0.0, 15.0, 0.0, 15.0, 15.0]
