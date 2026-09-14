import numpy as np
import pandas as pd
import pytest

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


def test_assign_block_labels_never_shows(monkeypatch, tmp_path):
    """A polygon without an interior point is reported by a saved figure, not plt.show."""
    import matplotlib.pyplot as plt

    import celeri

    monkeypatch.setattr(plt, "show", lambda *a, **k: pytest.fail("plt.show called"))

    config = celeri.get_config("./tests/configs/test_wna_config.json")
    config.repl = False
    segment, block, meshes, station, mogi, sar, los = celeri.read_data(config)
    station = celeri.process_station(station, config)
    segment = celeri.process_segment(segment, config, meshes)
    sar = celeri.process_sar(sar, config)
    # Move one block's interior point far away so that its polygon has none
    block = block.copy()
    block.loc[0, "interior_lon"] = block.loc[0, "interior_lon"] + 30.0

    celeri.assign_block_labels(
        segment=segment,
        station=station,
        block=block,
        mogi=mogi,
        sar=sar,
        los=los,
        debug_plot_dir=tmp_path,
    )

    assert list(tmp_path.glob("block_interior_points_polygon_*.png"))
