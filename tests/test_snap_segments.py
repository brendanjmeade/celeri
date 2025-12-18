from types import SimpleNamespace

import numpy as np
import pandas as pd

from celeri.scripts.snap_segments import make_default_segment, snap_segments


def test_make_default_segment_empty():
    """Test creating a default segment DataFrame with zero length."""
    seg = make_default_segment(0)
    assert len(seg) == 0
    assert "lon1" in seg.columns
    assert "lat1" in seg.columns
    assert "lon2" in seg.columns
    assert "lat2" in seg.columns
    assert "mesh_flag" in seg.columns
    assert "mesh_file_index" in seg.columns


def test_make_default_segment_with_length():
    """Test creating a default segment DataFrame with specified length."""
    seg = make_default_segment(5)
    assert len(seg) == 5
    assert all(seg.locking_depth == 15)
    assert all(seg.dip == 90)
    assert seg.loc[0, "name"] == "segment_0"
    assert seg.loc[4, "name"] == "segment_4"


def test_make_default_segment_default_values():
    """Test that numeric columns are initialized to zero except locking_depth and dip."""
    seg = make_default_segment(3)
    assert all(seg.ss_rate == 0)
    assert all(seg.ds_rate == 0)
    assert all(seg.ts_rate == 0)
    assert all(seg.mesh_flag == 0)
    assert all(seg.mesh_file_index == 0)


def make_simple_mesh():
    """Create a simple triangular mesh for testing."""
    return SimpleNamespace(
        top_elements=np.array([True]),
        ordered_edge_nodes=np.array([[0, 1], [1, 2], [2, 0]]),
        verts=np.array([[0, 1, 2]]),
        points=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        ),
    )


def test_snap_segments_no_mesh_segments_all_connected():
    """Test snap_segments with segments that don't reference mesh but share endpoints."""
    mesh = make_simple_mesh()

    # Create segments that form a closed loop, separate from the mesh
    # All endpoints appear twice (no hanging endpoints)
    segment = pd.DataFrame(
        {
            "name": ["seg1", "seg2", "seg3"],
            "lon1": [10.0, 11.0, 10.5],
            "lat1": [10.0, 10.0, 11.0],
            "lon2": [11.0, 10.5, 10.0],
            "lat2": [10.0, 11.0, 10.0],
            "dip": [90.0, 90.0, 90.0],
            "locking_depth": [15.0, 15.0, 15.0],
            "locking_depth_flag": [0, 0, 0],
            "ss_rate": [0.0, 0.0, 0.0],
            "ss_rate_sig": [0.0, 0.0, 0.0],
            "ss_rate_flag": [0, 0, 0],
            "ds_rate": [0.0, 0.0, 0.0],
            "ds_rate_sig": [0.0, 0.0, 0.0],
            "ds_rate_flag": [0, 0, 0],
            "ts_rate": [0.0, 0.0, 0.0],
            "ts_rate_sig": [0.0, 0.0, 0.0],
            "ts_rate_flag": [0, 0, 0],
            "mesh_file_index": [0, 0, 0],
            "mesh_flag": [0, 0, 0],
        }
    )

    result = snap_segments(segment, meshes=[mesh])

    # Original segments should be preserved (plus mesh edges added)
    # The 3 original segments have mesh_flag=0
    non_mesh_segs = result[result.mesh_flag == 0]
    assert len(non_mesh_segs) == 3
    np.testing.assert_array_almost_equal(non_mesh_segs.lon1.values, [10.0, 11.0, 10.5])
    np.testing.assert_array_almost_equal(non_mesh_segs.lat1.values, [10.0, 10.0, 11.0])


def test_snap_segments_replaces_mesh_segments():
    """Test that segments with mesh_flag!=0 are replaced with mesh edges."""
    # Create a simple triangular mesh with known geometry
    # Triangle with vertices at (0,0), (1,0), (0.5,1)
    mesh = SimpleNamespace(
        # Mark the first element as a top element
        top_elements=np.array([True]),
        # Edge nodes: edges of the triangle
        ordered_edge_nodes=np.array([[0, 1], [1, 2], [2, 0]]),
        # Single triangle with vertices 0,1,2
        verts=np.array([[0, 1, 2]]),
        # Vertex coordinates (lon, lat)
        points=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        ),
    )

    # Create a segment that traces the mesh (mesh_flag=1)
    # and another segment that connects to the mesh at (0,0) and extends away
    # The regular segment shares endpoint (0,0) with mesh, so only one end is hanging
    segment = pd.DataFrame(
        {
            "name": ["mesh_seg", "regular_seg"],
            "lon1": [0.0, 0.0],  # regular_seg starts at mesh vertex (0,0)
            "lat1": [0.0, 0.0],
            "lon2": [1.0, -1.0],  # regular_seg extends away from mesh
            "lat2": [0.0, -1.0],
            "dip": [90.0, 90.0],
            "locking_depth": [15.0, 15.0],
            "locking_depth_flag": [0, 0],
            "ss_rate": [0.0, 0.0],
            "ss_rate_sig": [0.0, 0.0],
            "ss_rate_flag": [0, 0],
            "ds_rate": [0.0, 0.0],
            "ds_rate_sig": [0.0, 0.0],
            "ds_rate_flag": [0, 0],
            "ts_rate": [0.0, 0.0],
            "ts_rate_sig": [0.0, 0.0],
            "ts_rate_flag": [0, 0],
            "mesh_file_index": [0, 0],
            "mesh_flag": [1, 0],  # First segment traces mesh, second doesn't
        }
    )

    result = snap_segments(segment, meshes=[mesh])

    # The mesh segment should be removed and replaced with mesh edge segments
    assert "mesh_flag" in result.columns

    # Check that we have the regular segment preserved (mesh_flag=0)
    regular_segs = result[result.mesh_flag == 0]
    assert len(regular_segs) == 1

    # The regular segment's endpoint at (0,0) should remain connected to mesh
    # Its hanging endpoint at (-1,-1) gets snapped to nearest mesh edge coord
    assert regular_segs.iloc[0].lon1 == 0.0
    assert regular_segs.iloc[0].lat1 == 0.0

    # Check that mesh edge segments were created (mesh_flag=1)
    mesh_segs = result[result.mesh_flag == 1]
    assert len(mesh_segs) > 0
    # All mesh segments should have locking_depth=-15 (set by snap_segments)
    assert all(mesh_segs.locking_depth == -15)


def test_snap_segments_hanging_endpoint_snapping():
    """Test that hanging endpoints get snapped to nearest mesh edge coordinate."""
    mesh = make_simple_mesh()

    # Create a segment with a hanging endpoint near the mesh
    # One end at (0,0) connects to mesh, other end at (-0.1, -0.1) is hanging
    segment = pd.DataFrame(
        {
            "name": ["hanging_seg"],
            "lon1": [0.0],
            "lat1": [0.0],
            "lon2": [-0.1],
            "lat2": [-0.1],
            "dip": [90.0],
            "locking_depth": [15.0],
            "locking_depth_flag": [0],
            "ss_rate": [0.0],
            "ss_rate_sig": [0.0],
            "ss_rate_flag": [0],
            "ds_rate": [0.0],
            "ds_rate_sig": [0.0],
            "ds_rate_flag": [0],
            "ts_rate": [0.0],
            "ts_rate_sig": [0.0],
            "ts_rate_flag": [0],
            "mesh_file_index": [0],
            "mesh_flag": [0],
        }
    )

    result = snap_segments(segment, meshes=[mesh])

    # The hanging endpoint should be snapped to the nearest mesh coordinate
    non_mesh_segs = result[result.mesh_flag == 0]
    assert len(non_mesh_segs) == 1

    # The hanging endpoint (-0.1, -0.1) should be snapped to (0, 0)
    # which is the nearest mesh edge coordinate
    assert non_mesh_segs.iloc[0].lon2 == 0.0
    assert non_mesh_segs.iloc[0].lat2 == 0.0
