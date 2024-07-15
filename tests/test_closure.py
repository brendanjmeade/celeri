import numpy as np
import os

from celeri.celeri_closure import run_block_closure, get_segment_labels, Polygon

import celeri


def test_closure():
    # First test a simple two triangle geometry.
    np_segments = np.array(
        [
            [[0, 0], [9, 1]],
            [[9, 1], [1, 10]],
            [[9, 1], [10, 11]],
            [[10, 11], [1, 10]],
            [[1, 10], [0, 0]],
        ],
        dtype=np.float64,
    )

    closure = run_block_closure(np_segments)
    labels = get_segment_labels(closure)

    correct_labels = np.array([[0, 1], [0, 2], [2, 1], [1, 2], [1, 0]])
    np.testing.assert_array_equal(labels, correct_labels)

    # Then shift the points to lie on the other side of the meridian by
    # subtracting 0.1 degree longitude.
    np_segments_meridian = np_segments.copy()
    np_segments_meridian[:, :, 0] -= 0.1
    np_segments_meridian[:, :, 0] = np.where(
        np_segments_meridian[:, :, 0] < 0,
        np_segments_meridian[:, :, 0] + 360,
        np_segments_meridian[:, :, 0],
    )

    closure_meridian = run_block_closure(np_segments_meridian)
    labels_meridian = get_segment_labels(closure_meridian)
    np.testing.assert_array_equal(labels_meridian, labels)
    # plot_segment_labels(np_segments, labels_meridian)

    assert closure_meridian.polygons[0].contains_point(np.array([0]), np.array([0.1]))
    assert closure_meridian.polygons[0].contains_point(np.array([2]), np.array([2]))
    assert closure_meridian.polygons[2].contains_point(np.array([8]), np.array([8]))
    assert closure_meridian.polygons[1].contains_point(np.array([50]), np.array([50]))


def test_interior_point_edge_crossing():
    # The first edge examined will be the one from (0,0) to (10,0). The
    # resulting interior pt should be ~(5, -1.5) which will intersect with the
    # opposite side of the rectangle and should be rejected. The next interior
    # pt tested will be ~(9.75, 0.5) which should be fine.
    vs = np.array([
        [0, 1],
        [10, 1],
        [10, 0],
        [0,0],
    ])
    p = Polygon(None, np.arange(4), vs)
    np.testing.assert_allclose(p.interior, (9.75, 0.5))

    vs = np.array([
        [0, 10],
        [10, 10],
        [10, 0],
        [0,0],
    ])
    p = Polygon(None, np.arange(4), vs)
    np.testing.assert_allclose(p.interior, (5, 7.5))


def test_exterior_block():
    # ordering has the outside on the right
    vs = np.array([
        [0, 1],
        [0,0],
        [10, 0],
        [10, 1],
    ])
    p = Polygon(None, np.arange(4), vs)

    # check that the block contains more than half the globe
    assert(p.area_steradians > 2 * np.pi)

    # check interior point tests.
    np.testing.assert_equal(
        p.contains_point(np.array([5.0, -5.0]), np.array([0.5, 0.5])),
        [False, True]
    )


def test_global_closure():
    """
    This check to make sure that the closure algorithm returns a known
    (and hopefully correct!) answer for the global closure problem.
    Right now all this does is check for the correct number of blocks and
    against one set of polygon edge indices
    """

    command_file_name = "./tests/test_closure_command.json"
    command = celeri.get_command(command_file_name)
    # logger = celeri.get_logger(command)
    segment, block, meshes, station, mogi, sar = celeri.read_data(command)

    station = celeri.process_station(station, command)
    segment = celeri.process_segment(segment, command, meshes)
    sar = celeri.process_sar(sar, command)
    closure, block = celeri.assign_block_labels(segment, station, block, mogi, sar)

    # Compare calculated edge indices with stored edge indices
    all_edge_idxs = np.array([])
    for i in range(closure.n_polygons()):
        all_edge_idxs = np.concatenate(
            (all_edge_idxs, np.array(closure.polygons[i].edge_idxs))
        )

    with open("./tests/test_closure_arrays.npy", "rb") as f:
        all_edge_idxs_stored = np.load(f)

    assert np.allclose(all_edge_idxs, all_edge_idxs_stored)

