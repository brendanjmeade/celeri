import numpy as np

from celeri.celeri_util import interleave2, interleave3


def test_interleave_keeps_float_values_from_any_argument():
    ints = np.array([1, 2], dtype=np.int64)
    floats = np.array([0.5, 0.7])

    np.testing.assert_array_equal(interleave2(ints, floats), [1.0, 0.5, 2.0, 0.7])
    np.testing.assert_array_equal(
        interleave3(ints, floats, ints), [1.0, 0.5, 1.0, 2.0, 0.7, 2.0]
    )
    assert interleave2(ints, ints).dtype == np.int64
    assert interleave3(ints == 1, floats > 0.6, ints == 2).dtype == bool
