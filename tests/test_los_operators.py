"""Tests for LOS (line-of-sight) operators.

These tests verify that the einsum-based projection in _project_operator_to_los
produces the same results as an explicit manual computation.
"""

import numpy as np

from celeri.operators import _project_operator_to_los


def _project_operator_to_los_manual(
    operator: np.ndarray, look_vectors: np.ndarray
) -> np.ndarray:
    """Manual implementation of LOS projection without np.einsum.

    This function computes the same result as _project_operator_to_los but uses
    explicit loops instead of einsum, making the computation more transparent.

    Given an operator G of shape (3 * n_los, n_cols) where rows are interleaved
    as [E0, N0, U0, E1, N1, U1, ...], and look vectors L of shape (n_los, 3),
    compute the projection so that:

        los_velocity[i] = L[i, 0] * G[3*i, :] + L[i, 1] * G[3*i+1, :] + L[i, 2] * G[3*i+2, :]

    Args:
        operator: Velocity operator, shape (3 * n_los, n_cols)
        look_vectors: Look vectors, shape (n_los, 3) with columns [E, N, U]

    Returns:
        Projected operator, shape (n_los, n_cols)
    """
    n_los = look_vectors.shape[0]
    n_cols = operator.shape[1]

    result = np.zeros((n_los, n_cols))

    for i in range(n_los):
        look_e = look_vectors[i, 0]
        look_n = look_vectors[i, 1]
        look_u = look_vectors[i, 2]

        row_e = operator[3 * i, :]
        row_n = operator[3 * i + 1, :]
        row_u = operator[3 * i + 2, :]

        result[i, :] = look_e * row_e + look_n * row_n + look_u * row_u

    return result


def test_project_operator_to_los_mixed_look_vectors():
    """Test projection with mixed look vectors."""
    n_los = 4
    n_cols = 6

    rng = np.random.default_rng(42)
    operator = rng.random((3 * n_los, n_cols))

    look_vectors = rng.random((n_los, 3))
    look_vectors /= np.linalg.norm(look_vectors, axis=1, keepdims=True)

    result_einsum = _project_operator_to_los(operator, look_vectors)
    result_manual = _project_operator_to_los_manual(operator, look_vectors)

    np.testing.assert_allclose(result_einsum, result_manual, rtol=1e-14)


def test_los_velocity_from_rotation_params():
    """Verify LOS velocity equals dot product of velocity with look vector.

    This is the key test: compute velocity at a point, then verify that
    projecting onto the look vector gives the same result as using the
    LOS operator directly.
    """
    n_los = 5
    n_blocks = 3
    n_cols = 3 * n_blocks  # omega_x, omega_y, omega_z per block

    rng = np.random.default_rng(999)

    velocity_operator = rng.random((3 * n_los, n_cols))

    look_vectors = rng.random((n_los, 3))
    look_vectors /= np.linalg.norm(look_vectors, axis=1, keepdims=True)

    rotation_params = rng.random(n_cols)

    velocities = velocity_operator @ rotation_params
    los_velocity_method1 = np.zeros(n_los)
    for i in range(n_los):
        vel_e = velocities[3 * i]
        vel_n = velocities[3 * i + 1]
        vel_u = velocities[3 * i + 2]
        los_velocity_method1[i] = (
            look_vectors[i, 0] * vel_e
            + look_vectors[i, 1] * vel_n
            + look_vectors[i, 2] * vel_u
        )

    los_operator = _project_operator_to_los(velocity_operator, look_vectors)
    los_velocity_method2 = los_operator @ rotation_params

    los_operator_manual = _project_operator_to_los_manual(
        velocity_operator, look_vectors
    )
    los_velocity_method3 = los_operator_manual @ rotation_params

    np.testing.assert_allclose(los_velocity_method1, los_velocity_method2, rtol=1e-14)
    np.testing.assert_allclose(los_velocity_method1, los_velocity_method3, rtol=1e-14)
    np.testing.assert_allclose(los_velocity_method2, los_velocity_method3, rtol=1e-14)
