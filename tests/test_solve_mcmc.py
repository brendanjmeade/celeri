"""Tests for solve_mcmc module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from celeri.solve_mcmc import _constrain_field, _unconstrain_field


def _eval_if_tensor(x):
    """Evaluate PyTensor tensor, or return numpy array as-is."""
    return x.eval() if hasattr(x, "eval") else x


class TestConstrainUnconstrainRoundtrip:
    """Test that _unconstrain_field inverts _constrain_field."""

    def test_no_bounds_identity(self):
        """With no bounds, both functions are identity."""
        values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        constrained = _eval_if_tensor(_constrain_field(values, None, None))
        unconstrained = _unconstrain_field(constrained, None, None)
        assert_allclose(unconstrained, values)

    @pytest.mark.parametrize("lower", [0.0, -5.0, 10.0])
    @pytest.mark.parametrize("softplus_lengthscale", [0.1, 1.0, 5.0])
    def test_lower_bound_roundtrip(self, lower, softplus_lengthscale):
        """Roundtrip through lower-bounded transform."""
        # Test values within ~10 lengthscales of the bound (avoid extreme values
        # that map to exactly the bound, making the inverse undefined)
        L = softplus_lengthscale
        values = np.array([lower - 5 * L, lower - L, lower, lower + L, lower + 10 * L])
        constrained = _eval_if_tensor(
            _constrain_field(values, lower, None, softplus_lengthscale)
        )
        # Constrained values should all be >= lower (approaches but never equals)
        assert np.all(constrained >= lower)
        unconstrained = _unconstrain_field(
            constrained, lower, None, softplus_lengthscale
        )
        assert_allclose(unconstrained, values, rtol=1e-6, atol=1e-12)

    @pytest.mark.parametrize("upper", [0.0, -5.0, 10.0])
    @pytest.mark.parametrize("softplus_lengthscale", [0.1, 1.0, 5.0])
    def test_upper_bound_roundtrip(self, upper, softplus_lengthscale):
        """Roundtrip through upper-bounded transform."""
        # Test values within ~10 lengthscales of the bound (avoid extreme values
        # that map to exactly the bound, making the inverse undefined)
        L = softplus_lengthscale
        values = np.array([upper - 10 * L, upper - L, upper, upper + L, upper + 5 * L])
        constrained = _eval_if_tensor(
            _constrain_field(values, None, upper, softplus_lengthscale)
        )
        # Constrained values should all be <= upper (approaches but never equals)
        assert np.all(constrained <= upper)
        unconstrained = _unconstrain_field(
            constrained, None, upper, softplus_lengthscale
        )
        assert_allclose(unconstrained, values, rtol=1e-6, atol=1e-12)

    @pytest.mark.parametrize("lower,upper", [(0.0, 1.0), (-10.0, 10.0), (5.0, 15.0)])
    def test_two_bounds_roundtrip(self, lower, upper):
        """Roundtrip through two-bounded sigmoid transform."""
        midpoint = (lower + upper) / 2
        scale = upper - lower
        # Test values spanning from below lower to above upper
        values = np.array(
            [
                midpoint - 2 * scale,
                midpoint - scale,
                midpoint,
                midpoint + scale,
                midpoint + 2 * scale,
            ]
        )
        constrained = _eval_if_tensor(_constrain_field(values, lower, upper))
        # Constrained values should be strictly within (lower, upper)
        assert np.all(constrained > lower)
        assert np.all(constrained < upper)
        unconstrained = _unconstrain_field(constrained, lower, upper)
        assert_allclose(unconstrained, values, rtol=1e-6, atol=1e-12)

    def test_two_bounds_midpoint_fixed_point(self):
        """Midpoint maps to itself for two-bounded case."""
        lower, upper = -5.0, 15.0
        midpoint = (lower + upper) / 2
        constrained = _eval_if_tensor(
            _constrain_field(np.array([midpoint]), lower, upper)
        )
        assert_allclose(constrained, [midpoint], rtol=1e-10)
        unconstrained = _unconstrain_field(constrained, lower, upper)
        assert_allclose(unconstrained, [midpoint], rtol=1e-10)


class TestConstrainFieldProperties:
    """Test specific properties of _constrain_field."""

    def test_two_bounds_unit_slope_at_midpoint(self):
        """Derivative at midpoint should be 1 for two-bounded case."""
        lower, upper = 0.0, 10.0
        midpoint = (lower + upper) / 2
        eps = 1e-6
        values = np.array([midpoint - eps, midpoint + eps])
        constrained = _eval_if_tensor(_constrain_field(values, lower, upper))
        slope = (constrained[1] - constrained[0]) / (2 * eps)
        assert_allclose(slope, 1.0, rtol=1e-4)

    def test_lower_bound_asymptotic_identity(self):
        """Large positive values should pass through unchanged for lower bound."""
        lower = 0.0
        softplus_lengthscale = 1.0
        large_value = 100.0
        constrained = _eval_if_tensor(
            _constrain_field(np.array([large_value]), lower, None, softplus_lengthscale)
        )
        assert_allclose(constrained, [large_value], rtol=1e-3)

    def test_upper_bound_asymptotic_identity(self):
        """Large negative values should pass through unchanged for upper bound."""
        upper = 0.0
        softplus_lengthscale = 1.0
        large_negative = -100.0
        constrained = _eval_if_tensor(
            _constrain_field(
                np.array([large_negative]), None, upper, softplus_lengthscale
            )
        )
        assert_allclose(constrained, [large_negative], rtol=1e-3)


class TestUnconstrainFieldErrors:
    """Test error handling in _unconstrain_field."""

    def test_lower_bound_requires_lengthscale(self):
        """Should raise if softplus_lengthscale not provided for lower bound."""
        with pytest.raises(ValueError, match="softplus_lengthscale is required"):
            _unconstrain_field(np.array([1.0]), lower=0.0, upper=None)

    def test_upper_bound_requires_lengthscale(self):
        """Should raise if softplus_lengthscale not provided for upper bound."""
        with pytest.raises(ValueError, match="softplus_lengthscale is required"):
            _unconstrain_field(np.array([1.0]), lower=None, upper=2.0)
