"""Hotfix module providing numerically stable log-probabilities for censored Normal distributions.

This is a temporary workaround until https://github.com/pymc-devs/pymc/pull/7996 is merged.

The fix monkey-patches the MeasurableClip logprob to use a stable log survival function
for Normal distributions instead of the numerically unstable log(1 - exp(logcdf)).

Usage:
    import stable_censored_hotfix  # Just import to apply the fix

    with pm.Model():
        normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        y = pm.Censored("y", normal_dist, lower=None, upper=40.0, observed=data)
"""

import numpy as np
import pytensor.tensor as pt
from pymc.distributions.dist_math import normal_lccdf
from pymc.logprob.abstract import _logcdf, _logprob
from pymc.logprob.censoring import MeasurableClip
from pymc.logprob.utils import CheckParameterValue
from pytensor.tensor.variable import TensorConstant


def _stable_normal_logccdf(mu, sigma, value):
    """Numerically stable log complementary CDF (log survival function) for Normal.

    Uses erfcx-based implementation that is stable even in extreme tails.
    """
    return normal_lccdf(mu, sigma, value)


def _get_stable_logccdf(base_rv_op, base_rv_inputs, value, logcdf_fallback):
    """Get numerically stable log complementary CDF if available.

    For Normal distribution, uses the stable erfcx-based implementation.
    For other distributions, falls back to log1mexp(logcdf).
    """
    from pytensor.tensor.random.basic import NormalRV

    if isinstance(base_rv_op, NormalRV):
        # Normal distribution: use stable implementation
        # base_rv_inputs are: rng, size, mu, sigma
        rng, size, mu, sigma = base_rv_inputs
        return _stable_normal_logccdf(mu, sigma, value)
    else:
        # Fall back to potentially unstable computation
        return pt.log1mexp(logcdf_fallback)


def _stable_clip_logprob(op, values, base_rv, lower_bound, upper_bound, **kwargs):
    r"""Stable logprob of a clipped censored distribution.

    The probability is given by
    .. math::
        \begin{cases}
            0 & \text{for } x < lower, \\
            \text{CDF}(lower, dist) & \text{for } x = lower, \\
            \text{P}(x, dist) & \text{for } lower < x < upper, \\
            1-\text{CDF}(upper, dist) & \text {for} x = upper, \\
            0 & \text{for } x > upper,
        \end{cases}

    """
    (value,) = values

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logprob = _logprob(base_rv_op, (value,), *base_rv_inputs, **kwargs)
    logcdf = _logcdf(base_rv_op, value, *base_rv_inputs, **kwargs)

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    is_lower_bounded, is_upper_bounded = False, False
    if not (
        isinstance(upper_bound, TensorConstant) and np.all(np.isinf(upper_bound.value))
    ):
        is_upper_bounded = True

        # Use stable logccdf for Normal distributions instead of pt.log1mexp(logcdf)
        logccdf = _get_stable_logccdf(base_rv_op, base_rv_inputs, value, logcdf)

        # For right clipped discrete RVs, we need to add an extra term
        # corresponding to the pmf at the upper bound
        if base_rv.dtype.startswith("int"):
            logccdf = pt.logaddexp(logccdf, logprob)

        logprob = pt.switch(
            pt.eq(value, upper_bound),
            logccdf,
            pt.switch(pt.gt(value, upper_bound), -np.inf, logprob),
        )
    if not (
        isinstance(lower_bound, TensorConstant)
        and np.all(np.isneginf(lower_bound.value))
    ):
        is_lower_bounded = True
        logprob = pt.switch(
            pt.eq(value, lower_bound),
            logcdf,
            pt.switch(pt.lt(value, lower_bound), -np.inf, logprob),
        )

    if is_lower_bounded and is_upper_bounded:
        logprob = CheckParameterValue("lower_bound <= upper_bound")(
            logprob, pt.all(pt.le(lower_bound, upper_bound))
        )

    return logprob


def _apply_fix():
    """Apply the fix by overriding the singledispatch registry."""
    # Use the register decorator to replace the existing function
    _logprob.register(MeasurableClip, _stable_clip_logprob)


# Apply the fix on import
_apply_fix()


def verify_fix():
    """Verify that the stable implementation works correctly."""
    import pymc as pm
    import scipy.stats as st

    with pm.Model() as model:
        normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)
        pm.Censored("y", normal_dist, lower=None, upper=40.0)

    logp_fn = model.compile_logp()
    result = logp_fn({"y": 40.0})
    expected = st.norm(0, 1).logsf(40.0)

    if not np.isfinite(result):
        raise RuntimeError(
            f"Stable censored fix not working: got {result}, expected {expected}"
        )

    if not np.isclose(result, expected, rtol=1e-6):
        raise RuntimeError(
            f"Stable censored result mismatch: got {result}, expected {expected}"
        )

    return True


if __name__ == "__main__":
    print("Verifying stable censored fix...")
    verify_fix()
    print("✓ Stable censored fix is working correctly!")
    print("\nUsage:")
    print("  import stable_censored_hotfix  # Just import to apply the fix")
    print("  ")
    print("  with pm.Model():")
    print("      normal_dist = pm.Normal.dist(mu=0.0, sigma=1.0)")
    print("      y = pm.Censored('y', normal_dist, lower=None, upper=40.0)")
