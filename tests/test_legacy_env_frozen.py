"""Canary guarding the frozen ``legacy-mcmc`` pixi environment.

``pixi.toml`` only constrains ``arviz<1`` for this environment; the rest of the
stack is frozen by the committed lockfile alone. This test asserts a curated set
of packages still match their locked versions, so an accidental relock (a stray
``pixi update``) fails CI loudly instead of drifting the stack out from under the
legacy branches of :mod:`celeri._arviz_compat`.

When intentionally refreshing the stack, update ``EXPECTED_VERSIONS`` to match
``pixi list -e legacy-mcmc --locked``. Outside the legacy environment this test
is skipped.

LEGACY-MCMC: delete this whole file when legacy ArviZ<1/PyMC<6 support is
dropped (see celeri/_arviz_compat.py for the full cleanup checklist).
"""

from importlib.metadata import PackageNotFoundError, version

import pytest

# Curated canary set, kept in sync with `pixi list -e legacy-mcmc --locked`.
# The ArviZ/PyMC/nutpie/pytensor quartet defines the legacy boundary; the rest
# are representative transitive dependencies (numeric, stats, plotting, storage)
# that a `pixi update` would otherwise silently bump.
EXPECTED_VERSIONS = {
    "arviz": "0.23.4",
    "pymc": "5.28.4",
    "pytensor": "2.38.2",
    "nutpie": "0.16.8",
    "numpy": "2.4.3",
    "scipy": "1.17.1",
    "pandas": "3.0.2",
    "xarray": "2026.4.0",
    "matplotlib": "3.10.8",
    "numba": "0.65.0",
    "zarr": "3.1.6",
}


def _is_legacy_stack() -> bool:
    """True when we are running inside the frozen ArviZ<1 legacy environment."""
    try:
        return int(version("arviz").split(".")[0]) < 1
    except PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_legacy_stack(),
    reason="canary only applies to the frozen legacy-mcmc environment (ArviZ<1)",
)


@pytest.mark.parametrize("package, expected", sorted(EXPECTED_VERSIONS.items()))
def test_legacy_canary_version_frozen(package: str, expected: str):
    """Fail loudly if a canary dependency drifted from the frozen legacy lock."""
    try:
        installed = version(package)
    except PackageNotFoundError:
        pytest.fail(
            f"{package!r} is not installed in the legacy-mcmc environment; the "
            "frozen stack changed shape. Update EXPECTED_VERSIONS to match "
            "`pixi list -e legacy-mcmc --locked`."
        )
    assert installed == expected, (
        f"{package} is {installed}, expected {expected}. The frozen legacy-mcmc "
        "stack drifted -- most likely an unintended `pixi update`. Either revert "
        "the lockfile change for this environment, or, if the refresh is "
        "intentional, update EXPECTED_VERSIONS in this file to match "
        "`pixi list -e legacy-mcmc --locked`."
    )
