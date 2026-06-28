"""Compatibility shims spanning the ArviZ/PyMC major-version boundary.

celeri supports two stacks simultaneously:

* **Legacy** -- ArviZ < 1 + PyMC < 6.  An MCMC trace is an
  ``arviz.InferenceData``.
* **Current** -- ArviZ >= 1 + PyMC >= 6.  ArviZ dropped ``InferenceData`` in
  favour of ``xarray.DataTree``, which is what nutpie/PyMC now return.

Every API divergence between the two stacks is isolated here so the rest of
the codebase stays version-agnostic.  Each helper feature-detects rather than
parsing version strings, so it keeps working across point releases.

DEPRECATION PATHWAY
-------------------
This whole module exists only to keep the legacy stack alive.  Everything
that needs to change when dropping it is reachable from a single grep --
no Git archaeology required::

    git grep -n _arviz_compat        # module + every call site + manifest pins

Cleanup checklist, once ArviZ < 1 / PyMC < 6 support is no longer interesting:

1. Tighten the version floors back to the current-only stack (these lines
   carry a ``celeri/_arviz_compat.py`` comment, so the grep above finds them):
   ``pyproject.toml`` -> ``arviz>=1``, ``nutpie>=0.16.10``, ``pymc>=6,<7``;
   ``pixi.toml`` -> the same.
2. At each call site the grep reports, inline the "current" branch of the
   helper and drop the import:
     * ``solve_mcmc.py``  -> ``adaptation="low_rank"`` and
       ``backend=model.config.mcmc_backend``
     * ``solve.py`` save  -> ``self.mcmc_trace.to_zarr(...)`` directly
     * ``solve.py`` load  -> use the ``xarray.DataTree`` from
       ``xr.open_datatree(...)`` directly
     * ``filter_mcmc_chains_by_waic.py`` -> ``float(loo[...].elpd)``
3. Delete this module and ``tests/test_arviz_compat.py``.
4. Re-run ``pixi lock`` and the test suite.
"""

from __future__ import annotations

import inspect
from typing import Any

import arviz as az


def nutpie_mass_matrix_kwargs() -> dict[str, Any]:
    """Kwargs requesting low-rank modified mass-matrix adaptation from nutpie.

    nutpie >= 0.16.9 replaced the boolean ``low_rank_modified_mass_matrix=True``
    with ``adaptation="low_rank"`` (the old kwarg lingers as a deprecated alias
    that emits a ``FutureWarning``).  We feed nutpie whichever it understands.
    """
    import nutpie

    if "adaptation" in inspect.signature(nutpie.sample).parameters:
        return {"adaptation": "low_rank"}
    return {"low_rank_modified_mass_matrix": True}


def compute_log_likelihood_backend_kwargs(backend: str) -> dict[str, Any]:
    """Kwargs selecting the compute backend for ``pm.compute_log_likelihood``.

    PyMC >= 6 accepts ``backend="numba"`` (etc.) directly; PyMC < 6 routed the
    backend through ``compile_kwargs={"mode": BACKEND.upper()}``.
    """
    import pymc as pm

    if "backend" in inspect.signature(pm.compute_log_likelihood).parameters:
        return {"backend": backend}
    return {"compile_kwargs": {"mode": backend.upper()}}


def trace_to_datatree(trace: Any) -> Any:
    """Return ``trace`` as an ``xarray.DataTree`` ready for ``.to_zarr()``.

    A legacy ``arviz.InferenceData`` exposes ``.to_datatree()``; a current
    ``xarray.DataTree`` is already in the right form.
    """
    to_datatree = getattr(trace, "to_datatree", None)
    if to_datatree is not None:
        return to_datatree()
    return trace


def trace_from_datatree(datatree: Any) -> Any:
    """Convert a loaded ``xarray.DataTree`` back to the native trace type.

    On the legacy stack the rest of the code expects an
    ``arviz.InferenceData``, recovered via ``arviz.from_datatree``; on the
    current stack the ``DataTree`` is the native type and is returned as-is.
    """
    # ``from_datatree`` only exists on ArviZ < 1.  (Avoid probing
    # ``az.InferenceData``: ArviZ >= 1 keeps it as a stub that emits a
    # MigrationWarning on access.)
    from_datatree = getattr(az, "from_datatree", None)
    if from_datatree is not None:
        return from_datatree(datatree)
    return datatree


def loo_elpd(loo_result: Any) -> float:
    """Extract the LOO ELPD point estimate from an ``az.loo`` result.

    ArviZ >= 1 names it ``elpd``; ArviZ < 1 named it ``elpd_loo``.
    """
    if hasattr(loo_result, "elpd"):
        return float(loo_result.elpd)
    return float(loo_result.elpd_loo)
