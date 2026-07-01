"""Tests for the ArviZ/PyMC version-compatibility shims.

The helpers feature-detect at runtime, so both the legacy (ArviZ<1/PyMC<6)
and current (ArviZ>=1/PyMC>=6) branches can be exercised here regardless of
which stack is actually installed -- the legacy branches are driven with
stubs/monkeypatching.

LEGACY-MCMC: delete this whole file when legacy ArviZ<1/PyMC<6 support is
dropped (see celeri/_arviz_compat.py for the full cleanup checklist).
"""

from types import SimpleNamespace

import arviz as az

from celeri import _arviz_compat


class TestLooElpd:
    def test_current_uses_elpd(self):
        assert _arviz_compat.loo_elpd(SimpleNamespace(elpd=1.5)) == 1.5

    def test_legacy_uses_elpd_loo(self):
        # No ``elpd`` attribute -> falls back to ArviZ<1's ``elpd_loo``.
        assert _arviz_compat.loo_elpd(SimpleNamespace(elpd_loo=2.5)) == 2.5

    def test_returns_python_float(self):
        import numpy as np

        result = _arviz_compat.loo_elpd(SimpleNamespace(elpd=np.float64(3.0)))
        assert isinstance(result, float)


class TestTraceToDatatree:
    def test_legacy_inferencedata_is_converted(self):
        sentinel = object()
        idata = SimpleNamespace(to_datatree=lambda: sentinel)
        assert _arviz_compat.trace_to_datatree(idata) is sentinel

    def test_current_datatree_passes_through(self):
        # A current xarray.DataTree has no ``to_datatree`` method.
        datatree = SimpleNamespace()
        assert _arviz_compat.trace_to_datatree(datatree) is datatree


class TestTraceFromDatatree:
    def test_legacy_converts_via_from_datatree(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(
            az, "from_datatree", lambda dt: (sentinel, dt), raising=False
        )
        datatree = object()
        assert _arviz_compat.trace_from_datatree(datatree) == (sentinel, datatree)

    def test_current_returns_datatree_unchanged(self, monkeypatch):
        # Emulate ArviZ>=1, where ``az.from_datatree`` does not exist.
        monkeypatch.delattr(az, "from_datatree", raising=False)
        datatree = object()
        assert _arviz_compat.trace_from_datatree(datatree) is datatree


class TestNutpieMassMatrixKwargs:
    def test_current_signature_uses_adaptation(self, monkeypatch):
        # A nutpie.sample that accepts ``adaptation`` -> current kwarg.
        import nutpie

        monkeypatch.setattr(nutpie, "sample", lambda model, *, adaptation=None: None)
        assert _arviz_compat.nutpie_mass_matrix_kwargs() == {"adaptation": "low_rank"}

    def test_legacy_signature_uses_boolean(self, monkeypatch):
        # A nutpie.sample without ``adaptation`` -> legacy boolean kwarg.
        import nutpie

        monkeypatch.setattr(
            nutpie,
            "sample",
            lambda model, *, low_rank_modified_mass_matrix=False: None,
        )
        assert _arviz_compat.nutpie_mass_matrix_kwargs() == {
            "low_rank_modified_mass_matrix": True
        }


class TestComputeLogLikelihoodBackendKwargs:
    def test_current_signature_uses_backend(self, monkeypatch):
        # A compute_log_likelihood that accepts ``backend`` -> current kwarg.
        import pymc as pm

        monkeypatch.setattr(
            pm, "compute_log_likelihood", lambda model, *, backend=None: None
        )
        assert _arviz_compat.compute_log_likelihood_backend_kwargs("numba") == {
            "backend": "numba"
        }

    def test_legacy_signature_uses_compile_kwargs(self, monkeypatch):
        # A compute_log_likelihood without ``backend`` -> legacy compile_kwargs.
        import pymc as pm

        monkeypatch.setattr(
            pm, "compute_log_likelihood", lambda model, *, compile_kwargs=None: None
        )
        assert _arviz_compat.compute_log_likelihood_backend_kwargs("numba") == {
            "compile_kwargs": {"mode": "NUMBA"}
        }
