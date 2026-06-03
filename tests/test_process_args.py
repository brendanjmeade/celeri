"""Tests for command-line argument processing (celeri.cli.process_args)."""

import argparse
import json
from pathlib import Path

from celeri import get_config, process_args

CONFIG = Path("tests/configs/test_japan_config.json")
ORIGINAL_MESH_PARAMS = Path("tests/data/mesh/test_japan_mesh_parameters.json")


def _write_alt_mesh_params(dest: Path, n_modes: int) -> None:
    """Write a copy of the test mesh-parameters file with a distinctive value."""
    params = json.loads(ORIGINAL_MESH_PARAMS.read_text())
    for entry in params:
        entry["n_modes_strike_slip"] = n_modes
    dest.write_text(json.dumps(params, indent=4))


def test_mesh_parameters_file_name_override_reloads_mesh_params(tmp_path):
    """--mesh_parameters_file_name override must re-load config.mesh_params.

    Regression test for issue #462: process_args updated the path field but the
    eagerly-loaded mesh_params list (consumed by build_model) still reflected
    the original file.
    """
    # The original file uses n_modes_strike_slip == 50; make the alternate use 7.
    original_modes = json.loads(ORIGINAL_MESH_PARAMS.read_text())[0][
        "n_modes_strike_slip"
    ]
    assert original_modes != 7
    alt = tmp_path / "alt_mesh_parameters.json"
    _write_alt_mesh_params(alt, n_modes=7)

    config = get_config(CONFIG)
    # Sanity: before override, mesh_params come from the original file.
    assert config.mesh_params[0].n_modes_strike_slip == original_modes

    args = argparse.Namespace(
        config_file_name=str(CONFIG),
        mesh_parameters_file_name=str(alt),
    )
    process_args(config, args)

    # Both the path field AND the actual loaded parameters must reflect the alt file.
    assert config.mesh_parameters_file_name == str(alt)
    assert config.mesh_params, "mesh_params should be non-empty"
    for mesh_param in config.mesh_params:
        assert mesh_param.n_modes_strike_slip == 7
        assert mesh_param.file_name == alt.resolve()


def test_no_mesh_override_leaves_mesh_params_untouched():
    """Without the override, mesh_params stay as loaded from the config file."""
    config = get_config(CONFIG)
    expected = [m.n_modes_strike_slip for m in config.mesh_params]

    args = argparse.Namespace(
        config_file_name=str(CONFIG),
        mesh_parameters_file_name=None,
    )
    process_args(config, args)

    assert [m.n_modes_strike_slip for m in config.mesh_params] == expected
