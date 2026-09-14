import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from celeri.config import Config
from celeri.model import locking_depth_manager


def _config_data(tmp_path, **overrides):
    config_file = Path("./tests/configs/test_japan_config.json").resolve()
    data = json.loads(config_file.read_text())
    data.update(
        file_name=config_file,
        output_path=tmp_path,
        mesh_params=[],
    )
    data.update(overrides)
    return data


def test_locking_depth_override_requires_value(tmp_path):
    with pytest.raises(ValidationError, match="locking_depth_override_value"):
        Config.model_validate(_config_data(tmp_path, locking_depth_override_flag=1))


def test_locking_depth_override_applies_to_every_segment(tmp_path):
    segment = pd.DataFrame(
        {"locking_depth": [5.0, 25.0, 40.0], "locking_depth_flag": [0, 2, 0]}
    )

    config = Config.model_validate(_config_data(tmp_path))
    managed = locking_depth_manager(segment, config)
    assert managed.locking_depth.tolist() == [5.0, config.locking_depth_flag2, 40.0]

    config = Config.model_validate(
        _config_data(
            tmp_path, locking_depth_override_flag=1, locking_depth_override_value=12.0
        )
    )
    overridden = locking_depth_manager(segment, config)
    assert overridden.locking_depth.tolist() == [12.0, 12.0, 12.0]


def test_solve_type_is_constrained(tmp_path):
    data = _config_data(tmp_path)
    data.pop("solve_type")
    assert Config.model_validate(data).solve_type == "dense"

    with pytest.raises(ValidationError, match="solve_type"):
        Config.model_validate(_config_data(tmp_path, solve_type="hmatrix"))
