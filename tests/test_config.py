import json
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger
from pydantic import ValidationError

import celeri
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


def _japan_config():
    return celeri.get_config("./tests/configs/test_japan_config.json")


def test_kinematic_smoothing_length_scale_propagates():
    config = _japan_config()
    assert config.mesh_default_kinematic_smoothing_length_scale == 25.0
    assert all(m.kinematic_smoothing_length_scale is None for m in config.mesh_params)

    config.mesh_params[0].kinematic_smoothing_length_scale = 0.0
    config.propagate_mesh_defaults()
    assert config.mesh_params[0].kinematic_smoothing_length_scale == 0.0
    assert all(
        m.kinematic_smoothing_length_scale == 25.0 for m in config.mesh_params[1:]
    )


def test_deprecated_degree_smoothing_length_scale_is_converted():
    config = _japan_config()
    config.mesh_params[0].iterative_coupling_smoothing_length_scale = 0.25
    messages = []
    handler_id = logger.add(
        lambda message: messages.append(str(message)), level="WARNING"
    )
    try:
        config.propagate_mesh_defaults()
    finally:
        logger.remove(handler_id)

    assert config.mesh_params[0].kinematic_smoothing_length_scale == pytest.approx(
        0.25 * 111.19
    )
    assert any("iterative_coupling_smoothing_length_scale" in m for m in messages)
    assert all(
        m.kinematic_smoothing_length_scale == 25.0 for m in config.mesh_params[1:]
    )


def test_poissons_ratio_must_match_lame_parameters(tmp_path):
    """poissons_ratio is derived from material_lambda and material_mu; a value
    given in the config is only a consistency check (issue #536).
    """
    # lambda == mu gives nu = 0.25
    config = Config.model_validate(_config_data(tmp_path, poissons_ratio=0.25))
    assert config.poissons_ratio == 0.25

    with pytest.raises(ValidationError, match="poissons_ratio"):
        Config.model_validate(_config_data(tmp_path, poissons_ratio=0.3))

    # lambda == 2 mu gives nu = 1/3
    Config.model_validate(
        _config_data(
            tmp_path,
            material_lambda=60_000_000_000,
            material_mu=30_000_000_000,
            poissons_ratio=1 / 3,
        )
    )
    with pytest.raises(ValidationError, match="poissons_ratio"):
        Config.model_validate(
            _config_data(
                tmp_path,
                material_lambda=60_000_000_000,
                material_mu=30_000_000_000,
                poissons_ratio=1 / 6,
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"mcmc_station_sigma_nu": 0.0},
        {"mcmc_station_sigma_nu": -2.0},
        {"mcmc_station_sigma_nu": float("inf")},
        {"mcmc_station_sigma_scale": 0.0},
        {"mcmc_station_sigma_scale": float("nan")},
    ],
)
def test_station_sigma_likelihood_settings_must_be_positive(tmp_path, overrides):
    (field,) = overrides
    with pytest.raises(ValidationError, match=field):
        Config.model_validate(_config_data(tmp_path, **overrides))


def test_station_sigma_likelihood_defaults(tmp_path):
    config = Config.model_validate(_config_data(tmp_path))
    assert config.mcmc_station_sigma_nu == 6.0
    assert config.mcmc_station_sigma_scale is None  # learned
    config = Config.model_validate(
        _config_data(tmp_path, mcmc_station_sigma_nu=4, mcmc_station_sigma_scale=1.5)
    )
    assert config.mcmc_station_sigma_nu == 4.0
    assert config.mcmc_station_sigma_scale == 1.5


def test_station_weighting_defaults_to_unweighted(tmp_path):
    config = Config.model_validate(_config_data(tmp_path))
    assert config.mcmc_station_weighting is None
    config = Config.model_validate(
        _config_data(tmp_path, mcmc_station_weighting="voronoi")
    )
    assert config.mcmc_station_weighting == "voronoi"


def test_station_covariance_defaults(tmp_path):
    config = Config.model_validate(_config_data(tmp_path))
    assert config.mcmc_station_covariance_file_name is None
    assert config.mcmc_station_covariance_taper_radius == 3.0
    assert config.mcmc_station_covariance_same_block is True


def test_station_covariance_mode_requires_file(tmp_path):
    with pytest.raises(ValidationError, match="mcmc_station_covariance_file_name"):
        Config.model_validate(
            _config_data(tmp_path, mcmc_station_weighting="covariance")
        )


def test_station_covariance_file_name_is_relative_to_config(tmp_path):
    config = Config.model_validate(
        _config_data(
            tmp_path,
            mcmc_station_weighting="covariance",
            mcmc_station_covariance_file_name="../covariance/cov.json",
        )
    )
    config_dir = Path("./tests/configs").resolve()
    expected = (config_dir / "../covariance/cov.json").resolve()
    assert config.mcmc_station_covariance_file_name == expected


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_station_covariance_taper_radius_must_be_positive(tmp_path, radius):
    with pytest.raises(ValidationError, match="mcmc_station_covariance_taper_radius"):
        Config.model_validate(
            _config_data(tmp_path, mcmc_station_covariance_taper_radius=radius)
        )
