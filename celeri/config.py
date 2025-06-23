from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from celeri.mesh import MeshConfig


class Config(BaseModel):
    # Required fields (no defaults)
    base_runs_folder: str
    block_file_name: str | Path
    mesh_parameters_file_name: str
    segment_file_name: str

    mesh_params: list[MeshConfig]

    # Runtime fields (not in JSON)
    file_name: str | Path
    run_name: str
    output_path: str | Path

    station_file_name: str | None = None
    mogi_file_name: str | None = None
    atol: float = 0.0001
    block_constraint_weight: float = 1e24
    block_constraint_weight_max: float = 1e20
    block_constraint_weight_min: float = 1e20
    block_constraint_weight_steps: int = 1
    btol: float = 0.0001
    global_elastic_cutoff_distance: int = 2000000
    global_elastic_cutoff_distance_flag: int = 0
    h_matrix_min_pts_per_box: int = 20
    h_matrix_min_separation: float = 1.25
    h_matrix_tol: float = 1e-06
    iterative_solver: str = "lsmr"
    lat_range: tuple[float, float] = (30, 45)
    locking_depth_flag2: int = 25
    locking_depth_flag3: int = 15
    locking_depth_flag4: int = 10
    locking_depth_flag5: int = 5
    locking_depth_overide_value: int = 15
    locking_depth_override_flag: int = 0
    lon_range: tuple[float, float] = (130, 150)
    material_lambda: int = 30000000000
    material_mu: int = 30000000000
    n_iterations: int = 1
    operators_folder: str = "../data/operators/"
    pickle_save: bool = True
    plot_estimation_summary: bool = True
    plot_input_summary: bool = True
    printslipcons: int = 0
    quiver_scale: int = 100
    repl: bool = False

    # TODO(Adrian): Would it be enough to just have reuse_elastic_file?
    # And assume that if reuse_elastic_file is None, then reuse_elastic is False?
    reuse_elastic: bool = True
    save_elastic: bool = True
    reuse_elastic_file: str | None = None
    save_elastic_file: str | None = None

    ridge_param: int = 0
    sar_file_name: str | None = None
    slip_constraint_weight: float = 100000
    slip_constraint_weight_max: float = 100000
    slip_constraint_weight_min: float = 100000
    slip_constraint_weight_steps: int = 1
    slip_file_names: str | None = None
    smooth_type: int = 1
    snap_segments: int = 0
    solution_method: str = "backslash"
    solve_type: str = "hmatrix"
    station_data_weight: int = 1
    station_data_weight_max: int = 1
    station_data_weight_min: int = 1
    station_data_weight_steps: int = 1
    strain_method: int = 1
    tri_con_weight: int = 1000000
    tri_depth_tolerance: int = 0
    tri_edge: list[int] = Field(default_factory=lambda: [0, 0, 0])
    tri_full_coupling: int = 0
    tri_slip_constraint_type: int = 0
    tri_slip_sign: list[int] = Field(default_factory=lambda: [0, 0])
    tri_smooth: float = 0.1
    unit_sigmas: int = 0

    coupling_bounds_total_percentage_satisfied_target: float | None = None
    coupling_bounds_max_iter: int | None = None

    # Only in tsts/global_config.json?
    patch_file_names: list[str] | None = None
    poissons_ratio: float | None = None

    @classmethod
    def from_file(cls, file_path: str) -> Config:
        """Read config from a JSON file and return a Config instance.

        Args:
            file_path: Path to the JSON config file

        Returns:
            Config: A validated Config instance
        """
        with open(file_path) as f:
            config_data = json.load(f)

        config_data["file_name"] = file_path
        config_data["run_name"] = _get_new_folder_name()

        mesh_params = MeshConfig.from_file(config_data["mesh_parameters_file_name"])
        config_data["mesh_params"] = mesh_params
        if "base_runs_folder" not in config_data:
            raise ValueError("base_runs_folder is required in the config file.")

        config_data["output_path"] = os.path.join(
            config_data["base_runs_folder"], config_data["run_name"]
        )

        return cls(**config_data)


def _get_new_folder_name() -> str:
    """Generate a new folder name based on existing numeric folder names.

    This function scans the current directory for folders with numeric names,
    identifies the highest number, and returns a new folder name that is one
    greater than the highest number, formatted as a zero-padded 10-digit string.

    Returns:
        str: A new folder name as a zero-padded 10-digit string.

    Raises:
        ValueError: If no numeric folder names are found in the current directory.

    Example:
        If the current directory contains folders named "0000000001", "0000000002",
        and "0000000003", the function will return "0000000004".
    """
    # Get all folder names
    folder_names = glob.glob("./../runs/*/")

    # Remove trailing slashes
    folder_names = [folder_name.rstrip(os.sep) for folder_name in folder_names]

    # Remove anything before numerical folder name
    folder_names = [folder_name[-10:] for folder_name in folder_names]

    # Check to see if the folder name is a native run number
    folder_names_runs = list()
    for folder_name in folder_names:
        try:
            folder_names_runs.append(int(folder_name))
        except ValueError:
            pass

    # Get new folder name
    if len(folder_names_runs) == 0:
        new_folder_name = "0000000001"
    else:
        new_folder_number = np.max(folder_names_runs) + 1
        new_folder_name = f"{new_folder_number:010d}"

    return new_folder_name


def get_config(config_file_name) -> Config:
    """Get the configuration from a JSON file.

    Args:
        config_file_name (str): Path to the JSON file.

    Returns:
        Config: A Config object with the loaded configuration.
    """
    return Config.from_file(config_file_name)
