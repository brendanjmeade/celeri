from __future__ import annotations

import datetime
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from celeri.mesh import MeshConfig


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # Base directory for runs
    base_runs_folder: Path
    # Where to store the output of estimation runs (subdir of base_runs_folder)
    output_path: Path
    # Location of the file containing the blocks
    block_file_name: Path
    # Location of the file containing the station data
    station_file_name: Path
    # Location of the file containing the segments
    segment_file_name: Path
    # Location of the file containing the mogi data (empty for no Mogi sources)
    mogi_file_name: Path | None = None
    # Location of the sar data (empty for no SAR data)
    sar_file_name: Path | None = None
    # Location of the mesh parameters file
    mesh_parameters_file_name: Path
    # Mesh specific parameters for each of the meshes
    mesh_params: list[MeshConfig]
    # Location of a hdf5 file to cache elastic operators
    elastic_operator_cache_dir: Path | None = None

    atol: float = 0.0001
    block_constraint_weight: float = 1e24
    block_constraint_weight_max: float = 1e20
    block_constraint_weight_min: float = 1e20
    block_constraint_weight_steps: int = 1
    btol: float = 0.0001
    global_elastic_cutoff_distance: int = 2000000
    global_elastic_cutoff_distance_flag: int = 0
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
    pickle_save: bool = True
    plot_estimation_summary: bool = True
    plot_input_summary: bool = True
    printslipcons: int = 0
    quiver_scale: int = 100
    repl: bool = False

    ridge_param: int = 0
    slip_constraint_weight: float = 100000
    slip_constraint_weight_max: float = 100000
    slip_constraint_weight_min: float = 100000
    slip_constraint_weight_steps: int = 1
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
    tri_edge: list[int] = [0, 0, 0]
    tri_full_coupling: int = 0
    tri_slip_constraint_type: int = 0
    tri_slip_sign: list[int] = [0, 0]
    tri_smooth: float = 0.1
    unit_sigmas: int = 0

    coupling_bounds_total_percentage_satisfied_target: float | None = None
    coupling_bounds_max_iter: int | None = None

    # Only in tsts/global_config.json?
    patch_file_names: list[Path] | None = None
    poissons_ratio: float | None = None

    @property
    def run_name(self) -> str:
        return self.output_path.name

    @classmethod
    def from_file(cls, file_path: Path | str) -> Config:
        """Read config from a JSON file and return a Config instance.

        Args:
            file_path: Path to the JSON config file

        Returns:
            Config: A validated Config instance
        """
        file_path = Path(file_path)
        with file_path.open("r") as file:
            config_data = json.load(file)

        base_runs_folder = config_data.get("base_runs_folder", None)
        if base_runs_folder is None:
            raise ValueError("`base_runs_folder` missing in config")
        base_runs_folder = Path(base_runs_folder)
        config_data["output_path"] = _get_output_path(base_runs_folder)

        mesh_parameters_file_name = config_data.get("mesh_parameters_file_name", None)
        if mesh_parameters_file_name is None:
            mesh_params = []
        else:
            mesh_params = MeshConfig.from_file(mesh_parameters_file_name)
        config_data["mesh_params"] = mesh_params

        return Config.model_validate(config_data)


def _get_output_path(base: Path) -> Path:
    """Generate a unique timestamped output path within the base directory.

    This function creates a new path using the current UTC timestamp in ISO 8601 format
    (YYYY-MM-DDTHH-MM-SSZ). If the path already exists, it adds a numeric suffix
    to ensure uniqueness.

    Args:
        base: Base directory path where the output folder should be created

    Returns:
        Path: A unique timestamped path within the base directory

    Example:
        If the base directory is "/path/to/runs", the function might return
        "/path/to/runs/2023-01-01T12-34-56Z" or "/path/to/runs/2023-01-01T12-34-56Z_1"
        if the first path already exists.
    """
    # Format: ISO 8601 in UTC (YYYY-MM-DDTHH:MM:SSZ)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )

    # Create the output path with the timestamp
    output_path = base / timestamp

    # Ensure the path is unique by adding a suffix if needed
    suffix = 1
    original_path = output_path
    while output_path.exists():
        output_path = original_path.with_name(f"{original_path.name}_{suffix}")
        suffix += 1

    return output_path


def get_config(config_file_name) -> Config:
    """Get the configuration from a JSON file.

    Args:
        config_file_name (str): Path to the JSON file.

    Returns:
        Config: A Config object with the loaded configuration.
    """
    return Config.from_file(config_file_name)
