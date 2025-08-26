from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from celeri.mesh import MeshConfig


class Config(BaseModel):
    # Forbid extra fields when reading from JSON
    model_config = ConfigDict(extra="forbid")

    file_name: Path | None
    """Location of the config file itself.

    Typically, this class is constructed via `get_config`, which will
    automatically populate this field. If this object is instead created
    directly, this field will default to `None`, in which case relative paths
    will be resolved relative to the current working directory.
    """

    base_runs_folder: Path
    """Base directory for runs."""

    output_path: Path
    """Where to store the output of estimation runs (subdir of base_runs_folder)"""

    block_file_name: Path
    """Location of the file containing the blocks"""

    station_file_name: Path
    """Location of the file containing the station data"""

    segment_file_name: Path
    """Location of the file containing the segments"""

    mogi_file_name: Path | None = None
    """Location of the file containing the mogi data (empty for no Mogi sources)"""

    sar_file_name: Path | None = None
    """Location of the sar data (empty for no SAR data)"""

    mesh_parameters_file_name: Path
    """Location of the mesh parameters file"""

    mesh_params: list[MeshConfig]
    """Mesh specific parameters for each of the meshes"""

    elastic_operator_cache_dir: Path | None = None
    """Location of a hdf5 file to cache elastic operators"""

    # Weights for various constraints and parameters in penalized linear inversion
    block_constraint_weight: float = 1e24
    block_constraint_weight_max: float = 1e20
    block_constraint_weight_min: float = 1e20
    block_constraint_weight_steps: int = 1
    slip_constraint_weight: float = 100000
    slip_constraint_weight_max: float = 100000
    slip_constraint_weight_min: float = 100000
    slip_constraint_weight_steps: int = 1
    station_data_weight: int = 1
    station_data_weight_max: int = 1
    station_data_weight_min: int = 1
    station_data_weight_steps: int = 1

    global_elastic_cutoff_distance: int = 2000000
    global_elastic_cutoff_distance_flag: int = 0

    # TODO(Brendan): They were marked as unused, but are still used in the code.
    locking_depth_flag2: int = 25
    locking_depth_flag3: int = 15
    locking_depth_flag4: int = 10
    locking_depth_flag5: int = 5
    locking_depth_overide_value: int = 15
    locking_depth_override_flag: int = 0

    # Plotting defaults
    lat_range: tuple[float, float] = (30, 45)
    lon_range: tuple[float, float] = (130, 150)
    plot_estimation_summary: bool = True
    plot_input_summary: bool = True
    quiver_scale: int = 100

    material_lambda: int = 30000000000
    material_mu: int = 30000000000
    poissons_ratio: float | None = None

    pickle_save: bool = True
    repl: bool = False

    snap_segments: int = 0
    solve_type: str = "hmatrix"
    strain_method: int = 1
    tri_con_weight: int = 1000000

    unit_sigmas: bool = False
    """Always report data uncertainties as one."""

    iterative_coupling_bounds_total_percentage_satisfied_target: float | None = None
    iterative_coupling_bounds_max_iter: int | None = None

    # Parameters of mcmc
    mcmc_tune: int | None = None
    mcmc_draws: int | None = None

    # Only in tsts/global_config.json?
    mesh_file_names: list[Path] | None = None

    @property
    def run_name(self) -> str:
        return self.output_path.name

    @classmethod
    def from_file(cls, file_name: Path | str) -> Config:
        """Read config from a JSON file and return a Config instance.

        Args:
            file_name: Path to the JSON config file

        Returns:
            Config: A validated Config instance
        """
        return get_config(file_name)

    @model_validator(mode="after")
    def relative_paths(self) -> Self:
        """Convert relative paths to absolute paths based on the config file location."""
        if self.file_name is not None:
            base_dir = self.file_name.parent

            for name in type(self).model_fields:
                if name == "file_name":
                    continue

                value = getattr(self, name)
                if isinstance(value, Path):
                    setattr(self, name, (base_dir / value).resolve())

        return self


def _get_output_path(base: Path) -> Path:
    """Generate a unique numbered output path within the base directory.

    This function creates a new path using an incremental counter format with 10 digits
    (e.g., 0000000001). It finds the highest existing numbered directory and creates
    the next one in sequence.

    Args:
        base: Base directory path where the output folder should be created

    Returns:
        Path: A unique numbered path within the base directory

    Example:
        If the base directory is "/path/to/runs" with existing directories "0000000001"
        and "0000000002", the function would return "/path/to/runs/0000000003".
    """
    base.mkdir(parents=True, exist_ok=True)
    names = [path.name for path in base.iterdir()]
    base_count = max((int(name) for name in names if name.isdigit()), default=0)
    for count in range(base_count + 1, base_count + 100):
        path = base / f"{count:010d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            count += 1
    else:
        raise RuntimeError(
            f"Failed to create a unique output path in {base} after 100 attempts."
        )


def get_config(file_name: Path | str) -> Config:
    """Read config from a JSON file and return a Config instance.

    Args:
        file_name: Path to the JSON config file

    Returns:
        Config: A validated Config instance
    """
    file_path = Path(file_name)
    with file_path.open("r") as file:
        config_data = json.load(file)

    base_runs_folder = config_data.get("base_runs_folder", None)
    if base_runs_folder is None:
        raise ValueError("`base_runs_folder` missing in config")
    base_runs_folder = (file_path.parent / Path(base_runs_folder)).resolve()
    config_data["output_path"] = _get_output_path(base_runs_folder)

    mesh_parameters_file_name = config_data.get("mesh_parameters_file_name", None)
    if mesh_parameters_file_name is None:
        mesh_params = []
    else:
        mesh_params = MeshConfig.from_file(file_path.parent / mesh_parameters_file_name)
    config_data["mesh_params"] = mesh_params
    config_data["file_name"] = file_path.resolve()

    return Config.model_validate(config_data)
