from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from celeri.mesh import MeshConfig

Sqp2Objective = Literal[
    "expanded_norm2",
    "sum_of_squares",
    "qr_sum_of_squares",
    "svd_sum_of_squares",
    "norm2",
    "norm1",
    "huber",
]

EigenvectorAlgorithm = Literal["eigh", "eigsh"]

McmcStationVelocityMethod = Literal[
    "direct",
    "low_rank",
    "project_to_eigen",
]

McmcStationWeighting = Literal["voronoi",]


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

    force_recompute: bool = False
    """When True, recomputes all operators even if cached versions are present."""

    # Weights for various constraints and parameters in penalized linear inversion
    block_constraint_weight: float = 1e24
    slip_constraint_weight: float = 100000

    segment_slip_rate_regularization: float = 1.0
    """Weight for regularizing slip rates towards 0.
    Applied to segments with *s_rate_flag = 2 in the segment file.

    A value of zero indicates no regularization.
    This is used in `solve_sqp2` to help stabilize the inversion.

    We can interpret this as adding pseudo-observations of zero slip rate
    at all segments. The weight represents the ratio of pseudo-observation variance
    to observation variance. Higher values indicate we trust the
    zero slip rate assumption more relative to the actual observations.

    Reasonable values might be in the range of 0.01 to 10.
    """

    segment_slip_rate_bound_weight: float = 100.0
    """Weight for enforcing slip rate bounds at segments.

    This is used in `solve_sqp2` to enforce slip rate bounds softly.
    """

    segment_slip_rate_hard_bounds: bool = False
    """Enforce hard slip rate bounds at segments.

    This should be disabled when using soft slip rate bounds
    via `segment_slip_rate_bound_sigma`.

    The mcmc solver does not support hard bounds.
    """

    segment_slip_rate_regularization_sigma: float | None = 100
    """Like `segment_slip_rate_regularization`, but for use in `solve_mcmc`.

    The regularization is implemented as a Student's t prior with this
    standard deviation in mm/yr, and 5 degrees of freedom. This means that
    sigma has an inverse relationship with the severity of the regularization.
    """

    segment_slip_rate_bound_sigma: float = 1.0
    """Standard deviation for slip rate bounds at segments in mm/yr.

    This is used in `solve_mcmc` to implement soft slip rate bounds.

    Hard slip rate bounds are implemented as a censored normal likelihood
    with this standard deviation. Small values approach hard bounds,
    while larger values allow more violation of the bounds.

    We can interpret this as a measurment error of the slip rate bound
    itself.

    This config value can be overridden on a per-segment basis by including
    a `slip_rate_bound_sigma` column in the segment file. If present, each
    segment will use its own sigma value from that column (defaults to 1.0
    if the column is missing).
    """

    sqp2_objective: Sqp2Objective = "qr_sum_of_squares"
    """Objective function to use in `solve_sqp2`."""

    # Default values for segment specified locking depth overrides
    locking_depth_flag2: int = 25
    locking_depth_flag3: int = 15
    locking_depth_flag4: int = 10
    locking_depth_flag5: int = 5
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

    save_operators: bool = True
    """Whether to save full operator arrays when writing output.

    If False, only saves model and index. Operators will be loaded
    from the elastic operator cache when the run is opened, saving several GBs per run.
    Requires elastic_operator_cache_dir to be set.
    """

    snap_segments: int = 0
    solve_type: str = "hmatrix"
    tri_con_weight: int = 1000000

    unit_sigmas: bool = False
    """Always report data uncertainties as one."""

    iterative_coupling_bounds_total_percentage_satisfied_target: float | None = None
    iterative_coupling_bounds_max_iter: int | None = None

    # Parameters of mcmc
    mcmc_tune: int | None = 1000
    """Number of tuning steps in MCMC before sampling."""

    mcmc_draws: int | None = None
    """Number of MCMC samples to draw after tuning."""

    mcmc_seed: int | None = None
    """Random seed for MCMC sampling."""

    mcmc_chains: int = 1
    """Number of parallel MCMC chains to run."""

    mcmc_backend: Literal["numba", "jax"] = "numba"
    """Backend to use for MCMC computations."""

    mesh_default_eigenvector_algorithm: EigenvectorAlgorithm = "eigh"
    """Default algorithm for computing eigenvectors in mesh processing.

    This value is used as a fallback when a mesh configuration does not
    specify its own `eigenvector_algorithm`.

    Options:
    - "eigh": Dense eigenvalue decomposition (scipy.linalg.eigh). Faster for many modes.
    - "eigsh": Sparse eigenvalue decomposition (scipy.sparse.linalg.eigsh). Faster for few modes.

    Both have equivalent accuracy, but eigenvector signs may differ between algorithms.
    """

    mcmc_station_velocity_method: McmcStationVelocityMethod = "project_to_eigen"
    """Method for computing station velocities from slip rates in MCMC.

    - "direct": Direct matrix-vector multiplication
    - "low_rank": Truncated SVD approximation
    - "project_to_eigen": Eigenmode projection (default)

    Currently, only "project_to_eigen" supports streaming mode, which loads
    and processes each mesh's G matrix individually rather than holding all
    in memory at once.
    """

    mcmc_station_weighting: McmcStationWeighting | None = "voronoi"
    """Method for weighting station observations in MCMC likelihood.

    Options:
    - None: All stations weighted equally with weight one.
    - "voronoi": Weight by Voronoi cell area to reduce over-representation of clusters (default)

    The "voronoi" option is a pragmatic approach to handle spatially clustered stations
    without the computational cost of full spatial correlation modeling. It down-weights
    stations in dense clusters proportionally to their spacing. Use None if you want
    standard unweighted likelihood or if your network has uniform spatial coverage.
    """

    mcmc_station_effective_area: float = 10_000**2
    """Effective area (in m²) for station likelihood weighting in MCMC.

    This parameter controls how station observations are weighted in the likelihood
    based on their spatial density. Stations are weighted by their Voronoi cell area
    (computed on a sphere), but areas larger than this threshold are clipped to avoid
    over-weighting isolated stations.

    Interpretation:
    - Smaller values: Give more uniform weight to all stations, regardless of spacing
    - Larger values: Weight stations more strongly based on their Voronoi cell area
    - Default (50000²): Stations separated by ~50 km or more get equal weight

    The default value of 2.5e9 m² corresponds to a square roughly 50 km on a side.
    This means stations that are more than ~50 km apart will receive equal weighting,
    while stations in dense clusters will be down-weighted proportionally to avoid
    over-representing those regions.

    Only used when mcmc_station_weighting is "voronoi".
    """

    sqp2_annealing_enabled: bool = False
    """Enable annealing to search for a more optimal solution.

    The SQP2 solver iteratively tightens coupling constraints until convergence
    (no out-of-bounds values). When annealing is enabled, the solver continues
    beyond this point: for each value in `sqp2_annealing_schedule`, it loosens
    the constraints by that amount and runs another SQP pass, potentially
    converging to a solution with a lower residual.

    Set to False for a standard solve (stops after initial convergence).
    Set to True to enable annealing (takes longer, but may find a better solution).
    """

    sqp2_annealing_schedule: list[float] = [0.125, 0.125, 0.125]
    """Looseness values (mm/yr) for each annealing pass.

    After the solver converges with no out-of-bounds values, each value in this
    list triggers an additional SQP pass where the coupling constraints are
    temporarily widened by that amount (added to upper bounds, subtracted from
    lower bounds). This allows the solver to escape local minima and potentially
    find a more optimal solution.

    The default [0.125, 0.125, 0.125] performs three annealing passes, each
    loosening constraints by 0.125 mm/yr. An empty list [] is equivalent to
    disabling annealing (single-shot solve).

    Only used when `sqp2_annealing_enabled` is True.
    """

    include_vertical_velocity: bool = False
    """When True, include vertical velocity component in station velocity predictions.
    
    By default, only horizontal (east and north) velocity components are included in the model.
    Setting this to True will include the vertical (up) component in forward predictions.
    """

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

    @model_validator(mode="after")
    def validate_save_operators(self) -> Self:
        """Validate that elastic_operator_cache_dir is set when save_operators is False."""
        if not self.save_operators and self.elastic_operator_cache_dir is None:
            raise ValueError(
                "elastic_operator_cache_dir must be set when save_operators is False. "
                "Either set elastic_operator_cache_dir or set save_operators to True."
            )
        return self

    @model_validator(mode="after")
    def validate_no_mixed_constraints(self) -> Self:
        """Validate that meshes don't have both elastic and coupling constraints.

        This check only applies when solve_type is 'mcmc', as the MCMC solver
        does not support mixed constraint types.
        """
        if self.solve_type != "mcmc":
            return self

        for mesh_config in self.mesh_params:
            error = mesh_config.has_mixed_constraints()
            if error:
                raise ValueError(error)
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

    # Apply the top-level default eigenvector algorithm to mesh configs
    # that don't explicitly set their own
    mesh_default_eigenvector_algorithm = config_data.get(
        "mesh_default_eigenvector_algorithm", None
    )
    if mesh_default_eigenvector_algorithm is not None:
        for mesh_param in mesh_params:
            if "eigenvector_algorithm" not in mesh_param.model_fields_set:
                mesh_param.eigenvector_algorithm = mesh_default_eigenvector_algorithm

    config_data["mesh_params"] = mesh_params
    config_data["file_name"] = file_path.resolve()

    return Config.model_validate(config_data)
