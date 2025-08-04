import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, overload

import addict
import h5py
import numpy as np
import scipy
from loguru import logger
from pandas import DataFrame
from scipy import spatial
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from celeri.celeri_util import (
    cartesian_vector_to_spherical_vector,
    euler_pole_covariance_to_rotation_vector_covariance,
    get_2component_index,
    get_cross_partials,
    get_keep_index_12,
    interleave2,
    interleave3,
    sph2cart,
)
from celeri.config import Config
from celeri.constants import (
    DEG_PER_MYR_TO_RAD_PER_YR,
    N_MESH_DIM,
)
from celeri.mesh import ByMesh, Mesh, MeshConfig
from celeri.model import (
    Model,
    merge_geodetic_data,
)
from celeri.output import dataclass_from_disk, dataclass_to_disk
from celeri.spatial import (
    get_block_motion_constraint_partials,
    get_block_strain_rate_to_velocities_partials,
    get_global_float_block_rotation_partials,
    get_mogi_to_velocities_partials,
    get_rotation_to_slip_rate_partials,
    get_rotation_to_velocities_partials,
    get_segment_station_operator_okada,
    get_shared_sides,
    get_tde_to_velocities_single_mesh,
    get_tri_shared_sides_distances,
    get_tri_smoothing_matrix,
    get_tri_smoothing_matrix_simple,
)


@dataclass
class TdeIndex:
    # One value per mesh
    n_tde: np.ndarray
    n_tde_constraints: np.ndarray
    start_tde_col: np.ndarray
    end_tde_col: np.ndarray
    start_tde_smoothing_row: np.ndarray
    end_tde_smoothing_row: np.ndarray
    start_tde_constraint_row: np.ndarray
    end_tde_constraint_row: np.ndarray
    start_tde_top_constraint_row: np.ndarray
    end_tde_top_constraint_row: np.ndarray
    start_tde_bot_constraint_row: np.ndarray
    end_tde_bot_constraint_row: np.ndarray
    start_tde_side_constraint_row: np.ndarray
    end_tde_side_constraint_row: np.ndarray
    start_tde_coup_constraint_row: np.ndarray
    end_tde_coup_constraint_row: np.ndarray
    start_tde_ss_slip_constraint_row: np.ndarray
    end_tde_ss_slip_constraint_row: np.ndarray
    start_tde_ds_slip_constraint_row: np.ndarray
    end_tde_ds_slip_constraint_row: np.ndarray

    @property
    def n_tde_total(self):
        return self.n_tde.sum()

    @property
    def n_tde_constraints_total(self):
        return self.n_tde_constraints.sum()

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir) / "tde_index"
        dataclass_to_disk(self, path)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "TdeIndex":
        """Load TDE operators from disk."""
        path = Path(input_dir) / "tde_index"
        return dataclass_from_disk(cls, path)


@dataclass
class EigenIndex:
    n_modes_mesh: np.ndarray
    start_col_eigen: np.ndarray
    end_col_eigen: np.ndarray
    start_tde_row_eigen: np.ndarray
    end_tde_row_eigen: np.ndarray
    start_tde_ss_slip_constraint_row_eigen: np.ndarray
    end_tde_ss_slip_constraint_row_eigen: np.ndarray
    start_tde_ds_slip_constraint_row_eigen: np.ndarray
    end_tde_ds_slip_constraint_row_eigen: np.ndarray
    start_tde_constraint_row_eigen: np.ndarray
    end_tde_constraint_row_eigen: np.ndarray
    qp_constraint_tde_rate_start_row_eigen: np.ndarray
    qp_constraint_tde_rate_end_row_eigen: np.ndarray
    qp_constraint_slip_rate_start_row_eigen: np.ndarray
    qp_constraint_slip_rate_end_row_eigen: np.ndarray

    # TODO(Brendan) Why no start_row_eigen?
    end_row_eigen: np.ndarray

    @property
    def n_eigen_total(self) -> int:
        return self.n_modes_mesh.sum()

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir)
        dataclass_to_disk(self, path)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "EigenIndex":
        """Load TDE operators from disk."""
        path = Path(input_dir)
        return dataclass_from_disk(cls, path)


@dataclass
class Index:
    n_blocks: int
    n_segments: int
    n_stations: int
    n_meshes: int
    n_mogis: int
    vertical_velocities: np.ndarray
    n_block_constraints: int
    station_row_keep_index: np.ndarray
    start_station_row: int
    end_station_row: int
    start_block_col: int
    end_block_col: int
    start_block_constraints_row: int
    end_block_constraints_row: int
    n_slip_rate_constraints: int
    start_slip_rate_constraints_row: int
    end_slip_rate_constraints_row: int

    n_strain_blocks: int
    n_block_strain_components: int
    start_block_strain_col: int
    end_block_strain_col: int

    start_mogi_col: int
    end_mogi_col: int
    slip_rate_bounds: np.ndarray
    tde: TdeIndex | None = None
    eigen: EigenIndex | None = None

    @property
    def n_slip_rate_bounds(self):
        return len(self.slip_rate_bounds)

    @property
    def n_operator_rows(self) -> int:
        base = (
            2 * self.n_stations
            + 3 * self.n_block_constraints
            + self.n_slip_rate_constraints
        )
        if self.tde is not None:
            base += 2 * self.tde.n_tde_total
            base += self.tde.n_tde_constraints_total
        return base

    @property
    def n_operator_cols(self) -> int:
        # TODO(Brendan): should there be the mogi/strain block terms here?
        # They were missing in one of the originial functions. I think in
        # most nodebooks those are zero.
        base = 3 * self.n_blocks + 3 * self.n_strain_blocks + self.n_mogis
        if self.tde is not None:
            base += 2 * self.tde.n_tde_total
        return base

    @property
    def n_operator_rows_eigen(self) -> int:
        assert self.eigen is not None
        base = (
            2 * self.n_stations
            + 3 * self.n_block_constraints
            + self.n_slip_rate_constraints
        )
        if self.tde is not None:
            base += self.tde.n_tde_constraints_total
        return base

    @property
    def n_operator_cols_eigen(self) -> int:
        assert self.eigen is not None
        return (
            3 * self.n_blocks
            + self.eigen.n_eigen_total
            + 3 * self.n_strain_blocks
            + self.n_mogis
        )

    @property
    def n_tde_total(self) -> int:
        if self.tde is None:
            return 0
        return self.tde.n_tde_total

    @property
    def n_tde_constraints_total(self) -> int:
        if self.tde is None:
            return 0
        return self.tde.n_tde_constraints_total

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir)

        if self.tde is not None:
            self.tde.to_disk(path / "tde")
        if self.eigen is not None:
            self.eigen.to_disk(path / "eigen")
        dataclass_to_disk(self, path, skip={"tde", "eigen"})

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "Index":
        """Load TDE operators from disk."""
        path = Path(input_dir)

        tde_path = path / "tde"
        if tde_path.exists():
            tde = TdeIndex.from_disk(tde_path)
        else:
            tde = None
        eigen_path = path / "eigen"
        if eigen_path.exists():
            eigen = EigenIndex.from_disk(eigen_path)
        else:
            eigen = None

        return dataclass_from_disk(cls, path, extra={"tde": tde, "eigen": eigen})


# TODO: Figure out what the types should be
@dataclass
class Assembly:
    data: Any
    sigma: Any
    index: Any


# TODO(Adrian): Maybe it would be better to use only one of
# FullTdeOperators or EigenTdeOperators?
# TODO(Adrian): Maybe some of the operators should be properties?
# We could add a reference to the model to Operators and then
# compute the operators on the fly when needed.


@dataclass
class TdeOperators:
    tde_to_velocities: ByMesh[np.ndarray]
    tde_slip_rate_constraints: ByMesh[np.ndarray]

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir)
        dataclass_to_disk(self, path)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "TdeOperators":
        """Load TDE operators from disk."""
        path = Path(input_dir)
        return dataclass_from_disk(cls, path)


@dataclass
class EigenOperators:
    eigenvectors_to_tde_slip: ByMesh[np.ndarray]
    eigen_to_velocities: ByMesh[np.ndarray]
    eigen_to_tde_bcs: ByMesh[np.ndarray]
    linear_gaussian_smoothing: ByMesh[np.ndarray]

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir)
        dataclass_to_disk(self, path)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "EigenOperators":
        """Load TDE operators from disk."""
        path = Path(input_dir)
        return dataclass_from_disk(cls, path)


@dataclass
class Operators:
    model: Model
    index: Index
    assembly: Assembly
    rotation_to_velocities: np.ndarray
    block_motion_constraints: np.ndarray
    slip_rate_constraints: np.ndarray
    rotation_to_slip_rate: np.ndarray
    block_strain_rate_to_velocities: np.ndarray
    mogi_to_velocities: np.ndarray
    slip_rate_to_okada_to_velocities: np.ndarray
    rotation_to_tri_slip_rate: dict[int, np.ndarray]
    rotation_to_slip_rate_to_okada_to_velocities: np.ndarray
    # TODO: Switch to csr_array?
    smoothing_matrix: dict[int, csr_matrix]
    global_float_block_rotation: np.ndarray
    tde: TdeOperators | None
    eigen: EigenOperators | None

    @overload
    def kinematic_slip_rate(
        self, parameters: np.ndarray, mesh_idx: int, smooth: bool
    ) -> np.ndarray:
        pass

    @overload
    def kinematic_slip_rate(
        self, parameters: np.ndarray, mesh_idx: None, smooth: bool
    ) -> dict[int, np.ndarray]:
        pass

    def kinematic_slip_rate(self, parameters, mesh_idx, smooth: bool):
        """Get the kinematic slip rates for a given mesh.

        Args:
            parameters: The model parameters to compute the slip rate from,
                as returned by `Operators.data_vector` or `Estimation.data_vector.
            mesh_idx: Index of the mesh to get the kinematic slip rate for.
                If None, return the kinematic slip rate for all meshes.
            smooth: Whether to apply smoothing to the slip rate.

        Returns:
            The kinematic slip rate as a numpy array.
        """
        if self.tde is None:
            raise ValueError("TDE operators are not set up.")
        if mesh_idx is None:
            return {
                idx: self.kinematic_slip_rate(parameters, mesh_idx=idx, smooth=smooth)
                for idx in self.model.segment_mesh_indices
            }
        if mesh_idx not in self.rotation_to_tri_slip_rate:
            raise ValueError(f"No kinematic velocities for mesh {mesh_idx}.")
        slip_rate = (
            self.rotation_to_tri_slip_rate[mesh_idx]
            @ parameters[0 : 3 * self.index.n_blocks]
        )
        if smooth:
            if mesh_idx not in self.smoothing_matrix:
                raise ValueError(f"No smoothing matrix for mesh {mesh_idx}.")
            if self.eigen is None:
                raise ValueError("Eigen operators are not set up.")
            smoothing_matrix = self.eigen.linear_gaussian_smoothing[mesh_idx]
            # Second dimension is for strike slip and dip slip
            slip_rate_ = slip_rate.reshape((smoothing_matrix.shape[-1], 2))
            slip_rate_smooth = smoothing_matrix @ slip_rate_
            slip_rate = slip_rate_smooth.ravel()
        return slip_rate

    # TODO: Maybe we can make it possible to always access
    # operators without tde or eigen, even if the operators
    # are set up for tde or eigen? We should make sure though
    # that the index is then correct for those use cases.
    @cached_property
    def full_dense_operator(self) -> np.ndarray:
        if self.tde is None:
            return _get_full_dense_operator_block_only(self)
        if self.eigen is not None:
            return get_full_dense_operator_eigen(self)
        return get_full_dense_operator(self)

    @property
    def data_vector(self) -> np.ndarray:
        if self.tde is None:
            return _get_data_vector_no_meshes(self.model, self.assembly, self.index)
        if self.eigen is not None:
            return _get_data_vector_eigen(self.model, self.assembly, self.index)
        return _get_data_vector(self.model, self.assembly, self.index)

    @property
    def weighting_vector(self) -> np.ndarray:
        if self.tde is None:
            return _get_weighting_vector_no_meshes(self.model, self.index)
        if self.eigen is not None:
            return _get_weighting_vector_eigen(self.model, self.index)
        return _get_weighting_vector(self.model, self.index)

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir)

        if self.tde is not None:
            self.tde.to_disk(path / "tde")
        if self.eigen is not None:
            self.eigen.to_disk(path / "eigen")

        self.index.to_disk(path / "index")
        self.model.to_disk(path / "model")

        # Save the smoothing matrix using scipy sparse format
        for mesh_idx, smoothing_matrix in self.smoothing_matrix.items():
            matrix_path = path / f"smoothing_matrix_{mesh_idx}.npz"
            scipy.sparse.save_npz(matrix_path, smoothing_matrix)

        # TODO: We ignore assembly for now. Do we need to save it?
        skip = {"tde", "eigen", "index", "model", "smoothing_matrix", "assembly"}

        dataclass_to_disk(self, path, skip=skip)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "Operators":
        """Load TDE operators from disk."""
        path = Path(input_dir)

        tde_path = path / "tde"
        if tde_path.exists():
            tde = TdeOperators.from_disk(tde_path)
        else:
            tde = None
        eigen_path = path / "eigen"
        if eigen_path.exists():
            eigen = EigenOperators.from_disk(eigen_path)
        else:
            eigen = None

        index = Index.from_disk(path / "index")
        model = Model.from_disk(path / "model")

        # Load smoothing matrices
        smoothing_matrix = {}
        for mesh_idx in range(index.n_meshes):
            matrix_path = path / f"smoothing_matrix_{mesh_idx}.npz"
            if matrix_path.exists():
                smoothing_matrix[mesh_idx] = scipy.sparse.load_npz(matrix_path)
            else:
                logger.warning(
                    f"Smoothing matrix for mesh {mesh_idx} not found at {matrix_path}"
                )

        extra = {
            "tde": tde,
            "eigen": eigen,
            "index": index,
            "model": model,
            "smoothing_matrix": smoothing_matrix,
            "assembly": Assembly(
                data=addict.Dict(), sigma=addict.Dict(), index=addict.Dict()
            ),
        }

        return dataclass_from_disk(cls, path, extra=extra)


@dataclass
class _OperatorBuilder:
    model: Model
    index: Index | None = None
    assembly: Assembly | None = None
    rotation_to_velocities: np.ndarray | None = None
    block_motion_constraints: np.ndarray | None = None
    slip_rate_constraints: np.ndarray | None = None
    rotation_to_slip_rate: np.ndarray | None = None
    block_strain_rate_to_velocities: np.ndarray | None = None
    mogi_to_velocities: np.ndarray | None = None
    slip_rate_to_okada_to_velocities: np.ndarray | None = None
    eigenvectors_to_tde_slip: dict[int, np.ndarray] = field(default_factory=dict)
    rotation_to_tri_slip_rate: dict[int, np.ndarray] = field(default_factory=dict)
    linear_gaussian_smoothing: dict[int, np.ndarray] = field(default_factory=dict)
    tde_to_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    smoothing_matrix: dict[int, csr_matrix] = field(default_factory=dict)
    tde_slip_rate_constraints: dict[int, np.ndarray] = field(default_factory=dict)
    eigen_to_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    eigen_to_tde_bcs: dict[int, np.ndarray] = field(default_factory=dict)
    rotation_to_slip_rate_to_okada_to_velocities: np.ndarray | None = None
    global_float_block_rotation: np.ndarray | None = None

    def finalize_basic(self) -> Operators:
        assert self.index is not None
        assert self.assembly is not None
        assert self.rotation_to_velocities is not None
        assert self.block_motion_constraints is not None
        assert self.slip_rate_constraints is not None
        assert self.rotation_to_slip_rate is not None
        assert self.block_strain_rate_to_velocities is not None
        assert self.mogi_to_velocities is not None
        assert self.slip_rate_to_okada_to_velocities is not None
        assert self.rotation_to_tri_slip_rate is not None
        assert self.smoothing_matrix is not None
        assert self.tde_slip_rate_constraints is not None
        assert self.global_float_block_rotation is not None
        assert self.rotation_to_slip_rate_to_okada_to_velocities is not None

        return Operators(
            model=self.model,
            index=self.index,
            assembly=self.assembly,
            rotation_to_velocities=self.rotation_to_velocities,
            block_motion_constraints=self.block_motion_constraints,
            slip_rate_constraints=self.slip_rate_constraints,
            rotation_to_slip_rate=self.rotation_to_slip_rate,
            block_strain_rate_to_velocities=self.block_strain_rate_to_velocities,
            mogi_to_velocities=self.mogi_to_velocities,
            slip_rate_to_okada_to_velocities=self.slip_rate_to_okada_to_velocities,
            rotation_to_tri_slip_rate=self.rotation_to_tri_slip_rate,
            smoothing_matrix=self.smoothing_matrix,
            global_float_block_rotation=self.global_float_block_rotation,
            rotation_to_slip_rate_to_okada_to_velocities=self.rotation_to_slip_rate_to_okada_to_velocities,
            eigen=None,
            tde=None,
        )

    def finalize_tde(self) -> Operators:
        operators = self.finalize_basic()

        assert self.tde_to_velocities is not None
        assert self.tde_slip_rate_constraints is not None

        tde = TdeOperators(
            tde_to_velocities=self.tde_to_velocities,
            tde_slip_rate_constraints=self.tde_slip_rate_constraints,
        )

        operators.tde = tde
        return operators

    def finalize_eigen(self) -> Operators:
        operators = self.finalize_tde()

        assert self.linear_gaussian_smoothing is not None
        assert self.eigen_to_velocities is not None
        assert self.eigen_to_tde_bcs is not None
        assert self.eigenvectors_to_tde_slip is not None

        eigen = EigenOperators(
            eigenvectors_to_tde_slip=self.eigenvectors_to_tde_slip,
            linear_gaussian_smoothing=self.linear_gaussian_smoothing,
            eigen_to_velocities=self.eigen_to_velocities,
            eigen_to_tde_bcs=self.eigen_to_tde_bcs,
        )

        operators.eigen = eigen
        return operators


def build_operators(model: Model, *, eigen: bool = True, tde: bool = True) -> Operators:
    if eigen and not tde:
        raise ValueError("eigen openrators require tde")
    assembly = Assembly(data=addict.Dict(), sigma=addict.Dict(), index=addict.Dict())
    operators = _OperatorBuilder(model)
    operators.assembly = assembly

    assembly = merge_geodetic_data(assembly, model.station, model.sar)

    # Get all elastic operators for segments and TDEs
    _store_elastic_operators(model, operators)

    # Get TDE smoothing operators
    _store_all_mesh_smoothing_matrices(model, operators)

    # Block rotation to velocity operator
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        model.station, len(model.block)
    )

    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        model.station
    )

    # Soft block motion constraints
    operators.block_motion_constraints = _store_block_motion_constraints(
        model, assembly
    )

    # Soft slip rate constraints
    operators.slip_rate_constraints = get_slip_rate_constraints(model, assembly)

    # Rotation vectors to slip rate operator
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        model.segment, model.block
    )

    # Internal block strain rate operator
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        model.block, model.station, model.segment
    )

    # Mogi source operator
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        model.mogi, model.station, model.config
    )

    # Soft TDE boundary condition constraints
    _store_tde_slip_rate_constraints(model, operators)

    if eigen:
        index = _get_index_eigen(model, assembly)
        operators.index = index

        # Get KL modes for each mesh
        _store_eigenvectors_to_tde_slip(model, operators)
    else:
        index = _get_index(model, assembly)
        operators.index = index

    # Get rotation to TDE kinematic slip rate operator for all meshes tied to segments
    _store_tde_coupling_constraints(model, operators)

    # Insert block rotations and elastic velocities from fully locked segments
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )

    if eigen:
        # EIGEN Eigenvector to velocity matrix
        for i in range(index.n_meshes):
            # Eliminate vertical elastic velocities
            tde_keep_row_index = get_keep_index_12(
                operators.tde_to_velocities[i].shape[0]
            )
            tde_keep_col_index = get_keep_index_12(
                operators.tde_to_velocities[i].shape[1]
            )

            # Create eigenvector to velocities operator
            operators.eigen_to_velocities[i] = (
                -operators.tde_to_velocities[i][tde_keep_row_index, :][
                    :, tde_keep_col_index
                ]
                @ operators.eigenvectors_to_tde_slip[i]
            )

    # Get smoothing operators for post-hoc smoothing of slip
    _store_gaussian_smoothing_operator(model.meshes, operators, index)
    if eigen:
        return operators.finalize_eigen()
    if tde:
        return operators.finalize_tde()
    return operators.finalize_basic()


def _store_gaussian_smoothing_operator(
    meshes: list[Mesh], operators: _OperatorBuilder, index: Index
):
    for i in range(index.n_meshes):
        points = np.vstack((meshes[i].lon_centroid, meshes[i].lat_centroid)).T

        length_scale = meshes[i].config.iterative_coupling_smoothing_length_scale

        # TODO(Adrian) this default should be in the config
        if length_scale is None:
            length_scale = 0.25

        # Compute pairwise Euclidean distance matrix
        D = spatial.distance_matrix(points, points)

        # Define Gaussian weight function
        W = np.exp(-(D**2) / (2 * length_scale**2))
        # TODO(Adrian) make this configurable
        W[W < 1e-8] = 0.0

        # Normalize rows so each row sums to 1
        W /= W.sum(axis=1, keepdims=True)

        operators.linear_gaussian_smoothing[i] = W


def _hash_elastic_operator_input(
    meshes: list[MeshConfig], segment: DataFrame, station: DataFrame, config: Config
):
    """Create a hash from segment, station DataFrames and elastic parameters from config.

    This allows us to check if we need to recompute elastic operators or can use cached ones.

    Args:
        segment: DataFrame containing fault segment information
        station: DataFrame containing station information
        config: Config object containing material parameters

    Returns:
        str: Hash string representing the input data
    """
    # Convert dataframes to string representations
    segment_str = segment.to_json()
    station_str = station.to_json()
    assert isinstance(segment_str, str)
    assert isinstance(station_str, str)

    # Get material parameters
    material_params = f"{config.material_mu}_{config.material_lambda}"

    mesh_configs = [mesh.model_dump_json() for mesh in meshes]

    # Combine all inputs and create hash
    combined_input = "_".join(
        [segment_str, station_str, material_params, *mesh_configs]
    )
    return hashlib.blake2b(combined_input.encode()).hexdigest()[:16]


def _store_elastic_operators(
    model: Model,
    operators: _OperatorBuilder,
    *,
    tde: bool = True,
):
    """Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces.

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        meshes (List): Geometries of meshes
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        config (Dict): All config data
    """
    config = model.config
    meshes = model.meshes
    segment = model.segment
    station = model.station

    cache = None

    if config.elastic_operator_cache_dir is not None:
        if tde:
            input_hash = _hash_elastic_operator_input(
                [mesh.config for mesh in meshes],
                segment,
                station,
                config,
            )
        else:
            input_hash = _hash_elastic_operator_input([], segment, station, config)
        cache = config.elastic_operator_cache_dir / f"{input_hash}.hdf5"
        if cache.exists():
            logger.info(f"Using precomputed elastic operators from {cache}")
            hdf5_file = h5py.File(cache, "r")
            operators.slip_rate_to_okada_to_velocities = np.array(
                hdf5_file.get("slip_rate_to_okada_to_velocities")
            )
            if tde:
                for i in range(len(meshes)):
                    operators.tde_to_velocities[i] = np.array(
                        hdf5_file.get("tde_to_velocities_" + str(i))
                    )
            hdf5_file.close()
            return

        logger.info("Precomputed elastic operator file not found, computing operators")
    else:
        logger.info(
            "No precomputed elastic operator file specified, computing operators"
        )

    operators.slip_rate_to_okada_to_velocities = get_segment_station_operator_okada(
        segment, station, config
    )

    if tde:
        for i in range(len(meshes)):
            logger.info(
                f"Start: TDE slip to velocity calculation for mesh: {meshes[i].file_name}"
            )
            operators.tde_to_velocities[i] = get_tde_to_velocities_single_mesh(
                meshes, station, config, mesh_idx=i
            )
            logger.success(
                f"Finish: TDE slip to velocity calculation for mesh: {meshes[i].file_name}"
            )

    # Save elastic to velocity matrices
    if cache is None:
        return
    logger.info("Saving elastic operators in cache")
    cache.parent.mkdir(parents=True, exist_ok=True)
    hdf5_file = h5py.File(cache, "w")

    hdf5_file.create_dataset(
        "slip_rate_to_okada_to_velocities",
        data=operators.slip_rate_to_okada_to_velocities,
    )
    if tde:
        for i in range(len(meshes)):
            hdf5_file.create_dataset(
                "tde_to_velocities_" + str(i),
                data=operators.tde_to_velocities[i],
            )
    hdf5_file.close()


def _store_all_mesh_smoothing_matrices(model: Model, operators: _OperatorBuilder):
    """Build smoothing matrices for each of the triangular meshes
    stored in meshes.
    """
    # TODO(Adrian): The first part should be in model.py?

    meshes = model.meshes
    for i in range(len(meshes)):
        # Get smoothing operator for a single mesh.
        meshes[i].share = get_shared_sides(meshes[i].verts)
        meshes[i].tri_shared_sides_distances = get_tri_shared_sides_distances(
            meshes[i].share,
            meshes[i].x_centroid,
            meshes[i].y_centroid,
            meshes[i].z_centroid,
        )
        operators.smoothing_matrix[i] = get_tri_smoothing_matrix(
            meshes[i].share, meshes[i].tri_shared_sides_distances
        )


def _store_all_mesh_smoothing_matrices_simple(
    model: Model, operators: _OperatorBuilder
):
    """Build smoothing matrices for each of the triangular meshes
    stored in meshes
    These are the simple not distance weighted meshes.
    """
    meshes = model.meshes
    for i in range(len(meshes)):
        # Get smoothing operator for a single mesh.
        meshes[i].share = get_shared_sides(meshes[i].verts)
        operators.smoothing_matrix[i] = get_tri_smoothing_matrix_simple(
            meshes[i].share, N_MESH_DIM
        )


def _store_tde_slip_rate_constraints(model: Model, operators: _OperatorBuilder):
    """Construct TDE slip rate constraint matrices for each mesh.
    These are identity matrices, used to set TDE slip rates on
    or coupling fractions on elements lining the edges of the mesh,
    as controlled by input parameters
    top_slip_rate_constraint,
    bot_slip_rate_constraint,
    side_slip_rate_constraint,.

    and at other elements with indices specified as
    ss_slip_constraint_idx,
    ds_slip_constraint_idx,

    Args:
        meshes (List): list of mesh dictionaries
        operators (Dict): dictionary of linear operators
    """
    meshes = model.meshes
    for i in range(len(meshes)):
        # Empty constraint matrix
        tde_slip_rate_constraints = np.zeros((2 * meshes[i].n_tde, 2 * meshes[i].n_tde))
        # Counting index
        start_row = 0
        end_row = 0

        # Process boundary constraints (top, bottom, side)
        boundary_constraints = [
            ("top", meshes[i].config.top_slip_rate_constraint, meshes[i].top_elements),
            ("bot", meshes[i].config.bot_slip_rate_constraint, meshes[i].bot_elements),
            (
                "side",
                meshes[i].config.side_slip_rate_constraint,
                meshes[i].side_elements,
            ),
        ]

        for name, constraint_value, elements in boundary_constraints:
            if constraint_value > 0:
                indices = np.asarray(np.where(elements))
                slip_idx = get_2component_index(indices)
                setattr(meshes[i], f"{name}_slip_idx", slip_idx)
                start_row = end_row
                end_row = start_row + len(slip_idx)
                tde_slip_rate_constraints[start_row:end_row, slip_idx] = np.eye(
                    len(slip_idx)
                )

        # Eliminate blank rows
        sum_constraint_columns = np.sum(tde_slip_rate_constraints, 1)
        tde_slip_rate_constraints = tde_slip_rate_constraints[
            sum_constraint_columns > 0, :
        ]
        operators.tde_slip_rate_constraints[i] = tde_slip_rate_constraints
        # Total number of slip constraints:
        # 2 for each element that has coupling constrained (top, bottom, side, specified indices)
        # 1 for each additional slip component that is constrained (specified indices)

        # TODO: Number of total constraints is determined by just finding 1 in the
        # constraint array. This could cause an error when the index Dict is constructed,
        # if an individual element has a constraint imposed, but that element is also
        # a constrained edge element. Need to build in some uniqueness tests.
        meshes[i].n_tde_constraints = np.sum(sum_constraint_columns > 0)


def _store_tde_coupling_constraints(model: Model, operators: _OperatorBuilder):
    """Get partials relating block motion to TDE slip rates for coupling constraints."""
    # for mesh_idx in range(len(meshes)):
    # Loop only over meshes that are tied to fault segments.  This *should*
    # eliminate touching CMI meshes which have problems with this function
    # becase it assumes that a mesh is tied to segments.
    for mesh_idx in range(np.max(model.segment.mesh_file_index) + 1):
        operators.rotation_to_tri_slip_rate[mesh_idx] = (
            get_rotation_to_tri_slip_rate_partials(model, mesh_idx)
        )
        # Trim tensile rows
        tri_keep_rows = get_keep_index_12(
            np.shape(operators.rotation_to_tri_slip_rate[mesh_idx])[0]
        )
        operators.rotation_to_tri_slip_rate[mesh_idx] = (
            operators.rotation_to_tri_slip_rate[mesh_idx][tri_keep_rows, :]
        )


def get_rotation_to_tri_slip_rate_partials(model: Model, mesh_idx: int) -> np.ndarray:
    """Calculate partial derivatives relating relative block motion to TDE slip rates
    for a single mesh. Called within a loop from get_tde_coupling_constraints().
    """
    # TODO(Adrian): This function modifies model.meshes[mesh_idx] in place.
    # Should part of this be moved to model.py?
    n_blocks = len(model.block)
    tri_slip_rate_partials = np.zeros((3 * model.meshes[mesh_idx].n_tde, 3 * n_blocks))

    # Generate strikes and dips for elements using same sign convention as segments
    # Element strikes are 0-180
    # Dip direction is 0-90 for E dips, 90-180 for W dips
    ew_switch = np.zeros_like(model.meshes[mesh_idx].strike)
    ns_switch = np.zeros_like(model.meshes[mesh_idx].strike)
    tristrike = np.array(model.meshes[mesh_idx].strike)
    tristrike[model.meshes[mesh_idx].strike > 180] -= 180
    tridip = np.array(model.meshes[mesh_idx].dip)
    tridip[model.meshes[mesh_idx].strike > 180] = (
        180 - tridip[model.meshes[mesh_idx].strike > 180]
    )
    # Find subset of segments that are replaced by this mesh
    seg_replace_idx = np.where(
        (model.segment.mesh_flag != 0) & (model.segment.mesh_file_index == mesh_idx)
    )
    # Find closest segment midpoint to each element centroid, using scipy.spatial.cdist
    model.meshes[mesh_idx].closest_segment_idx = seg_replace_idx[0][
        cdist(
            np.array(
                [
                    model.meshes[mesh_idx].lon_centroid,
                    model.meshes[mesh_idx].lat_centroid,
                ]
            ).T,
            np.array(
                [
                    model.segment.mid_lon[seg_replace_idx[0]],
                    model.segment.mid_lat[seg_replace_idx[0]],
                ]
            ).T,
        ).argmin(axis=1)
    ]
    # Add segment labels to elements
    model.meshes[mesh_idx].east_labels = np.array(
        model.segment.east_labels[model.meshes[mesh_idx].closest_segment_idx]
    )
    model.meshes[mesh_idx].west_labels = np.array(
        model.segment.west_labels[model.meshes[mesh_idx].closest_segment_idx]
    )

    # Find rotation partials for each element
    for el_idx in range(model.meshes[mesh_idx].n_tde):
        # Project velocities from Cartesian to spherical coordinates at element centroids
        row_idx = 3 * el_idx
        column_idx_east = 3 * model.meshes[mesh_idx].east_labels[el_idx]
        column_idx_west = 3 * model.meshes[mesh_idx].west_labels[el_idx]
        R = get_cross_partials(
            [
                model.meshes[mesh_idx].x_centroid[el_idx],
                model.meshes[mesh_idx].y_centroid[el_idx],
                model.meshes[mesh_idx].z_centroid[el_idx],
            ]
        )
        (
            vel_north_to_omega_x,
            vel_east_to_omega_x,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 0],
            R[1, 0],
            R[2, 0],
            model.meshes[mesh_idx].lon_centroid[el_idx],
            model.meshes[mesh_idx].lat_centroid[el_idx],
        )
        (
            vel_north_to_omega_y,
            vel_east_to_omega_y,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 1],
            R[1, 1],
            R[2, 1],
            model.meshes[mesh_idx].lon_centroid[el_idx],
            model.meshes[mesh_idx].lat_centroid[el_idx],
        )
        (
            vel_north_to_omega_z,
            vel_east_to_omega_z,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 2],
            R[1, 2],
            R[2, 2],
            model.meshes[mesh_idx].lon_centroid[el_idx],
            model.meshes[mesh_idx].lat_centroid[el_idx],
        )
        # This correction gives -1 for strikes > 90
        # Equivalent to the if statement in get_rotation_to_slip_rate_partials
        sign_corr = -np.sign(tristrike[el_idx] - 90)

        # Project about fault strike
        unit_x_parallel = sign_corr * np.cos(np.deg2rad(90 - tristrike[el_idx]))
        unit_y_parallel = sign_corr * np.sin(np.deg2rad(90 - tristrike[el_idx]))
        unit_x_perpendicular = sign_corr * np.sin(np.deg2rad(tristrike[el_idx] - 90))
        unit_y_perpendicular = sign_corr * np.cos(np.deg2rad(tristrike[el_idx] - 90))
        # Project by fault dip
        scale_factor = 1.0 / (np.cos(np.deg2rad(model.meshes[mesh_idx].dip[el_idx])))
        slip_rate_matrix = np.array(
            [
                [
                    (
                        unit_x_parallel * vel_east_to_omega_x
                        + unit_y_parallel * vel_north_to_omega_x
                    ),
                    (
                        unit_x_parallel * vel_east_to_omega_y
                        + unit_y_parallel * vel_north_to_omega_y
                    ),
                    (
                        unit_x_parallel * vel_east_to_omega_z
                        + unit_y_parallel * vel_north_to_omega_z
                    ),
                ],
                [
                    scale_factor
                    * (
                        unit_x_perpendicular * vel_east_to_omega_x
                        + unit_y_perpendicular * vel_north_to_omega_x
                    ),
                    scale_factor
                    * (
                        unit_x_perpendicular * vel_east_to_omega_y
                        + unit_y_perpendicular * vel_north_to_omega_y
                    ),
                    scale_factor
                    * (
                        unit_x_perpendicular * vel_east_to_omega_z
                        + unit_y_perpendicular * vel_north_to_omega_z
                    ),
                ],
                [0, 0, 0],
            ]
        )
        # Additional sign correction needs to compare TDE strike and corresponding segment strike
        # If they're on different sides of an E-W line, we need to apply a negative sign
        # This effectively flips the east and west labels
        ns_switch[el_idx] = np.sign(
            90
            - np.abs(
                tristrike[el_idx]
                - model.segment.azimuth[
                    model.meshes[mesh_idx].closest_segment_idx[el_idx]
                ]
            )
        )
        ew_switch[el_idx] = (
            ns_switch[el_idx]
            * np.sign(tristrike[el_idx] - 90)
            * np.sign(
                model.segment.azimuth[
                    model.meshes[mesh_idx].closest_segment_idx[el_idx]
                ]
                - 90
            )
        )
        # ew_switch = 1
        # Insert this element's partials into operator
        tri_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_east : column_idx_east + 3
        ] = ew_switch[el_idx] * slip_rate_matrix
        tri_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_west : column_idx_west + 3
        ] = -ew_switch[el_idx] * slip_rate_matrix
    return tri_slip_rate_partials


def _store_block_motion_constraints(model: Model, assembly: Assembly) -> np.ndarray:
    """Applying a priori block motion constraints."""
    # TODO This modifies the assembly object in place. Fix this.

    block = model.block
    block_constraint_partials = get_block_motion_constraint_partials(block)
    assembly.index.block_constraints_idx = np.where(block.rotation_flag == 1)[0]

    assembly.data.n_block_constraints = len(assembly.index.block_constraints_idx)
    assembly.data.block_constraints = np.zeros(block_constraint_partials.shape[0])
    assembly.sigma.block_constraints = np.zeros(block_constraint_partials.shape[0])
    if assembly.data.n_block_constraints > 0:
        (
            assembly.data.block_constraints[0::3],
            assembly.data.block_constraints[1::3],
            assembly.data.block_constraints[2::3],
        ) = sph2cart(
            block.euler_lon[assembly.index.block_constraints_idx],
            block.euler_lat[assembly.index.block_constraints_idx],
            np.deg2rad(block.rotation_rate[assembly.index.block_constraints_idx]),
        )
        euler_pole_covariance_all = np.diag(
            np.concatenate(
                (
                    np.deg2rad(
                        block.euler_lat_sig[assembly.index.block_constraints_idx]
                    ),
                    np.deg2rad(
                        block.euler_lon_sig[assembly.index.block_constraints_idx]
                    ),
                    np.deg2rad(
                        block.rotation_rate_sig[assembly.index.block_constraints_idx]
                    ),
                )
            )
        )
        (
            assembly.sigma.block_constraints[0::3],
            assembly.sigma.block_constraints[1::3],
            assembly.sigma.block_constraints[2::3],
        ) = euler_pole_covariance_to_rotation_vector_covariance(
            assembly.data.block_constraints[0::3],
            assembly.data.block_constraints[1::3],
            assembly.data.block_constraints[2::3],
            euler_pole_covariance_all,
        )
    assembly.sigma.block_constraint_weight = model.config.block_constraint_weight
    return block_constraint_partials


def get_slip_rate_constraints(model: Model, assembly: Assembly) -> np.ndarray:
    # TODO This modifies the assembly object in place. Fix this.

    segment = model.segment
    n_total_slip_rate_contraints = (
        np.sum(segment.ss_rate_flag.values)
        + np.sum(segment.ds_rate_flag.values)
        + np.sum(segment.ts_rate_flag.values)
    )
    if n_total_slip_rate_contraints > 0:
        logger.info(f"Found {n_total_slip_rate_contraints} slip rate constraints")
        for i in range(len(segment.lon1)):
            if segment.ss_rate_flag[i] == 1:
                logger.info(
                    "Strike-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + f"{segment.ss_rate[i]:.2f}"
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + f"{segment.ss_rate_sig[i]:.2f}"
                    + " (mm/yr)"
                )
            if segment.ds_rate_flag[i] == 1:
                logger.info(
                    "Dip-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + f"{segment.ds_rate[i]:.2f}"
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + f"{segment.ds_rate_sig[i]:.2f}"
                    + " (mm/yr)"
                )
            if segment.ts_rate_flag[i] == 1:
                logger.info(
                    "Tensile-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + f"{segment.ts_rate[i]:.2f}"
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + f"{segment.ts_rate_sig[i]:.2f}"
                    + " (mm/yr)"
                )
    else:
        logger.info("No slip rate constraints")

    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(
        segment, model.block
    )

    slip_rate_constraint_flag = interleave3(
        segment.ss_rate_flag, segment.ds_rate_flag, segment.ts_rate_flag
    )
    assembly.index.slip_rate_constraints = np.where(slip_rate_constraint_flag == 1)[0]
    assembly.data.n_slip_rate_constraints = len(assembly.index.slip_rate_constraints)

    assembly.data.slip_rate_constraints = interleave3(
        segment.ss_rate, segment.ds_rate, segment.ts_rate
    )

    assembly.data.slip_rate_constraints = assembly.data.slip_rate_constraints[
        assembly.index.slip_rate_constraints
    ]

    assembly.sigma.slip_rate_constraints = interleave3(
        segment.ss_rate_sig, segment.ds_rate_sig, segment.ts_rate_sig
    )

    assembly.sigma.slip_rate_constraints = assembly.sigma.slip_rate_constraints[
        assembly.index.slip_rate_constraints
    ]

    slip_rate_constraint_partials = slip_rate_constraint_partials[
        assembly.index.slip_rate_constraints, :
    ]
    assembly.sigma.slip_rate_constraint_weight = model.config.slip_constraint_weight
    return slip_rate_constraint_partials


def get_slip_rake_constraints(model: Model, assembly: Assembly) -> np.ndarray:
    segment = model.segment

    n_total_slip_rake_contraints = np.sum(segment.rake_flag.values)
    if n_total_slip_rake_contraints > 0:
        logger.info(f"Found {n_total_slip_rake_contraints} slip rake constraints")
        for i in range(len(segment.lon1)):
            if segment.rake_flag[i] == 1:
                logger.info(
                    "Rake constraint on "
                    + segment.name[i].strip()
                    + ": rake = "
                    + f"{segment.rake[i]:.2f}"
                    + ", constraint strike = "
                    + f"{segment.rake_strike[i]:.2f}"
                    + ", 1-sigma uncertainty = +/-"
                    + f"{segment.rake_sig[i]:.2f}"
                )
    else:
        logger.info("No slip rake constraints")
    # To keep this a standalone function, let's calculate the full set of slip rate partials
    # TODO: Check how get_slip_rate_constraints is called to see if we need to recalculate the full set of partials, or if we can reuse a previous calculation
    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(
        segment, model.block
    )
    # Figure out effective rake. This is a simple correction of the rake data by the calculated strike of the segment
    # The idea is that the source of the rake constraint will include its own strike (and dip), which may differ from the model segment geometry
    # TODO: Full three-dimensional rotation of rake vector, based on strike and dip of constraint source?
    effective_rakes = segment.rake[segment.rake_flag] + (
        segment.strike[segment.rake_flag] - segment.rake_strike[segment.rake_flag]
    )

    # Find indices of constrained segments
    assembly.index.slip_rake_constraints = np.where(segment.rake_flag == 1)[0]
    assembly.data.n_slip_rake_constraints = len(assembly.index.slip_rake_constraints)

    # Get component indices of slip rate partials
    rake_constraint_component_indices = get_2component_index(
        assembly.index.slip_rake_constraints
    )
    # Rotate slip partials about effective rake. We just want to use the second row (second basis vector) of a full rotation matrix, because we want to set slip in that direction to zero as a constraint
    slip_rake_constraint_partials = (
        np.cos(np.radians(effective_rakes))
        * slip_rate_constraint_partials[rake_constraint_component_indices[0::2]]
        + np.sin(np.radians(effective_rakes))
        * slip_rate_constraint_partials[rake_constraint_component_indices[1::2]]
    )

    # Constraint data is all zeros, because we're setting slip perpendicular to the rake direction equal to zero
    assembly.data.slip_rake_constraints = np.zeros(
        assembly.data.n_total_slip_rake_contraints
    )

    # Insert sigmas into assembly dict
    assembly.sigma.slip_rake_constraints = segment.rake_sig

    # Using the same weighting here as for slip rate constraints.
    assembly.sigma.slip_rake_constraint_weight = model.config.slip_constraint_weight
    return slip_rake_constraint_partials


def _get_data_vector_no_meshes(
    model: Model, assembly: Assembly, index: Index
) -> np.ndarray:
    data_vector = np.zeros(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )

    # Add GPS stations to data vector
    data_vector[index.start_station_row : index.end_station_row] = interleave2(
        assembly.data.east_vel, assembly.data.north_vel
    )

    # Add block motion constraints to data vector
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * assembly.data.block_constraints
    )

    # Add slip rate constraints to data vector
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = assembly.data.slip_rate_constraints

    return data_vector


def _get_data_vector(model: Model, assembly: Assembly, index: Index) -> np.ndarray:
    assert index.tde is not None

    data_vector = np.zeros(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )

    # Add GPS stations to data vector
    data_vector[index.start_station_row : index.end_station_row] = interleave2(
        assembly.data.east_vel, assembly.data.north_vel
    )

    # Add block motion constraints to data vector
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * assembly.data.block_constraints
    )

    # Add slip rate constraints to data vector
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = assembly.data.slip_rate_constraints
    return data_vector


def _get_data_vector_eigen(
    model: Model, assembly: Assembly, index: Index
) -> np.ndarray:
    assert index.tde is not None
    assert index.eigen is not None

    data_vector = np.zeros(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    # Add GPS stations to data vector
    data_vector[index.start_station_row : index.end_station_row] = interleave2(
        assembly.data.east_vel, assembly.data.north_vel
    )

    # Add block motion constraints to data vector
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * assembly.data.block_constraints
    )

    # Add slip rate constraints to data vector
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = assembly.data.slip_rate_constraints
    return data_vector


def _get_weighting_vector(model: Model, index: Index):
    assert index.tde is not None

    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (model.station.east_sig**2), 1 / (model.station.north_sig**2)
    )
    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = model.config.block_constraint_weight
    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = model.config.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    for i in range(len(model.meshes)):
        # Insert smoothing weight into weighting vector
        weighting_vector[
            index.tde.start_tde_smoothing_row[i] : index.tde.end_tde_smoothing_row[i]
        ] = model.meshes[i].config.smoothing_weight * np.ones(2 * index.tde.n_tde[i])
        weighting_vector[
            index.tde.start_tde_constraint_row[i] : index.tde.end_tde_constraint_row[i]
        ] = model.config.tri_con_weight * np.ones(index.tde.n_tde_constraints[i])
    return weighting_vector


def _get_weighting_vector_no_meshes(model: Model, index: Index) -> np.ndarray:
    station = model.station
    # NOTE: Consider combining with above
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )
    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )
    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = 1.0
    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = model.config.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    return weighting_vector


def get_weighting_vector_single_mesh_for_col_norms(
    model: Model, index: Index, mesh_index: int
) -> np.ndarray:
    assert index.tde is not None

    station = model.station
    mesh = model.meshes[mesh_index]
    config = model.config

    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 2 * index.tde.n_tde[mesh_index]
        + index.tde.n_tde_constraints[mesh_index]
    )

    weighting_vector[0 : 2 * index.n_stations] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )

    weighting_vector[
        2 * index.n_stations : 2 * index.n_stations + 2 * index.tde.n_tde[mesh_index]
    ] = mesh.config.smoothing_weight * np.ones(2 * index.tde.n_tde[mesh_index])

    weighting_vector[2 * index.n_stations + 2 * index.tde.n_tde[mesh_index] : :] = (
        config.tri_con_weight * np.ones(index.tde.n_tde_constraints[mesh_index])
    )

    return weighting_vector


def _get_weighting_vector_eigen(model: Model, index: Index) -> np.ndarray:
    assert index.tde is not None
    assert index.eigen is not None

    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (model.station.east_sig**2), 1 / (model.station.north_sig**2)
    )

    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = model.config.block_constraint_weight

    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = model.config.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    # TODO: Need to think about constraints weights
    # This is the only place where any individual constraint weights enter
    # I'm only using on of them: meshes[i].bot_slip_rate_weight
    for i in range(len(model.meshes)):
        weighting_vector[
            index.eigen.start_tde_constraint_row_eigen[
                i
            ] : index.eigen.end_tde_constraint_row_eigen[i]
        ] = model.meshes[i].config.bot_slip_rate_weight * np.ones(
            index.tde.n_tde_constraints[i]
        )

    return weighting_vector


def _get_full_dense_operator_block_only(operators: Operators) -> np.ndarray:
    index = operators.index

    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints,
            3 * index.n_blocks,
        )
    )

    # Insert block rotations and elastic velocities from fully locked segments
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_col : index.end_block_col,
    ] = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )

    # Insert block motion constraints
    operator[
        index.start_block_constraints_row : index.end_block_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.block_motion_constraints

    # Insert slip rate constraints
    operator[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.slip_rate_constraints
    return operator


def get_full_dense_operator(operators: Operators) -> np.ndarray:
    # TODO: This should either take an OperatorBuilder as input and
    # store the final operator *or* return the final operator, but not
    # modify the operator in place
    index = operators.index
    model = operators.model
    assert index.tde is not None

    assert operators.eigen is None
    assert operators.tde is not None

    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + 2 * index.n_tde_total
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.n_block_strain_components
            + index.n_mogis
            + 2 * index.n_tde_total,
        )
    )

    # DEBUG:
    # IPython.embed(banner1="")

    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_col : index.end_block_col,
    ] = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )

    # Insert block motion constraints
    operator[
        index.start_block_constraints_row : index.end_block_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.block_motion_constraints

    # Insert slip rate constraints
    operator[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.slip_rate_constraints

    # Insert all TDE operators
    for i in range(len(model.meshes)):
        # Insert TDE to velocity matrix
        tde_keep_row_index = get_keep_index_12(
            operators.tde.tde_to_velocities[i].shape[0]
        )
        tde_keep_col_index = get_keep_index_12(
            operators.tde.tde_to_velocities[i].shape[1]
        )
        operator[
            index.start_station_row : index.end_station_row,
            index.tde.start_tde_col[i] : index.tde.end_tde_col[i],
        ] = -operators.tde.tde_to_velocities[i][tde_keep_row_index, :][
            :, tde_keep_col_index
        ]

        # Insert TDE smoothing matrix
        smoothing_keep_index = get_keep_index_12(
            operators.tde.tde_to_velocities[i].shape[1]
        )
        operator[
            index.tde.start_tde_smoothing_row[i] : index.tde.end_tde_smoothing_row[i],
            index.tde.start_tde_col[i] : index.tde.end_tde_col[i],
        ] = operators.smoothing_matrix[i].toarray()[smoothing_keep_index, :][
            :, smoothing_keep_index
        ]

        # Insert TDE slip rate constraints into estimation operator
        # These are just the identity matrices, and we'll insert any block motion constraints next
        operator[
            index.tde.start_tde_constraint_row[i] : index.tde.end_tde_constraint_row[i],
            index.tde.start_tde_col[i] : index.tde.end_tde_col[i],
        ] = operators.tde.tde_slip_rate_constraints[i]
        # Insert block motion constraints for any coupling-constrained rows
        if model.meshes[i].config.top_slip_rate_constraint == 2:
            operator[
                index.tde.start_tde_top_constraint_row[
                    i
                ] : index.tde.end_tde_top_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                model.meshes[i].top_slip_idx,
                index.start_block_col : index.end_block_col,
            ]
        if model.meshes[i].config.bot_slip_rate_constraint == 2:
            operator[
                index.tde.start_tde_bot_constraint_row[
                    i
                ] : index.tde.end_tde_bot_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                model.meshes[i].bot_slip_idx,
                :,
            ]
        if model.meshes[i].config.side_slip_rate_constraint == 2:
            operator[
                index.tde.start_tde_side_constraint_row[
                    i
                ] : index.tde.end_tde_side_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                model.meshes[i].side_slip_idx,
                :,
            ]
    # Insert block strain operator
    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_strain_col : index.end_block_strain_col,
    ] = operators.block_strain_rate_to_velocities[index.station_row_keep_index, :]

    # Insert Mogi source operators
    operator[
        index.start_station_row : index.end_station_row,
        index.start_mogi_col : index.end_mogi_col,
    ] = operators.mogi_to_velocities[index.station_row_keep_index, :]
    # Insert TDE coupling constraints into estimation operator
    # The identity matrices were already inserted as part of the standard slip constraints,
    # so here we can just insert the rotation-to-slip partials into the block rotation columns
    # operator[
    #     index.start_tde_coup_constraint_row[i] : index.end_tde_coup_constraint_row[
    #         i
    #     ],
    #     index.start_block_col : index.end_block_col,
    # ] = operators.tde_coupling_constraints[i]

    return operator


def get_full_dense_operator_eigen(operators: Operators):
    # TODO deduplicate with get_full_dense_operator and get_full_dense_operator_block_only
    index = operators.index
    model = operators.model

    assert index.tde is not None
    assert index.eigen is not None

    assert operators.eigen is not None
    assert operators.tde is not None

    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.eigen.n_eigen_total
            + 3 * index.n_strain_blocks
            + index.n_mogis,
        )
    )

    operator[
        index.start_station_row : index.end_station_row,
        index.start_block_col : index.end_block_col,
    ] = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )

    # Insert block motion constraints
    operator[
        index.start_block_constraints_row : index.end_block_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.block_motion_constraints

    # Insert slip rate constraints
    operator[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row,
        index.start_block_col : index.end_block_col,
    ] = operators.slip_rate_constraints

    # EIGEN Eigenvector to velocity matrix
    for i in range(index.n_meshes):
        # Insert eigenvector to velocities operator
        operator[
            index.start_station_row : index.end_station_row,
            index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i],
        ] = operators.eigen.eigen_to_velocities[i]

    tde_keep_row_index = get_keep_index_12(
        list(operators.tde.tde_to_velocities.values())[-1].shape[0]
    )

    # EIGEN Eigenvector to TDE boundary conditions matrix
    for i in range(index.n_meshes):
        # Create eigenvector to TDE boundary conditions matrix
        operators.eigen.eigen_to_tde_bcs[i] = (
            model.meshes[i].config.eigenmode_slip_rate_constraint_weight
            * operators.tde.tde_slip_rate_constraints[i]
            @ operators.eigen.eigenvectors_to_tde_slip[i]
        )

        # Insert eigenvector to TDE boundary conditions matrix
        operator[
            index.eigen.start_tde_constraint_row_eigen[
                i
            ] : index.eigen.end_tde_constraint_row_eigen[i],
            index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i],
        ] = operators.eigen.eigen_to_tde_bcs[i]

    # EIGEN: Block strain operator
    operator[
        0 : 2 * index.n_stations,
        index.start_block_strain_col : index.end_block_strain_col,
    ] = operators.block_strain_rate_to_velocities[tde_keep_row_index, :]

    # EIGEN: Mogi operator
    operator[
        0 : 2 * index.n_stations,
        index.start_mogi_col : index.end_mogi_col,
    ] = operators.mogi_to_velocities[tde_keep_row_index, :]

    return operator


def get_slip_rate_bounds(segment, block):
    n_total_slip_rate_bounds = (
        np.sum(segment.ss_rate_bound_flag.values)
        + np.sum(segment.ds_rate_bound_flag.values)
        + np.sum(segment.ts_rate_bound_flag.values)
    )
    if n_total_slip_rate_bounds > 0:
        logger.info(f"Found {n_total_slip_rate_bounds} slip rate bounds")
        for i in range(len(segment.lon1)):
            if segment.ss_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP strike-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + f"{segment.ss_rate_bound_min[i]:.2f}"
                    + " (mm/yr), rate (upper bound) = "
                    + f"{segment.ss_rate_bound_max[i]:.2f}"
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert segment.ss_rate_bound_min[i] < segment.ss_rate_bound_max[i], (
                    "Bounds min max error"
                )

            if segment.ds_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP dip-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + f"{segment.ds_rate_bound_min[i]:.2f}"
                    + " (mm/yr), rate (upper bound) = "
                    + f"{segment.ds_rate_bound_max[i]:.2f}"
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert segment.ds_rate_bound_min[i] < segment.ds_rate_bound_max[i], (
                    "Bounds min max error"
                )

            if segment.ts_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP tensile-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + f"{segment.ts_rate_bound_min[i]:.2f}"
                    + " (mm/yr), rate (upper bound) = "
                    + f"{segment.ts_rate_bound_max[i]:.2f}"
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert segment.ts_rate_bound_min[i] < segment.ts_rate_bound_max[i], (
                    "Bounds min max error"
                )

    else:
        logger.info("No hard slip rate bounds")

    # Find 3-strided indices for slip rate bounds
    slip_rate_bounds_idx = np.where(
        interleave3(
            segment.ss_rate_bound_flag,
            segment.ds_rate_bound_flag,
            segment.ts_rate_bound_flag,
        )
        == 1
    )[0]

    # Data vector for minimum slip rate bounds
    slip_rate_bound_min = interleave3(
        segment.ss_rate_bound_min, segment.ds_rate_bound_min, segment.ts_rate_bound_min
    )[slip_rate_bounds_idx]

    # Data vector for maximum slip rate bounds
    slip_rate_bound_max = interleave3(
        segment.ss_rate_bound_max, segment.ds_rate_bound_max, segment.ts_rate_bound_max
    )[slip_rate_bounds_idx]

    # Linear opeartor for slip rate bounds
    slip_rate_bound_partials = get_rotation_to_slip_rate_partials(segment, block)[
        slip_rate_bounds_idx, :
    ]
    return slip_rate_bound_min, slip_rate_bound_max, slip_rate_bound_partials


def get_qp_tde_inequality_operator_and_data_vector(
    model: Model, operators: Operators, index: Index
) -> tuple[np.ndarray, np.ndarray]:
    assert index.eigen is not None
    assert index.tde is not None
    assert operators.eigen is not None

    qp_constraint_matrix = np.zeros(
        (4 * index.n_tde_total, index.n_operator_cols_eigen)
    )
    qp_constraint_data_vector = np.zeros(4 * index.n_tde_total)

    for i in range(index.n_meshes):
        # TDE strike- and dip-slip lower bounds
        lower_ss = model.meshes[i].config.elastic_constraints_ss.lower
        assert lower_ss is not None

        lower_ds = model.meshes[i].config.elastic_constraints_ds.lower
        assert lower_ds is not None

        lower_bound_current_mesh = interleave2(
            lower_ss * np.ones(index.tde.n_tde[i]),
            lower_ds * np.ones(index.tde.n_tde[i]),
        )

        # TDE strike- and dip-slip upper bounds
        upper_ss = model.meshes[i].config.elastic_constraints_ss.upper
        assert upper_ss is not None
        upper_ds = model.meshes[i].config.elastic_constraints_ds.upper
        assert upper_ds is not None
        upper_bound_current_mesh = interleave2(
            upper_ss * np.ones(index.tde.n_tde[i]),
            upper_ds * np.ones(index.tde.n_tde[i]),
        )

        # Insert TDE lower bounds into QP constraint data vector (note negative sign)
        qp_constraint_data_vector[
            index.eigen.qp_constraint_tde_rate_start_row_eigen[
                i
            ] : index.eigen.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.tde.n_tde[i]
        ] = -lower_bound_current_mesh

        # Insert TDE upper bounds into QP constraint data vector
        qp_constraint_data_vector[
            index.eigen.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.tde.n_tde[i] : index.eigen.qp_constraint_tde_rate_end_row_eigen[
                i
            ]
        ] = upper_bound_current_mesh

        # Insert eigenmode to TDE slip operator into QP constraint data vector for lower bounds (note negative sign)
        qp_constraint_matrix[
            index.eigen.qp_constraint_tde_rate_start_row_eigen[
                i
            ] : index.eigen.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.tde.n_tde[i],
            index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i],
        ] = -operators.eigen.eigenvectors_to_tde_slip[i]

        # Insert eigenmode to TDE slip operator into QP constraint data vector for lower bounds
        qp_constraint_matrix[
            index.eigen.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.tde.n_tde[i] : index.eigen.qp_constraint_tde_rate_end_row_eigen[
                i
            ],
            index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i],
        ] = operators.eigen.eigenvectors_to_tde_slip[i]

    return qp_constraint_matrix, qp_constraint_data_vector


def get_qp_all_inequality_operator_and_data_vector(
    model: Model, operators: Operators, index: Index, include_tde: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    # Create arrays and data vector of correct size for linear inequality constraints
    # Stack TDE slip rate bounds on top of slip rate bounds
    #   TDE slip rate bounds
    #   slip rate bounds

    # Get QP slip rate bounds
    qp_slip_rate_inequality_matrix, qp_slip_rate_inequality_data_vector = (
        get_qp_slip_rate_inequality_operator_and_data_vector(model, operators, index)
    )

    if include_tde:
        # Get QP TDE bounds
        qp_tde_inequality_matrix, qp_tde_inequality_data_vector = (
            get_qp_tde_inequality_operator_and_data_vector(model, operators, index)
        )

        # NOTE: This effectively doubles the memory requirements for the problem.
        # I could try creating qp_tde_inequality_matrix as sparse and casting
        # and to full only at the very end
        qp_inequality_constraints_matrix = np.vstack(
            (qp_tde_inequality_matrix, qp_slip_rate_inequality_matrix)
        )

        # Build data vector for QP inequality constraints
        qp_inequality_constraints_data_vector = np.hstack(
            (
                qp_tde_inequality_data_vector,
                qp_slip_rate_inequality_data_vector,
            )
        )
    else:
        # If TDE is not included, just use the slip rate inequality constraints
        qp_inequality_constraints_matrix = qp_slip_rate_inequality_matrix
        qp_inequality_constraints_data_vector = qp_slip_rate_inequality_data_vector

    return qp_inequality_constraints_matrix, qp_inequality_constraints_data_vector


def get_qp_slip_rate_inequality_operator_and_data_vector(
    model: Model, operators: Operators, index: Index
) -> tuple[np.ndarray, np.ndarray]:
    # Get slip rate bounds vectors and operators
    slip_rate_bound_min, slip_rate_bound_max, slip_rate_bound_partials = (
        get_slip_rate_bounds(model.segment, model.block)
    )

    # Combine minimum and maximum slip rate bound operator
    slip_rate_bound_matrix = np.zeros(
        (2 * index.n_slip_rate_bounds, index.n_operator_cols_eigen)
    )
    slip_rate_bound_matrix[0 : index.n_slip_rate_bounds, 0 : 3 * index.n_blocks] = (
        slip_rate_bound_partials
    )
    slip_rate_bound_matrix[
        index.n_slip_rate_bounds : 2 * index.n_slip_rate_bounds, 0 : 3 * index.n_blocks
    ] = -slip_rate_bound_partials

    slip_rate_bound_data_vector = np.hstack((slip_rate_bound_max, -slip_rate_bound_min))

    return slip_rate_bound_matrix, slip_rate_bound_data_vector


def get_eigenvalues_and_eigenvectors(n_eigenvalues, x, y, z, distance_exponent):
    n_tde = x.size

    # Calculate Cartesian distances between triangle centroids
    centroid_coordinates = np.array([x, y, z]).T
    distance_matrix = scipy.spatial.distance.cdist(
        centroid_coordinates, centroid_coordinates, "euclidean"
    )

    # Rescale distance matrix to the range 0-1
    distance_matrix = (distance_matrix - np.min(distance_matrix)) / np.ptp(
        distance_matrix
    )

    # Calculate correlation matrix
    correlation_matrix = np.exp(-(distance_matrix**distance_exponent))

    # https://stackoverflow.com/questions/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        correlation_matrix,
        subset_by_index=[n_tde - n_eigenvalues, n_tde - 1],
    )
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    eigenvectors[np.abs(eigenvectors) < 1e-6] = 0.0
    ordered_index = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[ordered_index]
    eigenvectors = eigenvectors[:, ordered_index]
    return eigenvalues, eigenvectors


def _store_eigenvectors_to_tde_slip(model: Model, operators: _OperatorBuilder):
    meshes = model.meshes

    for i in range(len(meshes)):
        logger.info(f"Start: Eigenvectors to TDE slip for mesh: {meshes[i].file_name}")
        # Get eigenvectors for curren mesh
        _, eigenvectors = get_eigenvalues_and_eigenvectors(
            meshes[i].n_modes,
            meshes[i].x_centroid,
            meshes[i].y_centroid,
            meshes[i].z_centroid,
            distance_exponent=1.0,  # Make this something set in mesh_parameters.json
        )

        # Create eigenvectors to TDE slip matrix
        operators.eigenvectors_to_tde_slip[i] = np.zeros(
            (
                2 * eigenvectors.shape[0],
                meshes[i].config.n_modes_strike_slip
                + meshes[i].config.n_modes_dip_slip,
            )
        )

        # Place strike-slip panel
        operators.eigenvectors_to_tde_slip[i][
            0::2, 0 : meshes[i].config.n_modes_strike_slip
        ] = eigenvectors[:, 0 : meshes[i].config.n_modes_strike_slip]

        # Place dip-slip panel
        operators.eigenvectors_to_tde_slip[i][
            1::2,
            meshes[i].config.n_modes_strike_slip : meshes[i].config.n_modes_strike_slip
            + meshes[i].config.n_modes_dip_slip,
        ] = eigenvectors[:, 0 : meshes[i].config.n_modes_dip_slip]
        logger.success(
            f"Finish: Eigenvectors to TDE slip for mesh: {meshes[i].file_name}"
        )


def rotation_vectors_to_euler_poles(
    rotation_vector_x, rotation_vector_y, rotation_vector_z
):
    def xyz_to_lon_lat(x, y, z):
        # TODO: Should I use proj and proper ellipsoid here?
        lon = np.arctan2(y, x)
        lat = np.arcsin(z)
        return lon, lat

    n_poles = len(rotation_vector_x)

    # Initialize arrays
    euler_lon = np.zeros(n_poles)
    euler_lat = np.zeros(n_poles)
    euler_rate = np.zeros(n_poles)

    # Loop over each pole
    for i in range(n_poles):
        euler_rate[i] = np.sqrt(
            rotation_vector_x[i] ** 2.0
            + rotation_vector_y[i] ** 2.0
            + rotation_vector_z[i] ** 2.0
        )
        unit_vec = (
            np.array([rotation_vector_x[i], rotation_vector_y[i], rotation_vector_z[i]])
            / euler_rate[i]
        )
        tlon, tlat = xyz_to_lon_lat(unit_vec[0], unit_vec[1], unit_vec[2])
        euler_lon[i] = tlon
        euler_lat[i] = tlat

    # Convert longitude and latitude from radians to degrees
    euler_lon = np.rad2deg(euler_lon)
    euler_lat = np.rad2deg(euler_lat)

    # Make sure we have west longitude
    euler_lon = np.where(euler_lon < 0, euler_lon + 360, euler_lon)

    # Convert the rotation rate from rad/yr to degrees per million years
    SCALE_TO_DEG_PER_MILLION_YEARS = 1e3  # TODO: Check this
    euler_rate = SCALE_TO_DEG_PER_MILLION_YEARS * np.rad2deg(euler_rate)

    return euler_lon, euler_lat, euler_rate


def rotation_vector_err_to_euler_pole_err(omega_x, omega_y, omega_z, omega_cov):
    # Linearized propagatin of rotation vector uncertainties to Euler pole uncertainties

    # Declare variables
    n_poles = len(omega_x)
    A = np.zeros((3 * n_poles, 3 * n_poles))

    # Loop over each set of estimates
    for i in range(n_poles):
        idx = 3 * i
        x = omega_x[i]
        y = omega_y[i]
        z = omega_z[i]

        # Calculate the partial derivatives
        dlat_dx = -z / (x**2 + y**2) ** (3 / 2) / (1 + z**2 / (x**2 + y**2)) * x
        dlat_dy = -z / (x**2 + y**2) ** (3 / 2) / (1 + z**2 / (x**2 + y**2)) * y
        dlat_dz = 1 / (x**2 + y**2) ** (1 / 2) / (1 + z**2 / (x**2 + y**2))
        dlon_dx = -y / x**2 / (1 + (y / x) ** 2)
        dlon_dy = 1 / x / (1 + (y / x) ** 2)
        dlon_dz = 0
        dmag_dx = x / np.sqrt(x**2 + y**2 + z**2)
        dmag_dy = y / np.sqrt(x**2 + y**2 + z**2)
        dmag_dz = z / np.sqrt(x**2 + y**2 + z**2)

        # Organize them into a matrix
        A_small = np.array(
            [
                [dlat_dx, dlat_dy, dlat_dz],
                [dlon_dx, dlon_dy, dlon_dz],
                [dmag_dx, dmag_dy, dmag_dz],
            ]
        )
        # Put the small set of partials into the big set
        A[idx : idx + 3, idx : idx + 3] = A_small

    # Propagate the uncertainties and the new covariance matrix
    euler_cov = A @ omega_cov @ A.T

    # Organize data for the return
    diag_vec = np.diag(euler_cov)
    euler_lat_err = np.sqrt(diag_vec[0::3])
    euler_lon_err = np.sqrt(diag_vec[1::3])
    euler_rate_err = np.sqrt(diag_vec[2::3])

    # Convert longitude and latitude from radians to degrees
    euler_lon_err = np.rad2deg(euler_lon_err)
    euler_lat_err = np.rad2deg(euler_lat_err)

    # Convert the rotation rate from rad/yr to degrees per million years
    SCALE_TO_DEG_PER_MILLION_YEARS = 1e3  # TODO: Check this
    euler_rate_err = SCALE_TO_DEG_PER_MILLION_YEARS * np.rad2deg(euler_rate_err)

    return euler_lon_err, euler_lat_err, euler_rate_err


def _get_index(model: Model, assembly: Assembly) -> Index:
    # TODO: Adapt this to use the dataclasses as in get_index_eigen
    # TODO: But better integrate it with the other get_index_* functions?

    index = _get_index_no_meshes(model, assembly)
    index.n_meshes = len(model.meshes)

    # Add TDE mesh indices
    tde = TdeIndex(
        n_tde=np.zeros(index.n_meshes, dtype=int),
        n_tde_constraints=np.zeros(index.n_meshes, dtype=int),
        start_tde_col=np.zeros(index.n_meshes, dtype=int),
        end_tde_col=np.zeros(index.n_meshes, dtype=int),
        start_tde_smoothing_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_smoothing_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_constraint_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_top_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_top_constraint_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_bot_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_bot_constraint_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_side_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_side_constraint_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_coup_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_coup_constraint_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_ss_slip_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_ss_slip_constraint_row=np.zeros(index.n_meshes, dtype=int),
        start_tde_ds_slip_constraint_row=np.zeros(index.n_meshes, dtype=int),
        end_tde_ds_slip_constraint_row=np.zeros(index.n_meshes, dtype=int),
    )
    index.tde = tde

    for i in range(len(model.meshes)):
        tde.n_tde[i] = model.meshes[i].n_tde
        tde.n_tde_constraints[i] = model.meshes[i].n_tde_constraints

        # Set column indices for current mesh
        tde.start_tde_col[i] = index.end_block_col if i == 0 else tde.end_tde_col[i - 1]
        tde.end_tde_col[i] = tde.start_tde_col[i] + 2 * tde.n_tde[i]

        # Set smoothing row indices for current mesh
        start_row = (
            index.end_slip_rate_constraints_row
            if i == 0
            else tde.end_tde_constraint_row[i - 1]
        )
        tde.start_tde_smoothing_row[i] = start_row
        tde.end_tde_smoothing_row[i] = tde.start_tde_smoothing_row[i] + 2 * tde.n_tde[i]

        # Set constraint row indices for current mesh
        tde.start_tde_constraint_row[i] = tde.end_tde_smoothing_row[i]
        tde.end_tde_constraint_row[i] = (
            tde.start_tde_constraint_row[i] + tde.n_tde_constraints[i]
        )

        # Set top constraint row indices and adjust count based on available data
        tde.start_tde_top_constraint_row[i] = tde.end_tde_smoothing_row[i]
        count = len(idx) if (idx := model.meshes[i].top_slip_idx) is not None else 0
        tde.end_tde_top_constraint_row[i] = tde.start_tde_top_constraint_row[i] + count

        # Set bottom constraint row indices
        tde.start_tde_bot_constraint_row[i] = tde.end_tde_top_constraint_row[i]
        count = len(idx) if (idx := model.meshes[i].bot_slip_idx) is not None else 0
        tde.end_tde_bot_constraint_row[i] = tde.start_tde_bot_constraint_row[i] + count

        # Set side constraint row indices
        tde.start_tde_side_constraint_row[i] = tde.end_tde_bot_constraint_row[i]
        count = len(idx) if (idx := model.meshes[i].side_slip_idx) is not None else 0
        tde.end_tde_side_constraint_row[i] = (
            tde.start_tde_side_constraint_row[i] + count
        )

        # Set coupling constraint row indices
        tde.start_tde_coup_constraint_row[i] = tde.end_tde_side_constraint_row[i]
        count = len(idx) if (idx := model.meshes[i].coup_idx) is not None else 0
        tde.end_tde_coup_constraint_row[i] = (
            tde.start_tde_coup_constraint_row[i] + count
        )

        # Set strike-slip constraint row indices
        tde.start_tde_ss_slip_constraint_row[i] = tde.end_tde_coup_constraint_row[i]
        count = len(idx) if (idx := model.meshes[i].ss_slip_idx) is not None else 0
        tde.end_tde_ss_slip_constraint_row[i] = (
            tde.start_tde_ss_slip_constraint_row[i] + count
        )

        # Set dip-slip constraint row indices
        tde.start_tde_ds_slip_constraint_row[i] = tde.end_tde_ss_slip_constraint_row[i]
        count = len(idx) if (idx := model.meshes[i].ds_slip_idx) is not None else 0
        tde.end_tde_ds_slip_constraint_row[i] = (
            tde.start_tde_ds_slip_constraint_row[i] + count
        )

    # Update some indices for the original index object

    # Index for block strain
    index.start_block_strain_col = tde.end_tde_col[-1]
    index.end_block_strain_col = (
        index.start_block_strain_col + index.n_block_strain_components
    )

    # Index for Mogi sources
    index.start_mogi_col = index.end_block_strain_col
    index.end_mogi_col = index.start_mogi_col + index.n_mogis

    # TODO(Brendan): There was this line in the original code:
    # index.n_operator_cols = 3 * index.n_blocks + 2 * index.n_tde_total
    # But that doesn't look right? Isn't that missing the n_strain_blocks and n_mogis?
    # The get_index_eigen function included those.
    # The code is now in a property of the index object.
    return index


def _get_index_no_meshes(model: Model, assembly: Assembly):
    # NOTE: Merge with above if possible.
    # Make sure empty meshes work
    n_blocks = len(model.block)
    n_stations = assembly.data.n_stations
    n_block_constraints = assembly.data.n_block_constraints
    n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    n_mogi = len(model.mogi)
    n_meshes = 0
    n_segments = len(model.segment)
    n_strain_blocks = model.block.strain_rate_flag.sum()

    return Index(
        n_blocks=n_blocks,
        n_segments=n_segments,
        n_stations=n_stations,
        n_meshes=n_meshes,
        n_mogis=n_mogi,
        vertical_velocities=np.arange(2, 3 * n_stations, 3),
        n_block_constraints=n_block_constraints,
        station_row_keep_index=get_keep_index_12(3 * n_stations),
        start_station_row=0,
        end_station_row=2 * n_stations,
        start_block_col=0,
        end_block_col=3 * n_blocks,
        start_block_constraints_row=2 * n_stations,
        end_block_constraints_row=(2 * n_stations + 3 * n_block_constraints),
        start_slip_rate_constraints_row=(2 * n_stations + 3 * n_block_constraints),
        end_slip_rate_constraints_row=(
            2 * n_stations + 3 * n_block_constraints + n_slip_rate_constraints
        ),
        n_slip_rate_constraints=n_slip_rate_constraints,
        start_block_strain_col=3 * n_blocks,
        end_block_strain_col=3 * n_blocks + n_slip_rate_constraints,
        start_mogi_col=3 * n_blocks + n_slip_rate_constraints,
        end_mogi_col=3 * n_blocks + n_slip_rate_constraints + n_mogi,
        slip_rate_bounds=np.where(
            interleave3(
                model.segment.ss_rate_bound_flag,
                model.segment.ds_rate_bound_flag,
                model.segment.ts_rate_bound_flag,
            )
            == 1
        )[0],
        n_block_strain_components=3 * n_strain_blocks,
        n_strain_blocks=n_strain_blocks,
        eigen=None,
        tde=None,
    )


def _get_index_eigen(model: Model, assembly: Assembly) -> Index:
    # Create dictionary to store indices and sizes for operator building
    index = _get_index(model, assembly)
    tde = index.tde
    assert tde is not None

    # EIGEN: Create index components for eigenmodes
    # TODO: Make this optional?
    eigen = EigenIndex(
        n_modes_mesh=np.zeros(index.n_meshes, dtype=int),
        start_col_eigen=np.zeros(index.n_meshes, dtype=int),
        end_col_eigen=np.zeros(index.n_meshes, dtype=int),
        start_tde_row_eigen=np.zeros(index.n_meshes, dtype=int),
        end_tde_row_eigen=np.zeros(index.n_meshes, dtype=int),
        start_tde_ss_slip_constraint_row_eigen=np.zeros(index.n_meshes, dtype=int),
        end_tde_ss_slip_constraint_row_eigen=np.zeros(index.n_meshes, dtype=int),
        start_tde_ds_slip_constraint_row_eigen=np.zeros(index.n_meshes, dtype=int),
        end_tde_ds_slip_constraint_row_eigen=np.zeros(index.n_meshes, dtype=int),
        start_tde_constraint_row_eigen=np.zeros(index.n_meshes, dtype=int),
        end_tde_constraint_row_eigen=np.zeros(index.n_meshes, dtype=int),
        qp_constraint_tde_rate_start_row_eigen=np.zeros(index.n_meshes, dtype=int),
        qp_constraint_tde_rate_end_row_eigen=np.zeros(index.n_meshes, dtype=int),
        qp_constraint_slip_rate_end_row_eigen=np.zeros(index.n_meshes, dtype=int),
        qp_constraint_slip_rate_start_row_eigen=np.zeros(index.n_meshes, dtype=int),
        end_row_eigen=np.zeros(index.n_meshes, dtype=int),
    )
    index.eigen = eigen

    # EIGEN: Count eigenmodes for each mesh
    for i in range(index.n_meshes):
        eigen.n_modes_mesh[i] = (
            model.meshes[i].config.n_modes_strike_slip
            + model.meshes[i].config.n_modes_dip_slip
        )

    # EIGEN: columns and rows for eigenmodes to velocity
    for i in range(index.n_meshes):
        # First mesh
        if i == 0:
            # Locations for eigenmodes to velocities
            eigen.start_col_eigen[i] = 3 * index.n_blocks
            eigen.end_col_eigen[i] = eigen.start_col_eigen[i] + eigen.n_modes_mesh[i]
            eigen.start_tde_row_eigen[i] = 0
            eigen.end_row_eigen[i] = 2 * index.n_stations

        # Meshes after first mesh
        else:
            # Locations for eigenmodes to velocities
            eigen.start_col_eigen[i] = eigen.end_col_eigen[i - 1]
            eigen.end_col_eigen[i] = eigen.start_col_eigen[i] + eigen.n_modes_mesh[i]
            eigen.start_tde_row_eigen[i] = 0
            eigen.end_row_eigen[i] = 2 * index.n_stations

    # EIGEN: Set initial values to follow segment slip rate constraints
    eigen.start_tde_constraint_row_eigen[0] = index.end_slip_rate_constraints_row

    eigen.end_tde_constraint_row_eigen[0] = (
        eigen.start_tde_constraint_row_eigen[0] + tde.n_tde_constraints[0]
    )

    # EIGEN: Rows for eigen to TDE boundary conditions
    for i in range(1, index.n_meshes):
        # All constraints for eigen to constraint matrix
        eigen.start_tde_constraint_row_eigen[i] = eigen.end_tde_constraint_row_eigen[
            i - 1
        ]
        eigen.end_tde_constraint_row_eigen[i] = (
            eigen.start_tde_constraint_row_eigen[i] + tde.n_tde_constraints[i]
        )

    # EIGEN: Rows for QP bounds
    # Create index components for linear inequality matrix and data vector
    eigen.qp_constraint_tde_rate_start_row_eigen[0] = 0
    eigen.qp_constraint_tde_rate_end_row_eigen[0] = (
        eigen.qp_constraint_tde_rate_start_row_eigen[0] + 4 * tde.n_tde[0]
    )

    for i in range(1, index.n_meshes):
        # Start row for current mesh
        eigen.qp_constraint_tde_rate_start_row_eigen[i] = (
            eigen.qp_constraint_tde_rate_end_row_eigen[i - 1]
        )

        # End row for current mesh
        eigen.qp_constraint_tde_rate_end_row_eigen[i] = (
            eigen.qp_constraint_tde_rate_start_row_eigen[i] + 4 * tde.n_tde[i]
        )

    # EIGEN: Index for block strain
    index.start_block_strain_col = eigen.end_col_eigen[-1]
    index.end_block_strain_col = (
        index.start_block_strain_col + index.n_block_strain_components
    )

    # EIGEN: Index for Mogi sources
    index.start_mogi_col = index.end_block_strain_col
    index.end_mogi_col = index.start_mogi_col + index.n_mogis

    return index
