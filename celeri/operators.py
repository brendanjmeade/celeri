import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, overload

import h5py
import numpy as np
import pandas as pd
import scipy
from loguru import logger
from pandas import DataFrame
from rich.progress import track
from scipy import spatial
from scipy.sparse import csr_matrix

from celeri.celeri_util import (
    cartesian_vector_to_spherical_vector,
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
)
from celeri.mesh import ByMesh, Mesh, MeshConfig
from celeri.model import (
    Model,
    assign_mesh_segment_labels,
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
    get_tde_to_velocities_single_mesh,
    get_tri_smoothing_matrix,
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
    """Index of the full dense linear operators comprising the forward model."""

    n_blocks: int
    """The number of blocks in the model."""
    n_segments: int
    """The number of segments in the model."""
    n_stations: int
    """The number of stations in the model."""
    n_meshes: int
    """The number of meshes in the model."""
    n_mogis: int
    """The number of Mogi sources in the model."""
    vertical_velocities: np.ndarray
    """The vertical velocities of the stations."""
    n_block_constraints: int
    """The number of block constraints in the model."""
    station_row_keep_index: np.ndarray
    """The indices comprising the horizontal (spherical plane) vector components acting on the stations.

    Used for assigning horizontal-only forces in the full operator, e.g. block strain.
    Length is (2 * n_stations). Created using `celeri.utils.get_keep_index_12`.
    """
    start_station_row: int
    """The starting index of the station rows in the full operator."""
    end_station_row: int
    """The ending index of the station rows in the full operator."""
    start_block_col: int
    """The starting index of the block columns in the full operator."""
    end_block_col: int
    """The ending index of the block columns in the full operator."""
    start_block_constraints_row: int
    """The starting index of the block constraints rows in the full operator."""
    end_block_constraints_row: int
    """The ending index of the block constraints rows in the full operator."""
    n_slip_rate_constraints: int
    """The number of slip rate constraints in the model."""
    start_slip_rate_constraints_row: int
    """The starting index of the slip rate constraints rows in the full operator."""
    end_slip_rate_constraints_row: int
    """The ending index of the slip rate constraints rows in the full operator."""

    n_strain_blocks: int
    """The number of strain blocks in the model."""
    n_block_strain_components: int
    """The number of block strain components in the model."""
    start_block_strain_col: int
    """The starting index of the block strain columns in the full operator."""
    end_block_strain_col: int
    """The ending index of the block strain columns in the full operator."""

    start_mogi_col: int
    """The starting index of the Mogi columns in the full operator."""
    end_mogi_col: int
    """The ending index of the Mogi columns in the full operator."""
    slip_rate_bounds: np.ndarray
    """The indices of the slip rate bounds in the model."""
    tde: TdeIndex | None = None
    """The TDE index."""
    eigen: EigenIndex | None = None
    """The Eigen index."""

    @property
    def n_slip_rate_bounds(self):
        return len(self.slip_rate_bounds)

    @property
    def n_operator_rows(self) -> int:
        base = (
            self.end_station_row
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
            self.end_station_row
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


# TODO(Adrian): Maybe it would be better to use only one of
# FullTdeOperators or EigenTdeOperators?
# TODO(Adrian): Maybe some of the operators should be properties?
# We could add a reference to the model to Operators and then
# compute the operators on the fly when needed.


@dataclass
class TdeOperators:
    tde_slip_rate_constraints: ByMesh[np.ndarray]
    tde_to_velocities: ByMesh[np.ndarray] | None = None

    def to_disk(self, output_dir: str | Path):
        """Save TDE operators to disk."""
        path = Path(output_dir)
        skip = set()
        if self.tde_to_velocities is None:
            skip.add("tde_to_velocities")
        dataclass_to_disk(self, path, skip=skip)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "TdeOperators":
        """Load TDE operators from disk."""
        path = Path(input_dir)
        return dataclass_from_disk(cls, path)


@dataclass
class EigenOperators:
    eigenvectors_to_tde_slip: ByMesh[np.ndarray]
    eigenvalues: ByMesh[np.ndarray]
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
    """Linear operators comprising the forward model."""

    model: Model
    """The model."""
    index: Index
    """Indices to access different parts of the full dense operator."""
    rotation_to_velocities: np.ndarray
    """Maps rotational vectors to velocities."""
    block_motion_constraints: np.ndarray
    """Constraints on block motions."""
    slip_rate_constraints: np.ndarray
    """Limitations on slip rates."""
    rotation_to_slip_rate: np.ndarray
    """Maps block rotations to kinematic slip rates along the segments."""
    block_strain_rate_to_velocities: np.ndarray
    """Computes predicted velocities on stations due to homogenous block strain rates.

    Has shape (3 * n_stations, 3 * n_strain_blocks).
    """
    mogi_to_velocities: np.ndarray
    """Computes predicted velocities on stations due to Mogi sources.

    Has shape (3 * n_stations, n_mogis).
    """
    slip_rate_to_okada_to_velocities: np.ndarray
    """Okada model slip rate to velocity mapping."""
    rotation_to_tri_slip_rate: dict[int, np.ndarray]
    """Rotation to triangular slip rate mapping."""
    rotation_to_slip_rate_to_okada_to_velocities: np.ndarray
    """Rotation to slip rate to Okada velocities transformation."""
    # TODO: Switch to csr_array?
    smoothing_matrix: dict[int, csr_matrix]
    """Smoothing matrices for various meshes."""
    global_float_block_rotation: np.ndarray
    """Global rotation operator for the block."""
    tde: TdeOperators | None
    """TDE-related operators."""
    eigen: EigenOperators | None
    """Operators related to eigenmodes for TDEs."""

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
            @ parameters[self.index.start_block_col : self.index.end_block_col]
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
        return get_full_dense_operator(self)

    @property
    def data_vector(self) -> np.ndarray:
        if self.tde is None:
            return _get_data_vector_no_meshes(self.model, self.index)
        if self.eigen is not None:
            return _get_data_vector_eigen(self.model, self.index)
        return _get_data_vector(self.model, self.index)

    @property
    def weighting_vector(self) -> np.ndarray:
        if self.tde is None:
            return _get_weighting_vector_no_meshes(self.model, self.index)
        if self.eigen is not None:
            return _get_weighting_vector_eigen(self.model, self.index)
        return _get_weighting_vector(self.model, self.index)

    def to_disk(self, output_dir: str | Path, *, save_arrays: bool = True):
        """Save operators to disk.

        Args:
            output_dir: Directory to save operators to.
            save_arrays: If True (default), save all operator arrays to disk.
                If False, only save model and index. Operators will be loaded from
                the elastic operator cache when opened, saving several GBs per run.
                Requires that the model's elastic_operator_cache_dir is set.
        """
        path = Path(output_dir)

        # Always save model and index (needed for cache loading)
        self.index.to_disk(path / "index")
        self.model.to_disk(path / "model")

        if not save_arrays:
            # Validate that cache is configured
            if self.model.config.elastic_operator_cache_dir is None:
                raise ValueError(
                    "Cannot save with save_arrays=False: elastic_operator_cache_dir "
                    "is not set in the model config. Either set it or use "
                    "save_arrays=True."
                )
            # Write a marker file so from_disk knows to load from cache
            (path / ".load_from_cache").touch()
            logger.info(
                f"Saved minimal operators to {path}. "
                "Full operators will be loaded from cache when opened."
            )
            return

        if self.tde is not None:
            self.tde.to_disk(path / "tde")
        if self.eigen is not None:
            self.eigen.to_disk(path / "eigen")

        # Save the smoothing matrix using scipy sparse format
        for mesh_idx, smoothing_matrix in self.smoothing_matrix.items():
            matrix_path = path / f"smoothing_matrix_{mesh_idx}.npz"
            scipy.sparse.save_npz(matrix_path, smoothing_matrix)

        skip = {"tde", "eigen", "index", "model", "smoothing_matrix"}

        dataclass_to_disk(self, path, skip=skip)

    @classmethod
    def from_disk(cls, input_dir: str | Path) -> "Operators":
        """Load operators from disk.

        If the operators were saved with save_arrays=False, they will be
        loaded from the elastic operator cache. This requires that
        the model's elastic_operator_cache_dir is set. If the cache is empty,
        operators are computed on first access.
        """
        path = Path(input_dir)

        # Check if we should rebuild from cache instead of loading arrays
        if (path / ".load_from_cache").exists():
            model = Model.from_disk(path / "model")
            index = Index.from_disk(path / "index")

            # Determine eigen/tde mode from saved index
            use_eigen = index.eigen is not None
            use_tde = index.tde is not None

            if model.config.elastic_operator_cache_dir is None:
                raise ValueError(
                    "Cannot load operators from cache: elastic_operator_cache_dir is not "
                    "set in the model config. Either set it or save operators with "
                    "save_arrays=True."
                )

            logger.info(
                f"Loading operators from cache at {model.config.elastic_operator_cache_dir}..."
            )
            return build_operators(model, eigen=use_eigen, tde=use_tde)

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
        }

        return dataclass_from_disk(cls, path, extra=extra)


@dataclass
class _OperatorBuilder:
    model: Model
    index: Index | None = None
    rotation_to_velocities: np.ndarray | None = None
    block_motion_constraints: np.ndarray | None = None
    slip_rate_constraints: np.ndarray | None = None
    rotation_to_slip_rate: np.ndarray | None = None
    block_strain_rate_to_velocities: np.ndarray | None = None
    mogi_to_velocities: np.ndarray | None = None
    slip_rate_to_okada_to_velocities: np.ndarray | None = None
    eigenvectors_to_tde_slip: dict[int, np.ndarray] = field(default_factory=dict)
    eigenvalues: dict[int, np.ndarray] = field(default_factory=dict)
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

    def finalize_eigen(self, *, discard_tde_to_velocities: bool = False) -> Operators:
        operators = self.finalize_tde()

        assert self.linear_gaussian_smoothing is not None
        assert self.eigen_to_velocities is not None
        assert self.eigen_to_tde_bcs is not None
        assert self.eigenvectors_to_tde_slip is not None
        assert self.eigenvalues is not None

        eigen = EigenOperators(
            eigenvectors_to_tde_slip=self.eigenvectors_to_tde_slip,
            eigenvalues=self.eigenvalues,
            linear_gaussian_smoothing=self.linear_gaussian_smoothing,
            eigen_to_velocities=self.eigen_to_velocities,
            eigen_to_tde_bcs=self.eigen_to_tde_bcs,
        )

        operators.eigen = eigen

        if discard_tde_to_velocities:
            assert operators.tde is not None
            operators.tde.tde_to_velocities = None

        return operators


def build_operators(
    model: Model,
    *,
    eigen: bool = True,
    tde: bool = True,
    discard_tde_to_velocities: bool = False,
) -> Operators:
    """Build linear operators for the forward model.

    Args:
        model: The model to build operators for.
        eigen: If True, build eigenmode operators for TDEs.
        tde: If True, build TDE operators.
        discard_tde_to_velocities: If True and eigen=True, discard the
            tde_to_velocities matrices after building eigen operators to save
            memory. This is safe when using mcmc_station_velocity_method="project_to_eigen"
            (the default). Set to False if you need the raw TDE operators for
            methods like "direct" or "low_rank".
    """
    if eigen and not tde:
        raise ValueError("eigen operators require tde")
    if discard_tde_to_velocities and not eigen:
        raise ValueError("discard_tde_to_velocities requires eigen=True")

    operators = _OperatorBuilder(model)

    # Get all elastic operators for segments and TDEs
    # When discard_tde_to_velocities is True, skip loading tde_to_velocities
    # as we'll compute eigen_to_velocities in streaming mode later
    _store_elastic_operators(
        model, operators, tde=tde, skip_tde_to_velocities=discard_tde_to_velocities
    )

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
    operators.block_motion_constraints = _store_block_motion_constraints(model)

    # Soft slip rate constraints
    operators.slip_rate_constraints = get_slip_rate_constraints(model)

    # Rotation vectors to slip rate operator
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        model.segment, model.block
    )

    # Internal block strain rate operator
    (
        operators.block_strain_rate_to_velocities,
        _strain_rate_block_index,
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
        index = _get_index_eigen(model)
        operators.index = index

        # Get KL modes for each mesh
        _store_eigenvectors_to_tde_slip(model, operators)
    elif tde:
        index = _get_index(model)
        operators.index = index
    else:
        index = _get_index_no_meshes(model)
        operators.index = index

    # Get rotation to TDE kinematic slip rate operator for all meshes tied to segments
    _store_tde_coupling_constraints(model, operators)

    # Insert block rotations and elastic velocities from fully locked segments
    assert operators.slip_rate_to_okada_to_velocities is not None
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )

    if eigen:
        operators.eigen_to_velocities = _compute_eigen_to_velocities(
            model, operators, index, streaming=discard_tde_to_velocities
        )

    # Get smoothing operators for post-hoc smoothing of slip
    _store_gaussian_smoothing_operator(model.meshes, operators, index)
    if eigen:
        return operators.finalize_eigen(
            discard_tde_to_velocities=discard_tde_to_velocities
        )
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
    meshes: list[MeshConfig], station: DataFrame, config: Config
):
    """Create a hash from geometric components of station DataFrames and elastic
    parameters from config. This allows us to check if we need to recompute elastic operators,
    or if we can use cached ones.

    Args:
        meshes: list[MeshConfig] containing mesh configuration information
        station: DataFrame containing station information
        config: Config object containing material parameters

    Returns:
        str: Hash string representing the input data
    """
    geometric_columns = ["lon", "lat", "depth", "x", "y", "z"]
    station_geom = station[geometric_columns].copy()
    station_str = station_geom.to_json()
    assert isinstance(station_str, str)

    # Get material parameters
    material_params = f"{config.material_mu}_{config.material_lambda}"

    constraint_fields = {
        "n_modes_strike_slip",  # n_modes do not affect the elastic operators
        "n_modes_dip_slip",
        "sqp_kinematic_slip_rate_hint_ss",
        "sqp_kinematic_slip_rate_hint_ds",
        "top_slip_rate_constraint",
        "bot_slip_rate_constraint",
        "top_elastic_constraint_sigma",
        "bot_elastic_constraint_sigma",
        "side_elastic_constraint_sigma",
        "side_slip_rate_constraint",
        "top_slip_rate_weight",
        "bot_slip_rate_weight",
        "side_slip_rate_weight",
        "eigenmode_slip_rate_constraint_weight",
        "a_priori_slip_filename",
        "coupling_constraints_ss",
        "coupling_constraints_ds",
        "coupling_mean",
        "coupling_mean_parameterization",
        "coupling_sigma",
        "elastic_constraints_ss",
        "elastic_constraints_ds",
        "elastic_mean",
        "elastic_mean_parameterization",
        "elastic_sigma",
        "smoothing_weight",
        "softplus_lengthscale",
    }

    mesh_configs = [mesh.model_dump_json(exclude=constraint_fields) for mesh in meshes]
    combined_input = "_".join([station_str, material_params, *mesh_configs])

    return hashlib.blake2b(combined_input.encode()).hexdigest()[:16]


def _save_segments_to_hdf5(segments: DataFrame, hdf5_file: h5py.File) -> None:
    """Save full segments DataFrame to HDF5 file.

    Args:
        segments: DataFrame containing segment information
        hdf5_file: Open HDF5 file handle to save to
    """
    if "name" in segments.columns:
        segment_no_name = segments.drop("name", axis=1)
        hdf5_file.create_dataset("segments", data=segment_no_name.to_numpy())
        string_dtype = h5py.string_dtype(encoding="utf-8")
        hdf5_file.create_dataset(
            "segments_names",
            data=segments["name"].to_numpy(dtype=object),
            dtype=string_dtype,
        )
        hdf5_file.attrs["segments_columns"] = np.array(
            segment_no_name.columns, dtype=h5py.string_dtype()
        )
    else:
        hdf5_file.create_dataset("segments", data=segments.to_numpy())
        hdf5_file.attrs["segments_columns"] = np.array(
            segments.columns, dtype=h5py.string_dtype()
        )
    hdf5_file.attrs["segments_index"] = segments.index.to_numpy()


def _load_segments_from_hdf5(hdf5_file: h5py.File) -> DataFrame | None:
    """Load segments DataFrame from HDF5 file.

    Args:
        hdf5_file: Open HDF5 file handle to load from

    Returns:
        DataFrame with segment information, or None if not found
    """
    if "segments" not in hdf5_file:
        return None
    try:
        segments_data = np.array(hdf5_file["segments"])
        columns_attr = hdf5_file.attrs.get("segments_columns")
        if columns_attr is None:
            return None
        columns_list = [
            col.decode() if isinstance(col, bytes) else str(col)
            for col in np.asarray(columns_attr)
        ]
        index_attr = hdf5_file.attrs.get("segments_index")
        if index_attr is None:
            index_array = np.arange(len(segments_data))
        else:
            index_array = np.array(index_attr)

        segments = pd.DataFrame(segments_data)
        segments.columns = pd.Index(columns_list)
        segments.index = pd.Index(index_array)

        if "segments_names" in hdf5_file:
            names_dataset = hdf5_file["segments_names"]
            names_data = np.array(names_dataset)
            segments["name"] = [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in names_data
            ]

        return segments
    except Exception:
        return None


def _compare_segments(
    cached_segments: DataFrame, current_segments: DataFrame
) -> list[int]:
    """Compare cached and current segments to find which indices have changed.

    Compares only geometric columns: lon1, lat1, lon2, lat2, locking_depth, dip, azimuth.

    Args:
        cached_segments: DataFrame with geometric columns from cache
        current_segments: DataFrame with current geometric columns

    Returns:
        List of segment indices where any geometric column differs
    """
    geometric_columns = [
        "lon1",
        "lat1",
        "lon2",
        "lat2",
        "locking_depth",
        "dip",
        "azimuth",
    ]

    cached_geom = cached_segments[geometric_columns].copy()
    current_geom = current_segments[geometric_columns].copy()

    tolerance = 1e-10
    changed_indices = []
    for i in range(len(current_geom)):
        cached_row = cached_geom.iloc[i]
        current_row = current_geom.iloc[i]
        if not np.allclose(
            cached_row.to_numpy(),
            current_row.to_numpy(),
            rtol=0,
            atol=tolerance,
            equal_nan=True,
        ):
            changed_indices.append(i)

    return changed_indices


def _store_elastic_operators(
    model: Model,
    operators: _OperatorBuilder,
    *,
    tde: bool = True,
    skip_tde_to_velocities: bool = False,
):
    """Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE-parameterized surfaces.

    Supports selective recomputation when only some source segment geometries
    change, in place. If the segment file has changed, rows have been added or removed,
    or `force_recompute` is True, the operators will be recomputed from scratch.

    Args:
        operators (_OperatorBuilder): Data structure which the elastic operators will be added to
        model (Model): Model instance
        tde (bool): Whether to compute TDE operators
        skip_tde_to_velocities (bool): If True, skip loading/computing tde_to_velocities.
            Use this when tde_to_velocities will be computed in streaming mode to save memory.
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
                station,
                config,
            )
        else:
            input_hash = _hash_elastic_operator_input([], station, config)

        cache = config.elastic_operator_cache_dir / f"{input_hash}.hdf5"
        logger.info(f"Cache: {cache}")
        if cache.exists() and config.force_recompute:
            logger.info(
                f"Force recompute enabled. Ignoring cached elastic operators at {cache}"
            )

        if cache.exists() and not config.force_recompute:
            logger.info(f"Found cached elastic operators at {cache}")
            hdf5_file = h5py.File(str(cache), "r")
            cached_operator: np.ndarray[Any, Any] | None = np.array(
                hdf5_file.get("slip_rate_to_okada_to_velocities")
            )
            cached_segments = _load_segments_from_hdf5(hdf5_file)
            hdf5_file.close()

            if cached_segments is None:
                logger.info(
                    "Metadata files not found (old cache format). Recomputing operators."
                )
                cached_operator = None
            else:
                # Check if segments have been added or removed
                if cached_segments["name"].tolist() != segment["name"].tolist():
                    logger.info(
                        "Segments have been added or removed since last computation. Recomputing operators."
                    )
                    cached_operator = None
                elif cached_operator is not None:
                    changed_segment_indices = _compare_segments(
                        cached_segments, segment
                    )
                    if len(changed_segment_indices) == 0:
                        logger.info(
                            "No source geometry changed since last computation. Using cached operator."
                        )
                        operators.slip_rate_to_okada_to_velocities = cached_operator

                        if tde and not skip_tde_to_velocities:
                            hdf5_file = h5py.File(str(cache), "r")
                            tde_data = hdf5_file.get("tde_to_velocities_0")
                            if tde_data is not None:
                                for i in range(len(meshes)):
                                    operators.tde_to_velocities[i] = np.array(
                                        hdf5_file.get("tde_to_velocities_" + str(i))
                                    )
                                hdf5_file.close()
                                return
                            hdf5_file.close()
                            logger.info(
                                "Cache missing tde_to_velocities. Computing from scratch."
                            )
                        else:
                            return

                    logger.info(f"Recomputing {len(changed_segment_indices)} segments")

                    operators.slip_rate_to_okada_to_velocities = cached_operator.copy()

                    if len(changed_segment_indices) > 0:
                        for seg_idx in track(
                            changed_segment_indices,
                            description="Recomputing changed segments",
                        ):
                            single_segment = segment.iloc[
                                seg_idx : seg_idx + 1
                            ].reset_index(drop=True)
                            new_columns = get_segment_station_operator_okada(
                                single_segment, station, config, progress_bar=False
                            )
                            col_start = 3 * seg_idx
                            col_end = col_start + 3
                            operators.slip_rate_to_okada_to_velocities[
                                :, col_start:col_end
                            ] = new_columns

                    tde_loaded_from_cache = False
                    if tde and not skip_tde_to_velocities:
                        hdf5_file = h5py.File(str(cache), "r")
                        if hdf5_file.get("tde_to_velocities_0") is not None:
                            for i in range(len(meshes)):
                                operators.tde_to_velocities[i] = np.array(
                                    hdf5_file.get("tde_to_velocities_" + str(i))
                                )
                            tde_loaded_from_cache = True
                        hdf5_file.close()

                    logger.info("Caching updated elastic operators")
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    hdf5_file = h5py.File(str(cache), "w")
                    hdf5_file.create_dataset(
                        "slip_rate_to_okada_to_velocities",
                        data=operators.slip_rate_to_okada_to_velocities,
                    )
                    if tde and tde_loaded_from_cache:
                        for i in range(len(meshes)):
                            hdf5_file.create_dataset(
                                "tde_to_velocities_" + str(i),
                                data=operators.tde_to_velocities[i],
                            )
                    _save_segments_to_hdf5(segment, hdf5_file)
                    hdf5_file.close()

                    return

        else:
            if not config.force_recompute:
                logger.info(
                    "Precomputed elastic operator file not found. Computing operators"
                )

    else:
        logger.info(
            "No precomputed elastic operator file specified in config. Computing operators."
        )

    operators.slip_rate_to_okada_to_velocities = get_segment_station_operator_okada(
        segment, station, config
    )

    if tde and not skip_tde_to_velocities:
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
    hdf5_file = h5py.File(str(cache), "w")

    hdf5_file.create_dataset(
        "slip_rate_to_okada_to_velocities",
        data=operators.slip_rate_to_okada_to_velocities,
    )
    if tde and not skip_tde_to_velocities:
        for i in range(len(meshes)):
            hdf5_file.create_dataset(
                "tde_to_velocities_" + str(i),
                data=operators.tde_to_velocities[i],
            )
    _save_segments_to_hdf5(segment, hdf5_file)
    hdf5_file.close()


def _store_all_mesh_smoothing_matrices(model: Model, operators: _OperatorBuilder):
    """Build smoothing matrices for each of the triangular meshes
    stored in meshes.
    """
    meshes = model.meshes
    for i in range(len(meshes)):
        operators.smoothing_matrix[i] = get_tri_smoothing_matrix(
            meshes[i].share, meshes[i].tri_shared_sides_distances
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
            meshes[i].top_slip_idx,
            meshes[i].bot_slip_idx,
            meshes[i].side_slip_idx,
        ]

        for slip_idx in boundary_constraints:
            if len(slip_idx) > 0:
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

    east_labels, west_labels, closest_segment_idx = assign_mesh_segment_labels(
        model, mesh_idx
    )

    for el_idx in range(model.meshes[mesh_idx].n_tde):
        # Project velocities from Cartesian to spherical coordinates at element centroids
        row_idx = 3 * el_idx
        column_idx_east = 3 * east_labels[el_idx]
        column_idx_west = 3 * west_labels[el_idx]
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
                tristrike[el_idx] - model.segment.azimuth[closest_segment_idx[el_idx]]
            )
        )
        ew_switch[el_idx] = (
            ns_switch[el_idx]
            * np.sign(tristrike[el_idx] - 90)
            * np.sign(model.segment.azimuth[closest_segment_idx[el_idx]] - 90)
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


def _store_block_motion_constraints(model: Model) -> np.ndarray:
    """Get block motion constraint partials."""
    block = model.block
    block_constraint_partials = get_block_motion_constraint_partials(block)
    return block_constraint_partials


def _get_block_constraints_data(model: Model) -> np.ndarray:
    """Compute block constraints data from model."""
    block = model.block
    block_constraint_partials = get_block_motion_constraint_partials(block)
    block_constraints_idx = np.where(block.rotation_flag == 1)[0]
    block_constraints = np.zeros(block_constraint_partials.shape[0])
    if len(block_constraints_idx) > 0:
        (
            block_constraints[0::3],
            block_constraints[1::3],
            block_constraints[2::3],
        ) = sph2cart(
            block.euler_lon[block_constraints_idx],
            block.euler_lat[block_constraints_idx],
            np.deg2rad(block.rotation_rate[block_constraints_idx]),
        )
    return block_constraints


def _get_slip_rate_constraints_index(model: Model) -> np.ndarray:
    """Compute slip rate constraints index from model."""
    segment = model.segment
    slip_rate_constraint_flag = interleave3(
        segment.ss_rate_flag, segment.ds_rate_flag, segment.ts_rate_flag
    )
    return np.where(slip_rate_constraint_flag == 1)[0]


def _get_slip_rate_constraints_data(model: Model) -> np.ndarray:
    """Compute slip rate constraints data from model."""
    segment = model.segment
    slip_rate_constraints_idx = _get_slip_rate_constraints_index(model)
    slip_rate_constraints_all = interleave3(
        segment.ss_rate, segment.ds_rate, segment.ts_rate
    )
    if len(slip_rate_constraints_idx) > 0:
        return slip_rate_constraints_all[slip_rate_constraints_idx]
    else:
        return np.array([], dtype=slip_rate_constraints_all.dtype)


def _get_slip_rake_constraints_index(model: Model) -> np.ndarray:
    """Compute slip rake constraints index from model."""
    segment = model.segment
    if "rake_flag" in segment.columns:
        return np.where(segment.rake_flag == 1)[0]
    else:
        return np.array([], dtype=int)


def get_slip_rate_constraints(model: Model) -> np.ndarray:
    """Get slip rate constraint partials."""
    segment = model.segment
    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(
        segment, model.block
    )

    # Filter partials to only include constrained segments
    slip_rate_constraints_idx = _get_slip_rate_constraints_index(model)
    slip_rate_constraint_partials = slip_rate_constraint_partials[
        slip_rate_constraints_idx, :
    ]
    return slip_rate_constraint_partials


def get_slip_rake_constraints(model: Model) -> np.ndarray:
    """Get slip rake constraint partials."""
    segment = model.segment

    # Get slip rake constraints index
    slip_rake_constraints_idx = _get_slip_rake_constraints_index(model)

    if len(slip_rake_constraints_idx) == 0:
        return np.array([])

    # Calculate the full set of slip rate partials
    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(
        segment, model.block
    )

    # Figure out effective rake. This is a simple correction of the rake data by the calculated strike of the segment
    # The idea is that the source of the rake constraint will include its own strike (and dip), which may differ from the model segment geometry
    # TODO: Full three-dimensional rotation of rake vector, based on strike and dip of constraint source?
    effective_rakes = segment.rake[segment.rake_flag] + (
        segment.strike[segment.rake_flag] - segment.rake_strike[segment.rake_flag]
    )

    # Get component indices of slip rate partials
    rake_constraint_component_indices = get_2component_index(slip_rake_constraints_idx)

    # Rotate slip partials about effective rake. We just want to use the second row (second basis vector) of a full rotation matrix, because we want to set slip in that direction to zero as a constraint
    slip_rake_constraint_partials = (
        np.cos(np.radians(effective_rakes))
        * slip_rate_constraint_partials[rake_constraint_component_indices[0::2]]
        + np.sin(np.radians(effective_rakes))
        * slip_rate_constraint_partials[rake_constraint_component_indices[1::2]]
    )

    return slip_rake_constraint_partials


def _get_data_vector_no_meshes(model: Model, index: Index) -> np.ndarray:
    """Constructs the data vector for an inversion run that does not use mesh-based constraints.

    The data vector is composed of:
    - GPS station velocities, interleaving east, north, and up components for all stations.
    - Block motion constraints, converted from degrees/Myr to radians/yr.
    - Slip rate constraints.

    Args:
        model (Model): The model object containing station and constraint data.
        index (Index): Index object with details about row regions in the data vector.

    Returns:
        np.ndarray: The assembled data vector.
    """
    data_vector = np.zeros(
        index.end_station_row
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )

    data_vector[index.start_station_row : index.end_station_row] = interleave3(
        model.station.east_vel.to_numpy(),
        model.station.north_vel.to_numpy(),
        model.station.up_vel.to_numpy(),
    )

    block_constraints = _get_block_constraints_data(model)
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * block_constraints
    )

    slip_rate_constraints = _get_slip_rate_constraints_data(model)
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = slip_rate_constraints

    return data_vector


def _get_data_vector(model: Model, index: Index) -> np.ndarray:
    """Constructs the data vector for an inversion run that uses mesh-based constraints.

    The data vector is composed of:
    - GPS station velocities, interleaving east, north, and up components for all stations.
    - Block motion constraints, converted from degrees/Myr to radians/yr.
    - Slip rate constraints.
    - TDE constraints.

    Args:
        model (Model): The model object containing station and constraint data.
        index (Index): Index object with details about row regions in the data vector.

    Returns:
        np.ndarray: The assembled data vector.
    """
    assert index.tde is not None

    data_vector = np.zeros(
        index.end_station_row
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )

    data_vector[index.start_station_row : index.end_station_row] = interleave3(
        model.station.east_vel.to_numpy(),
        model.station.north_vel.to_numpy(),
        model.station.up_vel.to_numpy(),
    )

    block_constraints = _get_block_constraints_data(model)
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * block_constraints
    )

    slip_rate_constraints = _get_slip_rate_constraints_data(model)
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = slip_rate_constraints
    return data_vector


def _get_data_vector_eigen(model: Model, index: Index) -> np.ndarray:
    """Constructs the data vector for an inversion with eigen operators.

    The data vector is composed of:
    - GPS station velocities, interleaving east, north, and up components for all stations.
    - Block motion constraints, converted from degrees/Myr to radians/yr.
    - Slip rate constraints.
    - TDE constraints.
    """
    assert index.tde is not None
    assert index.eigen is not None

    data_vector = np.zeros(
        index.end_station_row
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    data_vector[index.start_station_row : index.end_station_row] = interleave3(
        model.station.east_vel.to_numpy(),
        model.station.north_vel.to_numpy(),
        model.station.up_vel.to_numpy(),
    )

    block_constraints = _get_block_constraints_data(model)
    data_vector[index.start_block_constraints_row : index.end_block_constraints_row] = (
        DEG_PER_MYR_TO_RAD_PER_YR * block_constraints
    )

    slip_rate_constraints = _get_slip_rate_constraints_data(model)
    data_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = slip_rate_constraints
    return data_vector


def _get_weighting_vector(model: Model, index: Index):
    assert index.tde is not None

    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        index.end_station_row
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )

    if model.config.include_vertical_velocity:
        up_weight = 1 / (model.station.up_sig**2)
    else:
        up_weight = np.zeros(len(model.station))
    weighting_vector[index.start_station_row : index.end_station_row] = interleave3(
        1 / (model.station.east_sig**2), 1 / (model.station.north_sig**2), up_weight
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
        index.end_station_row
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )

    if model.config.include_vertical_velocity:
        up_weight = 1 / (station.up_sig**2)
    else:
        up_weight = np.zeros(len(station))
    weighting_vector[index.start_station_row : index.end_station_row] = interleave3(
        1 / (station.east_sig**2), 1 / (station.north_sig**2), up_weight
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
        index.end_station_row
        + 2 * index.tde.n_tde[mesh_index]
        + index.tde.n_tde_constraints[mesh_index]
    )

    if model.config.include_vertical_velocity:
        up_weight = 1 / (station.up_sig**2)
    else:
        up_weight = np.zeros(len(station))
    weighting_vector[0 : index.end_station_row] = interleave3(
        1 / (station.east_sig**2), 1 / (station.north_sig**2), up_weight
    )

    weighting_vector[
        index.end_station_row : index.end_station_row + 2 * index.tde.n_tde[mesh_index]
    ] = mesh.config.smoothing_weight * np.ones(2 * index.tde.n_tde[mesh_index])

    weighting_vector[index.end_station_row + 2 * index.tde.n_tde[mesh_index] : :] = (
        config.tri_con_weight * np.ones(index.tde.n_tde_constraints[mesh_index])
    )

    return weighting_vector


def _get_weighting_vector_eigen(model: Model, index: Index) -> np.ndarray:
    assert index.tde is not None
    assert index.eigen is not None

    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        index.end_station_row
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    if model.config.include_vertical_velocity:
        up_weight = 1 / (model.station.up_sig**2)
    else:
        up_weight = np.zeros(len(model.station))
    weighting_vector[index.start_station_row : index.end_station_row] = interleave3(
        1 / (model.station.east_sig**2), 1 / (model.station.north_sig**2), up_weight
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


def _insert_common_block_operators(
    operator: np.ndarray, operators: Operators, index: Index
) -> None:
    """Insert block rotation, block motion constraints, and slip rate constraints.

    This is common to all three operator types (block_only, tde, eigen).
    """
    # Insert block rotations and elastic velocities from fully locked segments
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


def _insert_block_strain_and_mogi(
    operator: np.ndarray, operators: Operators, index: Index
) -> None:
    """Insert block strain and Mogi source operators.

    This is common to tde and eigen operator types.
    """
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


def _get_full_dense_operator_block_only(operators: Operators) -> np.ndarray:
    index = operators.index
    operator = np.zeros(
        (
            index.end_station_row
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints,
            3 * index.n_blocks,
        )
    )

    _insert_common_block_operators(operator, operators, index)
    return operator


def _get_full_dense_operator_tde(operators: Operators) -> np.ndarray:
    """Build full dense operator for TDE (non-eigen) case."""
    index = operators.index
    model = operators.model
    assert index.tde is not None
    assert operators.tde is not None

    operator = np.zeros(
        (
            index.end_station_row
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

    _insert_common_block_operators(operator, operators, index)

    # Insert all TDE operators
    assert operators.tde.tde_to_velocities is not None
    for i in range(len(model.meshes)):
        # Insert TDE to velocity matrix
        # Use station_row_keep_index for rows to respect vertical flag
        # TDE columns are always 2-component (strike and dip slip)
        tde_keep_col_index = get_keep_index_12(
            operators.tde.tde_to_velocities[i].shape[1]
        )
        operator[
            index.start_station_row : index.end_station_row,
            index.tde.start_tde_col[i] : index.tde.end_tde_col[i],
        ] = -operators.tde.tde_to_velocities[i][index.station_row_keep_index, :][
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

    _insert_block_strain_and_mogi(operator, operators, index)
    return operator


def _get_full_dense_operator_eigen(operators: Operators) -> np.ndarray:
    """Build full dense operator for eigen case."""
    index = operators.index
    model = operators.model

    assert index.tde is not None
    assert index.eigen is not None
    assert operators.eigen is not None
    assert operators.tde is not None

    # Initialize linear operator
    operator = np.zeros(
        (
            index.end_station_row
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.eigen.n_eigen_total
            + 3 * index.n_strain_blocks
            + index.n_mogis,
        )
    )

    _insert_common_block_operators(operator, operators, index)

    # EIGEN Eigenvector to velocity matrix
    for i in range(index.n_meshes):
        # Insert eigenvector to velocities operator
        # Use station_row_keep_index to respect vertical flag
        operator[
            index.start_station_row : index.end_station_row,
            index.eigen.start_col_eigen[i] : index.eigen.end_col_eigen[i],
        ] = operators.eigen.eigen_to_velocities[i][index.station_row_keep_index, :]

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

    _insert_block_strain_and_mogi(operator, operators, index)
    return operator


def get_full_dense_operator(operators: Operators) -> np.ndarray:
    """Build the full dense operator matrix based on the operator type.

    Automatically determines which operator type to build based on the
    presence of eigen/tde operators.

    Args:
        operators: The Operators object containing all sub-operators.

    Returns:
        The full dense operator matrix.
    """
    if operators.eigen is not None:
        return _get_full_dense_operator_eigen(operators)
    elif operators.tde is not None:
        return _get_full_dense_operator_tde(operators)
    else:
        return _get_full_dense_operator_block_only(operators)


def get_full_dense_operator_eigen(operators: Operators) -> np.ndarray:
    """Deprecated: Use get_full_dense_operator instead."""
    return _get_full_dense_operator_eigen(operators)


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


def _store_eigenvectors_to_tde_slip(model: Model, operators: _OperatorBuilder):
    meshes = model.meshes

    for i, mesh in enumerate(meshes):
        logger.info(f"Start: Eigenvectors to TDE slip for mesh: {mesh.file_name}")
        assert mesh.eigenvectors is not None
        assert mesh.eigenvalues is not None
        eigenvectors = mesh.eigenvectors

        # Store eigenvalues for this mesh
        operators.eigenvalues[i] = mesh.eigenvalues

        operators.eigenvectors_to_tde_slip[i] = np.zeros(
            (
                2 * eigenvectors.shape[0],
                mesh.config.n_modes_strike_slip + mesh.config.n_modes_dip_slip,
            )
        )

        operators.eigenvectors_to_tde_slip[i][
            0::2, 0 : mesh.config.n_modes_strike_slip
        ] = eigenvectors[:, 0 : mesh.config.n_modes_strike_slip]

        operators.eigenvectors_to_tde_slip[i][
            1::2,
            mesh.config.n_modes_strike_slip : mesh.config.n_modes_strike_slip
            + mesh.config.n_modes_dip_slip,
        ] = eigenvectors[:, 0 : mesh.config.n_modes_dip_slip]
        logger.success(f"Finish: Eigenvectors to TDE slip for mesh: {mesh.file_name}")


def _compute_eigen_to_velocities(
    model: Model, operators: _OperatorBuilder, index: Index, streaming: bool
) -> dict[int, np.ndarray]:
    """Compute eigen_to_velocities for all meshes.

    Args:
        streaming: If True, processes one mesh at a time by loading/computing
            tde_to_velocities on demand, adding the result to the cache file, then
            discarding it before moving to the next mesh. This reduces peak memory
            usage from O(n_meshes * tde_matrix_size) to O(tde_matrix_size).
            If False, uses the pre-computed operators.tde_to_velocities.

    Returns:
        Dictionary mapping mesh index to eigen_to_velocities operator.
        Each operator has shape (2 * n_stations, n_modes_ss + n_modes_ds).
    """
    config = model.config
    meshes = model.meshes
    station = model.station

    cache = None
    if streaming and config.elastic_operator_cache_dir is not None:
        input_hash = _hash_elastic_operator_input(
            [mesh.config for mesh in meshes],
            station,
            config,
        )
        cache = config.elastic_operator_cache_dir / f"{input_hash}.hdf5"

    eigen_to_velocities: dict[int, np.ndarray] = {}

    for i in range(index.n_meshes):
        tde_computed = False
        if streaming:
            logger.info(f"Loading tde_to_velocities for mesh: {meshes[i].file_name}")

            tde_to_velocities = None
            if cache is not None and cache.exists():
                hdf5_file = h5py.File(str(cache), "r")
                cached_data = hdf5_file.get("tde_to_velocities_" + str(i))
                if cached_data is not None:
                    tde_to_velocities = np.array(cached_data)
                hdf5_file.close()

            if tde_to_velocities is None:
                tde_to_velocities = get_tde_to_velocities_single_mesh(
                    meshes, station, config, mesh_idx=i
                )
                tde_computed = True
        else:
            tde_to_velocities = operators.tde_to_velocities[i]

        assert tde_to_velocities is not None and tde_to_velocities.ndim == 2
        # Slice columns to only strike-slip and dip-slip (exclude tensile)
        tde_keep_col_index = get_keep_index_12(tde_to_velocities.shape[1])  # type: ignore[index]

        # Create eigenvector to velocities operator
        # Keep all 3 velocity components; use station_row_keep_index when
        # inserting into the full operator to handle vertical flag
        eigen_to_velocities[i] = (
            -tde_to_velocities[:, tde_keep_col_index]
            @ operators.eigenvectors_to_tde_slip[i]
        )

        if streaming:
            if tde_computed and cache is not None:
                logger.info(f"Saving tde_to_velocities to cache: {cache}")
                cache.parent.mkdir(parents=True, exist_ok=True)
                hdf5_file = h5py.File(str(cache), "a")
                key = "tde_to_velocities_" + str(i)
                if hdf5_file.get(key) is None:
                    hdf5_file.create_dataset(key, data=tde_to_velocities)
                hdf5_file.close()
            del tde_to_velocities

    return eigen_to_velocities


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


def _get_index(model: Model) -> Index:
    # TODO: Adapt this to use the dataclasses as in get_index_eigen
    # TODO: But better integrate it with the other get_index_* functions?

    index = _get_index_no_meshes(model)
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
        count = len(model.meshes[i].top_slip_idx)
        tde.end_tde_top_constraint_row[i] = tde.start_tde_top_constraint_row[i] + count

        # Set bottom constraint row indices
        tde.start_tde_bot_constraint_row[i] = tde.end_tde_top_constraint_row[i]
        count = len(model.meshes[i].bot_slip_idx)
        tde.end_tde_bot_constraint_row[i] = tde.start_tde_bot_constraint_row[i] + count

        # Set side constraint row indices
        tde.start_tde_side_constraint_row[i] = tde.end_tde_bot_constraint_row[i]
        count = len(model.meshes[i].side_slip_idx)
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


def _get_index_no_meshes(model: Model):
    # NOTE: Merge with above if possible.
    # Make sure empty meshes work
    segment = model.segment
    n_blocks = len(model.block)
    n_stations = len(model.station)
    n_block_constraints = len(np.where(model.block.rotation_flag == 1)[0])

    slip_rate_constraint_flag = interleave3(
        segment.ss_rate_flag, segment.ds_rate_flag, segment.ts_rate_flag
    )

    slip_rate_constraints = np.where(slip_rate_constraint_flag == 1)[0]
    n_slip_rate_constraints = len(slip_rate_constraints)

    n_mogi = len(model.mogi)
    n_meshes = 0
    n_segments = len(model.segment)
    n_strain_blocks = model.block.strain_rate_flag.sum()

    # Always use 3 components per station (east, north, up)
    # Horizontal-only solves are achieved via zero weights for vertical components
    n_station_rows = 3 * n_stations

    return Index(
        n_blocks=n_blocks,
        n_segments=n_segments,
        n_stations=n_stations,
        n_meshes=n_meshes,
        n_mogis=n_mogi,
        vertical_velocities=np.arange(2, 3 * n_stations, 3),
        n_block_constraints=n_block_constraints,
        station_row_keep_index=np.arange(3 * n_stations),
        start_station_row=0,
        end_station_row=n_station_rows,
        start_block_col=0,
        end_block_col=3 * n_blocks,
        start_block_constraints_row=n_station_rows,
        end_block_constraints_row=(n_station_rows + 3 * n_block_constraints),
        start_slip_rate_constraints_row=(n_station_rows + 3 * n_block_constraints),
        end_slip_rate_constraints_row=(
            n_station_rows + 3 * n_block_constraints + n_slip_rate_constraints
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


def _get_index_eigen(model: Model) -> Index:
    # Create dictionary to store indices and sizes for operator building
    index = _get_index(model)
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
            eigen.end_row_eigen[i] = index.end_station_row

        # Meshes after first mesh
        else:
            # Locations for eigenmodes to velocities
            eigen.start_col_eigen[i] = eigen.end_col_eigen[i - 1]
            eigen.end_col_eigen[i] = eigen.start_col_eigen[i] + eigen.n_modes_mesh[i]
            eigen.start_tde_row_eigen[i] = 0
            eigen.end_row_eigen[i] = index.end_station_row

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
