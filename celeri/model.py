from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import addict
import numpy as np
import pandas as pd
from scipy import spatial

from celeri.celeri import (
    assign_block_labels,
    create_output_folder,
    get_all_mesh_smoothing_matrices,
    get_block_motion_constraints,
    get_block_strain_rate_to_velocities_partials,
    get_data_vector_eigen,
    get_eigenvectors_to_tde_slip,
    get_elastic_operators,
    get_full_dense_operator_eigen,
    get_index_eigen,
    get_mogi_to_velocities_partials,
    get_rotation_to_slip_rate_partials,
    get_rotation_to_velocities_partials,
    get_slip_rate_constraints,
    get_tde_coupling_constraints,
    get_tde_slip_rate_constraints,
    get_weighting_vector_eigen,
    merge_geodetic_data,
    process_sar,
    process_segment,
    process_station,
    read_data,
)
from celeri.config import get_config
from celeri.mesh import Mesh

type ByMesh[T] = dict[int, T]


@dataclass
class Operators:
    meshes: list[Any] = field(default_factory=list)
    rotation_to_velocities: np.ndarray | None = None
    block_motion_constraints: np.ndarray | None = None
    slip_rate_constraints: np.ndarray | None = None
    rotation_to_slip_rate: np.ndarray | None = None
    block_strain_rate_to_velocities: np.ndarray | None = None
    mogi_to_velocities: np.ndarray | None = None
    eigen: np.ndarray | None = None
    slip_rate_to_skada_to_velocities: np.ndarray | None = None
    eigenvectors_to_tde_slip: dict[int, np.ndarray] = field(default_factory=dict)
    rotation_to_tri_slip_rate: dict[int, np.ndarray] = field(default_factory=dict)
    linear_guassian_smoothing: dict[int, np.ndarray] = field(default_factory=dict)
    tde_to_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    smoothing_matrix: dict[int, np.ndarray] = field(default_factory=dict)
    tde_slip_rate_constraints: dict[int, np.ndarray] = field(default_factory=dict)
    eigen_to_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    eigen_to_tde_bcs: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class Model:
    """Represents a problem configuration for Celeri fault slip rate modeling.

    Stores indices, meshes, operators, and various data components needed
    for solving interseismic coupling and fault slip rate problems.
    """

    index: dict
    meshes: ByMesh[Mesh]
    operators: Operators
    segment: pd.DataFrame
    block: pd.DataFrame
    station: pd.DataFrame
    assembly: addict.Dict
    mogi: pd.DataFrame
    command: dict[str, Any]

    @property
    def segment_mesh_indices(self):
        n_segment_meshes = np.max(self.segment.patch_file_name).astype(int) + 1
        return list(range(n_segment_meshes))

    @property
    def total_mesh_points(self):
        return sum([self.meshes[idx].n_tde for idx in self.segment_mesh_indices])


def _get_gaussian_smoothing_operator(meshes, operators, index):
    for i in range(index.n_meshes):
        points = np.vstack((meshes[i].lon_centroid, meshes[i].lat_centroid)).T

        length_scale = meshes[i].config.iterative_coupling_smoothing_length_scale
        if length_scale is None:
            length_scale = 0.25

        # Compute pairwise Euclidean distance matrix
        D = spatial.distance_matrix(points, points)

        # Define Gaussian weight function
        W = np.clip(np.exp(-(D**2) / (2 * length_scale**2)), 1e-6, np.inf)
        # W = np.exp(-(D**2) / (2 * length_scale**2))

        # Normalize rows so each row sums to 1
        W /= W.sum(axis=1, keepdims=True)

        operators.linear_guassian_smoothing[i] = W
    return operators


def build_model(command_path: str | Path) -> Model:
    command = get_config(command_path)
    create_output_folder(command)
    segment, block, meshes, station, mogi, sar = read_data(command)
    station = process_station(station, command)
    segment = process_segment(segment, command, meshes)
    sar = process_sar(sar, command)
    closure, block = assign_block_labels(segment, station, block, mogi, sar)
    assembly = addict.Dict()
    operators = Operators()
    operators.meshes = [addict.Dict() for _ in range(len(meshes))]
    assembly = merge_geodetic_data(assembly, station, sar)

    # Prepare the operators

    # Get all elastic operators for segments and TDEs
    get_elastic_operators(operators, meshes, segment, station, command)

    # Get TDE smoothing operators
    get_all_mesh_smoothing_matrices(meshes, operators)

    # Block rotation to velocity operator
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        station, len(block)
    )

    # Soft block motion constraints
    assembly, operators.block_motion_constraints = get_block_motion_constraints(
        assembly, block, command
    )

    # Soft slip rate constraints
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, segment, block, command
    )

    # Rotation vectors to slip rate operator
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(segment, block)

    # Internal block strain rate operator
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(block, station, segment)

    # Mogi source operator
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        mogi, station, command
    )

    # Soft TDE boundary condition constraints
    get_tde_slip_rate_constraints(meshes, operators)

    # Get index
    index = get_index_eigen(assembly, segment, station, block, meshes, mogi)

    # Get data vector for KL problem
    get_data_vector_eigen(meshes, assembly, index)

    # Get data vector for KL problem
    get_weighting_vector_eigen(command, station, meshes, index)

    # Get KL modes for each mesh
    get_eigenvectors_to_tde_slip(operators, meshes)

    # Get full operator including all blocks, KL modes, strain blocks, and mogis
    operators.eigen = get_full_dense_operator_eigen(operators, meshes, index)

    # Get rotation to TDE kinematic slip rate operator for all meshes tied to segments
    get_tde_coupling_constraints(meshes, segment, block, operators)

    # Get smoothing operators for post-hoc smoothing of slip
    operators = _get_gaussian_smoothing_operator(meshes, operators, index)

    return Model(
        index=index,
        meshes=meshes,
        operators=operators,
        segment=segment,
        block=block,
        station=station,
        assembly=assembly,
        command=command,
        mogi=mogi,
    )
