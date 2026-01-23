from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import meshio
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator
from sklearn.gaussian_process.kernels import Matern

from celeri import constants
from celeri.celeri_util import cart2sph, sph2cart, triangle_area, wrap2360
from celeri.output import dataclass_from_disk, dataclass_to_disk

# Should be once we support Python 3.12+
# type ByMesh[T] = dict[int, T]
T = TypeVar("T")
ByMesh = dict[int, T]


class ScalarBound(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    lower: float | None
    upper: float | None

    @model_validator(mode="before")
    @classmethod
    def from_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            if len(data) != 2:
                raise ValueError("ScalarBound should be a list of two values")
            lower, upper = data
            return {"lower": lower, "upper": upper}
        return data


class MeshConfig(BaseModel):
    """Configuration for the mesh.

    Attributes
    ----------
    file_name : Path
        The path to the mesh configuration file itself. All other paths in this
        configuration are relative to this file.
    mesh_filename : Path, optional
        The path to the mesh file, if any.
    smoothing_weight : float
        Weight for Laplacian smoothing of slip rates.
    n_modes_strike_slip : int
        Number of eigenmodes to use for strike- and dip-slips.
    n_modes_dip_slip : int
        Number of eigenmodes to use for dip-slip.
    top_slip_rate_constraint : Literal[0, 1, 2]
        Constraints for slip rates on the top boundary of the mesh.
    bot_slip_rate_constraint : Literal[0, 1, 2]
        Constraints for slip rates on the bottom boundary of the mesh.
    side_slip_rate_constraint : Literal[0, 1, 2]
        Constraints for slip rates on the side boundary of the mesh.
    top_slip_rate_weight : float
        Weight for top boundary zero-slip constraint loss during optimization.
    bot_slip_rate_weight : float
        Weight for bottom boundary zero-slip constraint loss during optimization.
    side_slip_rate_weight : float
        Weight for side boundary zero-slip constraint loss during optimization.
    eigenmode_slip_rate_constraint_weight : float
        Weight for TDE modes boundary conditions if TDE eigenmodes are used.
    a_priori_slip_filename : Path, optional
        Filename for fixed slip rates, not currently used.
    coupling_constraints_ss : ScalarBound
        Tuple containing the constrained upper and lower bounds for the coupling on the mesh for strike-slip. The
        coupling is the ratio of elastic to kinematic slip rates.
    coupling_constraints_ds : ScalarBound
        Tuple containing the constrained upper and lower bounds for the coupling on the mesh for dip-slip.
    elastic_constraints_ss : ScalarBound
        Tuple containing the constrained upper and lower bounds for the elastic rates on the mesh for strike-slip.
    elastic_constraints_ds : ScalarBound
        Tuple containing the constrained upper and lower bounds for the elastic rates on the mesh for dip-slip.
    matern_nu : float
        Matérn kernel smoothness parameter (default 1/2). Common values: 1/2 (exponential),
        3/2 (once-differentiable), 5/2 (twice-differentiable).
    matern_length_scale : float
        Matérn kernel length scale (default 1.0). Interpretation depends on matern_length_units.
    matern_length_units : Literal["absolute", "diameters"]
        Units for matern_length_scale: 'diameters' scales by mesh diameter (default),
        'absolute' uses the value directly in the same units as mesh coordinates.
    eigenvector_algorithm : Literal["eigh", "eigsh"]
        Algorithm for eigendecomposition (default "eigh"). 'eigh' (dense LAPACK) is faster for
        many modes, 'eigsh' (sparse ARPACK) is faster for few modes. Both have equivalent accuracy,
        but eigenvector signs may differ between algorithms.
    softplus_lengthscale : float
        Length scale for the softplus operations for sign constraints when only one bound (upper or lower) is present.
        Automatically set to 1.0 mm/yr if one bound is present.
            Softplus must operate on a unitless quantity; without a length scale divisor,
        the model would change with the units of the input values. As length scale approaches 0, the
        softplus approaches ReLU. Large length scales smooth out the softplus elbow.
    """

    # Forbid extra fields when reading from JSON
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # The path to the mesh configuration file itself.
    # All other paths in this configuration are relative to this file.
    file_name: Path
    mesh_filename: Path | None = None
    # Weight for Laplacian smooting of slip rates (TODO unit?)
    smoothing_weight: float = 1.0
    # Number of eigenmodes to use for strike-slip and dip-slip
    n_modes_strike_slip: int = 10
    n_modes_dip_slip: int = 10

    # TODO should be a string with Literal types or bool?
    # Constraints for slip rates on the boundary of the mesh.
    # These use mesh.top_slip_idx, mesh.bot_slip_idx, and mesh.side_slip_idx
    # to identify the slip rates to constrain.
    # 0: Don't constrain, 1: Constrain to zero, 2: ????
    top_slip_rate_constraint: Literal[0, 1, 2] = 0
    bot_slip_rate_constraint: Literal[0, 1, 2] = 0
    side_slip_rate_constraint: Literal[0, 1, 2] = 0

    # Weight for zero-slip constraint loss during optimization.
    # This will not be used if the elastic velocities are
    # computed using the TDE eigenmodes.
    top_slip_rate_weight: float = 1.0
    bot_slip_rate_weight: float = 1.0
    side_slip_rate_weight: float = 1.0

    # Weight for TDE modes boundary conditions.
    # This is used instead of `top_slip_rate_weight`, `bot_slip_rate_weight`, and
    # `side_slip_rate_weight` if the TDE eigenmodes are used.
    eigenmode_slip_rate_constraint_weight: float = 1.0

    # Sigma for the artificial observed 0s on elastic velocities at the
    # boundaries, used in MCMC sampling when the corresponding constraint == 1.
    top_elastic_constraint_sigma: float = 0.5
    bot_elastic_constraint_sigma: float = 0.5
    side_elastic_constraint_sigma: float = 0.5

    # Filename for fixed slip rates, not currently used
    a_priori_slip_filename: Path | None = None

    # Constraints for the coupling on the mesh, ie the ratio
    # of elastic and kinematic slip rates.
    # Specified independently for strike-slip and dip-slip
    coupling_constraints_ss: ScalarBound = ScalarBound(lower=None, upper=None)
    coupling_constraints_ds: ScalarBound = ScalarBound(lower=None, upper=None)

    # Constraints for elastic rates on the mesh.
    elastic_constraints_ss: ScalarBound = ScalarBound(lower=None, upper=None)
    elastic_constraints_ds: ScalarBound = ScalarBound(lower=None, upper=None)

    coupling_sigma: float = 1.0
    elastic_sigma: float = 1.0

    softplus_lengthscale: float = 1.0

    # Hint for the new sqp solver about the likely range of kinematic slip rates.
    sqp_kinematic_slip_rate_hint_ss: ScalarBound = ScalarBound(
        lower=-100.0, upper=100.0
    )
    sqp_kinematic_slip_rate_hint_ds: ScalarBound = ScalarBound(
        lower=-100.0, upper=100.0
    )

    # Parameters for SQP solver
    iterative_coupling_linear_slip_rate_reduction_factor: float = 0.025
    iterative_coupling_smoothing_length_scale: float | None = None
    iterative_coupling_kinematic_slip_regularization_scale: float = 1.0

    # GP kernel hyperparameters for eigenmode computation
    matern_nu: float = 0.5
    matern_length_scale: float = 1.0
    matern_length_units: Literal["absolute", "diameters"] = "diameters"
    eigenvector_algorithm: Literal["eigh", "eigsh"] = "eigh"

    @classmethod
    def from_file(cls, filename: str | Path) -> list[MeshConfig]:
        """Read mesh parameters from a JSON file and override defaults.

        Args:
            filename: Path to the JSON file containing mesh parameters

        Returns:
            list[MeshParams]: Instance with parameters from file overriding defaults
        """
        filename = Path(filename).resolve()
        with filename.open() as f:
            params_list = json.load(f)

        if not isinstance(params_list, list):
            raise ValueError(f"Expected a list of dictionaries in {filename}")

        for params in params_list:
            params["file_name"] = filename

        return [cls.model_validate(params) for params in params_list]

    @model_validator(mode="after")
    def relative_paths(self) -> MeshConfig:
        """Convert relative paths to absolute paths based on the config file location."""
        base_dir = self.file_name.parent

        for name in type(self).model_fields:
            if name == "file_name":
                continue

            value = getattr(self, name)
            if isinstance(value, Path) and not value.is_absolute():
                setattr(self, name, (base_dir / value).resolve())

        return self

    def has_mixed_constraints(self) -> str | None:
        """Check if mesh has both elastic and coupling constraints.

        Returns:
            Error message if mixed constraints found, None otherwise.
        """
        for kind in ("ss", "ds"):
            elastic = getattr(self, f"elastic_constraints_{kind}")
            coupling = getattr(self, f"coupling_constraints_{kind}")
            has_elastic = elastic.lower is not None or elastic.upper is not None
            has_coupling = coupling.lower is not None or coupling.upper is not None
            if has_elastic and has_coupling:
                kind_name = "strike-slip" if kind == "ss" else "dip-slip"
                return (
                    f"Mesh '{self.mesh_filename}' cannot have both elastic and coupling "
                    f"constraints for {kind_name}."
                )
        return None


def _compute_ordered_edge_nodes(mesh: dict):
    """Find exterior edges of each mesh and return them in the dictionary
    for each mesh.

    Args:
        meshes (List): list of mesh dictionaries
    """
    # Make side arrays containing vertex indices of sides
    vertices = mesh["verts"]
    side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
    side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
    side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)
    all_sides = np.vstack((side_1, side_2, side_3))
    unique_sides, sides_count = np.unique(all_sides, return_counts=True, axis=0)
    edge_nodes = unique_sides[np.where(sides_count == 1)]

    mesh["ordered_edge_nodes"] = np.zeros_like(edge_nodes)
    mesh["ordered_edge_nodes"][0, :] = edge_nodes[0, :]
    last_row = 0
    for j in range(1, len(edge_nodes)):
        idx = np.where(
            edge_nodes == mesh["ordered_edge_nodes"][j - 1, 1]
        )  # Edge node indices the same as previous row, second column
        next_idx = np.where(
            idx[0][:] != last_row
        )  # One of those indices is the last row itself. Find the other row index
        next_row = idx[0][next_idx]  # Index of the next ordered row
        next_col = idx[1][next_idx]  # Index of the next ordered column (1 or 2)
        if next_col == 1:
            next_col_ord = [1, 0]  # Flip edge ordering
        else:
            next_col_ord = [0, 1]
        mesh["ordered_edge_nodes"][j, :] = edge_nodes[next_row, next_col_ord]
        last_row = (
            next_row  # Update last_row so that it's excluded in the next iteration
        )


def _compute_mesh_edge_elements(mesh: dict):
    """Compute mesh boundary elements that are classified as top, bottom, or side.

    Modifies the `mesh` dictionary in-place by adding boolean arrays
    indicating which elements are on the top, bottom, or side of the mesh boundary.
    The classification is based on the depth of each element's vertices.
    Top elements are those that are relatively shallow, bottom elements are deepest,
    and sides are edges that are on the boundary but not classified as top or bottom.

    Args:
        mesh (dict): A mesh dictionary with keys including "points" (vertex coordinates),
            "verts" (triangular element vertex indices), and (on exit) "ordered_edge_nodes".

    Modifies:
        mesh["top_elements"]: np.ndarray of bool, True for elements on the top surface.
        mesh["bot_elements"]: np.ndarray of bool, True for elements on the bottom surface.
        mesh["side_elements"]: np.ndarray of bool, True for elements on mesh sides.
    """
    _compute_ordered_edge_nodes(mesh)

    coords = mesh["points"]
    vertices = mesh["verts"]

    # Get element centroid depths
    el_depths = coords[vertices, 2]
    centroid_depths = np.mean(el_depths, axis=1)

    # Arrays of all element side node pairs
    side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
    side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
    side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)

    # Sort edge node array
    sorted_edge_nodes = np.sort(mesh["ordered_edge_nodes"], 1)

    # Helper function to find matching rows
    def find_matching_rows(array1, array2):
        array1_set = {tuple(row) for row in array1}
        matches = [i for i, row in enumerate(array2) if tuple(row) in array1_set]
        return np.array(matches, dtype=int)

    # Indices of element sides that are in edge node array
    side_1_in_edge_idx = find_matching_rows(sorted_edge_nodes, side_1)
    side_2_in_edge_idx = find_matching_rows(sorted_edge_nodes, side_2)
    side_3_in_edge_idx = find_matching_rows(sorted_edge_nodes, side_3)

    # Depths of nodes
    side_1_depths = np.abs(
        coords[
            np.column_stack(
                (side_1[side_1_in_edge_idx, :], vertices[side_1_in_edge_idx, 2])
            ),
            2,
        ]
    )
    side_2_depths = np.abs(
        coords[
            np.column_stack(
                (side_2[side_2_in_edge_idx, :], vertices[side_2_in_edge_idx, 0])
            ),
            2,
        ]
    )
    side_3_depths = np.abs(
        coords[
            np.column_stack(
                (side_3[side_3_in_edge_idx, :], vertices[side_3_in_edge_idx, 1])
            ),
            2,
        ]
    )
    # Top elements are those where the depth difference between the non-edge node
    # and the mean of the edge nodes is greater than the depth difference between
    # the edge nodes themselves
    top1 = (side_1_depths[:, 2] - np.mean(side_1_depths[:, 0:2], 1)) > (
        np.abs(side_1_depths[:, 0] - side_1_depths[:, 1])
    )
    top2 = (side_2_depths[:, 2] - np.mean(side_2_depths[:, 0:2], 1)) > (
        np.abs(side_2_depths[:, 0] - side_2_depths[:, 1])
    )
    top3 = (side_3_depths[:, 2] - np.mean(side_3_depths[:, 0:2], 1)) > (
        np.abs(side_3_depths[:, 0] - side_3_depths[:, 1])
    )
    tops = np.full(len(vertices), False, dtype=bool)
    tops[side_1_in_edge_idx[top1]] = True
    tops[side_2_in_edge_idx[top2]] = True
    tops[side_3_in_edge_idx[top3]] = True
    # Make sure elements are really shallow
    # Depending on element shapes, some side elements can satisfy the depth difference criterion
    depth_count, depth_bins = np.histogram(centroid_depths[tops], bins="doane")
    depth_bin_min = depth_bins[0:-1]
    depth_bin_max = depth_bins[1:]
    zero_idx = np.where(depth_count == 0)[0]
    if len(zero_idx) > 0:
        if np.abs(depth_bin_max[zero_idx[0]] - depth_bin_min[zero_idx[-1] + 1]) > 10:
            tops[centroid_depths < depth_bins[zero_idx[0]]] = False
    # Assign in to meshes dict
    mesh["top_elements"] = tops

    # Bottom elements are those where the depth difference between the non-edge node
    # and the mean of the edge nodes is more negative than the depth difference between
    # the edge nodes themselves
    bot1 = side_1_depths[:, 2] - np.mean(side_1_depths[:, 0:2], 1) < -np.abs(
        side_1_depths[:, 0] - side_1_depths[:, 1]
    )
    bot2 = side_2_depths[:, 2] - np.mean(side_2_depths[:, 0:2], 1) < -np.abs(
        side_2_depths[:, 0] - side_2_depths[:, 1]
    )
    bot3 = side_3_depths[:, 2] - np.mean(side_3_depths[:, 0:2], 1) < -np.abs(
        side_3_depths[:, 0] - side_3_depths[:, 1]
    )
    bots = np.full(len(vertices), False, dtype=bool)
    bots[side_1_in_edge_idx[bot1]] = True
    bots[side_2_in_edge_idx[bot2]] = True
    bots[side_3_in_edge_idx[bot3]] = True
    # Make sure elements are really deep
    # Depending on element shapes, some side elements can satisfy the depth difference criterion
    depth_count, depth_bins = np.histogram(centroid_depths[bots], bins="doane")
    depth_bin_min = depth_bins[0:-1]
    depth_bin_max = depth_bins[1:]
    zero_idx = np.where(depth_count == 0)[0]
    if len(zero_idx) > 0:
        if abs(depth_bin_min[zero_idx[-1]] - depth_bin_max[zero_idx[0] - 1]) > 10:
            bots[centroid_depths > depth_bin_min[zero_idx[-1]]] = False
    # Assign in to meshes dict
    mesh["bot_elements"] = bots

    # Side elements are a set difference between all edges and tops, bottoms
    sides = np.full(len(vertices), False, dtype=bool)
    sides[side_1_in_edge_idx] = True
    sides[side_2_in_edge_idx] = True
    sides[side_3_in_edge_idx] = True
    sides[np.where(tops != 0)] = False
    sides[np.where(bots != 0)] = False
    mesh["side_elements"] = sides


def _compute_n_tde_constraints(
    n_tde: int,
    top_slip_idx: np.ndarray,
    bot_slip_idx: np.ndarray,
    side_slip_idx: np.ndarray,
) -> int:
    """Compute the total number of TDE constraints.

    Builds a constraint matrix and counts rows with at least one constraint,
    replicating the logic from operators._store_tde_slip_rate_constraints.

    Args:
        n_tde: Number of triangular elements
        top_slip_idx: Indices for top boundary constraints
        bot_slip_idx: Indices for bottom boundary constraints
        side_slip_idx: Indices for side boundary constraints

    Returns:
        Total number of constraint rows
    """
    tde_slip_rate_constraints = np.zeros((2 * n_tde, 2 * n_tde))
    end_row = 0

    boundary_slip_indices = [top_slip_idx, bot_slip_idx, side_slip_idx]

    for slip_idx in boundary_slip_indices:
        if len(slip_idx) > 0:
            start_row = end_row
            end_row = start_row + len(slip_idx)
            tde_slip_rate_constraints[start_row:end_row, slip_idx] = np.eye(
                len(slip_idx)
            )

    # Count rows with at least one constraint
    # Total number of slip constraints:
    # 2 for each element that has coupling constrained (top, bottom, side, specified indices)
    # 1 for each additional slip component that is constrained (specified indices)

    # TODO: Number of total constraints is determined by just finding 1 in the
    # constraint array. This could cause an error when the index Dict is constructed,
    # if an individual element has a constraint imposed, but that element is also
    # a constrained edge element. Need to build in some uniqueness tests.
    sum_constraint_columns = np.sum(tde_slip_rate_constraints, 1)
    return int(np.sum(sum_constraint_columns > 0))


def _compute_mesh_perimeter(mesh: dict):
    x_coords = mesh["points"][:, 0]
    y_coords = mesh["points"][:, 1]
    ordered_edge_nodes = mesh["ordered_edge_nodes"]
    mesh["x_perimeter"] = x_coords[ordered_edge_nodes[:, 0]]
    mesh["y_perimeter"] = y_coords[ordered_edge_nodes[:, 0]]
    mesh["x_perimeter"] = np.append(
        mesh["x_perimeter"], x_coords[ordered_edge_nodes[0, 0]]
    )
    mesh["y_perimeter"] = np.append(
        mesh["y_perimeter"], y_coords[ordered_edge_nodes[0, 0]]
    )


def _get_eigenvalues_and_eigenvectors(
    n_eigenvalues: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    matern_nu: float = 0.5,  # Smoothness parameter
    matern_length_scale: float = 1.0,
    matern_length_units: Literal["absolute", "diameters"] = "diameters",
    eigenvector_algorithm: Literal["eigh", "eigsh"] = "eigh",
) -> tuple[np.ndarray, np.ndarray]:
    """Get the eigenvalues and eigenvectors of the mesh Matérn kernel.

    The kernel is computed with unit amplitude scale parameter (sigma=1).
    Eigenvalues scale as sigma**2, so the amplitude can be reintroduced later
    by multiplying eigenvalues by sigma**2 (eigenvectors are unchanged).
    """
    n_tde = x.size

    # Triangle centroid coordinates (n_tde, 3)
    centroid_coordinates = np.array([x, y, z]).T

    if matern_length_units == "diameters":
        # Scale length_scale by mesh diameter (max pairwise distance)
        diameter = np.max(scipy.spatial.distance.pdist(centroid_coordinates))
        matern_length_scale *= diameter
    else:
        assert matern_length_units == "absolute"

    # Use sklearn's Matern kernel which handles all special cases (nu=0.5, 1.5, 2.5)
    # and the general case with proper numerical stability
    kernel = Matern(nu=matern_nu, length_scale=matern_length_scale)
    covariance_matrix = kernel(centroid_coordinates)

    # Algorithm choice: see https://github.com/brendanjmeade/celeri/pull/367#issuecomment-2690519498
    # and https://stackoverflow.com/questions/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with
    if eigenvector_algorithm == "eigh":
        eigenvalues_ascending, eigenvectors_ascending = scipy.linalg.eigh(
            covariance_matrix,
            subset_by_index=[n_tde - n_eigenvalues, n_tde - 1],
        )
    elif eigenvector_algorithm == "eigsh":
        # ARPACK is faster for small k; eigenvector signs may differ from eigh
        eigenvalues_ascending, eigenvectors_ascending = scipy.sparse.linalg.eigsh(
            covariance_matrix, k=n_eigenvalues, which="LM"
        )
    else:
        raise ValueError(f"Unknown eigenvector_algorithm: {eigenvector_algorithm}")

    assert np.all(eigenvalues_ascending > 0), (
        "Mesh kernel error: Some eigenvalues are negative"
    )
    eigenvalues_descending = eigenvalues_ascending[::-1]
    eigenvectors_descending = eigenvectors_ascending[:, ::-1]
    return eigenvalues_descending, eigenvectors_descending


@dataclass
class Mesh:
    """Triangular mesh for fault modeling.

    Parameters
    ----------
             points: np.ndarray
                 The coordinates of the vertices of the mesh.
             verts: np.ndarray
                 The indices of the vertices composing each triangle of the mesh.
             lon1: np.ndarray
                 The longitude of the vertex 1 of each triangle of the mesh.
             lat1: np.ndarray
                 The latitude of the vertex 1 of each triangle of the mesh.
             dep1: np.ndarray
                 The depth of the vertex 1 of each triangle of the mesh.
             lon2: np.ndarray
                 The longitude of the vertex 2 of each triangle of the mesh.
             lat2: np.ndarray
                 The latitude of the vertex 2 of each triangle of the mesh.
             dep2: np.ndarray
                 The depth of the vertex 2 of each triangle of the mesh.
             lon3: np.ndarray
                 The longitude of the vertex 3 of each triangle of the mesh.
             lat3: np.ndarray
                 The latitude of the vertex 3 of each triangle of the mesh.
             dep3: np.ndarray
                 The depth of the vertex 3 of each triangle of the mesh.
             centroids: np.ndarray
                 The centroids of the triangles.
             x_centroid: np.ndarray
                 The x-coordinates of the centroids of the triangles.
             y_centroid: np.ndarray
                 The y-coordinates of the centroids of the triangles.
             z_centroid: np.ndarray
                 The z-coordinates of the centroids of the triangles.
             nv: np.ndarray
                 Normal vectors of the triangles.
             strike: np.ndarray
                 Magnitude of the strike slip on each triangle.
             dip: np.ndarray
                 Magnitude of the dip slip on each triangle.
             dip_flag: np.ndarray
                 Bool indicating the presence of dip slip on each triangle.
             n_tde: int
                 The number of triangular elements constituting the mesh.
             areas: np.ndarray
                 The surface areas of the triangles.
             ordered_edge_nodes: np.ndarray
                 The edges constituting the perimeter of the mesh.
             side_elements: np.ndarray
                 Bool indicating the presence of side elements on each triangle.
             bot_elements: np.ndarray
                 Bool indicating if each triangle is a bottom element. Each triangle along the edge of the mesh will
                 naturally have two vertices which compose the outer edge that actually belongs to the perimeter of the mesh,
                 and a third "interior" vertex. A bottom element is defined as an edge triangle such that the depth
                 difference between the interior vertex and the midpoint of the outer edge is more negative than the negative
                 of the absolute depth difference between the two edge vertices. Additionally, elements are filtered to ensure
                 they are really deep using histogram analysis of centroid depths.
             top_elements: np.ndarray
                 Bool indicating top elements, the opposite of bottom elements.
             side_elements: np.ndarray
                 Bool indicating edge triangles which are neither top nor bottom elements.
             config: MeshConfig
                 Configuration of the mesh.
             share: np.ndarray
                 Array of shape (n_tde, 3) indicating the indices of the up to 3 triangles sharing a side with
                 each of the n_tde triangles.
             n_tde_constraints: int
                 Total number of slip rate constraints on the TDEs; equal to 2 * the number of TDEs with coupling
                 constraints (top, bottom, side, specified indices) + the number of additional slip components (specified indices)
             top_slip_idx: np.ndarray
                 Indices of the top TDEs which have constraints on their slip rates.
             coup_idx: np.ndarray
                 Indices of the TDEs which have constraints on their coupling.
             ss_slip_idx: np.ndarray
                 Indices of the TDEs which have constraints on their strike slip slip rates.
             ds_slip_idx: np.ndarray
                 Indices of the TDEs which have constraints on their dip slip slip rates.
    """

    points: np.ndarray
    verts: np.ndarray
    lon1: np.ndarray
    lon2: np.ndarray
    lon3: np.ndarray
    lat1: np.ndarray
    lat2: np.ndarray
    lat3: np.ndarray
    dep1: np.ndarray
    dep2: np.ndarray
    dep3: np.ndarray
    centroids: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    z1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    z2: np.ndarray
    x3: np.ndarray
    y3: np.ndarray
    z3: np.ndarray
    x_centroid: np.ndarray
    y_centroid: np.ndarray
    z_centroid: np.ndarray
    lon_centroid: np.ndarray
    lat_centroid: np.ndarray
    nv: np.ndarray
    strike: np.ndarray
    dip: np.ndarray
    dip_flag: np.ndarray
    n_tde: int
    areas: np.ndarray
    n_modes: int
    n_modes_total: int
    ordered_edge_nodes: np.ndarray
    top_elements: np.ndarray
    bot_elements: np.ndarray
    side_elements: np.ndarray
    x_perimeter: np.ndarray
    y_perimeter: np.ndarray
    config: MeshConfig
    share: np.ndarray
    tri_shared_sides_distances: np.ndarray
    top_slip_idx: np.ndarray
    bot_slip_idx: np.ndarray
    side_slip_idx: np.ndarray
    n_tde_constraints: int
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    coup_idx: np.ndarray | None = None
    ss_slip_idx: np.ndarray | None = None
    ds_slip_idx: np.ndarray | None = None
    closest_segment_idx: np.ndarray | None = None

    @classmethod
    def from_params(cls, config: MeshConfig):
        # Standalone reader for a single .msh file
        mesh = {}
        filename = config.mesh_filename

        # Suppress meshio's stdout output (it prints a newline)
        old_stdout = sys.stdout
        sys.stdout = Path(os.devnull).open("w")
        try:
            meshobj = meshio.read(filename)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        points = cast(np.ndarray, meshobj.points)
        mesh["points"] = points
        verts = meshio.CellBlock("triangle", meshobj.get_cells_type("triangle")).data
        verts = cast(np.ndarray, verts)
        mesh["verts"] = verts

        # Expand mesh coordinates
        mesh["lon1"] = points[verts[:, 0], 0]
        mesh["lon2"] = points[verts[:, 1], 0]
        mesh["lon3"] = points[verts[:, 2], 0]
        mesh["lat1"] = points[verts[:, 0], 1]
        mesh["lat2"] = points[verts[:, 1], 1]
        mesh["lat3"] = points[verts[:, 2], 1]
        mesh["dep1"] = points[verts[:, 0], 2]
        mesh["dep2"] = points[verts[:, 1], 2]
        mesh["dep3"] = points[verts[:, 2], 2]
        mesh["centroids"] = np.mean(mesh["points"][mesh["verts"], :], axis=1)
        # Cartesian coordinates in meters
        mesh["x1"], mesh["y1"], mesh["z1"] = sph2cart(
            mesh["lon1"],
            mesh["lat1"],
            constants.RADIUS_EARTH + constants.KM2M * mesh["dep1"],
        )
        mesh["x2"], mesh["y2"], mesh["z2"] = sph2cart(
            mesh["lon2"],
            mesh["lat2"],
            constants.RADIUS_EARTH + constants.KM2M * mesh["dep2"],
        )
        mesh["x3"], mesh["y3"], mesh["z3"] = sph2cart(
            mesh["lon3"],
            mesh["lat3"],
            constants.RADIUS_EARTH + constants.KM2M * mesh["dep3"],
        )

        # Cartesian triangle centroids
        mesh["x_centroid"] = (mesh["x1"] + mesh["x2"] + mesh["x3"]) / 3.0
        mesh["y_centroid"] = (mesh["y1"] + mesh["y2"] + mesh["y3"]) / 3.0
        mesh["z_centroid"] = (mesh["z1"] + mesh["z2"] + mesh["z3"]) / 3.0

        # Spherical triangle centroids
        mesh["lon_centroid"] = (mesh["lon1"] + mesh["lon2"] + mesh["lon3"]) / 3.0
        mesh["lat_centroid"] = (mesh["lat1"] + mesh["lat2"] + mesh["lat3"]) / 3.0

        # Cross products for orientations
        tri_leg1 = np.transpose(
            [
                np.deg2rad(mesh["lon2"] - mesh["lon1"]),
                np.deg2rad(mesh["lat2"] - mesh["lat1"]),
                (1 + constants.KM2M * mesh["dep2"] / constants.RADIUS_EARTH)
                - (1 + constants.KM2M * mesh["dep1"] / constants.RADIUS_EARTH),
            ]
        )
        tri_leg2 = np.transpose(
            [
                np.deg2rad(mesh["lon3"] - mesh["lon1"]),
                np.deg2rad(mesh["lat3"] - mesh["lat1"]),
                (1 + constants.KM2M * mesh["dep3"] / constants.RADIUS_EARTH)
                - (1 + constants.KM2M * mesh["dep1"] / constants.RADIUS_EARTH),
            ]
        )
        mesh["nv"] = np.cross(tri_leg1, tri_leg2)
        azimuth, elevation, _r = cart2sph(
            mesh["nv"][:, 0],
            mesh["nv"][:, 1],
            mesh["nv"][:, 2],
        )
        mesh["strike"] = wrap2360(-np.rad2deg(azimuth))
        mesh["dip"] = 90 - np.rad2deg(elevation)
        mesh["dip_flag"] = mesh["dip"] != 90

        mesh["n_tde"] = mesh["lon1"].size

        # Calcuate areas of each triangle in mesh
        triangle_vertex_array = np.zeros((mesh["n_tde"], 3, 3))
        triangle_vertex_array[:, 0, 0] = mesh["x1"]
        triangle_vertex_array[:, 1, 0] = mesh["x2"]
        triangle_vertex_array[:, 2, 0] = mesh["x3"]
        triangle_vertex_array[:, 0, 1] = mesh["y1"]
        triangle_vertex_array[:, 1, 1] = mesh["y2"]
        triangle_vertex_array[:, 2, 1] = mesh["y3"]
        triangle_vertex_array[:, 0, 2] = mesh["z1"]
        triangle_vertex_array[:, 1, 2] = mesh["z2"]
        triangle_vertex_array[:, 2, 2] = mesh["z3"]
        mesh["areas"] = triangle_area(triangle_vertex_array)

        # EIGEN: Calculate derived eigenmode parameters
        # Set n_modes to the greater of strike-slip or dip slip modes
        mesh["n_modes"] = np.max(
            [
                config.n_modes_strike_slip,
                config.n_modes_dip_slip,
            ]
        )
        mesh["n_modes_total"] = config.n_modes_strike_slip + config.n_modes_dip_slip
        mesh["config"] = config

        _compute_mesh_edge_elements(mesh)
        _compute_mesh_perimeter(mesh)

        from celeri.spatial import get_shared_sides, get_tri_shared_sides_distances

        mesh["share"] = get_shared_sides(mesh["verts"])
        mesh["tri_shared_sides_distances"] = get_tri_shared_sides_distances(
            mesh["share"],
            mesh["x_centroid"],
            mesh["y_centroid"],
            mesh["z_centroid"],
        )

        from celeri.celeri_util import get_2component_index

        boundary_constraints = [
            ("top", config.top_slip_rate_constraint, mesh["top_elements"]),
            ("bot", config.bot_slip_rate_constraint, mesh["bot_elements"]),
            ("side", config.side_slip_rate_constraint, mesh["side_elements"]),
        ]
        for name, constraint_value, elements in boundary_constraints:
            if constraint_value > 0:
                indices = np.asarray(np.where(elements))
                mesh[f"{name}_slip_idx"] = get_2component_index(indices)
            else:
                mesh[f"{name}_slip_idx"] = np.array([], dtype=np.int64)

        mesh["n_tde_constraints"] = _compute_n_tde_constraints(
            mesh["n_tde"],
            mesh["top_slip_idx"],
            mesh["bot_slip_idx"],
            mesh["side_slip_idx"],
        )

        mesh["eigenvalues"], mesh["eigenvectors"] = _get_eigenvalues_and_eigenvectors(
            mesh["n_modes"],
            mesh["x_centroid"],
            mesh["y_centroid"],
            mesh["z_centroid"],
            matern_nu=config.matern_nu,
            matern_length_scale=config.matern_length_scale,
            matern_length_units=config.matern_length_units,
            eigenvector_algorithm=config.eigenvector_algorithm,
        )

        logger.success(f"Read: {filename}")
        return cls(**mesh)

    @property
    def file_name(self) -> Path | None:
        return self.config.mesh_filename

    @property
    def name(self) -> str:
        """Return the name of the mesh configuration."""
        return self.file_name.stem if self.file_name is not None else "unknown"

    def to_disk(self, output_dir: str | Path):
        """Save the mesh configuration to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the MeshConfig separately
        config_file = output_dir / "mesh_config.json"
        with config_file.open("w") as f:
            f.write(self.config.model_dump_json(indent=4))

        # Use the general dataclass serialization function for the rest
        dataclass_to_disk(self, output_dir, skip={"config"})

        logger.success(f"Mesh configuration saved to {output_dir}")

    @classmethod
    def from_disk(cls, input_dir: str | Path):
        """Load the mesh configuration from a JSON file."""
        input_dir = Path(input_dir)
        config_file = input_dir / "mesh_config.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Mesh configuration file {config_file} not found.")

        with config_file.open() as f:
            config_data = json.load(f)

        config = MeshConfig(**config_data)

        # Use the general dataclass deserialization function with the config as extra data
        mesh = dataclass_from_disk(cls, input_dir, extra={"config": config})

        # Compute shared sides and distances if not already loaded (backward compatibility)
        from celeri.spatial import get_shared_sides, get_tri_shared_sides_distances

        mesh.share = get_shared_sides(mesh.verts)
        mesh.tri_shared_sides_distances = get_tri_shared_sides_distances(
            mesh.share,
            mesh.x_centroid,
            mesh.y_centroid,
            mesh.z_centroid,
        )

        # Compute slip indices for boundary constraints
        from celeri.celeri_util import get_2component_index

        boundary_constraints = [
            ("top", config.top_slip_rate_constraint, mesh.top_elements),
            ("bot", config.bot_slip_rate_constraint, mesh.bot_elements),
            ("side", config.side_slip_rate_constraint, mesh.side_elements),
        ]
        for name, constraint_value, elements in boundary_constraints:
            if constraint_value > 0:
                indices = np.asarray(np.where(elements))
                setattr(mesh, f"{name}_slip_idx", get_2component_index(indices))
            else:
                setattr(mesh, f"{name}_slip_idx", np.array([], dtype=np.int64))

        mesh.n_tde_constraints = _compute_n_tde_constraints(
            mesh.n_tde,
            mesh.top_slip_idx,
            mesh.bot_slip_idx,
            mesh.side_slip_idx,
        )

        if mesh.eigenvalues is None or mesh.eigenvectors is None:
            mesh.eigenvalues, mesh.eigenvectors = _get_eigenvalues_and_eigenvectors(
                mesh.n_modes,
                mesh.x_centroid,
                mesh.y_centroid,
                mesh.z_centroid,
                matern_nu=config.matern_nu,
                matern_length_scale=config.matern_length_scale,
                matern_length_units=config.matern_length_units,
                eigenvector_algorithm=config.eigenvector_algorithm,
            )

        return mesh
