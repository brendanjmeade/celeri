from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import meshio
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

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
    # Find indices of elements lining top, bottom, and sides of each mesh

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
    np.std(depth_bins)
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
    np.std(depth_bins)
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


@dataclass
class Mesh:
    """Represents a triangular mesh for fault modeling."""

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

    # TOOD(Adrian): Can we move those function into this module
    # and make them non-optional?

    # Computed in operators._store_all_mesh_smoothing_matrices
    share: np.ndarray | None = None
    tri_shared_sides_distances: np.ndarray | None = None
    n_tde_constraints: int | None = None
    # computed in operators._store_tde_slip_rate_constraints
    top_slip_idx: np.ndarray | None = None
    bot_slip_idx: np.ndarray | None = None
    side_slip_idx: np.ndarray | None = None
    coup_idx: np.ndarray | None = None
    ss_slip_idx: np.ndarray | None = None
    ds_slip_idx: np.ndarray | None = None
    # computed in operators.get_rotation_to_tri_slip_rate_partials
    closest_segment_idx: np.ndarray | None = None
    east_labels: np.ndarray | None = None
    west_labels: np.ndarray | None = None

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
        azimuth, elevation, r = cart2sph(
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

        logger.success(f"Read: {filename}")
        # Convert dict to Mesh dataclass
        return cls(**mesh)

    @property
    def file_name(self) -> Path:
        return self.config.mesh_filename

    @property
    def name(self) -> str:
        """Return the name of the mesh configuration."""
        return self.file_name.stem

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
        return dataclass_from_disk(cls, input_dir, extra={"config": config})
