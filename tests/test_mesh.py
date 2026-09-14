import meshio
import numpy as np
import pytest
from pydantic import ValidationError

from celeri.mesh import Mesh, MeshConfig, ScalarBound


def _strip_mesh(path, depths):
    """A two-column strip of triangles along latitude 40 with the given depths (km)."""
    points = []
    for depth in depths:
        for k in range(4):
            points.append((240.0 + 0.05 * k, 40.0 + 0.02 * len(points), depth))
    points = np.array(points, dtype=float)
    tris = []
    n = 4
    for row in range(len(depths) - 1):
        for k in range(n - 1):
            a, b, c, d = (
                row * n + k,
                row * n + k + 1,
                (row + 1) * n + k + 1,
                (row + 1) * n + k,
            )
            tris += [[a, b, c], [a, c, d]]
    meshio.write(
        path,
        meshio.Mesh(points, [("triangle", np.array(tris))]),
        file_format="gmsh22",
        binary=False,
    )
    return path


def _mesh_config(path, **overrides):
    params = dict(
        file_name=path.parent / "mesh_parameters.json",
        mesh_filename=str(path),
        smoothing_weight=1.0,
        n_modes_strike_slip=2,
        n_modes_dip_slip=2,
        top_slip_rate_constraint=0,
        bottom_slip_rate_constraint=0,
        side_slip_rate_constraint=0,
        matern_nu=2.5,
        matern_length_scale=0.2,
        matern_length_units="diameters",
        eigenvector_algorithm="eigh",
    )
    params.update(overrides)
    return MeshConfig.model_validate(params)


def test_positive_down_depths_are_rejected(tmp_path):
    path = _strip_mesh(tmp_path / "positive.msh", depths=(0.0, 10.0, 20.0))
    with pytest.raises(ValueError, match="negative below the surface"):
        Mesh.from_params(_mesh_config(path))


def test_more_modes_than_elements_is_rejected(tmp_path):
    path = _strip_mesh(tmp_path / "small.msh", depths=(0.0, -10.0, -20.0))
    Mesh.from_params(_mesh_config(path, n_modes_strike_slip=12, n_modes_dip_slip=12))
    with pytest.raises(ValueError, match="eigenmodes were requested"):
        Mesh.from_params(_mesh_config(path, n_modes_strike_slip=13, n_modes_dip_slip=1))


def test_scalar_bound_must_be_ordered():
    assert ScalarBound.model_validate([0, 1]).upper == 1
    assert ScalarBound.model_validate([None, 5]).lower is None
    with pytest.raises(ValidationError):
        ScalarBound.model_validate([1, 0])
    with pytest.raises(ValidationError):
        ScalarBound.model_validate({"lower": 0, "uper": 1})
