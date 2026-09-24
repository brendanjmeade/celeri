#!/usr/bin/env python3
# %%
import argparse
from pathlib import Path
from typing import cast

import meshio
import numpy as np
from loguru import logger

from celeri.celeri_util import (
    cart2sph,
    sph2cart,
)
from celeri.constants import KM2M, RADIUS_EARTH
from celeri.mesh import _triangle_normals_enu

WINDING_TOLERANCE = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh_file_name",
        type=Path,
        help="Name of .msh file.",
    )

    args = dict(vars(parser.parse_args()))

    # Standalone reader for a single .msh file
    mesh: dict = {}
    filename = Path(args["mesh_file_name"])

    # Suppress meshio's stdout output (it prints a newline)
    meshobj = meshio.read(filename)
    points = cast(np.ndarray, meshobj.points)
    points[:, 0] = np.where(points[:, 0] < 0, points[:, 0] + 360, points[:, 0])
    # if points[:, 2].min() >= 0.0 and points[:, 2].max() > 0.0:
    #     raise ValueError(
    #         f"Mesh {config.mesh_filename} has only non-negative depths; celeri "
    #         "expects mesh depths in km, negative below the surface"
    #     )
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
        RADIUS_EARTH + KM2M * mesh["dep1"],
    )
    mesh["x2"], mesh["y2"], mesh["z2"] = sph2cart(
        mesh["lon2"],
        mesh["lat2"],
        RADIUS_EARTH + KM2M * mesh["dep2"],
    )
    mesh["x3"], mesh["y3"], mesh["z3"] = sph2cart(
        mesh["lon3"],
        mesh["lat3"],
        RADIUS_EARTH + KM2M * mesh["dep3"],
    )

    # Cartesian triangle centroids
    mesh["x_centroid"] = (mesh["x1"] + mesh["x2"] + mesh["x3"]) / 3.0
    mesh["y_centroid"] = (mesh["y1"] + mesh["y2"] + mesh["y3"]) / 3.0
    mesh["z_centroid"] = (mesh["z1"] + mesh["z2"] + mesh["z3"]) / 3.0

    # Spherical triangle centroids, from the Cartesian centroid so that
    # triangles straddling the 0/360 meridian are handled
    centroid_lon, centroid_lat, _ = cart2sph(
        mesh["x_centroid"], mesh["y_centroid"], mesh["z_centroid"]
    )
    mesh["lon_centroid"] = np.rad2deg(centroid_lon) % 360.0
    mesh["lat_centroid"] = np.rad2deg(centroid_lat)

    # Element orientation from the Cartesian legs expressed in a local
    # east-north-up frame at each centroid (a plain (dlon, dlat) frame
    # would stretch the east leg by 1/cos(lat) and bias strike and dip)
    mesh["nv"] = _triangle_normals_enu(
        np.c_[mesh["x1"], mesh["y1"], mesh["z1"]],
        np.c_[mesh["x2"], mesh["y2"], mesh["z2"]],
        np.c_[mesh["x3"], mesh["y3"], mesh["z3"]],
        mesh["lon_centroid"],
        mesh["lat_centroid"],
    )

    # The vertex order of a triangle sets the direction of its normal, and
    # the sign of every dip-slip quantity on the element follows it: the
    # kinematic factor 1/cos(dip) and cutde's dip-slip direction both
    # reverse with the winding, so the element's physics is the same
    # either way but the stored dip-slip numbers change sign. The file's
    # winding is kept as is; report it rather than change it.
    unit_z = mesh["nv"][:, 2] / np.linalg.norm(mesh["nv"], axis=1)
    n_downward = int(np.sum(unit_z < -WINDING_TOLERANCE))
    n_upward = int(np.sum(unit_z > WINDING_TOLERANCE))
    if n_downward > 0 and n_upward > 0:
        logger.warning(
            f"Mesh {filename} has mixed vertex winding: "
            f"{n_downward} of {mesh['n_tde']} triangles have downward "
            "normals, so the sign of their dip-slip rates (kinematic and "
            "elastic) is opposite to the rest of the mesh and the Laplacian "
            "smoothing, eigenmodes and bounds mix two sign conventions on "
            "this mesh. Reorder those triangles in the mesh file."
        )
    elif n_downward > 0:
        logger.info(
            f"Mesh {filename}: every triangle has a downward "
            "normal (dip reported in (90, 180]); stored dip-slip rates are "
            "negative for reverse motion on this mesh"
        )

    # Swap CW-wound elements
    verts_ccw = np.vstack([verts[:, 2], verts[:, 1]]).T
    downward = unit_z < -WINDING_TOLERANCE
    verts[downward, 1:] = verts_ccw[downward, :]
    logger.info(f"Swapped nodes of {n_downward} elements to give CCW winding.")

    # Send updated verts back to mesh object
    cells = [("triangle", verts)]
    meshout = meshio.Mesh(points, cells)

    # Write to .msh file
    out_filename = Path(filename.parent, filename.stem + "_ccw.msh")
    meshio.gmsh.write(mesh=meshout, binary=False, filename=out_filename)
    logger.info(f"Wrote updated mesh to {out_filename}.")


if __name__ == "__main__":
    main()
