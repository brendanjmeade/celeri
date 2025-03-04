import addict
import argparse
import copy
import cvxopt
import datetime
import glob
import h5py
import json
import meshio
import scipy
import pyproj
import pytest
import os
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.path
import warnings
import shutil
import sys
import pickle
import timeit
import IPython
import numpy as np
import pandas as pd
import okada_wrapper
import cutde.halfspace as cutde_halfspace
import scipy.io as sio

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm

from matplotlib.colors import Normalize

from ismember import ismember
from loguru import logger
from tqdm import tqdm

from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import scipy.sparse
import scipy.sparse.linalg


from . import celeri_closure
from .celeri_util import sph2cart, cart2sph
from celeri.hmatrix import build_hmatrix_from_mesh_tdes


###############################################################################################################################
#                                                                                                                             #
#    ,ad8888ba,    ,ad8888ba,    888b      88   ad88888ba  888888888888    db         888b      88  888888888888  ad88888ba   #
#   d8"'    `"8b  d8"'    `"8b   8888b     88  d8"     "8b      88        d88b        8888b     88       88      d8"     "8b  #
#  d8'           d8'        `8b  88 `8b    88  Y8,              88       d8'`8b       88 `8b    88       88      Y8,          #
#  88            88          88  88  `8b   88  `Y8aaaaa,        88      d8'  `8b      88  `8b   88       88      `Y8aaaaa,    #
#  88            88          88  88   `8b  88    `"""""8b,      88     d8YaaaaY8b     88   `8b  88       88        `"""""8b,  #
#  Y8,           Y8,        ,8P  88    `8b 88          `8b      88    d8""""""""8b    88    `8b 88       88              `8b  #
#   Y8a.    .a8P  Y8a.    .a8P   88     `8888  Y8a     a8P      88   d8'        `8b   88     `8888       88      Y8a     a8P  #
#    `"Y8888Y"'    `"Y8888Y"'    88      `888   "Y88888P"       88  d8'          `8b  88      `888       88       "Y88888P"   #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################


GEOID = pyproj.Geod(ellps="WGS84")
KM2M = 1.0e3
M2MM = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)
DEG_PER_MYR_TO_RAD_PER_YR = 1 / 1e3
# The conversion should be 1 / 1e3. Linear units for Cartesian conversions are
# in meters, but we need to convert them to mm to be consistent with mm/yr
# geodetic constraints units. Rotation constraints are expressed in deg/Myr,
# and when applied in celeri we are effectively using m*rad/Myr. To convert
# to the right rate units, we need 1e-3*m*rad/Myr. This conversion is applied in
# get_data_vector (JPL 12/31/23)
N_MESH_DIM = 3
EPS = np.finfo(float).eps


###############################################################
#                                                             #
#                                                             #
#  88  888b      88  88888888ba   88        88  888888888888  #
#  88  8888b     88  88      "8b  88        88       88       #
#  88  88 `8b    88  88      ,8P  88        88       88       #
#  88  88  `8b   88  88aaaaaa8P'  88        88       88       #
#  88  88   `8b  88  88""""""'    88        88       88       #
#  88  88    `8b 88  88           88        88       88       #
#  88  88     `8888  88           Y8a.    .a8P       88       #
#  88  88      `888  88            `"Y8888Y"'        88       #
#                                                             #
#                                                             #
###############################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command_file_name", type=str, help="Name of *_command.json file"
    )
    parser.add_argument(
        "--segment_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_segment.csv file",
    )
    parser.add_argument(
        "--station_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_station.csv file",
    )
    parser.add_argument(
        "--block_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_block.csv file",
    )
    parser.add_argument(
        "--mesh_parameters_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_mesh_parameters.json file",
    )
    parser.add_argument(
        "--los_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_los.csv file",
    )
    parser.add_argument(
        "--solve_type",
        type=str,
        default=None,
        required=False,
        help="Solution type (dense | hmatrix)",
    )
    parser.add_argument(
        "--repl",
        type=int,
        default=0,
        required=False,
        help="Flag for dropping into REPL (0 | 1)",
    )
    parser.add_argument(
        "--pickle_save",
        type=int,
        default=0,
        required=False,
        help="Flag for saving major data structures in pickle file (0 | 1)",
    )
    parser.add_argument(
        "--plot_input_summary",
        type=int,
        default=0,
        required=False,
        help="Flag for saving summary plot of input data (0 | 1)",
    )
    parser.add_argument(
        "--plot_estimation_summary",
        type=int,
        default=0,
        required=False,
        help="Flag for saving summary plot of model results (0 | 1)",
    )
    parser.add_argument(
        "--save_elastic",
        type=int,
        default=0,
        required=False,
        help="Flag for saving elastic calculations (0 | 1)",
    )
    parser.add_argument(
        "--reuse_elastic",
        type=int,
        default=0,
        required=False,
        help="Flag for reusing elastic calculations (0 | 1)",
    )
    parser.add_argument(
        "--snap_segments",
        type=int,
        default=0,
        required=False,
        help="Flag for snapping segments (0 | 1)",
    )
    parser.add_argument(
        "--atol",
        type=int,
        default=None,
        required=False,
        help="Primary tolerance for H-matrix solve",
    )
    parser.add_argument(
        "--btol",
        type=int,
        default=None,
        required=False,
        help="Secondary tolerance for H-matrix solve",
    )
    parser.add_argument(
        "--iterative_solver",
        type=str,
        default=None,
        required=False,
        help="Interative solver type (lsqr | lsmr)",
    )

    args = addict.Dict(vars(parser.parse_args()))
    return args


def process_args(command: Dict, args: Dict):
    for key in command:
        if key in args:
            if args[key] is not None:
                logger.warning(f"ORIGINAL: command.{key}: {command[key]}")
                command[key] = args[key]
                logger.warning(f"REPLACED: command.{key}: {command[key]}")
            else:
                logger.info(f"command.{key}: {command[key]}")


def get_command(command_file_name):
    # NOTE: Rename to `read_command`?
    """Read *command.json file and return contents as a dictionary

    Args:
        command_file_name (string): Path to command file

    Returns:
        command (Dict): Dictionary with content of command file
    """
    with open(command_file_name, "r") as f:
        command = json.load(f)
    command = addict.Dict(command)  # Convert to dot notation dictionary
    command.file_name = command_file_name

    # Add run_name and output_path
    # command.run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    command.run_name = get_new_folder_name()
    command.output_path = os.path.join(command.base_runs_folder, command.run_name)
    # command.file_name = command_file_name

    # Sort command keys alphabetically for readability
    command = addict.Dict(sorted(command.items()))

    return command


def get_new_folder_name():
    """
    Generate a new folder name based on existing numeric folder names.

    This function scans the current directory for folders with numeric names,
    identifies the highest number, and returns a new folder name that is one
    greater than the highest number, formatted as a zero-padded 10-digit string.

    Returns:
        str: A new folder name as a zero-padded 10-digit string.

    Raises:
        ValueError: If no numeric folder names are found in the current directory.

    Example:
        If the current directory contains folders named "0000000001", "0000000002",
        and "0000000003", the function will return "0000000004".
    """
    # Get all folder names
    folder_names = glob.glob("./../runs/*/")

    # Remove trailing slashes
    folder_names = [folder_name.rstrip(os.sep) for folder_name in folder_names]

    # Remove anything before numerical folder name
    folder_names = [folder_name[-10:] for folder_name in folder_names]

    # Check to see if the folder name is a native run number
    folder_names_runs = list()
    for folder_name in folder_names:
        try:
            folder_names_runs.append(int(folder_name))
        except ValueError as ex:
            pass

    # Get new folder name
    if len(folder_names_runs) == 0:
        new_folder_name = "0000000001"
    else:
        new_folder_number = np.max(folder_names_runs) + 1
        new_folder_name = f"{new_folder_number:010d}"

    return new_folder_name


def get_default_mesh_parameters():
    # Empty/default arrays for mesh parameters
    default_mesh_parameters = {
        "mesh_filename": "",
        "smoothing_weight": 1e0,
        "n_modes_strike_slip": 10,
        "n_modes_dip_slip": 10,
        "top_slip_rate_constraint": 0,
        "bot_slip_rate_constraint": 0,
        "side_slip_rate_constraint": 0,
        "top_slip_rate_weight": 1e0,
        "bot_slip_rate_weight": 1e0,
        "side_slip_rate_weight": 1e0,
        "a_priori_slip_filename": "",
        "ss_slip_constraint_idx": [],
        "ss_slip_constraint_rate": [],
        "ss_slip_constraint_sig": [],
        "ss_slip_constraint_weight": [],
        "ds_slip_constraint_idx": [],
        "ds_slip_constraint_rate": [],
        "ds_slip_constraint_sig": [],
        "ds_slip_constraint_weight": [],
        "coupling_constraint_idx": [],
        "coupling_constraint_frac": [1.0],
        "coupling_constraint_sigma": [1.0],
        "coupling_constraint_weight": [1e0],
        "mesh_tde_bound": [1],
        "mesh_tde_slip_rate_bound_lower_ss": ["-inf"],
        "mesh_tde_slip_rate_bound_upper_ss": ["inf"],
        "mesh_tde_slip_rate_bound_lower_ds": [0],
        "mesh_tde_slip_rate_bound_upper_ds": [1],
        "mesh_tde_coupling_bound": [0],
        "mesh_tde_coupling_bound_lower_ss": ["-inf"],
        "mesh_tde_coupling_upper_ss": ["inf"],
        "mesh_tde_coupling_lower_ds": [0],
        "mesh_tde_coupling_upper_ds": [1],
        "mesh_tde_modes_bc_weight": 1e0,
    }
    return default_mesh_parameters


def read_mesh(filename):
    # Standalone reader for a single .msh file
    # Outputs to mesh (Dict)
    mesh = addict.Dict()
    meshobj = meshio.read(filename)
    mesh.file_name = filename
    mesh.points = meshobj.points
    mesh.verts = meshio.CellBlock("triangle", meshobj.get_cells_type("triangle")).data

    # Expand mesh coordinates
    mesh.lon1 = mesh.points[mesh.verts[:, 0], 0]
    mesh.lon2 = mesh.points[mesh.verts[:, 1], 0]
    mesh.lon3 = mesh.points[mesh.verts[:, 2], 0]
    mesh.lat1 = mesh.points[mesh.verts[:, 0], 1]
    mesh.lat2 = mesh.points[mesh.verts[:, 1], 1]
    mesh.lat3 = mesh.points[mesh.verts[:, 2], 1]
    mesh.dep1 = mesh.points[mesh.verts[:, 0], 2]
    mesh.dep2 = mesh.points[mesh.verts[:, 1], 2]
    mesh.dep3 = mesh.points[mesh.verts[:, 2], 2]
    mesh.centroids = np.mean(mesh.points[mesh.verts, :], axis=1)
    # Cartesian coordinates in meters
    mesh.x1, mesh.y1, mesh.z1 = sph2cart(
        mesh.lon1,
        mesh.lat1,
        RADIUS_EARTH + KM2M * mesh.dep1,
    )
    mesh.x2, mesh.y2, mesh.z2 = sph2cart(
        mesh.lon2,
        mesh.lat2,
        RADIUS_EARTH + KM2M * mesh.dep2,
    )
    mesh.x3, mesh.y3, mesh.z3 = sph2cart(
        mesh.lon3,
        mesh.lat3,
        RADIUS_EARTH + KM2M * mesh.dep3,
    )

    # Cartesian triangle centroids
    mesh.x_centroid = (mesh.x1 + mesh.x2 + mesh.x3) / 3.0
    mesh.y_centroid = (mesh.y1 + mesh.y2 + mesh.y3) / 3.0
    mesh.z_centroid = (mesh.z1 + mesh.z2 + mesh.z3) / 3.0

    # Spherical triangle centroids
    mesh.lon_centroid = (mesh.lon1 + mesh.lon2 + mesh.lon3) / 3.0
    mesh.lat_centroid = (mesh.lat1 + mesh.lat2 + mesh.lat3) / 3.0

    # Cross products for orientations
    tri_leg1 = np.transpose(
        [
            np.deg2rad(mesh.lon2 - mesh.lon1),
            np.deg2rad(mesh.lat2 - mesh.lat1),
            (1 + KM2M * mesh.dep2 / RADIUS_EARTH)
            - (1 + KM2M * mesh.dep1 / RADIUS_EARTH),
        ]
    )
    tri_leg2 = np.transpose(
        [
            np.deg2rad(mesh.lon3 - mesh.lon1),
            np.deg2rad(mesh.lat3 - mesh.lat1),
            (1 + KM2M * mesh.dep3 / RADIUS_EARTH)
            - (1 + KM2M * mesh.dep1 / RADIUS_EARTH),
        ]
    )
    mesh.nv = np.cross(tri_leg1, tri_leg2)
    azimuth, elevation, r = cart2sph(
        mesh.nv[:, 0],
        mesh.nv[:, 1],
        mesh.nv[:, 2],
    )
    mesh.strike = wrap2360(-np.rad2deg(azimuth))
    mesh.dip = 90 - np.rad2deg(elevation)
    mesh.dip_flag = mesh.dip != 90

    # Assign all mesh parameters to this mesh
    # for key, value in mesh_param[i].items():
    #     mesh[key] = value

    # # Assign empty arrays for any unspecified parameters
    # mesh_default = get_default_mesh_parameters()
    # for key, value in mesh_default.items():
    #     if key not in mesh:
    #         mesh[key] = value

    mesh.n_tde = mesh.lon1.size

    # Calcuate areas of each triangle in mesh
    triangle_vertex_array = np.zeros((mesh.n_tde, 3, 3))
    triangle_vertex_array[:, 0, 0] = mesh.x1
    triangle_vertex_array[:, 1, 0] = mesh.x2
    triangle_vertex_array[:, 2, 0] = mesh.x3
    triangle_vertex_array[:, 0, 1] = mesh.y1
    triangle_vertex_array[:, 1, 1] = mesh.y2
    triangle_vertex_array[:, 2, 1] = mesh.y3
    triangle_vertex_array[:, 0, 2] = mesh.z1
    triangle_vertex_array[:, 1, 2] = mesh.z2
    triangle_vertex_array[:, 2, 2] = mesh.z3
    mesh.areas = triangle_area(triangle_vertex_array)

    # EIGEN: Calculate derived eigenmode parameters
    # Set n_modes to the greater of strike-slip or dip slip modes
    # mesh.n_modes = np.max([mesh.n_modes_strike_slip, mesh.n_modes_dip_slip])
    # mesh.n_modes_total = mesh.n_modes_strike_slip + mesh.n_modes_dip_slip

    logger.success(f"Read: {filename}")
    return mesh


def read_data(command: Dict):
    logger.info("Reading data files")
    # Read segment data
    segment = pd.read_csv(command.segment_file_name)
    segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]
    logger.success(f"Read: {command.segment_file_name}")

    # Read block data
    block = pd.read_csv(command.block_file_name)
    block = block.loc[:, ~block.columns.str.match("Unnamed")]
    logger.success(f"Read: {command.block_file_name}")

    # Read mesh data - List of dictionary version
    meshes = []
    if command.mesh_parameters_file_name != "":
        with open(command.mesh_parameters_file_name) as f:
            mesh_param = json.load(f)
            logger.success(f"Read: {command.mesh_parameters_file_name}")

        if len(mesh_param) > 0:
            for i in range(len(mesh_param)):
                meshes.append(addict.Dict())
                # Read current mesh using separate read_mesh function
                this_mesh = read_mesh(mesh_param[i]["mesh_filename"])
                meshes[i] = this_mesh

                # Assign all mesh parameters to this mesh
                for key, value in mesh_param[i].items():
                    meshes[i][key] = value

                # Assign empty arrays for any unspecified parameters
                mesh_default = get_default_mesh_parameters()
                for key, value in mesh_default.items():
                    if key not in meshes[i]:
                        meshes[i][key] = value

                # EIGEN: Calculate derived eigenmode parameters
                # Set n_modes to the greater of strike-slip or dip slip modes
                meshes[i].n_modes = np.max(
                    [
                        mesh_param[i]["n_modes_strike_slip"],
                        mesh_param[i]["n_modes_dip_slip"],
                    ]
                )
                meshes[i].n_modes_total = (
                    mesh_param[i]["n_modes_strike_slip"]
                    + mesh_param[i]["n_modes_dip_slip"]
                )
            # Get mesh perimeter information for all meshes
            get_mesh_edge_elements(meshes)
            get_mesh_perimeter(meshes)

    # Read station data
    if (
        not command.__contains__("station_file_name")
        or len(command.station_file_name) == 0
    ):
        station = pd.DataFrame(
            columns=[
                "lon",
                "lat",
                "corr",
                "other1",
                "name",
                "east_vel",
                "north_vel",
                "east_sig",
                "north_sig",
                "flag",
                "up_vel",
                "up_sig",
                "east_adjust",
                "north_adjust",
                "up_adjust",
                "depth",
                "x",
                "y",
                "z",
                "block_label",
            ]
        )
        logger.info(f"No station_file_name")
    else:
        station = pd.read_csv(command.station_file_name)
        station = station.loc[:, ~station.columns.str.match("Unnamed")]
        logger.success(f"Read: {command.station_file_name}")

    # Read Mogi source data
    if not command.__contains__("mogi_file_name") or len(command.mogi_file_name) == 0:
        mogi = pd.DataFrame(
            columns=[
                "name",
                "lon",
                "lat",
                "depth",
                "volume_change_flag",
                "volume_change",
                "volume_change_sig",
            ]
        )
        logger.info(f"No mogi_file_name")
    else:
        mogi = pd.read_csv(command.mogi_file_name)
        mogi = mogi.loc[:, ~mogi.columns.str.match("Unnamed")]
        logger.success(f"Read: {command.mogi_file_name}")

    # Read SAR data
    if not command.__contains__("sar_file_name") or len(command.sar_file_name) == 0:
        sar = pd.DataFrame(
            columns=[
                "lon",
                "lat",
                "depth",
                "line_of_sight_change_val",
                "line_of_sight_change_sig",
                "look_vector_x",
                "look_vector_y",
                "look_vector_z",
                "reference_point_x",
                "reference_point_y",
            ]
        )
        logger.info(f"No sar_file_name")
    else:
        sar = pd.read_csv(command.sar_file_name)
        sar = sar.loc[:, ~sar.columns.str.match("Unnamed")]
        logger.success(f"Read: {command.sar_file_name}")
    return segment, block, meshes, station, mogi, sar


####################################################################################################
#                                                                                                  #
#                                                                                                  #
#  88888888ba   88888888ba     ,ad8888ba,      ,ad8888ba,   88888888888  ad88888ba    ad88888ba    #
#  88      "8b  88      "8b   d8"'    `"8b    d8"'    `"8b  88          d8"     "8b  d8"     "8b   #
#  88      ,8P  88      ,8P  d8'        `8b  d8'            88          Y8,          Y8,           #
#  88aaaaaa8P'  88aaaaaa8P'  88          88  88             88aaaaa     `Y8aaaaa,    `Y8aaaaa,     #
#  88""""""'    88""""88'    88          88  88             88"""""       `"""""8b,    `"""""8b,   #
#  88           88    `8b    Y8,        ,8P  Y8,            88                  `8b          `8b   #
#  88           88     `8b    Y8a.    .a8P    Y8a.    .a8P  88          Y8a     a8P  Y8a     a8P   #
#  88           88      `8b    `"Y8888Y"'      `"Y8888Y"'   88888888888  "Y88888P"    "Y88888P"    #
#                                                                                                  #
#                                                                                                  #
####################################################################################################
def triangle_area(triangles):
    # The norm of the cross product of two sides is twice the area
    # https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
    return np.linalg.norm(triangle_normal(triangles), axis=1) / 2.0


def get_mesh_perimeter(meshes):
    for i in range(len(meshes)):
        x_coords = meshes[i].points[:, 0]
        y_coords = meshes[i].points[:, 1]
        meshes[i].x_perimeter = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        meshes[i].y_perimeter = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        meshes[i].x_perimeter = np.append(
            meshes[i].x_perimeter, x_coords[meshes[i].ordered_edge_nodes[0, 0]]
        )
        meshes[i].y_perimeter = np.append(
            meshes[i].y_perimeter, y_coords[meshes[i].ordered_edge_nodes[0, 0]]
        )


def triangle_normal(triangles):
    # The cross product of two sides is a normal vector
    # https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
    return np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1
    )


def process_station(station, command):
    if bool(command.unit_sigmas):  # Assign unit uncertainties, if requested
        station.east_sig = np.ones_like(station.east_sig)
        station.north_sig = np.ones_like(station.north_sig)
        station.up_sig = np.ones_like(station.up_sig)

    station["depth"] = np.zeros_like(station.lon)
    station["x"], station["y"], station["z"] = sph2cart(
        station.lon, station.lat, RADIUS_EARTH
    )
    station = station.drop(np.where(station.flag == 0)[0])
    station = station.reset_index(drop=True)
    return station


def process_sar(sar, command):
    """
    Preprocessing of SAR data.
    """
    if sar.empty:
        sar["depth"] = np.zeros_like(sar.lon)

        # Set the uncertainties to reflect the weights specified in the command file
        # In constructing the data weight vector, the value is 1./Sar.dataSig.^2, so
        # the adjustment made here is sar.dataSig / np.sqrt(command.sarWgt)
        sar.line_of_sight_change_sig = sar.line_of_sight_change_sig / np.sqrt(
            command.sar_weight
        )
        sar["x"], sar["y"], sar["z"] = sph2cart(sar.lon, sar.lat, RADIUS_EARTH)
        sar["block_label"] = -1 * np.ones_like(sar.x)
    else:
        sar["dep"] = []
        sar["x"] = []
        sar["y"] = []
        sar["x"] = []
        sar["block_label"] = []
    return sar


def merge_geodetic_data(assembly, station, sar):
    """
    Merge GPS and InSAR data to a single assembly object
    """
    assembly.data.n_stations = len(station)
    assembly.data.n_sar = len(sar)
    assembly.data.east_vel = station.east_vel.to_numpy()
    assembly.sigma.east_sig = station.east_sig.to_numpy()
    assembly.data.north_vel = station.north_vel.to_numpy()
    assembly.sigma.north_sig = station.north_sig.to_numpy()
    assembly.data.up_vel = station.up_vel.to_numpy()
    assembly.sigma.up_sig = station.up_sig.to_numpy()
    assembly.data.sar_line_of_sight_change_val = sar.line_of_sight_change_val.to_numpy()
    assembly.sigma.sar_line_of_sight_change_sig = (
        sar.line_of_sight_change_sig.to_numpy()
    )
    assembly.data.lon = np.concatenate((station.lon.to_numpy(), sar.lon.to_numpy()))
    assembly.data.lat = np.concatenate((station.lat.to_numpy(), sar.lat.to_numpy()))
    assembly.data.depth = np.concatenate(
        (station.depth.to_numpy(), sar.depth.to_numpy())
    )
    assembly.data.x = np.concatenate((station.x.to_numpy(), sar.x.to_numpy()))
    assembly.data.y = np.concatenate((station.y.to_numpy(), sar.y.to_numpy()))
    assembly.data.z = np.concatenate((station.z.to_numpy(), sar.z.to_numpy()))
    assembly.data.block_label = np.concatenate(
        (station.block_label.to_numpy(), sar.block_label.to_numpy())
    )
    assembly.index.sar_coordinate_idx = np.arange(
        len(station), len(station) + len(sar)
    )  # TODO: Not sure this is correct
    return assembly


def process_segment(segment, command, meshes):
    """
    Add derived fields to segment dataframe
    """

    if bool(command.snap_segments):
        segment = snap_segments(segment, meshes)

    segment["x1"], segment["y1"], segment["z1"] = sph2cart(
        segment.lon1, segment.lat1, RADIUS_EARTH
    )
    segment["x2"], segment["y2"], segment["z2"] = sph2cart(
        segment.lon2, segment.lat2, RADIUS_EARTH
    )

    segment = order_endpoints_sphere(segment)

    segment["length"] = np.zeros(len(segment))
    segment["azimuth"] = np.zeros(len(segment))
    for i in range(len(segment)):
        segment.azimuth.values[i], _, segment.length.values[i] = GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )  # Segment azimuth, Segment length in meters

    # This calculation needs to account for the periodic nature of longitude.
    # Calculate the periodic longitudinal separation.
    # @BJM: Is this better done with GEIOD?
    sep = segment.lon2 - segment.lon1
    periodic_lon_separation = np.where(
        sep > 180, sep - 360, np.where(sep < -180, sep + 360, sep)
    )
    segment["mid_lon_plate_carree"] = (
        segment.lon1.values + periodic_lon_separation / 2.0
    )

    # No worries for latitude because there's no periodicity.
    segment["mid_lat_plate_carree"] = (segment.lat1.values + segment.lat2.values) / 2.0
    segment["mid_lon"] = np.zeros_like(segment.lon1)
    segment["mid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment.mid_lon.values[i], segment.mid_lat.values[i] = GEOID.npts(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i], 1
        )[0]
    segment.mid_lon.values[segment.mid_lon < 0.0] += 360.0

    segment["mid_x"], segment["mid_y"], segment["mid_z"] = sph2cart(
        segment.mid_lon, segment.mid_lat, RADIUS_EARTH
    )
    segment = locking_depth_manager(segment, command)
    segment = zero_mesh_segment_locking_depth(segment, meshes)
    segment = segment_centroids(segment)
    return segment


def order_endpoints_sphere(segment):
    """
    Endpoint ordering function, placing west point first.
    This converts the endpoint coordinates from spherical to Cartesian,
    then takes the cross product to test for ordering (i.e., a positive z
    component of cross(point1, point2) means that point1 is the western
    point). This method works for both (-180, 180) and (0, 360) longitude
    conventions.
    BJM: Not sure why cross product approach was definitely not working in
    python so I revereted to relative longitude check which sould be fine because
    we're always in 0-360 space.
    """
    segment_copy = copy.deepcopy(segment)
    endpoints1 = np.transpose(np.array([segment.x1, segment.y1, segment.z1]))
    endpoints2 = np.transpose(np.array([segment.x2, segment.y2, segment.z2]))
    cross_product = np.cross(endpoints1, endpoints2)
    swap_endpoint_idx = np.where(cross_product[:, 2] < 0)
    segment_copy.lon1.values[swap_endpoint_idx] = segment.lon2.values[swap_endpoint_idx]
    segment_copy.lat1.values[swap_endpoint_idx] = segment.lat2.values[swap_endpoint_idx]
    segment_copy.lon2.values[swap_endpoint_idx] = segment.lon1.values[swap_endpoint_idx]
    segment_copy.lat2.values[swap_endpoint_idx] = segment.lat1.values[swap_endpoint_idx]
    return segment_copy


def locking_depth_manager(segment, command):
    """
    This function assigns the locking depths given in the command file to any
    segment that has the same locking depth flag.  Segments with flag =
    0, 1 are untouched.
    """
    segment = segment.copy(deep=True)
    segment.locking_depth.values[segment.locking_depth_flag == 2] = (
        command.locking_depth_flag2
    )
    segment.locking_depth.values[segment.locking_depth_flag == 3] = (
        command.locking_depth_flag3
    )
    segment.locking_depth.values[segment.locking_depth_flag == 4] = (
        command.locking_depth_flag4
    )
    segment.locking_depth.values[segment.locking_depth_flag == 5] = (
        command.locking_depth_flag5
    )

    if bool(command.locking_depth_override_flag):
        segment.locking_depth.values = command.locking_depth_override_value
    return segment


def zero_mesh_segment_locking_depth(segment, meshes):
    """
    This function sets the locking depths of any segments that trace
    a mesh to zero, so that they have no rectangular elastic strain
    contribution, as the elastic strain is accounted for by the mesh.

    To have its locking depth set to zero, the segment's patch_flag
    and patch_file_name fields must not be equal to zero but also
    less than the number of available mesh files.
    """
    segment = segment.copy(deep=True)
    toggle_off = np.where(
        (segment.patch_flag != 0)
        & (segment.patch_file_name >= 0)
        & (segment.patch_file_name <= len(meshes))
    )[0]
    segment.locking_depth.values[toggle_off] = 0
    segment.burial_depth.values[toggle_off] = 0
    return segment


def segment_centroids(segment):
    """Calculate segment centroids."""
    segment["centroid_x"] = np.zeros_like(segment.lon1)
    segment["centroid_y"] = np.zeros_like(segment.lon1)
    segment["centroid_z"] = np.zeros_like(segment.lon1)
    segment["centroid_lon"] = np.zeros_like(segment.lon1)
    segment["centroid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment_forward_azimuth, _, _ = GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )
        segment_down_dip_azimuth = segment_forward_azimuth + 90.0 * np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        )
        azimuth_xy_cartesian = (segment.y2[i] - segment.y1[i]) / (
            segment.x2[i] - segment.x1[i]
        )
        azimuth_xy_cartesian = np.arctan(-1.0 / azimuth_xy_cartesian)
        segment.centroid_z.values[i] = (
            segment.locking_depth[i] - segment.burial_depth[i]
        ) / 2.0
        segment_down_dip_distance = segment.centroid_z[i] / np.abs(
            np.tan(np.deg2rad(segment.dip[i]))
        )
        (
            segment.centroid_lon.values[i],
            segment.centroid_lat.values[i],
            _,
        ) = GEOID.fwd(
            segment.mid_lon[i],
            segment.mid_lat[i],
            segment_down_dip_azimuth,
            segment_down_dip_distance,
        )
        segment.centroid_x.values[i] = segment.mid_x[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.cos(azimuth_xy_cartesian)
        segment.centroid_y.values[i] = segment.mid_y[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.sin(azimuth_xy_cartesian)
    segment.centroid_lon.values[segment.centroid_lon < 0.0] += 360.0
    return segment


def snap_segments(segment, meshes):
    """
    Replace segments tracing meshes with the actual top edges of those meshes
    """
    # For each mesh, find associated segments
    cut_segment_idx = []
    all_edge_segment = make_default_segment(0)
    for i in range(len(meshes)):
        these_segments = np.where(
            (segment.patch_flag != 0) & (segment.patch_file_name == i)
        )[0]
        cut_segment_idx = np.append(cut_segment_idx, these_segments)
        # Get top coordinates of the mesh
        top_el_indices = np.where(meshes[i].top_elements)
        edges = np.sort(meshes[i].ordered_edge_nodes[:-1], axis=1)
        top_verts = np.sort(meshes[i].verts[top_el_indices], axis=1)
        # Concatenate edges with vertex pairs
        edges1 = np.vstack((edges, top_verts[:, 0:2]))
        # Find unique edges
        unique_edges1, unique_indices1, unique_counts1 = np.unique(
            edges1, axis=0, return_index=True, return_counts=True
        )
        # But keep those edges that appear twice
        top_edge_indices1 = unique_indices1[np.where(unique_counts1 == 2)]
        # Same process with 2nd and 3rd columns of the mesh vertex array
        edges2 = np.vstack((edges, top_verts[:, 1:3]))
        unique_edges2, unique_indices2, unique_counts2 = np.unique(
            edges2, axis=0, return_index=True, return_counts=True
        )
        top_edge_indices2 = unique_indices2[np.where(unique_counts2 == 2)]
        # Final selection
        top_edge_indices = np.sort(np.hstack((top_edge_indices1, top_edge_indices2)))
        # Get new segment coordinates from these indices
        edge_segs = make_default_segment(len(top_edge_indices))
        edge_segs.lon1 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 0], 0
        ]
        edge_segs.lat1 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 0], 1
        ]
        edge_segs.lon2 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 1], 0
        ]
        edge_segs.lat2 = meshes[i].points[
            meshes[i].ordered_edge_nodes[top_edge_indices, 1], 1
        ]
        edge_segs.locking_depth = +15
        edge_segs.patch_flag = +1
        edge_segs.patch_file_name = +i + 1
        all_edge_segment = all_edge_segment.append(edge_segs)

    # Get indices of segments to keep
    keep_segment_idx = np.setdiff1d(range(len(segment.lon1)), cut_segment_idx)
    # Isolate kept segments and reindex
    keep_segment = segment.loc[keep_segment_idx]
    new_index = range(len(keep_segment_idx))
    keep_segment.index = new_index
    # Find hanging endpoints; these mark terminations of mesh-replaced segments
    lons = np.hstack((keep_segment.lon1, keep_segment.lon2))
    lats = np.hstack((keep_segment.lat1, keep_segment.lat2))
    coords = np.array([lons, lats])
    unique_coords, indices, counts = np.unique(
        coords, axis=1, return_index=True, return_counts=True
    )
    hanging_idx = indices[np.where(counts == 1)]
    # Calculate distance to all mesh edge coordinates
    # Can't just use the terminations because we might have triple junctions in the middle of a mesh
    elons = np.hstack((all_edge_segment.lon1, all_edge_segment.lon2))
    elats = np.hstack((all_edge_segment.lat1, all_edge_segment.lat2))
    ecoords = np.array([elons, elats])
    hang_to_mesh_dist = cdist(coords[:, hanging_idx].T, ecoords.T)
    # Find closest edge coordinate
    closest_edge_idx = np.argmin(hang_to_mesh_dist, axis=1)
    # Replace segment coordinates with closest mesh coordinate
    # Using a loop because we need to evaluate whether to replace endpoint 1 or 2
    for i in range(len(closest_edge_idx)):
        if hanging_idx[i] < len(keep_segment.lon1):
            keep_segment.loc[hanging_idx[i], "lon1"] = ecoords[0, closest_edge_idx[i]]
            keep_segment.loc[hanging_idx[i], "lat1"] = ecoords[1, closest_edge_idx[i]]
        else:
            keep_segment.loc[hanging_idx[i] - len(keep_segment.lon1), "lon2"] = ecoords[
                0, closest_edge_idx[i]
            ]
            keep_segment.loc[hanging_idx[i] - len(keep_segment.lon1), "lat2"] = ecoords[
                1, closest_edge_idx[i]
            ]
    # Merge with mesh edge segments
    new_segment = keep_segment.append(all_edge_segment)
    new_index = range(len(new_segment))
    new_segment.index = new_index
    return new_segment


def assign_block_labels(segment, station, block, mogi, sar):
    """
    Ben Thompson's implementation of the half edge approach to the
    block labeling problem and east/west assignment.
    """
    # segment = split_segments_crossing_meridian(segment)

    np_segments = np.zeros((len(segment), 2, 2))
    np_segments[:, 0, 0] = segment.lon1.to_numpy()
    np_segments[:, 1, 0] = segment.lon2.to_numpy()
    np_segments[:, 0, 1] = segment.lat1.to_numpy()
    np_segments[:, 1, 1] = segment.lat2.to_numpy()

    closure = celeri_closure.run_block_closure(np_segments)
    labels = celeri_closure.get_segment_labels(closure)

    segment["west_labels"] = labels[:, 0]
    segment["east_labels"] = labels[:, 1]

    # Check for unprocessed indices
    unprocessed_indices = np.union1d(
        np.where(segment["east_labels"] < 0),
        np.where(segment["west_labels"] < 0),
    )
    if len(unprocessed_indices) > 0:
        logger.warning("Found unproccessed indices")

    # Find relative areas of each block to identify an external block
    block["area_steradians"] = -1 * np.ones(len(block))
    block["area_plate_carree"] = -1 * np.ones(len(block))
    for i in range(closure.n_polygons()):
        vs = closure.polygons[i].vertices
        block.area_steradians.values[i] = closure.polygons[i].area_steradians
        block.area_plate_carree.values[i] = polygon_area(vs[:, 0], vs[:, 1])

    # Assign block labels points to block interior points
    block["block_label"] = closure.assign_points(
        block.interior_lon.to_numpy(), block.interior_lat.to_numpy()
    )

    # I copied this from the bottom of:
    # https://stackoverflow.com/questions/39992502/rearrange-rows-of-pandas-dataframe-based-on-list-and-keeping-the-order
    # and I definitely don't understand it all but emperically it seems to work.
    block = (
        block.set_index(block.block_label, append=True)
        .sort_index(level=1)
        .reset_index(1, drop=True)
    )
    block = block.reset_index()
    block = block.loc[:, ~block.columns.str.match("index")]

    # Assign block labels to GPS stations
    if not station.empty:
        station["block_label"] = closure.assign_points(
            station.lon.to_numpy(), station.lat.to_numpy()
        )

    # Assign block labels to SAR locations
    if not sar.empty:
        sar["block_label"] = closure.assign_points(
            sar.lon.to_numpy(), sar.lat.to_numpy()
        )

    # Assign block labels to Mogi sources
    if not mogi.empty:
        mogi["block_label"] = closure.assign_points(
            mogi.lon.to_numpy(), mogi.lat.to_numpy()
        )

    return closure, block


def station_row_keep(assembly):
    """
    Determines which station rows should be retained based on up velocities
    TODO: I do not understand this!!!
    TODO: The logic in the first conditional seems to indicate that if there are
    no vertical velocities as a part of the data then they should be eliminated.
    TODO: Perhaps it would be better to make this a flag in command???
    """
    if np.sum(np.abs(assembly.data.up_vel)) == 0:
        assembly.index.station_row_keep = np.setdiff1d(
            np.arange(0, assembly.index.sz_rotation[0]),
            np.arange(2, assembly.index.sz_rotation[0], 3),
        )
    else:
        assembly.index.station_row_keep = np.arange(0, assembly.index.sz_rotation[1])
    return assembly


def get_block_centroid(segment, block_idx):
    """
    Calculate centroid of a block based on boundary polygon
    We take all block vertices (including duplicates) and estimate
    the centroid by taking the average of longitude and latitude
    weighted by the length of the segment that each vertex is
    attached to.
    """
    segments_with_block_idx = np.union1d(
        np.where(segment.west_labels == block_idx)[0],
        np.where(segment.east_labels == block_idx)[0],
    )
    lon0 = np.concatenate(
        (segment.lon1[segments_with_block_idx], segment.lon2[segments_with_block_idx])
    )
    lat0 = np.concatenate(
        (segment.lat1[segments_with_block_idx], segment.lat2[segments_with_block_idx])
    )
    lengths = np.concatenate(
        (
            segment.length[segments_with_block_idx],
            segment.length[segments_with_block_idx],
        )
    )
    block_centroid_lon = np.average(lon0, weights=lengths)
    block_centroid_lat = np.average(lat0, weights=lengths)
    return block_centroid_lon, block_centroid_lat


def get_shared_sides(vertices):
    """
    Determine the indices of the triangular elements sharing
    one side with a particular element.
    Inputs:
    vertices: n x 3 array containing the 3 vertex indices of the n elements,
        assumes that values increase monotonically from 1:n

    Outputs:
    share: n x 3 array containing the indices of the m elements sharing a
        side with each of the n elements.  "-1" values in the array
        indicate elements with fewer than m neighbors (i.e., on
        the edge of the geometry).

    In general, elements will have 1 (mesh corners), 2 (mesh edges), or 3
    (mesh interiors) neighbors, but in the case of branching faults that
    have been adjusted with mergepatches, it's for edges and corners to
    also up to 3 neighbors.
    """
    # Make side arrays containing vertex indices of sides
    side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
    side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
    side_3 = np.sort(np.vstack((vertices[:, 0], vertices[:, 2])).T, 1)
    sides_all = np.vstack((side_1, side_2, side_3))

    # Find the unique sides - each side can part of at most 2 elements
    _, first_occurence_idx = np.unique(sides_all, return_index=True, axis=0)
    _, last_occurence_idx = np.unique(np.flipud(sides_all), return_index=True, axis=0)
    last_occurence_idx = sides_all.shape[0] - last_occurence_idx - 1

    # Shared sides are those whose first and last indices are not equal
    shared = np.where((last_occurence_idx - first_occurence_idx) != 0)[0]

    # These are the indices of the shared sides
    sside1 = first_occurence_idx[shared]  # What should I name these variables?
    sside2 = last_occurence_idx[shared]

    el1, sh1 = np.unravel_index(
        sside1, vertices.shape, order="F"
    )  # "F" is for fortran ordering.  What should I call this variables?
    el2, sh2 = np.unravel_index(sside2, vertices.shape, order="F")
    share = -1 * np.ones((vertices.shape[0], 3))
    for i in range(el1.size):
        share[el1[i], sh1[i]] = el2[i]
        share[el2[i], sh2[i]] = el1[i]
    share = share.astype(int)
    return share


def get_tri_shared_sides_distances(share, x_centroid, y_centroid, z_centroid):
    """
    Calculates the distances between the centroids of adjacent triangular
    elements, for use in smoothing algorithms.

    Inputs:
    share: n x 3 array output from ShareSides, containing the indices
        of up to 3 elements that share a side with each of the n elements.
    x_centroid: x coordinates of element centroids
    y_centroid: y coordinates of element centroids
    z_centroid: z coordinates of element centroids

    Outputs:
    dists: n x 3 array containing distance between each of the n elements
        and its 3 or fewer neighbors.  A distance of 0 does not imply
        collocated elements, but rather implies that there are fewer
        than 3 elements that share a side with the element in that row.
    """
    tri_shared_sides_distances = np.zeros(share.shape)
    for i in range(share.shape[0]):
        tri_shared_sides_distances[i, :] = np.sqrt(
            (x_centroid[i] - x_centroid[share[i, :]]) ** 2.0
            + (y_centroid[i] - y_centroid[share[i, :]]) ** 2.0
            + (z_centroid[i] - z_centroid[share[i, :]]) ** 2.0
        )
    tri_shared_sides_distances[np.where(share == -1)] = 0
    return tri_shared_sides_distances


def get_ordered_edge_nodes(meshes: List):
    """Find exterior edges of each mesh and return them in the dictionary
    for each mesh.

    Args:
        meshes (List): list of mesh dictionaries
    """

    for i in range(len(meshes)):
        # Make side arrays containing vertex indices of sides
        vertices = meshes[i].verts
        side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
        side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
        side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)
        all_sides = np.vstack((side_1, side_2, side_3))
        unique_sides, sides_count = np.unique(all_sides, return_counts=True, axis=0)
        edge_nodes = unique_sides[np.where(sides_count == 1)]

        meshes[i].ordered_edge_nodes = np.zeros_like(edge_nodes)
        meshes[i].ordered_edge_nodes[0, :] = edge_nodes[0, :]
        last_row = 0
        for j in range(1, len(edge_nodes)):
            idx = np.where(
                (edge_nodes == meshes[i].ordered_edge_nodes[j - 1, 1])
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
            meshes[i].ordered_edge_nodes[j, :] = edge_nodes[next_row, next_col_ord]
            last_row = (
                next_row  # Update last_row so that it's excluded in the next iteration
            )


def get_mesh_edge_elements(meshes: List):
    # Find indices of elements lining top, bottom, and sides of each mesh

    get_ordered_edge_nodes(meshes)

    for i in range(len(meshes)):
        coords = meshes[i].points
        vertices = meshes[i].verts

        # Get element centroid depths
        el_depths = coords[vertices, 2]
        centroid_depths = np.mean(el_depths, axis=1)

        # Arrays of all element side node pairs
        side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
        side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
        side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)

        # Sort edge node array
        sorted_edge_nodes = np.sort(meshes[i].ordered_edge_nodes, 1)

        # Indices of element sides that are in edge node array
        side_1_in_edge, side_1_in_edge_idx = ismember(sorted_edge_nodes, side_1, "rows")
        side_2_in_edge, side_2_in_edge_idx = ismember(sorted_edge_nodes, side_2, "rows")
        side_3_in_edge, side_3_in_edge_idx = ismember(sorted_edge_nodes, side_3, "rows")

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
        bin_std = np.std(depth_bins)
        zero_idx = np.where(depth_count == 0)[0]
        if len(zero_idx) > 0:
            if (
                np.abs(depth_bin_max[zero_idx[0]] - depth_bin_min[zero_idx[-1] + 1])
                > 10
            ):
                tops[centroid_depths < depth_bins[zero_idx[0]]] = False
        # Assign in to meshes dict
        meshes[i].top_elements = tops

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
        bin_std = np.std(depth_bins)
        zero_idx = np.where(depth_count == 0)[0]
        if len(zero_idx) > 0:
            if abs(depth_bin_min[zero_idx[-1]] - depth_bin_max[zero_idx[0] - 1]) > 10:
                bots[centroid_depths > depth_bin_min[zero_idx[-1]]] = False
        # Assign in to meshes dict
        meshes[i].bot_elements = bots

        # Side elements are a set difference between all edges and tops, bottoms
        sides = np.full(len(vertices), False, dtype=bool)
        sides[side_1_in_edge_idx] = True
        sides[side_2_in_edge_idx] = True
        sides[side_3_in_edge_idx] = True
        sides[np.where(tops != 0)] = False
        sides[np.where(bots != 0)] = False
        meshes[i].side_elements = sides


def get_index(assembly, station, block, meshes, mogi):
    # Create dictionary to store indices and sizes for operator building
    index = addict.Dict()
    index.n_stations = assembly.data.n_stations
    # index.n_stations = len(station)
    index.vertical_velocities = np.arange(2, 3 * index.n_stations, 3)
    index.n_blocks = len(block)
    index.n_block_constraints = assembly.data.n_block_constraints
    index.station_row_keep_index = get_keep_index_12(3 * len(station))
    index.start_station_row = 0
    index.end_station_row = 2 * len(station)
    index.start_block_col = 0
    index.end_block_col = 3 * len(block)
    index.start_block_constraints_row = index.end_station_row
    index.end_block_constraints_row = (
        index.start_block_constraints_row + 3 * index.n_block_constraints
    )
    index.n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    index.start_slip_rate_constraints_row = index.end_block_constraints_row
    index.end_slip_rate_constraints_row = (
        index.start_slip_rate_constraints_row + index.n_slip_rate_constraints
    )
    index.n_tde_total = 0
    index.n_tde_constraints_total = 0
    # A bunch of declarations of TDE-related indices as arrays, otherwise they're dicts
    index.n_tde = np.zeros((len(meshes),), dtype=int)
    index.n_tde_constraints = np.zeros((len(meshes),), dtype=int)
    index.start_tde_col = np.zeros((len(meshes),), dtype=int)
    index.end_tde_col = np.zeros((len(meshes),), dtype=int)
    index.start_tde_smoothing_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_smoothing_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_top_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_top_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_bot_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_bot_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_side_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_side_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_coup_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_coup_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_ss_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_ss_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_ds_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.end_tde_ds_slip_constraint_row = np.zeros((len(meshes),), dtype=int)
    index.start_tde_col_eigen = np.zeros((len(meshes),), dtype=int)
    index.end_tde_col_eigen = np.zeros((len(meshes),), dtype=int)
    index.start_tde_constraint_row_eigen = np.zeros((len(meshes),), dtype=int)
    index.end_tde_constraint_row_eigen = np.zeros((len(meshes),), dtype=int)
    for i in range(len(meshes)):
        index.n_tde[i] = meshes[i].n_tde
        index.n_tde_total += index.n_tde[i]
        index.n_tde_constraints[i] = meshes[i].n_tde_constraints
        index.n_tde_constraints_total += index.n_tde_constraints[i]
        if i == 0:
            index.start_tde_col[i] = index.end_block_col
            index.end_tde_col[i] = index.start_tde_col[i] + 2 * index.n_tde[i]
            index.start_tde_smoothing_row[i] = index.end_slip_rate_constraints_row
            index.end_tde_smoothing_row[i] = (
                index.start_tde_smoothing_row[i] + 2 * index.n_tde[i]
            )
            index.start_tde_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_constraint_row[i] = (
                index.start_tde_constraint_row[i] + index.n_tde_constraints[i]
            )
            index.start_tde_top_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_top_constraint_row[i] = index.start_tde_top_constraint_row[
                i
            ] + len(meshes[i].top_slip_idx)
            index.start_tde_bot_constraint_row[i] = index.end_tde_top_constraint_row[i]
            index.end_tde_bot_constraint_row[i] = index.start_tde_bot_constraint_row[
                i
            ] + len(meshes[i].bot_slip_idx)
            index.start_tde_side_constraint_row[i] = index.end_tde_bot_constraint_row[i]
            index.end_tde_side_constraint_row[i] = index.start_tde_side_constraint_row[
                i
            ] + len(meshes[i].side_slip_idx)
            index.start_tde_coup_constraint_row[i] = index.end_tde_side_constraint_row[
                i
            ]
            index.end_tde_coup_constraint_row[i] = index.start_tde_coup_constraint_row[
                i
            ] + len(meshes[i].coup_idx)
            index.start_tde_ss_slip_constraint_row[i] = (
                index.end_tde_coup_constraint_row[i]
            )
            index.end_tde_ss_slip_constraint_row[i] = (
                index.start_tde_ss_slip_constraint_row[i] + len(meshes[i].ss_slip_idx)
            )
            index.start_tde_ds_slip_constraint_row[i] = (
                index.end_tde_ss_slip_constraint_row[i]
            )
            index.end_tde_ds_slip_constraint_row[i] = (
                index.start_tde_ds_slip_constraint_row[i] + len(meshes[i].ds_slip_idx)
            )
        else:
            index.start_tde_col[i] = index.end_tde_col[i - 1]
            index.end_tde_col[i] = index.start_tde_col[i] + 2 * index.n_tde[i]
            index.start_tde_smoothing_row[i] = index.end_tde_constraint_row[i - 1]
            index.end_tde_smoothing_row[i] = (
                index.start_tde_smoothing_row[i] + 2 * index.n_tde[i]
            )
            index.start_tde_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_constraint_row[i] = (
                index.start_tde_constraint_row[i] + index.n_tde_constraints[i]
            )
            index.start_tde_top_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_top_constraint_row[i] = index.start_tde_top_constraint_row[
                i
            ] + len(meshes[i].top_slip_idx)
            index.start_tde_bot_constraint_row[i] = index.end_tde_top_constraint_row[i]
            index.end_tde_bot_constraint_row[i] = index.start_tde_bot_constraint_row[
                i
            ] + len(meshes[i].bot_slip_idx)
            index.start_tde_side_constraint_row[i] = index.end_tde_bot_constraint_row[i]
            index.end_tde_side_constraint_row[i] = index.start_tde_side_constraint_row[
                i
            ] + len(meshes[i].side_slip_idx)
            index.start_tde_coup_constraint_row[i] = index.end_tde_side_constraint_row[
                i
            ]
            index.end_tde_coup_constraint_row[i] = index.start_tde_coup_constraint_row[
                i
            ] + len(meshes[i].coup_idx)
            index.start_tde_ss_slip_constraint_row[i] = (
                index.end_tde_coup_constraint_row[i]
            )
            index.end_tde_ss_slip_constraint_row[i] = (
                index.start_tde_ss_slip_constraint_row[i] + len(meshes[i].ss_slip_idx)
            )
            index.start_tde_ds_slip_constraint_row[i] = (
                index.end_tde_ss_slip_constraint_row[i]
            )
            index.end_tde_ds_slip_constraint_row[i] = (
                index.start_tde_ds_slip_constraint_row[i] + len(meshes[i].ds_slip_idx)
            )

            # # Add eigen specific entries to index
            # index.start_tde_col_eigen = np.zeros(len(meshes), dtype=int)
            # index.end_tde_col_eigen = np.zeros(len(meshes), dtype=int)
            # index.start_tde_constraint_row_eigen = np.zeros(len(meshes), dtype=int)
            # index.end_tde_constraint_row_eigen = np.zeros(len(meshes), dtype=int)

            # # TODO: Double-check this:
            # # Eigenvalue indexing should be unchanged by the above specification
            # # of indices for individual constraint styles, because n_tde_constraints
            # # is tracked correctly as the total number of constraint rows (2 each for
            # # coupling constraints, one each for slip component constraints)
            # if len(meshes) > 0:
            #     for i in range(len(meshes)):
            #         if i == 0:
            #             index.start_tde_col_eigen[i] = 3 * len(block)
            #             index.end_tde_col_eigen[i] = (
            #                 index.start_tde_col_eigen[i] + 2 * meshes[i].n_eigen
            #             )
            #             if meshes[i].n_tde_constraints > 0:
            #                 index.start_tde_constraint_row_eigen[i] = (
            #                     index.end_slip_rate_constraints_row
            #                 )
            #                 index.end_tde_constraint_row_eigen[i] = (
            #                     index.start_tde_constraint_row_eigen[i]
            #                     + meshes[i].n_tde_constraints
            #                 )
            #             else:
            #                 index.start_tde_constraint_row_eigen[i] = 0
            #                 index.end_tde_constraint_row_eigen[i] = 0
            #         else:
            #             index.start_tde_col_eigen[i] = index.end_tde_col_eigen[i - 1]
            #             index.end_tde_col_eigen[i] = (
            #                 index.start_tde_col_eigen[i] + 2 * meshes[i].n_eigen
            #             )
            #             if meshes[i].n_tde_constraints > 0:
            #                 index.start_tde_constraint_row_eigen[i] = (
            #                     index.end_tde_constraint_row_eigen[i - 1]
            #                 )
            #                 index.end_tde_constraint_row_eigen[i] = (
            #                     index.start_tde_constraint_row_eigen[i]
            #                     + meshes[i].n_tde_constraints
            #                 )
            #             else:
            #                 index.start_tde_constraint_row_eigen[i] = (
            #                     index.end_tde_constraint_row_eigen[i - 1]
            #                 )
            #                 index.end_tde_constraint_row_eigen[i] = (
            #                     index.end_tde_constraint_row_eigen[i - 1]
            #                 )
    # Index for block strain
    index.n_block_strain_components = 3 * np.sum(block.strain_rate_flag)
    index.start_block_strain_col = index.end_tde_col[-1]
    index.end_block_strain_col = (
        index.start_block_strain_col + index.n_block_strain_components
    )
    # Index for Mogi sources
    index.n_mogi = len(mogi)
    index.start_mogi_col = index.end_block_strain_col
    index.end_mogi_col = index.start_mogi_col + index.n_mogi

    index.n_operator_rows = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    index.n_operator_cols = 3 * index.n_blocks + 2 * index.n_tde_total
    return index


def get_index_no_meshes(assembly, station, block):
    # NOTE: Merge with above if possible.
    # Make sure empty meshes work

    # Create dictionary to store indices and sizes for operator building
    index = addict.Dict()
    index.n_stations = assembly.data.n_stations
    # index.n_stations = len(station)
    index.vertical_velocities = np.arange(2, 3 * index.n_stations, 3)
    index.n_blocks = len(block)
    index.n_block_constraints = assembly.data.n_block_constraints
    index.station_row_keep_index = get_keep_index_12(3 * len(station))
    index.start_station_row = 0
    index.end_station_row = 2 * len(station)
    index.start_block_col = 0
    index.end_block_col = 3 * len(block)
    index.start_block_constraints_row = index.end_station_row
    index.end_block_constraints_row = (
        index.start_block_constraints_row + 3 * index.n_block_constraints
    )
    index.n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    index.start_slip_rate_constraints_row = index.end_block_constraints_row
    index.end_slip_rate_constraints_row = (
        index.start_slip_rate_constraints_row + index.n_slip_rate_constraints
    )

    index.n_tde_total = 0
    index.n_tde_constraints_total = 0
    index.n_operator_rows = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
    )
    index.n_operator_cols = 3 * index.n_blocks
    return index


def get_index_eigen(assembly, segment, station, block, meshes, mogi):
    # Create dictionary to store indices and sizes for operator building
    index = addict.Dict()
    index.n_blocks = len(block)
    index.n_segments = len(segment)
    index.n_stations = assembly.data.n_stations
    index.n_meshes = len(meshes)
    index.n_mogis = len(mogi)
    index.vertical_velocities = np.arange(2, 3 * index.n_stations, 3)
    index.n_block_constraints = assembly.data.n_block_constraints
    index.station_row_keep_index = get_keep_index_12(3 * index.n_stations)
    index.start_station_row = 0
    index.end_station_row = 2 * index.n_stations
    index.start_block_col = 0
    index.end_block_col = 3 * index.n_blocks
    index.start_block_constraints_row = index.end_station_row
    index.end_block_constraints_row = (
        index.start_block_constraints_row + 3 * index.n_block_constraints
    )
    index.n_slip_rate_constraints = assembly.data.slip_rate_constraints.size
    index.start_slip_rate_constraints_row = index.end_block_constraints_row
    index.end_slip_rate_constraints_row = (
        index.start_slip_rate_constraints_row + index.n_slip_rate_constraints
    )
    index.n_tde_total = 0
    index.n_tde_constraints_total = 0

    # A bunch of declarations of TDE-related indices as arrays, otherwise they're dicts
    index.n_tde = np.zeros(index.n_meshes, dtype=int)
    index.n_tde_constraints = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_col = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_col = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_smoothing_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_smoothing_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_top_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_top_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_bot_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_bot_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_side_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_side_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_coup_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_coup_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ss_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ss_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ds_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ds_slip_constraint_row = np.zeros(index.n_meshes, dtype=int)

    for i in range(index.n_meshes):
        index.n_tde[i] = meshes[i].n_tde
        index.n_tde_total += index.n_tde[i]
        index.n_tde_constraints[i] = meshes[i].n_tde_constraints
        index.n_tde_constraints_total += index.n_tde_constraints[i]
        if i == 0:
            index.start_tde_col[i] = index.end_block_col
            index.end_tde_col[i] = index.start_tde_col[i] + 2 * index.n_tde[i]
            index.start_tde_smoothing_row[i] = index.end_slip_rate_constraints_row
            index.end_tde_smoothing_row[i] = (
                index.start_tde_smoothing_row[i] + 2 * index.n_tde[i]
            )
            index.start_tde_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_constraint_row[i] = (
                index.start_tde_constraint_row[i] + index.n_tde_constraints[i]
            )
            index.start_tde_top_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_top_constraint_row[i] = index.start_tde_top_constraint_row[
                i
            ] + len(meshes[i].top_slip_idx)
            index.start_tde_bot_constraint_row[i] = index.end_tde_top_constraint_row[i]
            index.end_tde_bot_constraint_row[i] = index.start_tde_bot_constraint_row[
                i
            ] + len(meshes[i].bot_slip_idx)
            index.start_tde_side_constraint_row[i] = index.end_tde_bot_constraint_row[i]
            index.end_tde_side_constraint_row[i] = index.start_tde_side_constraint_row[
                i
            ] + len(meshes[i].side_slip_idx)
            index.start_tde_coup_constraint_row[i] = index.end_tde_side_constraint_row[
                i
            ]
            index.end_tde_coup_constraint_row[i] = index.start_tde_coup_constraint_row[
                i
            ] + len(meshes[i].coup_idx)
            index.start_tde_ss_slip_constraint_row[i] = (
                index.end_tde_coup_constraint_row[i]
            )
            index.end_tde_ss_slip_constraint_row[i] = (
                index.start_tde_ss_slip_constraint_row[i] + len(meshes[i].ss_slip_idx)
            )
            index.start_tde_ds_slip_constraint_row[i] = (
                index.end_tde_ss_slip_constraint_row[i]
            )
            index.end_tde_ds_slip_constraint_row[i] = (
                index.start_tde_ds_slip_constraint_row[i] + len(meshes[i].ds_slip_idx)
            )
        else:
            index.start_tde_col[i] = index.end_tde_col[i - 1]
            index.end_tde_col[i] = index.start_tde_col[i] + 2 * index.n_tde[i]
            index.start_tde_smoothing_row[i] = index.end_tde_constraint_row[i - 1]
            index.end_tde_smoothing_row[i] = (
                index.start_tde_smoothing_row[i] + 2 * index.n_tde[i]
            )
            index.start_tde_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_constraint_row[i] = (
                index.start_tde_constraint_row[i] + index.n_tde_constraints[i]
            )
            index.start_tde_top_constraint_row[i] = index.end_tde_smoothing_row[i]
            index.end_tde_top_constraint_row[i] = index.start_tde_top_constraint_row[
                i
            ] + len(meshes[i].top_slip_idx)
            index.start_tde_bot_constraint_row[i] = index.end_tde_top_constraint_row[i]
            index.end_tde_bot_constraint_row[i] = index.start_tde_bot_constraint_row[
                i
            ] + len(meshes[i].bot_slip_idx)
            index.start_tde_side_constraint_row[i] = index.end_tde_bot_constraint_row[i]
            index.end_tde_side_constraint_row[i] = index.start_tde_side_constraint_row[
                i
            ] + len(meshes[i].side_slip_idx)
            index.start_tde_coup_constraint_row[i] = index.end_tde_side_constraint_row[
                i
            ]
            index.end_tde_coup_constraint_row[i] = index.start_tde_coup_constraint_row[
                i
            ] + len(meshes[i].coup_idx)
            index.start_tde_ss_slip_constraint_row[i] = (
                index.end_tde_coup_constraint_row[i]
            )
            index.end_tde_ss_slip_constraint_row[i] = (
                index.start_tde_ss_slip_constraint_row[i] + len(meshes[i].ss_slip_idx)
            )
            index.start_tde_ds_slip_constraint_row[i] = (
                index.end_tde_ss_slip_constraint_row[i]
            )
            index.end_tde_ds_slip_constraint_row[i] = (
                index.start_tde_ds_slip_constraint_row[i] + len(meshes[i].ds_slip_idx)
            )

    # EIGEN: Get total number of eigenmodes
    index.n_eigen_total = 0
    for i in range(index.n_meshes):
        index.n_eigen_total += meshes[i].n_modes_strike_slip
        index.n_eigen_total += meshes[i].n_modes_dip_slip

    # EIGEN: Count eigenmodes for each mesh
    index.n_modes_mesh = np.zeros(index.n_meshes, dtype=int)
    for i in range(index.n_meshes):
        index.n_modes_mesh[i] = (
            meshes[i].n_modes_strike_slip + meshes[i].n_modes_dip_slip
        )

    # EIGEN: Column and row indices
    index.start_col_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_col_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ss_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ss_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_ds_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_ds_slip_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.start_tde_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.end_tde_constraint_row_eigen = np.zeros(index.n_meshes, dtype=int)

    # EIGEN: columns and rows for eigenmodes to velocity
    for i in range(index.n_meshes):
        # First mesh
        if i == 0:
            # Locations for eigenmodes to velocities
            index.start_col_eigen[i] = 3 * index.n_blocks
            index.end_col_eigen[i] = index.start_col_eigen[i] + index.n_modes_mesh[i]
            index.start_tde_row_eigen[i] = 0
            index.end_row_eigen[i] = 2 * index.n_stations

        # Meshes after first mesh
        else:
            # Locations for eigenmodes to velocities
            index.start_col_eigen[i] = index.end_col_eigen[i - 1]
            index.end_col_eigen[i] = index.start_col_eigen[i] + index.n_modes_mesh[i]
            index.start_tde_row_eigen[i] = 0
            index.end_row_eigen[i] = 2 * index.n_stations

    # EIGEN: Set initial values to follow segment slip rate constraints
    index.start_tde_constraint_row_eigen[0] = index.end_slip_rate_constraints_row

    index.end_tde_constraint_row_eigen[0] = (
        index.start_tde_constraint_row_eigen[0] + index.n_tde_constraints[0]
    )

    # EIGEN: Rows for eigen to TDE boundary conditions
    for i in range(1, index.n_meshes):
        # All constraints for eigen to constraint matrix
        index.start_tde_constraint_row_eigen[i] = index.end_tde_constraint_row_eigen[
            i - 1
        ]
        index.end_tde_constraint_row_eigen[i] = (
            index.start_tde_constraint_row_eigen[i] + index.n_tde_constraints[i]
        )

    # EIGEN: Rows for QP bounds
    # Create index components for linear inequality matrix and data vector
    index.qp_constraint_tde_rate_start_row_eigen = np.zeros(index.n_meshes, dtype=int)
    index.qp_constraint_slip_rate_end_row_eigen = np.zeros(index.n_meshes, dtype=int)

    # Create index components for slip rate constraints
    index.qp_constraint_slip_rate_start_row_eigen = np.zeros(index.n_meshes, dtype=int)

    index.qp_constraint_tde_rate_start_row_eigen[0] = 0
    index.qp_constraint_tde_rate_end_row_eigen[0] = (
        index.qp_constraint_tde_rate_start_row_eigen[0] + 4 * index.n_tde[0]
    )

    for i in range(1, index.n_meshes):
        # Start row for current mesh
        index.qp_constraint_tde_rate_start_row_eigen[i] = (
            index.qp_constraint_tde_rate_end_row_eigen[i - 1]
        )

        # End row for current mesh
        index.qp_constraint_tde_rate_end_row_eigen[i] = (
            index.qp_constraint_tde_rate_start_row_eigen[i] + 4 * index.n_tde[i]
        )

    # Index for block strain
    index.n_strain_blocks = np.sum(block.strain_rate_flag)
    index.n_block_strain_components = 3 * index.n_strain_blocks
    index.start_block_strain_col = index.end_tde_col[-1]
    index.end_block_strain_col = (
        index.start_block_strain_col + index.n_block_strain_components
    )

    # EIGEN: Index for block strain
    index.start_block_strain_col_eigen = index.end_col_eigen[-1]
    index.end_block_strain_col_eigen = (
        index.start_block_strain_col_eigen + index.n_block_strain_components
    )

    # Index for Mogi sources
    index.start_mogi_col = index.end_block_strain_col
    index.end_mogi_col = index.start_mogi_col + index.n_mogis

    # EIGEN: Index for Mogi sources
    index.start_mogi_col_eigen = index.end_block_strain_col_eigen
    index.end_mogi_col_eigen = index.start_mogi_col_eigen + index.n_mogis

    index.n_operator_rows = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    index.n_operator_cols = (
        3 * index.n_blocks
        + 2 * index.n_tde_total
        + 3 * index.n_strain_blocks
        + index.n_mogis
    )

    # Total number of columns for eigenmode problem
    index.n_operator_cols_eigen = (
        3 * index.n_blocks
        + index.n_eigen_total
        + 3 * index.n_strain_blocks
        + index.n_mogis
    )

    # Total number of rows for eigenmode problem
    index.n_operator_rows_eigen = (
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    # Indices for QP hard slip rate bounds
    index.slip_rate_bounds = np.where(
        interleave3(
            segment.ss_rate_bound_flag,
            segment.ds_rate_bound_flag,
            segment.ts_rate_bound_flag,
        )
        == 1
    )[0]
    index.n_slip_rate_bounds = len(index.slip_rate_bounds)

    return index


def get_processed_data_structures(command):
    # NOTE: Used in celeri_solve.py
    data = addict.Dict()
    assembly = addict.Dict()
    operators = addict.Dict()
    (
        data.segment,
        data.block,
        data.meshes,
        data.station,
        data.mogi,
        data.sar,
    ) = read_data(command)
    data.station = process_station(data.station, command)
    data.segment = process_segment(data.segment, command, data.meshes)
    data.sar = process_sar(data.sar, command)
    data.closure, data.block = assign_block_labels(
        data.segment, data.station, data.block, data.mogi, data.sar
    )
    operators.meshes = [addict.Dict()] * len(data.meshes)
    assembly = merge_geodetic_data(
        assembly, data.station, data.sar
    )  # TODO: Not sure this works correctly

    # Quick input plot
    if bool(command.plot_input_summary):
        plot_input_summary(
            command,
            data.segment,
            data.station,
            data.block,
            data.meshes,
            data.mogi,
            data.sar,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )
    return data, assembly, operators


##############################################################################################################################
#                                                                                                                            #
#                                                                                                                            #
#    ,ad8888ba,    88888888ba   88888888888  88888888ba          db    888888888888  ,ad8888ba,    88888888ba    ad88888ba   #
#   d8"'    `"8b   88      "8b  88           88      "8b        d88b        88      d8"'    `"8b   88      "8b  d8"     "8b  #
#  d8'        `8b  88      ,8P  88           88      ,8P       d8'`8b       88     d8'        `8b  88      ,8P  Y8,          #
#  88          88  88aaaaaa8P'  88aaaaa      88aaaaaa8P'      d8'  `8b      88     88          88  88aaaaaa8P'  `Y8aaaaa,    #
#  88          88  88""""""'    88"""""      88""""88'       d8YaaaaY8b     88     88          88  88""""88'      `"""""8b,  #
#  Y8,        ,8P  88           88           88    `8b      d8""""""""8b    88     Y8,        ,8P  88    `8b            `8b  #
#   Y8a.    .a8P   88           88           88     `8b    d8'        `8b   88      Y8a.    .a8P   88     `8b   Y8a     a8P  #
#    `"Y8888Y"'    88           88888888888  88      `8b  d8'          `8b  88       `"Y8888Y"'    88      `8b   "Y88888P"   #
#                                                                                                                            #
#                                                                                                                            #
##############################################################################################################################


def get_rotation_to_velocities_partials(station, n_blocks):
    """
    Calculate block rotation partials operator for stations in dataframe
    station.
    """
    # n_blocks = (
    #     np.max(station.block_label.values) + 1
    # )  # +1 required so that a single block with index zero still propagates
    block_rotation_operator = np.zeros((3 * len(station), 3 * n_blocks))
    for i in range(n_blocks):
        station_idx = np.where(station.block_label == i)[0]
        (
            vel_east_omega_x,
            vel_north_omega_x,
            vel_up_omega_x,
        ) = get_rotation_displacements(
            station.lon.values[station_idx],
            station.lat.values[station_idx],
            omega_x=1,
            omega_y=0,
            omega_z=0,
        )
        (
            vel_east_omega_y,
            vel_north_omega_y,
            vel_up_omega_y,
        ) = get_rotation_displacements(
            station.lon.values[station_idx],
            station.lat.values[station_idx],
            omega_x=0,
            omega_y=1,
            omega_z=0,
        )
        (
            vel_east_omega_z,
            vel_north_omega_z,
            vel_up_omega_z,
        ) = get_rotation_displacements(
            station.lon.values[station_idx],
            station.lat.values[station_idx],
            omega_x=0,
            omega_y=0,
            omega_z=1,
        )
        block_rotation_operator[3 * station_idx, 3 * i] = vel_east_omega_x
        block_rotation_operator[3 * station_idx, 3 * i + 1] = vel_east_omega_y
        block_rotation_operator[3 * station_idx, 3 * i + 2] = vel_east_omega_z
        block_rotation_operator[3 * station_idx + 1, 3 * i] = vel_north_omega_x
        block_rotation_operator[3 * station_idx + 1, 3 * i + 1] = vel_north_omega_y
        block_rotation_operator[3 * station_idx + 1, 3 * i + 2] = vel_north_omega_z
        block_rotation_operator[3 * station_idx + 2, 3 * i] = vel_up_omega_x
        block_rotation_operator[3 * station_idx + 2, 3 * i + 1] = vel_up_omega_y
        block_rotation_operator[3 * station_idx + 2, 3 * i + 2] = vel_up_omega_z
    return block_rotation_operator


def get_rotation_displacements(lon_obs, lat_obs, omega_x, omega_y, omega_z):
    """
    # TODO: Consider renaming to: get_rotation_velocities
    Get displacments at at longitude and latitude coordinates given rotation
    vector components (omega_x, omega_y, omega_z)
    """
    vel_east = np.zeros(lon_obs.size)
    vel_north = np.zeros(lon_obs.size)
    vel_up = np.zeros(lon_obs.size)
    x, y, z = sph2cart(lon_obs, lat_obs, RADIUS_EARTH)
    for i in range(lon_obs.size):
        cross_product_operator = get_cross_partials([x[i], y[i], z[i]])
        (
            vel_north_from_omega_x,
            vel_east_from_omega_x,
            vel_up_from_omega_x,
        ) = cartesian_vector_to_spherical_vector(
            cross_product_operator[0, 0],
            cross_product_operator[1, 0],
            cross_product_operator[2, 0],
            lon_obs[i],
            lat_obs[i],
        )
        (
            vel_north_from_omega_y,
            vel_east_from_omega_y,
            vel_up_from_omega_y,
        ) = cartesian_vector_to_spherical_vector(
            cross_product_operator[0, 1],
            cross_product_operator[1, 1],
            cross_product_operator[2, 1],
            lon_obs[i],
            lat_obs[i],
        )
        (
            vel_north_from_omega_z,
            vel_east_from_omega_z,
            vel_up_from_omega_z,
        ) = cartesian_vector_to_spherical_vector(
            cross_product_operator[0, 2],
            cross_product_operator[1, 2],
            cross_product_operator[2, 2],
            lon_obs[i],
            lat_obs[i],
        )
        vel_east[i] = (
            omega_x * vel_east_from_omega_x
            + omega_y * vel_east_from_omega_y
            + omega_z * vel_east_from_omega_z
        )
        vel_north[i] = (
            omega_x * vel_north_from_omega_x
            + omega_y * vel_north_from_omega_y
            + omega_z * vel_north_from_omega_z
        )
    return vel_east, vel_north, vel_up


def get_elastic_operators(
    operators: Dict,
    meshes: List,
    segment: pd.DataFrame,
    station: pd.DataFrame,
    command: Dict,
):
    """
    Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        meshes (List): Geometries of meshes
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        command (Dict): All command data
    """
    if bool(command.reuse_elastic) and os.path.exists(command.reuse_elastic_file):
        logger.info("Using precomputed elastic operators")
        hdf5_file = h5py.File(command.reuse_elastic_file, "r")

        operators.slip_rate_to_okada_to_velocities = np.array(
            hdf5_file.get("slip_rate_to_okada_to_velocities")
        )
        for i in range(len(meshes)):
            operators.tde_to_velocities[i] = np.array(
                hdf5_file.get("tde_to_velocities_" + str(i))
            )
        hdf5_file.close()

    else:
        if not os.path.exists(command.reuse_elastic_file):
            logger.warning("Precomputed elastic operator file not found")
        logger.info("Computing elastic operators")

        # Calculate Okada partials for all segments
        operators.slip_rate_to_okada_to_velocities = get_segment_station_operator_okada(
            segment, station, command
        )

        for i in range(len(meshes)):
            logger.info(
                f"Start: TDE slip to velocity calculation for mesh: {meshes[i].file_name}"
            )
            operators.tde_to_velocities[i] = get_tde_to_velocities_single_mesh(
                meshes, station, command, mesh_idx=i
            )
            logger.success(
                f"Finish: TDE slip to velocity calculation for mesh: {meshes[i].file_name}"
            )

        # Save elastic to velocity matrices
        if bool(command.save_elastic):
            # Check to see if "data/operators" folder exists and if not create it
            if not os.path.exists(command.operators_folder):
                os.mkdir(command.operators_folder)

            logger.info(
                "Saving elastic to velocity matrices to :" + command.save_elastic_file
            )
            hdf5_file = h5py.File(command.save_elastic_file, "w")

            hdf5_file.create_dataset(
                "slip_rate_to_okada_to_velocities",
                data=operators.slip_rate_to_okada_to_velocities,
            )
            for i in range(len(meshes)):
                hdf5_file.create_dataset(
                    "tde_to_velocities_" + str(i),
                    data=operators.tde_to_velocities[i],
                )
            hdf5_file.close()


def get_elastic_operators_okada(
    operators: Dict,
    segment: pd.DataFrame,
    station: pd.DataFrame,
    command: Dict,
):
    """
    NOTE: This is for the case with no TDEs.  May be redundant.  Consider

    Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        command (Dict): All command data
    """
    if bool(command.reuse_elastic) and os.path.exists(command.reuse_elastic_file):
        logger.info("Using precomputed elastic operators")
        hdf5_file = h5py.File(command.reuse_elastic_file, "r")

        operators.slip_rate_to_okada_to_velocities = np.array(
            hdf5_file.get("slip_rate_to_okada_to_velocities")
        )
        hdf5_file.close()

    else:
        if not os.path.exists(command.reuse_elastic_file):
            logger.warning("Precomputed elastic operator file not found")
        logger.info("Computing elastic operators")

        # Calculate Okada partials for all segments
        operators.slip_rate_to_okada_to_velocities = get_segment_station_operator_okada(
            segment, station, command
        )

        # Save elastic to velocity matrices
        if bool(command.save_elastic):
            # Check to see if "data/operators" folder exists and if not create it
            if not os.path.exists(command.operators_folder):
                os.mkdir(command.operators_folder)

            logger.info(
                "Saving elastic to velocity matrices to :" + command.save_elastic_file
            )
            hdf5_file = h5py.File(command.save_elastic_file, "w")

            hdf5_file.create_dataset(
                "slip_rate_to_okada_to_velocities",
                data=operators.slip_rate_to_okada_to_velocities,
            )
            hdf5_file.close()


def get_segment_station_operator_okada(segment, station, command):
    """
    Calculates the elastic displacement partial derivatives based on the Okada
    formulation, using the source and receiver geometries defined in
    dicitonaries segment and stations. Before calculating the partials for
    each segment, a local oblique Mercator project is done.

    The linear operator is structured as ():

                ss(segment1)  ds(segment1) ts(segment1) ... ss(segmentN) ds(segmentN) ts(segmentN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if not station.empty:
        okada_segment_operator = np.ones((3 * len(station), 3 * len(segment)))
        # Loop through each segment and calculate displacements for each slip component
        for i in tqdm(
            range(len(segment)),
            desc="Calculating Okada partials for segments",
            colour="cyan",
        ):
            (
                u_east_strike_slip,
                u_north_strike_slip,
                u_up_strike_slip,
            ) = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                command.material_lambda,
                command.material_mu,
                1,
                0,
                0,
                station.lon,
                station.lat,
            )
            (
                u_east_dip_slip,
                u_north_dip_slip,
                u_up_dip_slip,
            ) = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                command.material_lambda,
                command.material_mu,
                0,
                1,
                0,
                station.lon,
                station.lat,
            )
            (
                u_east_tensile_slip,
                u_north_tensile_slip,
                u_up_tensile_slip,
            ) = get_okada_displacements(
                segment.lon1[i],
                segment.lat1[i],
                segment.lon2[i],
                segment.lat2[i],
                segment.locking_depth[i],
                segment.burial_depth[i],
                segment.dip[i],
                segment.azimuth[i],
                command.material_lambda,
                command.material_mu,
                0,
                0,
                1,
                station.lon,
                station.lat,
            )
            segment_column_start_idx = 3 * i
            okada_segment_operator[0::3, segment_column_start_idx] = np.squeeze(
                u_east_strike_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx] = np.squeeze(
                u_north_strike_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx] = np.squeeze(
                u_up_strike_slip
            )
            okada_segment_operator[0::3, segment_column_start_idx + 1] = np.squeeze(
                u_east_dip_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx + 1] = np.squeeze(
                u_north_dip_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx + 1] = np.squeeze(
                u_up_dip_slip
            )
            okada_segment_operator[0::3, segment_column_start_idx + 2] = np.squeeze(
                u_east_tensile_slip
            )
            okada_segment_operator[1::3, segment_column_start_idx + 2] = np.squeeze(
                u_north_tensile_slip
            )
            okada_segment_operator[2::3, segment_column_start_idx + 2] = np.squeeze(
                u_up_tensile_slip
            )
    else:
        okada_segment_operator = np.empty(1)
    return okada_segment_operator


def get_okada_displacements(
    segment_lon1,
    segment_lat1,
    segment_lon2,
    segment_lat2,
    segment_locking_depth,
    segment_burial_depth,
    segment_dip,
    segment_azimuth,
    material_lambda,
    material_mu,
    strike_slip,
    dip_slip,
    tensile_slip,
    station_lon,
    station_lat,
):
    """
    Caculate elastic displacements in a homogeneous elastic half-space.
    Inputs are in geographic coordinates and then projected into a local
    xy-plane using a oblique Mercator projection that is tangent and parallel
    to the trace of the fault segment.  The elastic calculation is the
    original Okada 1992 Fortran code acceccesed through T. Ben Thompson's
    okada_wrapper: https://github.com/tbenthompson/okada_wrapper
    """
    segment_locking_depth *= KM2M
    segment_burial_depth *= KM2M

    # Make sure depths are expressed as positive
    segment_locking_depth = np.abs(segment_locking_depth)
    segment_burial_depth = np.abs(segment_burial_depth)

    # Correct sign of dip-slip based on fault dip, as noted on p. 1023 of Okada (1992)
    dip_slip *= np.sign(90 - segment_dip)

    # Project coordinates to flat space using a local oblique Mercator projection
    projection = get_segment_oblique_projection(
        segment_lon1, segment_lat1, segment_lon2, segment_lat2
    )
    station_x, station_y = projection(station_lon, station_lat)
    segment_x1, segment_y1 = projection(segment_lon1, segment_lat1)
    segment_x2, segment_y2 = projection(segment_lon2, segment_lat2)

    # Calculate geometric fault parameters
    segment_strike = np.arctan2(
        segment_y2 - segment_y1, segment_x2 - segment_x1
    )  # radians
    segment_length = np.sqrt(
        (segment_y2 - segment_y1) ** 2.0 + (segment_x2 - segment_x1) ** 2.0
    )
    segment_up_dip_width = (segment_locking_depth - segment_burial_depth) / np.sin(
        np.deg2rad(segment_dip)
    )

    # Translate stations and segment so that segment mid-point is at the origin
    segment_x_mid = (segment_x1 + segment_x2) / 2.0
    segment_y_mid = (segment_y1 + segment_y2) / 2.0
    station_x -= segment_x_mid
    station_y -= segment_y_mid
    segment_x1 -= segment_x_mid
    segment_x2 -= segment_x_mid
    segment_y1 -= segment_y_mid
    segment_y2 -= segment_y_mid

    # Unrotate coordinates to eliminate strike, segment will lie along y = 0
    rotation_matrix = np.array(
        [
            [np.cos(segment_strike), -np.sin(segment_strike)],
            [np.sin(segment_strike), np.cos(segment_strike)],
        ]
    )
    station_x_rotated, station_y_rotated = np.hsplit(
        np.einsum("ij,kj->ik", np.dstack((station_x, station_y))[0], rotation_matrix.T),
        2,
    )

    # Shift station y coordinates by surface projection of locking depth
    # y_shift will be positive for dips <90 and negative for dips > 90
    y_shift = np.cos(np.deg2rad(segment_dip)) * segment_up_dip_width
    station_y_rotated += y_shift

    # Elastic displacements from Okada 1992
    alpha = (material_lambda + material_mu) / (material_lambda + 2 * material_mu)
    u_x = np.zeros_like(station_x)
    u_y = np.zeros_like(station_x)
    u_up = np.zeros_like(station_x)
    for i in range(len(station_x)):
        _, u, _ = okada_wrapper.dc3dwrapper(
            alpha,  # (lambda + mu) / (lambda + 2 * mu)
            [
                station_x_rotated[i],
                station_y_rotated[i],
                0,
            ],  # (meters) observation point
            segment_locking_depth,  # (meters) depth of the fault origin
            segment_dip,  # (degrees) the dip-angle of the rectangular dislocation surface
            [
                -segment_length / 2,
                segment_length / 2,
            ],  # (meters) the along-strike range of the surface (al1,al2 in the original)
            [
                0,
                segment_up_dip_width,
            ],  # (meters) along-dip range of the surface (aw1, aw2 in the original)
            [strike_slip, dip_slip, tensile_slip],
        )  # (meters) strike-slip, dip-slip, tensile-slip
        u_x[i] = u[0]
        u_y[i] = u[1]
        u_up[i] = u[2]

    # Un-rotate displacement to account for projected fault strike
    # u_east, u_north = np.hsplit(
    #     np.einsum("ij,kj->ik", np.dstack((u_x, u_y))[0], rotation_matrix), 2
    # )

    # Rotate x, y displacements about geographic azimuth to yield east, north displacements
    cosstrike = np.cos(np.radians(90 - segment_azimuth))
    sinstrike = np.sin(np.radians(90 - segment_azimuth))
    u_north = sinstrike * u_x + cosstrike * u_y
    u_east = cosstrike * u_x - sinstrike * u_y
    return u_east, u_north, u_up


def get_rotation_to_slip_rate_partials(segment, block):
    """
    Calculate partial derivatives relating relative block motion to fault slip rates
    """
    n_segments = len(segment)
    n_blocks = len(block)
    fault_slip_rate_partials = np.zeros((3 * n_segments, 3 * n_blocks))
    for i in range(n_segments):
        # Project velocities from Cartesian to spherical coordinates at segment mid-points
        row_idx = 3 * i
        column_idx_east = 3 * segment.east_labels[i]
        column_idx_west = 3 * segment.west_labels[i]
        R = get_cross_partials([segment.mid_x[i], segment.mid_y[i], segment.mid_z[i]])
        (
            vel_north_to_omega_x,
            vel_east_to_omega_x,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 0], R[1, 0], R[2, 0], segment.mid_lon[i], segment.mid_lat[i]
        )
        (
            vel_north_to_omega_y,
            vel_east_to_omega_y,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 1], R[1, 1], R[2, 1], segment.mid_lon[i], segment.mid_lat[i]
        )
        (
            vel_north_to_omega_z,
            vel_east_to_omega_z,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 2], R[1, 2], R[2, 2], segment.mid_lon[i], segment.mid_lat[i]
        )

        # Build unit vector for the fault
        # Fault strike calculated in process_segment
        # TODO: Need to check this vs. matlab azimuth for consistency
        unit_x_parallel = np.cos(np.deg2rad(90 - segment.azimuth[i]))
        unit_y_parallel = np.sin(np.deg2rad(90 - segment.azimuth[i]))
        unit_x_perpendicular = np.sin(np.deg2rad(segment.azimuth[i] - 90))
        unit_y_perpendicular = np.cos(np.deg2rad(segment.azimuth[i] - 90))

        # Projection onto fault dip
        if segment.lat2[i] < segment.lat1[i]:
            unit_x_parallel = -unit_x_parallel
            unit_y_parallel = -unit_y_parallel
            unit_x_perpendicular = -unit_x_perpendicular
            unit_y_perpendicular = -unit_y_perpendicular

        # This is the logic for dipping vs. non-dipping faults
        # If fault is dipping make it so that the dip slip rate has a fault normal
        # component equal to the fault normal differential plate velocity.  This
        # is kinematically consistent in the horizontal but *not* in the vertical.
        if segment.dip[i] != 90:
            scale_factor = 1 / abs(np.cos(np.deg2rad(segment.dip[i])))
            slip_rate_matrix = np.array(
                [
                    [
                        unit_x_parallel * vel_east_to_omega_x
                        + unit_y_parallel * vel_north_to_omega_x,
                        unit_x_parallel * vel_east_to_omega_y
                        + unit_y_parallel * vel_north_to_omega_y,
                        unit_x_parallel * vel_east_to_omega_z
                        + unit_y_parallel * vel_north_to_omega_z,
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
        else:
            scale_factor = (
                -1
            )  # This is for consistency with the Okada convention for tensile faulting
            slip_rate_matrix = np.array(
                [
                    [
                        unit_x_parallel * vel_east_to_omega_x
                        + unit_y_parallel * vel_north_to_omega_x,
                        unit_x_parallel * vel_east_to_omega_y
                        + unit_y_parallel * vel_north_to_omega_y,
                        unit_x_parallel * vel_east_to_omega_z
                        + unit_y_parallel * vel_north_to_omega_z,
                    ],
                    [0, 0, 0],
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
                ]
            )

        fault_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_east : column_idx_east + 3
        ] = slip_rate_matrix
        fault_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_west : column_idx_west + 3
        ] = -slip_rate_matrix
    return fault_slip_rate_partials


def get_tde_to_velocities(meshes, station, command):
    """
    Calculates the elastic displacement partial derivatives based on the
    T. Ben Thompson cutde of the Nikhool and Walters (2015) equations
    for the displacements resulting from slip on a triangular
    dislocation in a homogeneous elastic half space.

    The linear operator is structured as ():

                ss(tri1)  ds(tri1) ts(tri1) ... ss(triN) ds(triN) ts(triN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if len(meshes) > 0:
        n_tris = meshes[0].lon1.size
        if not station.empty:
            tri_operator = np.zeros((3 * len(station), 3 * n_tris))

            # Loop through each segment and calculate displacements for each slip component
            for i in tqdm(
                range(n_tris),
                desc="Calculating cutde partials for triangles",
                colour="green",
            ):
                (
                    vel_east_strike_slip,
                    vel_north_strike_slip,
                    vel_up_strike_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=1,
                    dip_slip=0,
                    tensile_slip=0,
                )
                (
                    vel_east_dip_slip,
                    vel_north_dip_slip,
                    vel_up_dip_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=1,
                    tensile_slip=0,
                )
                (
                    vel_east_tensile_slip,
                    vel_north_tensile_slip,
                    vel_up_tensile_slip,
                ) = get_tri_displacements(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=0,
                    tensile_slip=1,
                )
                tri_operator[0::3, 3 * i] = np.squeeze(vel_east_strike_slip)
                tri_operator[1::3, 3 * i] = np.squeeze(vel_north_strike_slip)
                tri_operator[2::3, 3 * i] = np.squeeze(vel_up_strike_slip)
                tri_operator[0::3, 3 * i + 1] = np.squeeze(vel_east_dip_slip)
                tri_operator[1::3, 3 * i + 1] = np.squeeze(vel_north_dip_slip)
                tri_operator[2::3, 3 * i + 1] = np.squeeze(vel_up_dip_slip)
                tri_operator[0::3, 3 * i + 2] = np.squeeze(vel_east_tensile_slip)
                tri_operator[1::3, 3 * i + 2] = np.squeeze(vel_north_tensile_slip)
                tri_operator[2::3, 3 * i + 2] = np.squeeze(vel_up_tensile_slip)
        else:
            tri_operator = np.empty(0)
    else:
        tri_operator = np.empty(0)
    return tri_operator


def get_tde_to_velocities_single_mesh(meshes, station, command, mesh_idx):
    """
    Calculates the elastic displacement partial derivatives based on the
    T. Ben Thompson cutde of the Nikhool and Walters (2015) equations
    for the displacements resulting from slip on a triangular
    dislocation in a homogeneous elastic half space.

    The linear operator is structured as ():

                ss(tri1)  ds(tri1) ts(tri1) ... ss(triN) ds(triN) ts(triN)
    ve(station 1)
    vn(station 1)
    vu(station 1)
    .
    .
    .
    ve(station N)
    vn(station N)
    vu(station N)

    """
    if len(meshes) > 0:
        n_tris = meshes[mesh_idx].lon1.size
        if not station.empty:
            tri_operator = np.zeros((3 * len(station), 3 * n_tris))

            # Loop through each segment and calculate displacements for each slip component
            for i in tqdm(
                range(n_tris),
                desc=f"Calculating cutde partials for triangles",
                colour="green",
            ):
                (
                    vel_east_strike_slip,
                    vel_north_strike_slip,
                    vel_up_strike_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=1,
                    dip_slip=0,
                    tensile_slip=0,
                    mesh_idx=mesh_idx,
                )
                (
                    vel_east_dip_slip,
                    vel_north_dip_slip,
                    vel_up_dip_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=1,
                    tensile_slip=0,
                    mesh_idx=mesh_idx,
                )
                (
                    vel_east_tensile_slip,
                    vel_north_tensile_slip,
                    vel_up_tensile_slip,
                ) = get_tri_displacements_single_mesh(
                    station.lon.to_numpy(),
                    station.lat.to_numpy(),
                    meshes,
                    command.material_lambda,
                    command.material_mu,
                    tri_idx=i,
                    strike_slip=0,
                    dip_slip=0,
                    tensile_slip=1,
                    mesh_idx=mesh_idx,
                )
                tri_operator[0::3, 3 * i] = np.squeeze(vel_east_strike_slip)
                tri_operator[1::3, 3 * i] = np.squeeze(vel_north_strike_slip)
                tri_operator[2::3, 3 * i] = np.squeeze(vel_up_strike_slip)
                tri_operator[0::3, 3 * i + 1] = np.squeeze(vel_east_dip_slip)
                tri_operator[1::3, 3 * i + 1] = np.squeeze(vel_north_dip_slip)
                tri_operator[2::3, 3 * i + 1] = np.squeeze(vel_up_dip_slip)
                tri_operator[0::3, 3 * i + 2] = np.squeeze(vel_east_tensile_slip)
                tri_operator[1::3, 3 * i + 2] = np.squeeze(vel_north_tensile_slip)
                tri_operator[2::3, 3 * i + 2] = np.squeeze(vel_up_tensile_slip)
        else:
            tri_operator = np.empty(0)
    else:
        tri_operator = np.empty(0)
    return tri_operator


def get_tri_displacements(
    obs_lon,
    obs_lat,
    meshes,
    material_lambda,
    material_mu,
    tri_idx,
    strike_slip,
    dip_slip,
    tensile_slip,
):
    """
    Calculate surface displacments due to slip on a triangular dislocation
    element in a half space.  Includes projection from longitude and
    latitude to locally tangent planar coordinate system.
    """
    poissons_ratio = material_mu / (2 * (material_mu + material_lambda))

    # Project coordinates
    tri_centroid_lon = meshes[0].centroids[tri_idx, 0]
    tri_centroid_lat = meshes[0].centroids[tri_idx, 1]
    projection = get_transverse_projection(tri_centroid_lon, tri_centroid_lat)
    obs_x, obs_y = projection(obs_lon, obs_lat)
    tri_x1, tri_y1 = projection(meshes[0].lon1[tri_idx], meshes[0].lat1[tri_idx])
    tri_x2, tri_y2 = projection(meshes[0].lon2[tri_idx], meshes[0].lat2[tri_idx])
    tri_x3, tri_y3 = projection(meshes[0].lon3[tri_idx], meshes[0].lat3[tri_idx])
    tri_z1 = KM2M * meshes[0].dep1[tri_idx]
    tri_z2 = KM2M * meshes[0].dep2[tri_idx]
    tri_z3 = KM2M * meshes[0].dep3[tri_idx]

    # Package coordinates for cutde call
    obs_coords = np.vstack((obs_x, obs_y, np.zeros_like(obs_x))).T
    tri_coords = np.array(
        [[tri_x1, tri_y1, tri_z1], [tri_x2, tri_y2, tri_z2], [tri_x3, tri_y3, tri_z3]]
    )

    # Call cutde, multiply by displacements, and package for the return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mat = cutde_halfspace.disp_matrix(
            obs_pts=obs_coords, tris=np.array([tri_coords]), nu=poissons_ratio
        )
    slip = np.array([[strike_slip, dip_slip, tensile_slip]])
    disp = disp_mat.reshape((-1, 3)).dot(slip.flatten())
    vel_east = disp[0::3]
    vel_north = disp[1::3]
    vel_up = disp[2::3]
    return vel_east, vel_north, vel_up


def get_tri_displacements_single_mesh(
    obs_lon,
    obs_lat,
    meshes,
    material_lambda,
    material_mu,
    tri_idx,
    strike_slip,
    dip_slip,
    tensile_slip,
    mesh_idx,
):
    """
    Calculate surface displacments due to slip on a triangular dislocation
    element in a half space.  Includes projection from longitude and
    latitude to locally tangent planar coordinate system.
    """
    poissons_ratio = material_mu / (2 * (material_mu + material_lambda))

    # Project coordinates
    tri_centroid_lon = meshes[mesh_idx].centroids[tri_idx, 0]
    tri_centroid_lat = meshes[mesh_idx].centroids[tri_idx, 1]
    projection = get_transverse_projection(tri_centroid_lon, tri_centroid_lat)
    obs_x, obs_y = projection(obs_lon, obs_lat)
    tri_x1, tri_y1 = projection(
        meshes[mesh_idx].lon1[tri_idx], meshes[mesh_idx].lat1[tri_idx]
    )
    tri_x2, tri_y2 = projection(
        meshes[mesh_idx].lon2[tri_idx], meshes[mesh_idx].lat2[tri_idx]
    )
    tri_x3, tri_y3 = projection(
        meshes[mesh_idx].lon3[tri_idx], meshes[mesh_idx].lat3[tri_idx]
    )
    tri_z1 = KM2M * meshes[mesh_idx].dep1[tri_idx]
    tri_z2 = KM2M * meshes[mesh_idx].dep2[tri_idx]
    tri_z3 = KM2M * meshes[mesh_idx].dep3[tri_idx]

    # Package coordinates for cutde call
    obs_coords = np.vstack((obs_x, obs_y, np.zeros_like(obs_x))).T
    tri_coords = np.array(
        [[tri_x1, tri_y1, tri_z1], [tri_x2, tri_y2, tri_z2], [tri_x3, tri_y3, tri_z3]]
    )

    # Call cutde, multiply by displacements, and package for the return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mat = cutde_halfspace.disp_matrix(
            obs_pts=obs_coords, tris=np.array([tri_coords]), nu=poissons_ratio
        )
    slip = np.array([[strike_slip, dip_slip, tensile_slip]])
    disp = disp_mat.reshape((-1, 3)).dot(slip.flatten())
    vel_east = disp[0::3]
    vel_north = disp[1::3]
    vel_up = disp[2::3]
    return vel_east, vel_north, vel_up


def get_tri_smoothing_matrix(share, tri_shared_sides_distances):
    """
    Produces a smoothing matrix based on the scale-dependent
    umbrella operator (e.g., Desbrun et al., 1999; Resor, 2004).

    Inputs:
    share: n x 3 array of indices of the up to 3 elements sharing a side
        with each of the n elements
    tri_shared_sides_distances: n x 3 array of distances between each of the
        n elements and its up to 3 neighbors

    Outputs:
    smoothing matrix: 3n x 3n smoothing matrix
    """

    # Allocate sparse matrix for contructing smoothing matrix
    n_shared_tris = share.shape[0]
    smoothing_matrix = scipy.sparse.lil_matrix((3 * n_shared_tris, 3 * n_shared_tris))

    # Create a design matrix for Laplacian construction
    share_copy = copy.deepcopy(share)
    share_copy[np.where(share == -1)] = 0
    share_copy[np.where(share != -1)] = 1

    # Sum the distances between each element and its neighbors
    share_distances = np.sum(tri_shared_sides_distances, axis=1)
    leading_coefficient = 2.0 / share_distances

    # Replace zero distances with 1 to avoid divide by zero
    tri_shared_sides_distances[np.where(tri_shared_sides_distances == 0)] = 1

    # Take the reciprocal of the distances
    inverse_tri_shared_sides_distances = 1.0 / tri_shared_sides_distances

    # Diagonal terms # TODO: Defnitely not sure about his line!!!
    diagonal_terms = -leading_coefficient * np.sum(
        inverse_tri_shared_sides_distances * share_copy,
        axis=1,
    )

    # Off-diagonal terms
    off_diagonal_terms = (
        np.vstack((leading_coefficient, leading_coefficient, leading_coefficient)).T
        * inverse_tri_shared_sides_distances
        * share_copy
    )

    # Place the weights into the smoothing operator
    for j in range(3):
        for i in range(n_shared_tris):
            smoothing_matrix[3 * i + j, 3 * i + j] = diagonal_terms[i]
            if share[i, j] != -1:
                k = 3 * i + np.array([0, 1, 2])
                m = 3 * share[i, j] + np.array([0, 1, 2])
                smoothing_matrix[k, m] = off_diagonal_terms[i, j]
    return smoothing_matrix


def get_all_mesh_smoothing_matrices(meshes: List, operators: Dict):
    """
    Build smoothing matrices for each of the triangular meshes
    stored in meshes
    """
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


def get_all_mesh_smoothing_matrices_simple(meshes: List, operators: Dict):
    """
    Build smoothing matrices for each of the triangular meshes
    stored in meshes
    These are the simple not distance weighted meshes
    """
    for i in range(len(meshes)):
        # Get smoothing operator for a single mesh.
        meshes[i].share = get_shared_sides(meshes[i].verts)
        operators.smoothing_matrix[i] = get_tri_smoothing_matrix_simple(
            meshes[i].share, N_MESH_DIM
        )


def get_tri_smoothing_matrix_simple(share, n_dim):
    """
    Produces a smoothing matrix based without scale-dependent
    weighting.

    Inputs:
    share: n x 3 array of indices of the up to 3 elements sharing a side
        with each of the n elements

    Outputs:
    smoothing matrix: n_dim * n x n_dim * n smoothing matrix
    """

    # Allocate sparse matrix for contructing smoothing matrix
    n_shared_tri = share.shape[0]
    smoothing_matrix = scipy.sparse.lil_matrix(
        (n_dim * n_shared_tri, n_dim * n_shared_tri)
    )

    for j in range(n_dim):
        for i in range(n_shared_tri):
            smoothing_matrix[n_dim * i + j, n_dim * i + j] = 3
            if share[i, j] != -1:
                k = n_dim * i + np.arange(n_dim)
                m = n_dim * share[i, j] + np.arange(n_dim)
                smoothing_matrix[k, m] = -1
    return smoothing_matrix


def get_tde_slip_rate_constraints(meshes: List, operators: Dict):
    """Construct TDE slip rate constraint matrices for each mesh.
    These are identity matrices, used to set TDE slip rates on
    or coupling fractions on elements lining the edges of the mesh,
    as controlled by input parameters
    top_slip_rate_constraint,
    bot_slip_rate_constraint,
    side_slip_rate_constraint,

    and at other elements with indices specified as
    ss_slip_constraint_idx,
    ds_slip_constraint_idx,
    coupling_constraint_idx

    Args:
        meshes (List): list of mesh dictionaries
        operators (Dict): dictionary of linear operators
    """
    for i in range(len(meshes)):
        # Empty constraint matrix
        tde_slip_rate_constraints = np.zeros((2 * meshes[i].n_tde, 2 * meshes[i].n_tde))
        # Counting index
        start_row = 0
        end_row = 0
        # Top constraints
        # A value of 1 (free slip) or 2 (full coupling) will satisfy the following condition
        if meshes[i].top_slip_rate_constraint > 0:
            # Indices of top elements
            top_indices = np.asarray(np.where(meshes[i].top_elements))
            # Indices of top elements' 2 slip components
            meshes[i].top_slip_idx = get_2component_index(top_indices)
            end_row = len(meshes[i].top_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].top_slip_idx] = (
                np.eye(len(meshes[i].top_slip_idx))
            )
        # Bottom constraints
        if meshes[i].bot_slip_rate_constraint > 0:
            # Indices of bottom elements
            bot_indices = np.asarray(np.where(meshes[i].bot_elements))
            # Indices of bottom elements' 2 slip components
            meshes[i].bot_slip_idx = get_2component_index(bot_indices)
            start_row = end_row
            end_row = start_row + len(meshes[i].bot_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].bot_slip_idx] = (
                np.eye(len(meshes[i].bot_slip_idx))
            )
        # Side constraints
        if meshes[i].side_slip_rate_constraint > 0:
            # Indices of side elements
            side_indices = np.asarray(np.where(meshes[i].side_elements))
            # Indices of side elements' 2 slip components
            meshes[i].side_slip_idx = get_2component_index(side_indices)
            start_row = end_row
            end_row = start_row + len(meshes[i].side_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].side_slip_idx] = (
                np.eye(len(meshes[i].side_slip_idx))
            )
        # Other element indices
        if len(meshes[i].coupling_constraint_idx) > 0:
            meshes[i].coup_idx = get_2component_index(
                np.asarray(meshes[i].coupling_constraint_idx)
            )
            start_row += end_row
            end_row += 2 * len(meshes[i].coupling_constraint_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].coup_idx] = np.eye(
                2 * len(meshes[i].coupling_constraint_idx)
            )
        if len(meshes[i].ss_slip_constraint_idx) > 0:
            ss_idx = get_2component_index(np.asarray(meshes[i].ss_slip_constraint_idx))
            meshes[i].ss_slip_idx = ss_idx[0::2]
            start_row += end_row
            end_row += len(meshes[i].ss_slip_idx)
            # Component slip needs identities placed every other row, column
            tde_slip_rate_constraints[start_row:end_row, meshes[i].ss_slip_idx] = (
                np.eye(len(meshes[i].ss_slip_idx))
            )
        if len(meshes[i].ds_slip_constraint_idx) > 0:
            ds_idx = get_2component_index(np.asarray(meshes[i].ds_slip_constraint_idx))
            meshes[i].ds_slip_idx = ds_idx[1::2]
            start_row += end_row
            end_row += len(meshes[i].ds_slip_idx)
            tde_slip_rate_constraints[start_row:end_row, meshes[i].ds_slip_idx] = (
                np.eye(len(meshes[i].ds_slip_idx))
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


def get_tde_coupling_constraints(meshes, segment, block, operators):
    """
    Get partials relating block motion to TDE slip rates for coupling constraints
    """
    # for mesh_idx in range(len(meshes)):
    # Loop only over meshes that are tied to fault segments.  This *should*
    # eliminate touching CMI meshes which have problems with this function
    # becase it assumes that a mesh is tied to segments.
    for mesh_idx in range(np.max(segment.patch_file_name) + 1):
        operators.rotation_to_tri_slip_rate[mesh_idx] = (
            get_rotation_to_tri_slip_rate_partials(
                meshes[mesh_idx], mesh_idx, segment, block
            )
        )
        # Trim tensile rows
        tri_keep_rows = get_keep_index_12(
            np.shape(operators.rotation_to_tri_slip_rate[mesh_idx])[0]
        )
        operators.rotation_to_tri_slip_rate[mesh_idx] = (
            operators.rotation_to_tri_slip_rate[mesh_idx][tri_keep_rows, :]
        )


def get_rotation_to_tri_slip_rate_partials(meshes, mesh_idx, segment, block):
    """
    Calculate partial derivatives relating relative block motion to TDE slip rates
    for a single mesh. Called within a loop from get_tde_coupling_constraints()
    """
    n_blocks = len(block)
    tri_slip_rate_partials = np.zeros((3 * meshes.n_tde, 3 * n_blocks))

    # Generate strikes for elements using same sign convention as segments
    dipdir = np.array(
        meshes.strike + 90
    )  # Dip direction. Not wrapping to 360 so we can use it as a threshold
    # dipdir(dipdir > 360) -= 360
    tristrike = np.array(meshes.strike)
    tristrike[meshes.strike > 180] -= 180
    tristrike2 = np.array(meshes.strike)
    tristrike2[meshes.strike > 180] -= 360
    tridip = np.array(meshes.dip)
    tridip[meshes.strike > 180] = 180 - tridip[meshes.strike > 180]
    # Find subset of segments that are replaced by this mesh
    seg_replace_idx = np.where(
        (segment.patch_flag != 0) & (segment.patch_file_name == mesh_idx)
    )
    # Find closest segment midpoint to each element centroid, using scipy.spatial.cdist
    meshes.closest_segment_idx = seg_replace_idx[0][
        cdist(
            np.array([meshes.lon_centroid, meshes.lat_centroid]).T,
            np.array(
                [
                    segment.mid_lon[seg_replace_idx[0]],
                    segment.mid_lat[seg_replace_idx[0]],
                ]
            ).T,
        ).argmin(axis=1)
    ]
    # Add segment labels to elements
    meshes.east_labels = np.array(segment.east_labels[meshes.closest_segment_idx])
    meshes.west_labels = np.array(segment.west_labels[meshes.closest_segment_idx])

    # Check for switching of block labels
    seg_dip_dir = np.array(segment.azimuth)
    seg_dip_dir = seg_dip_dir + np.sign(np.cos(np.deg2rad(segment.dip))) * 90
    seg_dip_dir_x = np.cos(np.deg2rad(90 - seg_dip_dir))
    seg_dip_dir_y = np.sin(np.deg2rad(90 - seg_dip_dir))
    seg_comps = np.vstack(
        [seg_dip_dir_x[:], seg_dip_dir_y[:], np.zeros_like(seg_dip_dir_x)]
    ).T
    tri_dip_dir = np.array(meshes.strike) + 90
    tri_dip_dir_x = np.cos(np.deg2rad(90 - tri_dip_dir))
    tri_dip_dir_y = np.sin(np.deg2rad(90 - tri_dip_dir))
    tri_comps = np.vstack(
        [tri_dip_dir_x[:], tri_dip_dir_y[:], np.zeros_like(tri_dip_dir_x)]
    ).T
    north_tri_cross = np.cross(
        np.array([0, 1, 0]),
        tri_comps,
    )
    north_seg_cross = np.cross(
        np.array([0, 1, 0]),
        seg_comps,
    )

    # Find rotation partials for each element
    for el_idx in range(meshes.n_tde):
        # Project velocities from Cartesian to spherical coordinates at element centroids
        row_idx = 3 * el_idx
        column_idx_east = 3 * meshes.east_labels[el_idx]
        column_idx_west = 3 * meshes.west_labels[el_idx]
        R = get_cross_partials(
            [
                meshes.x_centroid[el_idx],
                meshes.y_centroid[el_idx],
                meshes.z_centroid[el_idx],
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
            meshes.lon_centroid[el_idx],
            meshes.lat_centroid[el_idx],
        )
        (
            vel_north_to_omega_y,
            vel_east_to_omega_y,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 1],
            R[1, 1],
            R[2, 1],
            meshes.lon_centroid[el_idx],
            meshes.lat_centroid[el_idx],
        )
        (
            vel_north_to_omega_z,
            vel_east_to_omega_z,
            _,
        ) = cartesian_vector_to_spherical_vector(
            R[0, 2],
            R[1, 2],
            R[2, 2],
            meshes.lon_centroid[el_idx],
            meshes.lat_centroid[el_idx],
        )
        # This correction gives -1 for strikes > 90
        # Equivalent to the if statement in get_rotation_to_slip_rate_partials
        sign_corr = -np.sign(tristrike[el_idx] - (90 + 1e-5))
        # sign_corr = np.sign(meshes.strike[el_idx] - (180 + 1e-5))
        # Sign correction based on dip direction. Sign is flipped for south-dipping elements
        # sign_corr = -np.sign(dipdir[el_idx] - (270 + 1e-5))

        # sign_corr = 1
        # Project about fault strike
        unit_x_parallel = sign_corr * np.cos(np.deg2rad(90 - tristrike[el_idx]))
        unit_y_parallel = sign_corr * np.sin(np.deg2rad(90 - tristrike[el_idx]))
        unit_x_perpendicular = sign_corr * np.sin(np.deg2rad(tristrike[el_idx] - 90))
        unit_y_perpendicular = sign_corr * np.cos(np.deg2rad(tristrike[el_idx] - 90))
        # Project by fault dip
        scale_factor = 1.0 / (np.cos(np.deg2rad(meshes.dip[el_idx])))
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
        # Equivalently, this situation is where the dip directions of the segment and triangle are
        # on different sides of a N-S line
        # ew_switch = np.sign(meshes.strike[el_idx] - 270) * np.sign(
        #     segment.azimuth[meshes.closest_segment_idx[el_idx]] - 90
        # )
        # ew_switch = np.sign(90 - tristrike2[el_idx]) * np.sign(
        #     90 - segment.azimuth[meshes.closest_segment_idx[el_idx]]
        # )
        ew_switch = np.sign(north_tri_cross[el_idx, 2]) * np.sign(
            north_seg_cross[meshes.closest_segment_idx[el_idx], 2]
        )
        # ew_switch = 1
        # Insert this element's partials into operator
        tri_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_east : column_idx_east + 3
        ] = (ew_switch * slip_rate_matrix)
        tri_slip_rate_partials[
            row_idx : row_idx + 3, column_idx_west : column_idx_west + 3
        ] = (-ew_switch * slip_rate_matrix)
    return tri_slip_rate_partials


def get_block_motion_constraints(assembly: Dict, block: pd.DataFrame, command: Dict):
    """
    Applying a priori block motion constraints
    """
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
    assembly.sigma.block_constraint_weight = command.block_constraint_weight
    return assembly, block_constraint_partials


def get_block_motion_constraint_partials(block):
    """
    Partials for a priori block motion constraints.
    Essentially a set of eye(3) matrices
    """
    # apriori_block_idx = np.where(block.apriori_flag.to_numpy() == 1)[0]
    apriori_rotation_block_idx = np.where(block.rotation_flag.to_numpy() == 1)[0]

    operator = np.zeros((3 * len(apriori_rotation_block_idx), 3 * len(block)))
    for i in range(len(apriori_rotation_block_idx)):
        start_row = 3 * i
        start_column = 3 * apriori_rotation_block_idx[i]
        operator[start_row : start_row + 3, start_column : start_column + 3] = np.eye(3)
    return operator


def get_slip_rate_constraints(assembly, segment, block, command):
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
                    + "{:.2f}".format(segment.ss_rate[i])
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + "{:.2f}".format(segment.ss_rate_sig[i])
                    + " (mm/yr)"
                )
            if segment.ds_rate_flag[i] == 1:
                logger.info(
                    "Dip-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + "{:.2f}".format(segment.ds_rate[i])
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + "{:.2f}".format(segment.ds_rate_sig[i])
                    + " (mm/yr)"
                )
            if segment.ts_rate_flag[i] == 1:
                logger.info(
                    "Tensile-slip rate constraint on "
                    + segment.name[i].strip()
                    + ": rate = "
                    + "{:.2f}".format(segment.ts_rate[i])
                    + " (mm/yr), 1-sigma uncertainty = +/-"
                    + "{:.2f}".format(segment.ts_rate_sig[i])
                    + " (mm/yr)"
                )
    else:
        logger.info("No slip rate constraints")

    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(segment, block)

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
    assembly.sigma.slip_rate_constraint_weight = command.slip_constraint_weight
    return assembly, slip_rate_constraint_partials


def get_slip_rake_constraints(assembly, segment, block, command):
    n_total_slip_rake_contraints = np.sum(segment.rake_flag.values)
    if n_total_slip_rake_contraints > 0:
        logger.info(f"Found {n_total_slip_rake_contraints} slip rake constraints")
        for i in range(len(segment.lon1)):
            if segment.rake_flag[i] == 1:
                logger.info(
                    "Rake constraint on "
                    + segment.name[i].strip()
                    + ": rake = "
                    + "{:.2f}".format(segment.rake[i])
                    + ", constraint strike = "
                    + "{:.2f}".format(segment.rake_strike[i])
                    + ", 1-sigma uncertainty = +/-"
                    + "{:.2f}".format(segment.rake_sig[i])
                )
    else:
        logger.info("No slip rake constraints")
    # To keep this a standalone function, let's calculate the full set of slip rate partials
    # TODO: Check how get_slip_rate_constraints is called to see if we need to recalculate the full set of partials, or if we can reuse a previous calculation
    slip_rate_constraint_partials = get_rotation_to_slip_rate_partials(segment, block)
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
    assembly.sigma.slip_rake_constraint_weight = command.slip_constraint_weight
    return assembly, slip_rake_constraint_partials


def get_mogi_to_velocities_partials(mogi, station, command):
    """
    Mogi volume change to station displacement operator
    """
    if mogi.empty:
        mogi_operator = np.zeros((3 * len(station), 0))
    else:
        poissons_ratio = command.material_mu / (
            2 * (command.material_lambda + command.material_mu)
        )
        mogi_operator = np.zeros((3 * len(station), len(mogi)))
        for i in range(len(mogi)):
            mogi_depth = KM2M * mogi.depth[i]
            u_east, u_north, u_up = mogi_forward(
                mogi.lon[i],
                mogi.lat[i],
                mogi_depth,
                poissons_ratio,
                station.lon,
                station.lat,
            )

            # Insert components into partials matrix
            mogi_operator[0::3, i] = u_east
            mogi_operator[1::3, i] = u_north
            mogi_operator[2::3, i] = u_up
    return mogi_operator


def mogi_forward(mogi_lon, mogi_lat, mogi_depth, poissons_ratio, obs_lon, obs_lat):
    """
    Calculate displacements from a single Mogi source using
    equation 7.14 from "Earthquake and Volcano Deformation" by Paul Segall
    """
    u_east = np.zeros_like(obs_lon)
    u_north = np.zeros_like(obs_lon)
    u_up = np.zeros_like(obs_lon)
    for i in range(obs_lon.size):
        # Find angle between source and observation as well as distance betwen them
        source_to_obs_forward_azimuth, _, source_to_obs_distance = GEOID.inv(
            mogi_lon, mogi_lat, obs_lon[i], obs_lat[i]
        )

        # Mogi displacements in cylindrical coordinates
        u_up[i] = (
            (1 - poissons_ratio)
            / np.pi
            * mogi_depth
            / ((source_to_obs_distance**2.0 + mogi_depth**2) ** 1.5)
        )
        u_radial = (
            (1 - poissons_ratio)
            / np.pi
            * source_to_obs_distance
            / ((source_to_obs_distance**2 + mogi_depth**2.0) ** 1.5)
        )

        # Convert radial displacement to east and north components
        u_east[i] = u_radial * np.sin(np.deg2rad(source_to_obs_forward_azimuth))
        u_north[i] = u_radial * np.cos(np.deg2rad(source_to_obs_forward_azimuth))
    return u_east, u_north, u_up


def get_strain_rate_displacements(
    lon_obs,
    lat_obs,
    centroid_lon,
    centroid_lat,
    strain_rate_lon_lon,
    strain_rate_lat_lat,
    strain_rate_lon_lat,
):
    """
    Calculate displacements due to three strain rate components.
    Equations are from Savage (2001) and expressed concisely in McCaffrey (2005)
    In McCaffrey (2005) these are the two unnumbered equations at the bottom
    of page 2
    """
    centroid_lon = np.deg2rad(centroid_lon)
    centroid_lat = latitude_to_colatitude(centroid_lat)
    centroid_lat = np.deg2rad(centroid_lat)
    lon_obs = np.deg2rad(lon_obs)
    lat_obs = latitude_to_colatitude(lat_obs)
    lat_obs = np.deg2rad(lat_obs)
    # Calculate displacements from homogeneous strain
    u_up = np.zeros(
        lon_obs.size
    )  # Always zero here because we're assuming plane strain on the sphere
    u_east = strain_rate_lon_lon * (
        RADIUS_EARTH * (lon_obs - centroid_lon) * np.sin(centroid_lat)
    ) + strain_rate_lon_lat * (RADIUS_EARTH * (lat_obs - centroid_lat))
    u_north = strain_rate_lon_lat * (
        RADIUS_EARTH * (lon_obs - centroid_lon) * np.sin(centroid_lat)
    ) + strain_rate_lat_lat * (RADIUS_EARTH * (lat_obs - centroid_lat))
    return u_east, u_north, u_up


def get_block_strain_rate_to_velocities_partials(block, station, segment):
    """
    Calculate strain partial derivatives assuming a strain centroid at the center of each block
    """
    strain_rate_block_idx = np.where(block.strain_rate_flag.to_numpy() > 0)[0]
    # Allocate space. Zero width, if no blocks should have strain estimated, helps with indexing
    block_strain_rate_operator = np.zeros(
        (3 * len(station), 3 * strain_rate_block_idx.size)
    )
    if strain_rate_block_idx.size > 0:
        for i in range(strain_rate_block_idx.size):
            # Find centroid of current block
            block_centroid_lon, block_centroid_lat = get_block_centroid(
                segment, strain_rate_block_idx[i]
            )
            # Find stations on current block
            station_idx = np.where(station.block_label == strain_rate_block_idx[i])[0]
            stations_block_lon = station.lon[station_idx].to_numpy()
            stations_block_lat = station.lat[station_idx].to_numpy()

            # Calculate partials for each component of strain rate
            (
                vel_east_lon_lon,
                vel_north_lon_lon,
                vel_up_lon_lon,
            ) = get_strain_rate_displacements(
                stations_block_lon,
                stations_block_lat,
                block_centroid_lon,
                block_centroid_lat,
                strain_rate_lon_lon=1,
                strain_rate_lat_lat=0,
                strain_rate_lon_lat=0,
            )
            (
                vel_east_lat_lat,
                vel_north_lat_lat,
                vel_up_lat_lat,
            ) = get_strain_rate_displacements(
                stations_block_lon,
                stations_block_lat,
                block_centroid_lon,
                block_centroid_lat,
                strain_rate_lon_lon=0,
                strain_rate_lat_lat=1,
                strain_rate_lon_lat=0,
            )
            (
                vel_east_lon_lat,
                vel_north_lon_lat,
                vel_up_lon_lat,
            ) = get_strain_rate_displacements(
                stations_block_lon,
                stations_block_lat,
                block_centroid_lon,
                block_centroid_lat,
                strain_rate_lon_lon=0,
                strain_rate_lat_lat=0,
                strain_rate_lon_lat=1,
            )
            # The sign convention established here (with negative signs on the lat_lat components) is that
            # positive longitudinal strain is extensional, and
            # positive shear strain is sinistral (counterclockwise)
            block_strain_rate_operator[3 * station_idx, 3 * i] = vel_east_lon_lon
            block_strain_rate_operator[3 * station_idx, 3 * i + 1] = -vel_east_lat_lat
            block_strain_rate_operator[3 * station_idx, 3 * i + 2] = vel_east_lon_lat
            block_strain_rate_operator[3 * station_idx + 1, 3 * i] = vel_north_lon_lon
            block_strain_rate_operator[3 * station_idx + 1, 3 * i + 1] = (
                -vel_north_lat_lat
            )
            block_strain_rate_operator[3 * station_idx + 1, 3 * i + 2] = (
                vel_north_lon_lat
            )
            block_strain_rate_operator[3 * station_idx + 2, 3 * i] = vel_up_lon_lon
            block_strain_rate_operator[3 * station_idx + 2, 3 * i + 1] = -vel_up_lat_lat
            block_strain_rate_operator[3 * station_idx + 2, 3 * i + 2] = vel_up_lon_lat
    return block_strain_rate_operator, strain_rate_block_idx


def get_global_float_block_rotation_partials(station):
    """
    Return a linear operator for the rotations of all stations assuming they
    are the on the same block (i.e., the globe). The purpose of this is to
    allow all of the stations to "float" in the inverse problem reducing the
    dependence on reference frame specification. This is done by making a
    copy of the station data frame, setting "block_label" for all stations
    equal to zero and then calling the standard block rotation operator
    function. The matrix returned here only has 3 columns
    """
    station_all_on_one_block = station.copy()
    station_all_on_one_block.block_label.values[:] = (
        0  # Force all stations to be on one block
    )
    global_float_block_rotation_operator = get_rotation_to_velocities_partials(
        station_all_on_one_block, 1
    )
    return global_float_block_rotation_operator


def get_data_vector(assembly, index, meshes):
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

    # Add TDE slip rate constraints to data vector
    # Coupling fractions remain zero but slip component constraints are as specified in mesh_param
    for i in range(len(meshes)):
        data_vector[
            index.start_tde_ss_slip_constraint_row[
                i
            ] : index.end_tde_ss_slip_constraint_row[i]
        ] = meshes[i].ss_slip_constraint_rate

        data_vector[
            index.start_tde_ds_slip_constraint_row[
                i
            ] : index.end_tde_ds_slip_constraint_row[i]
        ] = meshes[i].ds_slip_constraint_rate
    return data_vector


def get_weighting_vector(command, station, meshes, index):
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + 2 * index.n_tde_total
        + index.n_tde_constraints_total
    )
    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )
    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = command.block_constraint_weight
    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = command.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    for i in range(len(meshes)):
        # Insert smoothing weight into weighting vector
        weighting_vector[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i]
        ] = meshes[i].smoothing_weight * np.ones(2 * index.n_tde[i])
        weighting_vector[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i]
        ] = command.tri_con_weight * np.ones(index.n_tde_constraints[i])
    return weighting_vector


def get_weighting_vector_no_meshes(command, station, index):
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
    ] = command.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    return weighting_vector


def get_weighting_vector_single_mesh_for_col_norms(
    command, station, meshes, index, mesh_index: np.int_
):
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 2 * index.n_tde[mesh_index]
        + index.n_tde_constraints[mesh_index]
    )

    weighting_vector[0 : 2 * index.n_stations] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )

    weighting_vector[
        2 * index.n_stations : 2 * index.n_stations + 2 * index.n_tde[mesh_index]
    ] = meshes[mesh_index].smoothing_weight * np.ones(2 * index.n_tde[mesh_index])

    weighting_vector[2 * index.n_stations + 2 * index.n_tde[mesh_index] : :] = (
        command.tri_con_weight * np.ones(index.n_tde_constraints[mesh_index])
    )

    return weighting_vector


def get_data_vector_eigen(meshes, assembly, index):
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

    # Add TDE boundary condtions constraints
    for i in range(index.n_meshes):
        # Place strike-slip TDE BCs
        data_vector[
            index.start_tde_constraint_row_eigen[
                i
            ] : index.end_tde_constraint_row_eigen[i] : 2
        ] = meshes[i].ss_slip_constraint_rate

        # Place dip-slip TDE BCs
        data_vector[
            index.start_tde_constraint_row_eigen[i]
            + 1 : index.end_tde_constraint_row_eigen[i] : 2
        ] = meshes[i].ds_slip_constraint_rate

    return data_vector


def get_weighting_vector_eigen(command, station, meshes, index):
    # Initialize and build weighting matrix
    weighting_vector = np.ones(
        2 * index.n_stations
        + 3 * index.n_block_constraints
        + index.n_slip_rate_constraints
        + index.n_tde_constraints_total
    )

    weighting_vector[index.start_station_row : index.end_station_row] = interleave2(
        1 / (station.east_sig**2), 1 / (station.north_sig**2)
    )

    weighting_vector[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] = command.block_constraint_weight

    weighting_vector[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] = command.slip_constraint_weight * np.ones(index.n_slip_rate_constraints)

    # TODO: Need to think about constraints weights
    # This is the only place where any individual constraint weights enter
    # I'm only using on of them: meshes[i].bot_slip_rate_weight
    for i in range(len(meshes)):
        weighting_vector[
            index.start_tde_constraint_row_eigen[
                i
            ] : index.end_tde_constraint_row_eigen[i]
        ] = meshes[i].bot_slip_rate_weight * np.ones(index.n_tde_constraints[i])

    return weighting_vector


def get_full_dense_operator_block_only(operators, index):
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


def get_full_dense_operator(operators, meshes, index):
    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + 2 * index.n_tde_total
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.n_block_strain_components
            + index.n_mogi
            + 2 * index.n_tde_total,
        )
    )

    # DEBUG:
    # IPython.embed(banner1="")

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

    # Insert all TDE operators
    for i in range(len(meshes)):
        # Insert TDE to velocity matrix
        tde_keep_row_index = get_keep_index_12(operators.tde_to_velocities[i].shape[0])
        tde_keep_col_index = get_keep_index_12(operators.tde_to_velocities[i].shape[1])
        operator[
            index.start_station_row : index.end_station_row,
            index.start_tde_col[i] : index.end_tde_col[i],
        ] = -operators.tde_to_velocities[i][tde_keep_row_index, :][
            :, tde_keep_col_index
        ]

        # Insert TDE smoothing matrix
        smoothing_keep_index = get_keep_index_12(
            operators.tde_to_velocities[i].shape[1]
        )
        operator[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i],
            index.start_tde_col[i] : index.end_tde_col[i],
        ] = operators.smoothing_matrix[i].toarray()[smoothing_keep_index, :][
            :, smoothing_keep_index
        ]

        # Insert TDE slip rate constraints into estimation operator
        # These are just the identity matrices, and we'll insert any block motion constraints next
        operator[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i],
            index.start_tde_col[i] : index.end_tde_col[i],
        ] = operators.tde_slip_rate_constraints[i]
        # Insert block motion constraints for any coupling-constrained rows
        if meshes[i].top_slip_rate_constraint == 2:
            operator[
                index.start_tde_top_constraint_row[
                    i
                ] : index.end_tde_top_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                meshes[i].top_slip_idx,
                index.start_block_col : index.end_block_col,
            ]
        if meshes[i].bot_slip_rate_constraint == 2:
            operator[
                index.start_tde_bot_constraint_row[
                    i
                ] : index.end_tde_bot_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                meshes[i].bot_slip_idx,
                :,
            ]
        if meshes[i].side_slip_rate_constraint == 2:
            operator[
                index.start_tde_side_constraint_row[
                    i
                ] : index.end_tde_side_constraint_row[i],
                index.start_block_col : index.end_block_col,
            ] = -operators.rotation_to_tri_slip_rate[i][
                meshes[i].side_slip_idx,
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


def get_full_dense_operator_eigen(operators, meshes, index):
    # Initialize linear operator
    operator = np.zeros(
        (
            2 * index.n_stations
            + 3 * index.n_block_constraints
            + index.n_slip_rate_constraints
            + index.n_tde_constraints_total,
            3 * index.n_blocks
            + index.n_eigen_total
            + 3 * index.n_strain_blocks
            + index.n_mogis,
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

    # EIGEN Eigenvector to velocity matrix
    for i in range(index.n_meshes):
        # Eliminate vertical elastic velocities
        tde_keep_row_index = get_keep_index_12(operators.tde_to_velocities[i].shape[0])
        tde_keep_col_index = get_keep_index_12(operators.tde_to_velocities[i].shape[1])

        # Create eigenvector to velocities operator
        operators.eigen_to_velocities[i] = (
            -operators.tde_to_velocities[i][tde_keep_row_index, :][
                :, tde_keep_col_index
            ]
            @ operators.eigenvectors_to_tde_slip[i]
        )

        # Insert eigenvector to velocities operator
        operator[
            index.start_station_row : index.end_station_row,
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = operators.eigen_to_velocities[i]

    # EIGEN Eigenvector to TDE boundary conditions matrix
    for i in range(index.n_meshes):
        # Create eigenvector to TDE boundary conditions matrix
        operators.eigen_to_tde_bcs[i] = (
            meshes[i].mesh_tde_modes_bc_weight
            * operators.tde_slip_rate_constraints[i]
            @ operators.eigenvectors_to_tde_slip[i]
        )

        # Insert eigenvector to TDE boundary conditions matrix
        operator[
            index.start_tde_constraint_row_eigen[
                i
            ] : index.end_tde_constraint_row_eigen[i],
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = operators.eigen_to_tde_bcs[i]

    # EIGEN: Block strain operator
    operator[
        0 : 2 * index.n_stations,
        index.start_block_strain_col_eigen : index.end_block_strain_col_eigen,
    ] = operators.block_strain_rate_to_velocities[tde_keep_row_index, :]

    # EIGEN: Mogi operator
    operator[
        0 : 2 * index.n_stations,
        index.start_mogi_col_eigen : index.end_mogi_col_eigen,
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
                    + "{:.2f}".format(segment.ss_rate_bound_min[i])
                    + " (mm/yr), rate (upper bound) = "
                    + "{:.2f}".format(segment.ss_rate_bound_max[i])
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert (
                    segment.ss_rate_bound_min[i] < segment.ss_rate_bound_max[i]
                ), "Bounds min max error"

            if segment.ds_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP dip-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + "{:.2f}".format(segment.ds_rate_bound_min[i])
                    + " (mm/yr), rate (upper bound) = "
                    + "{:.2f}".format(segment.ds_rate_bound_max[i])
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert (
                    segment.ds_rate_bound_min[i] < segment.ds_rate_bound_max[i]
                ), "Bounds min max error"

            if segment.ts_rate_bound_flag[i] == 1:
                logger.info(
                    "Hard QP tensile-slip rate bounds on "
                    + segment.name[i].strip()
                    + ": rate (lower bound) = "
                    + "{:.2f}".format(segment.ts_rate_bound_min[i])
                    + " (mm/yr), rate (upper bound) = "
                    + "{:.2f}".format(segment.ts_rate_bound_max[i])
                    + " (mm/yr)"
                )

                # Fail if min bound not less than max bound
                assert (
                    segment.ts_rate_bound_min[i] < segment.ts_rate_bound_max[i]
                ), "Bounds min max error"

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


def get_qp_tde_inequality_operator_and_data_vector(index, meshes, operators):
    qp_constraint_matrix = np.zeros(
        (4 * index.n_tde_total, index.n_operator_cols_eigen)
    )
    qp_constraint_data_vector = np.zeros(4 * index.n_tde_total)

    for i in range(index.n_meshes):
        # TDE strike- and dip-slip lower bounds
        lower_bound_current_mesh = interleave2(
            meshes[i].qp_mesh_tde_slip_rate_lower_bound_ss * np.ones(index.n_tde[i]),
            meshes[i].qp_mesh_tde_slip_rate_lower_bound_ds * np.ones(index.n_tde[i]),
        )

        # TDE strike- and dip-slip upper bounds
        upper_bound_current_mesh = interleave2(
            meshes[i].qp_mesh_tde_slip_rate_upper_bound_ss * np.ones(index.n_tde[i]),
            meshes[i].qp_mesh_tde_slip_rate_upper_bound_ds * np.ones(index.n_tde[i]),
        )

        # Insert TDE lower bounds into QP constraint data vector (note negative sign)
        qp_constraint_data_vector[
            index.qp_constraint_tde_rate_start_row_eigen[
                i
            ] : index.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.n_tde[i]
        ] = -lower_bound_current_mesh

        # Insert TDE upper bounds into QP constraint data vector
        qp_constraint_data_vector[
            index.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.n_tde[i] : index.qp_constraint_tde_rate_end_row_eigen[i]
        ] = upper_bound_current_mesh

        # Insert eigenmode to TDE slip operator into QP constraint data vector for lower bounds (note negative sign)
        qp_constraint_matrix[
            index.qp_constraint_tde_rate_start_row_eigen[
                i
            ] : index.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.n_tde[i],
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = -operators.eigenvectors_to_tde_slip[i]

        # Insert eigenmode to TDE slip operator into QP constraint data vector for lower bounds
        qp_constraint_matrix[
            index.qp_constraint_tde_rate_start_row_eigen[i]
            + 2 * index.n_tde[i] : index.qp_constraint_tde_rate_end_row_eigen[i],
            index.start_col_eigen[i] : index.end_col_eigen[i],
        ] = operators.eigenvectors_to_tde_slip[i]

    return qp_constraint_matrix, qp_constraint_data_vector


def get_qp_all_inequality_operator_and_data_vector(
    index, meshes, operators, segment, block
):
    # Create arrays and data vector of correct size for linear inequality constraints
    # Stack TDE slip rate bounds on top of slip rate bounds
    #   TDE slip rate bounds
    #   slip rate bounds

    # Get QP TDE bounds
    qp_tde_inequality_matrix, qp_tde_inequality_data_vector = (
        get_qp_tde_inequality_operator_and_data_vector(index, meshes, operators)
    )

    # Get QP slip rate bounds
    qp_slip_rate_inequality_matrix, qp_slip_rate_inequality_data_vector = (
        get_qp_slip_rate_inequality_operator_and_data_vector(index, segment, block)
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

    return qp_inequality_constraints_matrix, qp_inequality_constraints_data_vector


def get_qp_slip_rate_inequality_operator_and_data_vector(index, segment, block):
    # Get slip rate bounds vectors and operators
    slip_rate_bound_min, slip_rate_bound_max, slip_rate_bound_partials = (
        get_slip_rate_bounds(segment, block)
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


def get_elastic_operator_single_mesh(
    meshes: List, station: pd.DataFrame, command: Dict, mesh_index: np.int_
):
    """
    Calculate (or load previously calculated) elastic operators from
    both fully locked segments and TDE parameterizes surfaces

    Args:
        operators (Dict): Elastic operators will be added to this data structure
        meshes (List): Geometries of meshes
        segment (pd.DataFrame): All segment data
        station (pd.DataFrame): All station data
        command (Dict): All command data
    """
    if bool(command.reuse_elastic) and os.path.exists(command.reuse_elastic_file):
        logger.info("Using precomputed elastic operators")
        hdf5_file = h5py.File(command.reuse_elastic_file, "r")
        tde_to_velocities = np.array(
            hdf5_file.get("tde_to_velocities_" + str(mesh_index))
        )
        hdf5_file.close()

    else:
        if not os.path.exists(command.reuse_elastic_file):
            logger.warning("Precomputed elastic operator file not found")
        logger.info("Computing elastic operators")
        logger.info(
            f"Start: TDE slip to velocity calculation for mesh: {meshes[mesh_index].file_name}"
        )
        tde_to_velocities = get_tde_to_velocities_single_mesh(
            meshes, station, command, mesh_idx=mesh_index
        )
        logger.success(
            f"Finish: TDE slip to velocity calculation for mesh: {meshes[mesh_index].file_name}"
        )

    # Save tde to velocity matrix for current mesh
    if bool(command.save_elastic):
        # Check to see if "data/operators" folder exists and if not create it
        if not os.path.exists(command.operators_folder):
            os.mkdir(command.operators_folder)

        logger.info(
            "Saving elastic to velocity matrices to :" + command.save_elastic_file
        )

        # Check if file exists.  If it does append.
        if os.path.exists(command.save_elastic_file):
            hdf5_file = h5py.File(command.save_elastic_file, "a")
            current_mesh_label = "tde_to_velocities_" + str(mesh_index)
            if current_mesh_label in hdf5_file:
                hdf5_file[current_mesh_label][...] = tde_to_velocities
            else:
                hdf5_file.create_dataset(current_mesh_label, data=tde_to_velocities)
        else:
            hdf5_file = h5py.File(command.save_elastic_file, "w")
            hdf5_file.create_dataset(
                "tde_to_velocities_" + str(mesh_index), data=tde_to_velocities
            )
        hdf5_file.close()
    return tde_to_velocities


def get_h_matrices_for_tde_meshes(
    command, meshes, station, operators, index, col_norms
):
    # Create lists for all TDE matrices per mesh
    H = []
    for i in range(len(meshes)):
        # Get full TDE to velocity matrix for current mesh
        tde_to_velocities = get_elastic_operator_single_mesh(
            meshes, station, command, i
        )

        # H-matrix representation
        H.append(
            build_hmatrix_from_mesh_tdes(
                meshes[i],
                station,
                -tde_to_velocities,
                command.h_matrix_tol,
                command.h_matrix_min_separation,
                command.h_matrix_min_pts_per_box,
            )
        )

        logger.info(
            f"mesh {i} ({meshes[i].file_name}) H-matrix compression ratio: {H[i].report_compression_ratio():0.4f}"
        )

        # Case smoothing matrices and tde slip rate constraints to sparse
        smoothing_keep_index = get_keep_index_12(operators.smoothing_matrix[i].shape[0])
        operators.smoothing_matrix[i] = csr_matrix(
            operators.smoothing_matrix[i][smoothing_keep_index, :][
                :, smoothing_keep_index
            ]
        )
        operators.tde_slip_rate_constraints[i] = csr_matrix(
            operators.tde_slip_rate_constraints[i]
        )

        # Eliminate unused columns and rows of TDE to velocity matrix
        tde_to_velocities = np.delete(
            tde_to_velocities, np.arange(2, tde_to_velocities.shape[0], 3), axis=0
        )
        tde_to_velocities = np.delete(
            tde_to_velocities, np.arange(2, tde_to_velocities.shape[1], 3), axis=1
        )

        # Calculate column normalization vector current TDE mesh
        weighting_vector_no_zero_rows = get_weighting_vector_single_mesh_for_col_norms(
            command, station, meshes, index, i
        )
        current_tde_mesh_columns_full_no_zero_rows = np.vstack(
            (
                -tde_to_velocities,
                operators.smoothing_matrix[i].toarray(),
                operators.tde_slip_rate_constraints[i].toarray(),
            )
        ) * np.sqrt(weighting_vector_no_zero_rows[:, None])

        # Concatenate everthing we need for col_norms
        col_norms_current_tde_mesh = np.linalg.norm(
            current_tde_mesh_columns_full_no_zero_rows, axis=0
        )
        col_norms = np.hstack((col_norms, col_norms_current_tde_mesh))

        # Free memory.  We have the Hmatrix version of this.
        del tde_to_velocities
    return H, col_norms


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
    ordered_index = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[ordered_index]
    eigenvectors = eigenvectors[:, ordered_index]
    return eigenvalues, eigenvectors


def get_eigenvectors_to_tde_slip(operators, meshes):
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
                meshes[i].n_modes_strike_slip + meshes[i].n_modes_dip_slip,
            )
        )

        # Place strike-slip panel
        operators.eigenvectors_to_tde_slip[i][
            0::2, 0 : meshes[i].n_modes_strike_slip
        ] = eigenvectors[:, 0 : meshes[i].n_modes_strike_slip]

        # Place dip-slip panel
        operators.eigenvectors_to_tde_slip[i][
            1::2,
            meshes[i].n_modes_strike_slip : meshes[i].n_modes_strike_slip
            + meshes[i].n_modes_dip_slip,
        ] = eigenvectors[:, 0 : meshes[i].n_modes_dip_slip]
        logger.success(
            f"Finish: Eigenvectors to TDE slip for mesh: {meshes[i].file_name}"
        )


######################################################################
#                                                                    #
#                                                                    #
#   ad88888ba     ,ad8888ba,    88     8b           d8  88888888888  #
#  d8"     "8b   d8"'    `"8b   88     `8b         d8'  88           #
#  Y8,          d8'        `8b  88      `8b       d8'   88           #
#  `Y8aaaaa,    88          88  88       `8b     d8'    88aaaaa      #
#    `"""""8b,  88          88  88        `8b   d8'     88"""""      #
#          `8b  Y8,        ,8P  88         `8b d8'      88           #
#  Y8a     a8P   Y8a.    .a8P   88          `888'       88           #
#   "Y88888P"     `"Y8888Y"'    88888888888  `8'        88888888888  #
#                                                                    #
#                                                                    #
######################################################################


def assemble_and_solve_dense(
    command, assembly, operators, station, block, meshes, mogi
):
    index = get_index(assembly, station, block, meshes, mogi)
    estimation = addict.Dict()
    estimation.data_vector = get_data_vector(assembly, index, meshes)
    estimation.weighting_vector = get_weighting_vector(command, station, meshes, index)
    estimation.operator = get_full_dense_operator(operators, meshes, index)

    # DEBUG HERE:
    # IPython.embed(banner1="")

    # Solve the overdetermined linear system using only a weighting vector rather than matrix
    estimation.state_covariance_matrix = np.linalg.inv(
        estimation.operator.T * estimation.weighting_vector @ estimation.operator
    )
    estimation.state_vector = (
        estimation.state_covariance_matrix
        @ estimation.operator.T
        * estimation.weighting_vector
        @ estimation.data_vector
    )
    return index, estimation


def post_process_estimation(
    estimation: Dict, operators: Dict, station: pd.DataFrame, index: Dict
):
    """Calculate derived values derived from the block model linear estimate (e.g., velocities, undertainties)

    Args:
        estimation (Dict): Estimated state vector and model covariance
        operators (Dict): All linear operators
        station (pd.DataFrame): GPS station data
        idx (Dict): Indices and counts of data and array sizes
    """
    # Convert rotation vectors to Euler poles

    estimation.predictions = estimation.operator @ estimation.state_vector
    estimation.vel = estimation.predictions[0 : 2 * index.n_stations]
    estimation.east_vel = estimation.vel[0::2]
    estimation.north_vel = estimation.vel[1::2]

    # Estimate slip rate uncertainties
    estimation.slip_rate_sigma = np.sqrt(
        np.diag(
            operators.rotation_to_slip_rate
            @ estimation.state_covariance_matrix[
                0 : 3 * index.n_blocks, 0 : 3 * index.n_blocks
            ]
            @ operators.rotation_to_slip_rate.T
        )
    )  # I don't think this is correct because for the case when there is a rotation vector a priori
    estimation.strike_slip_rate_sigma = estimation.slip_rate_sigma[0::3]
    estimation.dip_slip_rate_sigma = estimation.slip_rate_sigma[1::3]
    estimation.tensile_slip_rate_sigma = estimation.slip_rate_sigma[2::3]

    # Calculate mean squared residual velocity
    estimation.east_vel_residual = estimation.east_vel - station.east_vel
    estimation.north_vel_residual = estimation.north_vel - station.north_vel

    # Extract TDE slip rates from state vector
    estimation.tde_rates = estimation.state_vector[
        3 * index.n_blocks : 3 * index.n_blocks + 2 * index.n_tde_total
    ]
    estimation.tde_strike_slip_rates = estimation.tde_rates[0::2]
    estimation.tde_dip_slip_rates = estimation.tde_rates[1::2]

    # Extract segment slip rates from state vector
    estimation.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.strike_slip_rates = estimation.slip_rates[0::3]
    estimation.dip_slip_rates = estimation.slip_rates[1::3]
    estimation.tensile_slip_rates = estimation.slip_rates[2::3]

    # Extract estimated block strain rates
    estimation.block_strain_rates = estimation.state_vector[
        3 * index.n_blocks
        + 2 * index.n_tde_total : 3 * index.n_blocks
        + 2 * index.n_tde_total
        + index.n_block_strain_components
    ]

    # Extract Mogi parameters
    estimation.mogi_volume_change_rates = estimation.state_vector[
        +3 * index.n_blocks
        + 2 * index.n_tde_total
        + index.n_block_strain_components : +3 * index.n_blocks
        + 2 * index.n_tde_total
        + index.n_block_strain_components
        + index.n_mogi
    ]

    # Calculate rotation only velocities
    estimation.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_rotation = estimation.vel_rotation[0::2]
    estimation.north_vel_rotation = estimation.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_elastic_segment = estimation.vel_elastic_segment[0::2]
    estimation.north_vel_elastic_segment = estimation.vel_elastic_segment[1::2]

    # TODO: Calculate block strain rate velocities
    estimation.vel_block_strain_rate = (
        operators.block_strain_rate_to_velocities[index.station_row_keep_index, :]
        @ estimation.block_strain_rates
    )
    estimation.east_vel_block_strain_rate = estimation.vel_block_strain_rate[0::2]
    estimation.north_vel_block_strain_rate = estimation.vel_block_strain_rate[1::2]

    # Calculate Mogi source velocities
    estimation.vel_mogi = (
        operators.mogi_to_velocities[index.station_row_keep_index, :]
        @ estimation.mogi_volume_change_rates
    )
    estimation.east_vel_mogi = estimation.vel_mogi[0::2]
    estimation.north_vel_mogi = estimation.vel_mogi[1::2]

    # Calculate TDE velocities
    estimation.vel_tde = np.zeros(2 * index.n_stations)
    for i in range(len(operators.tde_to_velocities)):
        tde_keep_row_index = get_keep_index_12(operators.tde_to_velocities[i].shape[0])
        tde_keep_col_index = get_keep_index_12(operators.tde_to_velocities[i].shape[1])
        estimation.vel_tde += (
            operators.tde_to_velocities[i][tde_keep_row_index, :][:, tde_keep_col_index]
            @ estimation.state_vector[index.start_tde_col[i] : index.end_tde_col[i]]
        )
    estimation.east_vel_tde = estimation.vel_tde[0::2]
    estimation.north_vel_tde = estimation.vel_tde[1::2]


def post_process_estimation_eigen(estimation_eigen, operators, station, index):
    """Calculate derived values derived from the block model linear estimate (e.g., velocities, undertainties)

    Args:
        estimation (Dict): Estimated state vector and model covariance
        operators (Dict): All linear operators
        station (pd.DataFrame): GPS station data
        idx (Dict): Indices and counts of data and array sizes
    """

    # Isolate eigenvalues (weights for eigenmodes)
    estimation_eigen.eigenvalues = estimation_eigen.state_vector[
        index.start_col_eigen[0] : index.end_col_eigen[-1]
    ]

    estimation_eigen.predictions = (
        estimation_eigen.operator @ estimation_eigen.state_vector
    )

    estimation_eigen.vel = estimation_eigen.predictions[0 : 2 * index.n_stations]
    estimation_eigen.east_vel = estimation_eigen.vel[0::2]
    estimation_eigen.north_vel = estimation_eigen.vel[1::2]

    # Estimate slip rate uncertainties
    # Set to ones because we have no meaningful uncertainties for
    # The quadratic programming solve
    estimation_eigen.strike_slip_rate_sigma = np.zeros(index.n_segments)
    estimation_eigen.dip_slip_rate_sigma = np.zeros(index.n_segments)
    estimation_eigen.tensile_slip_rate_sigma = np.zeros(index.n_segments)

    # Calculate mean squared residual velocity
    estimation_eigen.east_vel_residual = estimation_eigen.east_vel - station.east_vel
    estimation_eigen.north_vel_residual = estimation_eigen.north_vel - station.north_vel

    # Extract TDE slip rates from state vector
    estimation_eigen.tde_rates = np.zeros(2 * index.n_tde_total)
    for i in range(index.n_meshes):
        # Temporary indices for easier reading
        start_idx = index.start_tde_col[i] - 3 * index.n_blocks
        end_idx = index.end_tde_col[i] - 3 * index.n_blocks

        # Calcuate estimated TDE rates for the current mesh
        estimation_eigen.tde_rates[start_idx:end_idx] = (
            operators.eigenvectors_to_tde_slip[i]
            @ estimation_eigen.state_vector[
                index.start_col_eigen[i] : index.end_col_eigen[i]
            ]
        )

    # Isolate strike- and dip-slip rates
    estimation_eigen.tde_strike_slip_rates = estimation_eigen.tde_rates[0::2]
    estimation_eigen.tde_dip_slip_rates = estimation_eigen.tde_rates[1::2]

    # Create a pseudo state vector that is the length of a TDE state vector
    estimation_eigen.pseudo_tde_state_vector = np.zeros(
        3 * index.n_blocks + 2 * index.n_tde_total
    )
    estimation_eigen.pseudo_tde_state_vector[0 : 3 * index.n_blocks] = (
        estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )

    # Insert estimated TDE rates into pseudo state vector
    estimation_eigen.pseudo_tde_state_vector[
        index.start_tde_col[0] : index.end_tde_col[-1]
    ] = estimation_eigen.tde_rates

    # Extract segment slip rates from state vector
    estimation_eigen.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_eigen.strike_slip_rates = estimation_eigen.slip_rates[0::3]
    estimation_eigen.dip_slip_rates = estimation_eigen.slip_rates[1::3]
    estimation_eigen.tensile_slip_rates = estimation_eigen.slip_rates[2::3]

    # Calculate rotation only velocities
    estimation_eigen.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_eigen.east_vel_rotation = estimation_eigen.vel_rotation[0::2]
    estimation_eigen.north_vel_rotation = estimation_eigen.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation_eigen.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation_eigen.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_eigen.east_vel_elastic_segment = estimation_eigen.vel_elastic_segment[
        0::2
    ]
    estimation_eigen.north_vel_elastic_segment = estimation_eigen.vel_elastic_segment[
        1::2
    ]

    # Extract estimated block strain rates
    estimation_eigen.block_strain_rates = estimation_eigen.state_vector[
        index.start_block_strain_col_eigen : index.end_block_strain_col_eigen
    ]

    # Calculate block strain rate velocities
    estimation_eigen.vel_block_strain_rate = (
        operators.block_strain_rate_to_velocities[index.station_row_keep_index, :]
        @ estimation_eigen.block_strain_rates
    )
    estimation_eigen.east_vel_block_strain_rate = (
        estimation_eigen.vel_block_strain_rate[0::2]
    )
    estimation_eigen.north_vel_block_strain_rate = (
        estimation_eigen.vel_block_strain_rate[1::2]
    )

    # Extract Mogi parameters
    estimation_eigen.mogi_volume_change_rates = estimation_eigen.state_vector[
        index.start_mogi_col_eigen : index.end_mogi_col_eigen
    ]

    # Calculate Mogi source velocities
    estimation_eigen.vel_mogi = (
        operators.mogi_to_velocities[index.station_row_keep_index, :]
        @ estimation_eigen.mogi_volume_change_rates
    )
    estimation_eigen.east_vel_mogi = estimation_eigen.vel_mogi[0::2]
    estimation_eigen.north_vel_mogi = estimation_eigen.vel_mogi[1::2]

    # Calculate elastic TDE velocities from eigenmodes
    estimation_eigen.vel_tde = np.zeros(2 * index.n_stations)
    for i in range(len(operators.tde_to_velocities)):
        estimation_eigen.vel_tde += (
            -operators.eigen_to_velocities[i]
            @ estimation_eigen.state_vector[
                index.start_col_eigen[i] : index.end_col_eigen[i]
            ]
        )

    estimation_eigen.east_vel_tde = estimation_eigen.vel_tde[0::2]
    estimation_eigen.north_vel_tde = estimation_eigen.vel_tde[1::2]


def euler_pole_covariance_to_rotation_vector_covariance(
    omega_x, omega_y, omega_z, euler_pole_covariance_all
):
    """
    This function takes the model parameter covariance matrix
    in terms of the Euler pole and rotation rate and linearly
    propagates them to rotation vector space.
    """
    omega_x_sig = np.zeros_like(omega_x)
    omega_y_sig = np.zeros_like(omega_y)
    omega_z_sig = np.zeros_like(omega_z)
    for i in range(len(omega_x)):
        x = omega_x[i]
        y = omega_y[i]
        z = omega_z[i]
        euler_pole_covariance_current = euler_pole_covariance_all[
            3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
        ]

        """
        There may be cases where x, y and z are all zero.  This leads to /0 errors.  To avoid this  %%
        we check for these cases and Let A = b * I where b is a small constant (10^-4) and I is     %%
        the identity matrix
        """
        if (x == 0) and (y == 0):
            euler_to_cartsian_operator = 1e-4 * np.eye(
                3
            )  # Set a default small value for rotation vector uncertainty
        else:
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
            euler_to_cartsian_operator = np.array(
                [
                    [dlat_dx, dlat_dy, dlat_dz],
                    [dlon_dx, dlon_dy, dlon_dz],
                    [dmag_dx, dmag_dy, dmag_dz],
                ]
            )

        # Propagate the Euler pole covariance matrix to a rotation rate
        # covariance matrix
        rotation_vector_covariance = (
            np.linalg.inv(euler_to_cartsian_operator)
            * euler_pole_covariance_current
            * np.linalg.inv(euler_to_cartsian_operator).T
        )

        # Organized data for the return
        main_diagonal_values = np.diag(rotation_vector_covariance)
        omega_x_sig[i] = np.sqrt(main_diagonal_values[0])
        omega_y_sig[i] = np.sqrt(main_diagonal_values[1])
        omega_z_sig[i] = np.sqrt(main_diagonal_values[2])
    return omega_x_sig, omega_y_sig, omega_z_sig


def lsqlin_qp(
    C,
    d,
    reg=0,
    A=None,
    b=None,
    Aeq=None,
    beq=None,
    lb=None,
    ub=None,
    x0=None,
    opts=None,
):
    """
    Solve linear constrained l2-regularized least squares. Can
    handle both dense and sparse matrices. Call modeled after Matlab's
    lsqlin. It is actually wrapper around CVXOPT QP solver.

        min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
        s.t.  A * x <= b
              Aeq * x = beq
              lb <= x <= ub

    Input arguments:
        C   is m x n dense or sparse matrix
        d   is n x 1 dense matrix
        reg is regularization parameter
        A   is p x n dense or sparse matrix
        b   is p x 1 dense matrix
        Aeq is q x n dense or sparse matrix
        beq is q x 1 dense matrix
        lb  is n x 1 matrix or scalar
        ub  is n x 1 matrix or scalar

    Output arguments:
        Return dictionary, the output of CVXOPT QP.

    Dont pass matlab-like empty lists to avoid setting parameters,
    just use None:
        lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
        lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!

    Provenance notes:
    Found a few places on Github:
    - https://github.com/KasparP/PSI_simulations/blob/master/Python/SLAPMi/lsqlin.py
    - https://github.com/geospace-code/airtools/blob/main/src/airtools/lsqlin.py

    Some attribution:
    __author__ = "Valeriy Vishnevskiy", "Michael Hirsch"
    __email__ = "valera.vishnevskiy@yandex.ru"
    __version__ = "1.0"
    __date__ = "22.11.2013"
    __license__ = "MIT"
    """

    # Helper functions
    def scipy_sparse_to_spmatrix(A):
        coo = A.tocoo()
        SP = cvxopt.spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
        return SP

    def spmatrix_sparse_to_scipy(A):
        data = np.array(A.V).squeeze()
        rows = np.array(A.I).squeeze()
        cols = np.array(A.J).squeeze()
        return scipy.sparse.coo_matrix((data, (rows, cols)))

    def sparse_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return scipy.sparse.vstack([A1, A2])

    def numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        elif isinstance(A1, np.ndarray):
            return np.vstack([A1, A2])
        elif isinstance(A1, cvxopt.spmatrix):
            return np.vstack([cvxopt_to_numpy_matrix(A1).todense(), A2])

    def numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    def numpy_to_cvxopt_matrix(A):
        if A is None:
            return

        if scipy.sparse.issparse(A):
            if isinstance(A, scipy.sparse.spmatrix):
                return scipy_sparse_to_spmatrix(A)
            else:
                return A
        else:
            if isinstance(A, np.ndarray):
                if A.ndim == 1:
                    return cvxopt.matrix(A, (A.shape[0], 1), "d")
                else:
                    return cvxopt.matrix(A, A.shape, "d")
            else:
                return A

    def cvxopt_to_numpy_matrix(A):
        if A is None:
            return
        if isinstance(A, cvxopt.spmatrix):
            return spmatrix_sparse_to_scipy(A)
        elif isinstance(A, cvxopt.matrix):
            return np.asarray(A).squeeze()
        else:
            return np.asarray(A).squeeze()

    # Main function body
    if scipy.sparse.issparse(A):  # detects both np and cxopt sparse
        sparse_case = True
        # We need A to be scipy sparse, as I couldn't find how
        # CVXOPT spmatrix can be vstacked
        if isinstance(A, cvxopt.spmatrix):
            A = spmatrix_sparse_to_scipy(A)
    else:
        sparse_case = False

    C = numpy_to_cvxopt_matrix(C)
    d = numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = -d.T * C
    nvars = C.size[1]

    if reg > 0:
        if sparse_case:
            i = scipy_sparse_to_spmatrix(scipy.sparse.eye(nvars, nvars, format="coo"))
        else:
            i = cvxopt.matrix(np.eye(nvars), (nvars, nvars), "d")
        Q = Q + reg * i

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b = cvxopt_to_numpy_matrix(b)

    if lb is not None:  # Modify 'A' and 'b' to add lb inequalities
        if lb.size == 1:
            lb = np.repeat(lb, nvars)

        if sparse_case:
            lb_A = -scipy.sparse.eye(nvars, nvars, format="coo")
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:  # Modify 'A' and 'b' to add ub inequalities
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = scipy.sparse.eye(nvars, nvars, format="coo")
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    # Convert data to CVXOPT format
    A = numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b = numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    # Set up options
    if opts is not None:
        for k, v in opts.items():
            cvxopt.solvers.options[k] = v

    # Run CVXOPT.SQP solver
    sol = cvxopt.solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
    return sol


def matvec_wrapper(h_matrix_solve_parameters):
    def matvec_caller(x):
        return matvec(x, h_matrix_solve_parameters)

    return matvec_caller


def rmatvec_wrapper(h_matrix_solve_parameters):
    def rmatvec_caller(x):
        return rmatvec(x, h_matrix_solve_parameters)

    return rmatvec_caller


def matvec(v, h_matrix_solve_parameters):
    """Build matvec (matrix vector product) operator for
    scipy.sparse.linalg.LinearOperator.  This returns A * u

    Args:
        u (nd.array): Candidate state vector

    Returns:
        out (nd.array): Predicted data vector
    """

    # Unpack parameters
    (
        index,
        meshes,
        H,
        operators,
        weighting_vector,
        col_norms,
        sparse_block_motion_okada_faults,
        sparse_block_motion_constraints,
        sparse_block_slip_rate_constraints,
    ) = h_matrix_solve_parameters

    # Column normalize the state vector
    v_scaled = v / col_norms

    # Make storage for output
    out = np.zeros(index.n_operator_rows)
    block_rotations = v_scaled[index.start_block_col : index.end_block_col]

    # Okada
    out[
        index.start_station_row : index.end_station_row
    ] += sparse_block_motion_okada_faults.dot(block_rotations)

    # Block motion constraints
    out[
        index.start_block_constraints_row : index.end_block_constraints_row
    ] += sparse_block_motion_constraints.dot(block_rotations)

    # Slip rate constraints
    out[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ] += sparse_block_slip_rate_constraints.dot(block_rotations)

    # Loop over TDE meshes
    # for i in range(len(meshes)):
    for i in range(len(meshes)):
        tde_velocities = v_scaled[index.start_tde_col[i] : index.end_tde_col[i]]

        # Insert TDE to velocity matrix
        out[index.start_station_row : index.end_station_row] += H[i].dot(tde_velocities)

        # TDE smoothing
        out[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i]
        ] += operators.smoothing_matrix[i].dot(tde_velocities)

        # TDE slip rate constraints
        out[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i]
        ] += operators.tde_slip_rate_constraints[i].dot(tde_velocities)

    # Weight!
    return out * np.sqrt(weighting_vector)


def rmatvec(u, h_matrix_solve_parameters):
    """Build rmatvec (matrix vector product) operator for
    scipy.sparse.linalg.LinearOperator.  This returns:
    Returns A^H * v, where A^H is the conjugate transpose of A
    for a candidate state vector, u.  We do this because
    with the h-matrix approach we no longer have the full matrix
    so we can't take the transpose all at once.

    Args:
        u (nd.array): Candidate state vector

    Returns:
        out (nd.array): Predicted data vector
    """

    # Unpack parameters
    (
        index,
        meshes,
        H,
        operators,
        weighting_vector,
        col_norms,
        sparse_block_motion_okada_faults,
        sparse_block_motion_constraints,
        sparse_block_slip_rate_constraints,
    ) = h_matrix_solve_parameters

    # Weight the data vector
    u_weighted = u * np.sqrt(weighting_vector)

    # Storage for output
    # out = np.zeros(X.shape[1])
    out = np.zeros(index.n_operator_cols)

    # Select subset of weighted data for the observed velocities
    station_rows = u_weighted[index.start_station_row : index.end_station_row]
    block_constraints = u_weighted[
        index.start_block_constraints_row : index.end_block_constraints_row
    ]

    # Select subset of weighted data for the fault slip rate constraints
    slip_rate_constraints = u_weighted[
        index.start_slip_rate_constraints_row : index.end_slip_rate_constraints_row
    ]

    # Okada and block rotation contribution to data vector
    out[index.start_block_col : index.end_block_col] += (
        station_rows @ sparse_block_motion_okada_faults
    )

    # Block motion constraints contribution to data vector
    out[index.start_block_col : index.end_block_col] += (
        block_constraints @ sparse_block_motion_constraints
    )

    # Fault slip rate constraints contribution to data vector
    out[index.start_block_col : index.end_block_col] += (
        slip_rate_constraints @ sparse_block_slip_rate_constraints
    )

    for i in range(len(meshes)):
        # Select subset of weighted data for the TDE smoothing
        tde_smoothing = u_weighted[
            index.start_tde_smoothing_row[i] : index.end_tde_smoothing_row[i]
        ]

        # Select subset of weighted data for the TDE slip rate constraints
        tde_slip_rate = u_weighted[
            index.start_tde_constraint_row[i] : index.end_tde_constraint_row[i]
        ]

        # Hmatrix (TDEs to velocities)
        out[index.start_tde_col[i] : index.end_tde_col[i]] += H[i].transpose_dot(
            station_rows
        )

        # TDE smoothing contribution to data vector
        out[index.start_tde_col[i] : index.end_tde_col[i]] += (
            tde_smoothing @ operators.smoothing_matrix[i]
        )

        # TDE slip rate constraint contributions to data vector
        out[index.start_tde_col[i] : index.end_tde_col[i]] += (
            tde_slip_rate @ operators.tde_slip_rate_constraints[i]
        )

    # Weight
    return out / col_norms


def build_and_solve_hmatrix(command, assembly, operators, data):
    logger.info("build_and_solve_hmatrix")

    # Calculate Okada partials for all segments
    get_elastic_operators_okada(operators, data.segment, data.station, command)

    # Get TDE smoothing operators
    get_all_mesh_smoothing_matrices(data.meshes, operators)

    # Get non elastic operators
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        data.station, data.block.shape[0]
    )
    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        data.station
    )
    assembly, operators.block_motion_constraints = get_block_motion_constraints(
        assembly, data.block, command
    )
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, data.segment, data.block, command
    )
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        data.segment, data.block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        data.block, data.station, data.segment
    )
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        data.mogi, data.station, command
    )
    operators.rotation_to_slip_rate_to_okada_to_velocities = (
        operators.slip_rate_to_okada_to_velocities @ operators.rotation_to_slip_rate
    )
    get_tde_slip_rate_constraints(data.meshes, operators)

    index = get_index(assembly, data.station, data.block, data.meshes)

    # Data and data weighting vector
    weighting_vector = get_weighting_vector(command, data.station, data.meshes, index)
    data_vector = get_data_vector(assembly, index)

    # Apply data weighting
    data_vector = data_vector * np.sqrt(weighting_vector)

    # Cast all block submatrices to sparse
    sparse_block_motion_okada_faults = csr_matrix(
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        - operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
    )
    sparse_block_motion_constraints = csr_matrix(operators.block_motion_constraints)
    sparse_block_slip_rate_constraints = csr_matrix(operators.slip_rate_constraints)

    # Calculate column normalization vector for blocks
    operator_block_only = get_full_dense_operator_block_only(operators, index)
    weighting_vector_block_only = weighting_vector[0 : operator_block_only.shape[0]][
        :, None
    ]
    col_norms = np.linalg.norm(
        operator_block_only * np.sqrt(weighting_vector_block_only), axis=0
    )

    # Hmatrix decompositon for each TDE mesh
    logger.info("Start: H-matrix build")
    H, col_norms = get_h_matrices_for_tde_meshes(
        command, data.meshes, data.station, operators, index, col_norms
    )
    logger.success("Finish: H-matrix build")

    # Package parameters that matvec and rmatvec need for the iterative solve
    h_matrix_solve_parameters = (
        index,
        data.meshes,
        H,
        operators,
        weighting_vector,
        col_norms,
        sparse_block_motion_okada_faults,
        sparse_block_motion_constraints,
        sparse_block_slip_rate_constraints,
    )

    # Instantiate the scipy the linear operator for the iterative solver to use
    operator_hmatrix = scipy.sparse.linalg.LinearOperator(
        (index.n_operator_rows, index.n_operator_cols),
        matvec=matvec_wrapper(h_matrix_solve_parameters),
        rmatvec=rmatvec_wrapper(h_matrix_solve_parameters),
    )

    # Solve the linear system
    logger.info("Start: H-matrix solve")
    start_solve_time = timeit.default_timer()

    if command.iterative_solver == "lsmr":
        logger.info("Using LSMR solver")
        sparse_hmatrix_solution = scipy.sparse.linalg.lsmr(
            operator_hmatrix, data_vector, atol=command.atol, btol=command.btol
        )
    elif command.iterative_solver == "lsqr":
        logger.info("Using LSQR solver")
        sparse_hmatrix_solution = scipy.sparse.linalg.lsqr(
            operator_hmatrix, data_vector, atol=command.atol, btol=command.btol
        )

    end_solve_time = timeit.default_timer()
    logger.success(
        f"Finish: H-matrix solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    )

    # Correct the solution for the col_norms preconditioning.
    estimation = addict.Dict()

    sparse_hmatrix_state_vector = sparse_hmatrix_solution[0] / col_norms
    if command.iterative_solver == "lsqr":
        sparse_hmatrix_state_vector_sigma = (
            np.sqrt(sparse_hmatrix_solution[9]) / col_norms
        )
        estimation.state_vector_sigma = sparse_hmatrix_state_vector_sigma

    estimation.data_vector = data_vector
    estimation.weighting_vector = weighting_vector
    estimation.operator = operator_hmatrix
    estimation.state_vector = sparse_hmatrix_state_vector

    post_process_estimation_hmatrix(
        command,
        data.block,
        estimation,
        operators,
        data.meshes,
        H,
        data.station,
        index,
        col_norms,
        h_matrix_solve_parameters,
    )
    write_output(
        command, estimation, data.station, data.segment, data.block, data.meshes
    )

    if bool(command.plot_estimation_summary):
        plot_estimation_summary(
            command,
            data.segment,
            data.station,
            data.meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    return estimation, operators, index


def post_process_estimation_hmatrix(
    command: Dict,
    block: pd.DataFrame,
    estimation_hmatrix: Dict,
    operators: Dict,
    meshes: List,
    H: List,
    station: pd.DataFrame,
    index: Dict,
    col_norms: np.array,
    h_matrix_solve_parameters: Tuple,
):
    """Calculate derived values derived from the block model linear estimate (e.g., velocities, undertainties)

    Args:
        estimation (Dict): Estimated state vector and model covariance
        operators (Dict): All linear operators
        meshes (List): Mesh geometries
        H (List): Hmatrix decompositions for each TDE mesh
        station (pd.DataFrame): GPS station data
        index (Dict): Indices and counts of data and array sizes
        col_norms (np.array): Column preconditining vector
        h_matrix_solve_parameters (Tuple): Package of sparse and hmatrix operators
    """

    estimation_hmatrix.predictions = matvec(
        estimation_hmatrix.state_vector * col_norms, h_matrix_solve_parameters
    ) / np.sqrt(estimation_hmatrix.weighting_vector)
    estimation_hmatrix.vel = estimation_hmatrix.predictions[0 : 2 * index.n_stations]
    estimation_hmatrix.east_vel = estimation_hmatrix.vel[0::2]
    estimation_hmatrix.north_vel = estimation_hmatrix.vel[1::2]

    # Calculate mean squared residual velocity
    estimation_hmatrix.east_vel_residual = (
        estimation_hmatrix.east_vel - station.east_vel
    )
    estimation_hmatrix.north_vel_residual = (
        estimation_hmatrix.north_vel - station.north_vel
    )

    # Extract TDE slip rates from state vector
    estimation_hmatrix.tde_rates = estimation_hmatrix.state_vector[
        3 * index.n_blocks : 3 * index.n_blocks + 2 * index.n_tde_total
    ]
    estimation_hmatrix.tde_strike_slip_rates = estimation_hmatrix.tde_rates[0::2]
    estimation_hmatrix.tde_dip_slip_rates = estimation_hmatrix.tde_rates[1::2]

    # Extract segment slip rates from state vector
    estimation_hmatrix.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation_hmatrix.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_hmatrix.strike_slip_rates = estimation_hmatrix.slip_rates[0::3]
    estimation_hmatrix.dip_slip_rates = estimation_hmatrix.slip_rates[1::3]
    estimation_hmatrix.tensile_slip_rates = estimation_hmatrix.slip_rates[2::3]

    if command.iterative_solver == "lsmr":
        # All uncertainties set to 1 because lsmr doesn't calculate variance
        logger.warning(
            "Slip rate uncertainty estimates set to 1 because LSMR doesn't provide variance estimates"
        )
        estimation_hmatrix.strike_slip_rate_sigma = np.ones_like(
            estimation_hmatrix.strike_slip_rates
        )
        estimation_hmatrix.dip_slip_rate_sigma = np.ones_like(
            estimation_hmatrix.dip_slip_rates
        )
        estimation_hmatrix.tensile_slip_rate_sigma = np.ones_like(
            estimation_hmatrix.tensile_slip_rates
        )
    elif command.iterative_solver == "lsqr":
        # TODO: Block motion uncertainties
        estimation_hmatrix.slip_rate_sigma = np.sqrt(
            np.diag(
                operators.rotation_to_slip_rate
                @ np.diag(estimation_hmatrix.state_vector_sigma[0 : 3 * index.n_blocks])
                @ operators.rotation_to_slip_rate.T
            )
        )
        estimation_hmatrix.strike_slip_rate_sigma = estimation_hmatrix.slip_rate_sigma[
            0::3
        ]
        estimation_hmatrix.dip_slip_rate_sigma = estimation_hmatrix.slip_rate_sigma[
            1::3
        ]
        estimation_hmatrix.tensile_slip_rate_sigma = estimation_hmatrix.slip_rate_sigma[
            2::3
        ]

    # Calculate rotation only velocities
    estimation_hmatrix.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation_hmatrix.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_hmatrix.east_vel_rotation = estimation_hmatrix.vel_rotation[0::2]
    estimation_hmatrix.north_vel_rotation = estimation_hmatrix.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation_hmatrix.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation_hmatrix.state_vector[0 : 3 * index.n_blocks]
    )
    estimation_hmatrix.east_vel_elastic_segment = (
        estimation_hmatrix.vel_elastic_segment[0::2]
    )
    estimation_hmatrix.north_vel_elastic_segment = (
        estimation_hmatrix.vel_elastic_segment[1::2]
    )

    # TODO: Calculate block strain rate velocities
    estimation_hmatrix.east_vel_block_strain_rate = np.zeros(len(station))
    estimation_hmatrix.north_vel_block_strain_rate = np.zeros(len(station))

    # Calculate TDE velocities
    estimation_hmatrix.vel_tde = np.zeros(2 * index.n_stations)
    for i in range(len(meshes)):
        estimation_hmatrix.vel_tde += H[i].dot(
            estimation_hmatrix.state_vector[
                index.start_tde_col[i] : index.end_tde_col[i]
            ]
        )
    estimation_hmatrix.east_vel_tde = estimation_hmatrix.vel_tde[0::2]
    estimation_hmatrix.north_vel_tde = estimation_hmatrix.vel_tde[1::2]


def build_and_solve_dense(command, assembly, operators, data):
    # NOTE: Used in celeri_solve.py
    logger.info("build_and_solve_dense")

    # Get all elastic operators for segments and TDEs
    get_elastic_operators(operators, data.meshes, data.segment, data.station, command)

    # Get TDE smoothing operators
    get_all_mesh_smoothing_matrices(data.meshes, operators)

    # Get non-elastic operators
    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        data.station, data.block.shape[0]
    )
    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        data.station
    )
    assembly, operators.block_motion_constraints = get_block_motion_constraints(
        assembly, data.block, command
    )
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, data.segment, data.block, command
    )
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        data.segment, data.block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        data.block, data.station, data.segment
    )
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        data.mogi, data.station, command
    )
    get_tde_slip_rate_constraints(data.meshes, operators)

    # Direct solve dense linear system
    logger.info("Start: Dense assemble and solve")
    start_solve_time = timeit.default_timer()
    index, estimation = assemble_and_solve_dense(
        command, assembly, operators, data.station, data.block, data.meshes, data.mogi
    )
    end_solve_time = timeit.default_timer()
    logger.success(
        f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    )

    post_process_estimation(estimation, operators, data.station, index)

    write_output(
        command, estimation, data.station, data.segment, data.block, data.meshes
    )

    if bool(command.plot_estimation_summary):
        plot_estimation_summary(
            command,
            data.segment,
            data.station,
            data.meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    return estimation, operators, index


def build_and_solve_dense_no_meshes(command, assembly, operators, data):
    # NOTE: Used in celeri_solve.py
    logger.info("build_and_solve_dense_no_meshes")

    # Get all elastic operators for segments and TDEs
    get_elastic_operators(operators, data.meshes, data.segment, data.station, command)

    operators.rotation_to_velocities = get_rotation_to_velocities_partials(
        data.station, data.block.shape[0]
    )
    operators.global_float_block_rotation = get_global_float_block_rotation_partials(
        data.station
    )
    assembly, operators.block_motion_constraints = get_block_motion_constraints(
        assembly, data.block, command
    )
    assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
        assembly, data.segment, data.block, command
    )
    operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
        data.segment, data.block
    )
    (
        operators.block_strain_rate_to_velocities,
        strain_rate_block_index,
    ) = get_block_strain_rate_to_velocities_partials(
        data.block, data.station, data.segment
    )
    operators.mogi_to_velocities = get_mogi_to_velocities_partials(
        data.mogi, data.station, command
    )

    # Blocks only operator
    index = get_index_no_meshes(assembly, data.station, data.block)

    # TODO: Clean up!
    logger.error(operators.keys())

    operator_block_only = get_full_dense_operator_block_only(operators, index)
    # weighting_vector = get_weighting_vector(command, data.station, data.meshes, index)
    weighting_vector = get_weighting_vector_no_meshes(command, data.station, index)
    data_vector = get_data_vector(assembly, index)
    weighting_vector_block_only = weighting_vector[0 : operator_block_only.shape[0]]

    # Solve the overdetermined linear system using only a weighting vector rather than matrix
    estimation = addict.Dict()
    estimation.operator = operator_block_only
    estimation.weighting_vector = weighting_vector_block_only

    estimation.state_covariance_matrix = np.linalg.inv(
        operator_block_only.T * weighting_vector_block_only @ operator_block_only
    )
    estimation.state_vector = (
        estimation.state_covariance_matrix
        @ operator_block_only.T
        * weighting_vector_block_only
        @ data_vector[0 : weighting_vector_block_only.size]
    )

    # Post-processing

    estimation.predictions = estimation.operator @ estimation.state_vector
    estimation.vel = estimation.predictions[0 : 2 * index.n_stations]
    estimation.east_vel = estimation.vel[0::2]
    estimation.north_vel = estimation.vel[1::2]

    # Estimate slip rate uncertainties
    estimation.slip_rate_sigma = np.sqrt(
        np.diag(
            operators.rotation_to_slip_rate
            @ estimation.state_covariance_matrix[
                0 : 3 * index.n_blocks, 0 : 3 * index.n_blocks
            ]
            @ operators.rotation_to_slip_rate.T
        )
    )  # I don't think this is correct because for the case when there is a rotation vector a priori
    estimation.strike_slip_rate_sigma = estimation.slip_rate_sigma[0::3]
    estimation.dip_slip_rate_sigma = estimation.slip_rate_sigma[1::3]
    estimation.tensile_slip_rate_sigma = estimation.slip_rate_sigma[2::3]

    # Calculate mean squared residual velocity
    estimation.east_vel_residual = estimation.east_vel - data.station.east_vel
    estimation.north_vel_residual = estimation.north_vel - data.station.north_vel

    # Extract segment slip rates from state vector
    estimation.slip_rates = (
        operators.rotation_to_slip_rate
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.strike_slip_rates = estimation.slip_rates[0::3]
    estimation.dip_slip_rates = estimation.slip_rates[1::3]
    estimation.tensile_slip_rates = estimation.slip_rates[2::3]

    # Calculate rotation only velocities
    estimation.vel_rotation = (
        operators.rotation_to_velocities[index.station_row_keep_index, :]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_rotation = estimation.vel_rotation[0::2]
    estimation.north_vel_rotation = estimation.vel_rotation[1::2]

    # Calculate fully locked segment velocities
    estimation.vel_elastic_segment = (
        operators.rotation_to_slip_rate_to_okada_to_velocities[
            index.station_row_keep_index, :
        ]
        @ estimation.state_vector[0 : 3 * index.n_blocks]
    )
    estimation.east_vel_elastic_segment = estimation.vel_elastic_segment[0::2]
    estimation.north_vel_elastic_segment = estimation.vel_elastic_segment[1::2]

    # TODO: Calculate block strain rate velocities
    estimation.east_vel_block_strain_rate = np.zeros(len(data.station))
    estimation.north_vel_block_strain_rate = np.zeros(len(data.station))

    # # Get all elastic operators for segments and TDEs
    # get_elastic_operators(operators, data.meshes, data.segment, data.station, command)

    # # Get TDE smoothing operators
    # get_all_mesh_smoothing_matrices(data.meshes, operators)

    # # Get non-elastic operators
    # operators.rotation_to_velocities = get_rotation_to_velocities_partials(data.station, data.block.shape[0])
    # operators.global_float_block_rotation = get_global_float_block_rotation_partials(
    #     data.station
    # )
    # assembly, operators.block_motion_constraints = get_block_motion_constraints(
    #     assembly, data.block, command
    # )
    # assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
    #     assembly, data.segment, data.block, command
    # )
    # operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
    #     data.segment, data.block
    # )
    # (
    #     operators.block_strain_rate_to_velocities,
    #     strain_rate_block_index,
    # ) = get_block_strain_rate_to_velocities_partials(
    #     data.block, data.station, data.segment
    # )
    # operators.mogi_to_velocities = get_mogi_to_velocities_partials(
    #     data.mogi, data.station, command
    # )
    # get_tde_slip_rate_constraints(data.meshes, operators)

    # # Direct solve dense linear system
    # logger.info("Start: Dense assemble and solve")
    # start_solve_time = timeit.default_timer()
    # index, estimation = assemble_and_solve_dense(
    #     command, assembly, operators, data.station, data.block, data.meshes
    # )
    # end_solve_time = timeit.default_timer()
    # logger.success(
    #     f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    # )

    # post_process_estimation(estimation, operators, data.station, index)

    write_output(
        command, estimation, data.station, data.segment, data.block, data.meshes
    )

    if bool(command.plot_estimation_summary):
        plot_estimation_summary(
            command,
            data.segment,
            data.station,
            data.meshes,
            estimation,
            lon_range=command.lon_range,
            lat_range=command.lat_range,
            quiver_scale=command.quiver_scale,
        )

    return estimation, operators, index


def build_and_solve_qp_kl(command, assembly, operators, data):
    # NOTE: Used in celeri_solve.py
    logger.info("build_and_solve_qp_kl")
    logger.info("PLACEHOLDER")

    # # Get all elastic operators for segments and TDEs
    # get_elastic_operators(operators, data.meshes, data.segment, data.station, command)

    # # Get TDE smoothing operators
    # get_all_mesh_smoothing_matrices(data.meshes, operators)

    # # Get non-elastic operators
    # operators.rotation_to_velocities = get_rotation_to_velocities_partials(
    #     data.station, data.block.shape[0]
    # )
    # operators.global_float_block_rotation = get_global_float_block_rotation_partials(
    #     data.station
    # )
    # assembly, operators.block_motion_constraints = get_block_motion_constraints(
    #     assembly, data.block, command
    # )
    # assembly, operators.slip_rate_constraints = get_slip_rate_constraints(
    #     assembly, data.segment, data.block, command
    # )
    # operators.rotation_to_slip_rate = get_rotation_to_slip_rate_partials(
    #     data.segment, data.block
    # )
    # (
    #     operators.block_strain_rate_to_velocities,
    #     strain_rate_block_index,
    # ) = get_block_strain_rate_to_velocities_partials(
    #     data.block, data.station, data.segment
    # )
    # operators.mogi_to_velocities = get_mogi_to_velocities_partials(
    #     data.mogi, data.station, command
    # )
    # get_tde_slip_rate_constraints(data.meshes, operators)

    # # Direct solve dense linear system
    # logger.info("Start: Dense assemble and solve")
    # start_solve_time = timeit.default_timer()
    # index, estimation = assemble_and_solve_dense(
    #     command, assembly, operators, data.station, data.block, data.meshes
    # )
    # end_solve_time = timeit.default_timer()
    # logger.success(
    #     f"Finish: Dense assemble and solve: {end_solve_time - start_solve_time:0.2f} seconds for solve"
    # )

    # post_process_estimation(estimation, operators, data.station, index)

    # write_output(
    #     command, estimation, data.station, data.segment, data.block, data.meshes
    # )

    # if bool(command.plot_estimation_summary):
    #     plot_estimation_summary(
    #         command,
    #         data.segment,
    #         data.station,
    #         data.meshes,
    #         estimation,
    #         lon_range=command.lon_range,
    #         lat_range=command.lat_range,
    #         quiver_scale=command.quiver_scale,
    #     )

    # return estimation, operators, index


#########################################################################################
#                                                                                       #
#                                                                                       #
#    ,ad8888ba,    88        88  888888888888  88888888ba   88        88  888888888888  #
#   d8"'    `"8b   88        88       88       88      "8b  88        88       88       #
#  d8'        `8b  88        88       88       88      ,8P  88        88       88       #
#  88          88  88        88       88       88aaaaaa8P'  88        88       88       #
#  88          88  88        88       88       88""""""'    88        88       88       #
#  Y8,        ,8P  88        88       88       88           88        88       88       #
#   Y8a.    .a8P   Y8a.    .a8P       88       88           Y8a.    .a8P       88       #
#    `"Y8888Y"'     `"Y8888Y"'        88       88            `"Y8888Y"'        88       #
#                                                                                       #
#                                                                                       #
#########################################################################################
@pytest.mark.skip(reason="Writing output to disk")
def create_output_folder(command: Dict):
    # Check to see if "runs" folder exists and if not create it
    if not os.path.exists(command.base_runs_folder):
        os.mkdir(command.base_runs_folder)

    # Make output folder for current run
    os.mkdir(command.output_path)


@pytest.mark.skip(reason="Writing output files")
def write_output(
    command: Dict,
    estimation: Dict,
    station: pd.DataFrame,
    segment: pd.DataFrame,
    block: pd.DataFrame,
    meshes: Dict,
    mogi=[],
):
    # Add model velocities to station dataframe and write .csv
    station["model_east_vel"] = estimation.east_vel
    station["model_north_vel"] = estimation.north_vel
    station["model_east_vel_residual"] = estimation.east_vel_residual
    station["model_north_vel_residual"] = estimation.north_vel_residual
    station["model_east_vel_rotation"] = estimation.east_vel_rotation
    station["model_north_vel_rotation"] = estimation.north_vel_rotation
    station["model_east_elastic_segment"] = estimation.east_vel_elastic_segment
    station["model_north_elastic_segment"] = estimation.north_vel_elastic_segment
    if command.solve_type != "dense_no_meshes":
        station["model_east_vel_tde"] = estimation.east_vel_tde
        station["model_north_vel_tde"] = estimation.north_vel_tde
    station["model_east_vel_block_strain_rate"] = estimation.east_vel_block_strain_rate
    station["model_north_vel_block_strain_rate"] = (
        estimation.north_vel_block_strain_rate
    )
    station["model_east_vel_mogi"] = estimation.east_vel_mogi
    station["model_north_vel_mogi"] = estimation.north_vel_mogi
    station_output_file_name = command.output_path + "/" + "model_station.csv"
    station.to_csv(station_output_file_name, index=False, float_format="%0.4f")

    # Add estimated slip rates to segment dataframe and write .csv
    segment["model_strike_slip_rate"] = estimation.strike_slip_rates
    segment["model_dip_slip_rate"] = estimation.dip_slip_rates
    segment["model_tensile_slip_rate"] = estimation.tensile_slip_rates
    segment["model_strike_slip_rate_uncertainty"] = estimation.strike_slip_rate_sigma
    segment["model_dip_slip_rate_uncertainty"] = estimation.strike_slip_rate_sigma
    segment["model_tensile_slip_rate_uncertainty"] = estimation.strike_slip_rate_sigma
    segment_output_file_name = command.output_path + "/" + "model_segment.csv"
    segment.to_csv(segment_output_file_name, index=False, float_format="%0.4f")

    # TODO: Add rotation rates and block strain rate block dataframe and write .csv
    # block["euler_lon"] =
    # block["euler_lon_sig"] =
    # block["euler_lat"] =
    # block["euler_lat_sig"] =
    # block["rotation_rate"]=
    # block["rotation_rate_sig"]=
    # block["rotation_flag"]=
    # block["apriori_flag"]=

    # Add volume change rates to Mogi source dataframe
    # mogi["volume_change"] = estimation.mogi_volume_change_rates
    # mogi["volume_change_sig"] = estimation.mogi_volume_change_rates

    # Construct mesh geometry dataframe
    if command.solve_type != "dense_no_meshes":
        mesh_outputs = pd.DataFrame()
        for i in range(len(meshes)):
            this_mesh_output = {
                "lon1": meshes[i].lon1,
                "lat1": meshes[i].lat1,
                "dep1": meshes[i].dep1,
                "lon2": meshes[i].lon2,
                "lat2": meshes[i].lat2,
                "dep2": meshes[i].dep2,
                "lon3": meshes[i].lon3,
                "lat3": meshes[i].lat3,
                "dep3": meshes[i].dep3,
            }
            this_mesh_output = pd.DataFrame(this_mesh_output)
            # mesh_outputs = mesh_outputs.append(this_mesh_output)
            mesh_outputs = pd.concat([mesh_outputs, this_mesh_output])

        # Append slip rates
        mesh_outputs["strike_slip_rate"] = estimation.tde_strike_slip_rates
        mesh_outputs["dip_slip_rate"] = estimation.tde_dip_slip_rates

        # Write to CSV
        mesh_output_file_name = command.output_path + "/" + "model_meshes.csv"
        mesh_outputs.to_csv(mesh_output_file_name, index=False)

        # Write a lot to a single hdf file
        hdf_output_file_name = command.output_path + "/" + f"model_{command.run_name}.hdf5"
        with h5py.File(hdf_output_file_name, "w") as hdf:
            # Meta data
            hdf.create_dataset("run_name", data=command.run_name.encode("utf-8"), dtype=h5py.string_dtype(encoding="utf-8"))
            hdf.create_dataset("earth_radius", data=6371.0)

            # Write command dictionary
            grp = hdf.create_group("command")
            for key, value in command.items():
                if isinstance(value, str):
                    # Handle strings specially
                    grp.create_dataset(key, data=value.encode("utf-8"), 
                                    dtype=h5py.string_dtype(encoding="utf-8"))
                else:
                    # Handle numeric values
                    grp.create_dataset(key, data=value)


            # Write meshes
            for i in range(len(meshes)):
                grp = hdf.create_group(f"meshes/mesh_{i:05}")
                mesh_name = os.path.splitext(os.path.basename(meshes[i].file_name))[0]
                grp.create_dataset("mesh_name", data=mesh_name.encode("utf-8"), dtype=h5py.string_dtype(encoding="utf-8"))
                # grp.create_dataset("n_time_steps", data=1)
                
                # Write mesh geometry
                grp.create_dataset(f"coordinates", data=meshes[i].points)
                grp.create_dataset(f"verts", data=meshes[i].verts)

                # Calculate and write Cartesian mesh geometry
                # meshes[i].points_cartesian = np.zeros(meshes[i].points.shape)
                # meshes[i].points_cartesian[:, 0], meshes[i].points_cartesian[:, 1], meshes[i].points_cartesian[:, 2] = sph2cart(
                #         meshes[i].points[:, 0],
                #         meshes[i].points[:, 1],
                #         RADIUS_EARTH + KM2M * meshes[i].points[:, 2],
                #     )
                # grp.create_dataset(f"points_cartesian", data=meshes[i].points_cartesian)

                # Write mesh scalars (we'll add more later)
                if i == 0:
                    mesh_start_idx = 0
                    mesh_end_idx = meshes[i].n_tde
                else:
                    mesh_start_idx = mesh_end_idx
                    mesh_end_idx = mesh_start_idx + meshes[i].n_tde
                grp.create_dataset(
                    f"strike_slip",
                    data=estimation.tde_strike_slip_rates[mesh_start_idx:mesh_end_idx],
                )
                grp.create_dataset(
                    f"dip_slip",
                    data=estimation.tde_dip_slip_rates[mesh_start_idx:mesh_end_idx],
                )
                grp.create_dataset(
                    f"tensile_slip",
                    data=np.zeros_like(
                        estimation.tde_dip_slip_rates[mesh_start_idx:mesh_end_idx]
                    ),
                )

            # Save segment information
            segment_no_name = segment.drop("name", axis=1)
            # Store the segment data
            hdf.create_dataset("segment", data=segment_no_name.to_numpy())
            # Store the segment column as a separate dataset
            string_dtype = h5py.string_dtype(encoding="utf-8")  # Variable-length UTF-8 strings
            hdf.create_dataset(
                "segment_names", data=segment["name"].to_numpy(dtype=object), dtype=string_dtype
            )
            # Store the column names as attributes
            hdf.attrs["columns"] = np.array(segment_no_name.columns, dtype=h5py.string_dtype())
            # Store the index as an attribute
            hdf.attrs["index"] = segment_no_name.index.to_numpy()

            # Save station information
            station_no_name = station.drop("name", axis=1)
            # Store the station data
            hdf.create_dataset("station", data=station_no_name.to_numpy())
            # Store the segment column as a separate dataset
            string_dtype = h5py.string_dtype(encoding="utf-8")  # Variable-length UTF-8 strings
            hdf.create_dataset(
                "station_names", data=station["name"].to_numpy(dtype=object), dtype=string_dtype
            )
            # Store the column names as attributes
            hdf.attrs["columns"] = np.array(station_no_name.columns, dtype=h5py.string_dtype())
            # Store the index as an attribute
            hdf.attrs["index"] = station_no_name.index.to_numpy()

    # Write the command dict to an a json file
    args_command_output_file_name = (
        command.output_path + "/args_" + os.path.basename(command.file_name)
    )
    with open(args_command_output_file_name, "w") as f:
        json.dump(command, f, indent=4)

    # Write all major variables to .pkl file in output folder
    if bool(command.pickle_save):
        with open(os.path.join(command.output_path, "output" + ".pkl"), "wb") as f:
            pickle.dump([command, estimation, station, segment, block, meshes], f)


def write_output_supplemental(
    args, command, index, data, operators, estimation, assembly
):
    # Copy all input files to output folder
    file_names = [
        "segment_file_name",
        "station_file_name",
        "block_file_name",
        "mesh_parameters_file_name",
        "los_file_name",
        "file_name",
    ]
    for file_name in file_names:
        try:
            shutil.copyfile(
                command[file_name],
                os.path.join(
                    command.output_path,
                    os.path.basename(os.path.normpath(command[file_name])),
                ),
            )
        except:
            logger.warning(f"No {file_name} to copy to output folder")

    # Copy .msh files to output foler
    if len(data.meshes) > 0:
        for i in range(len(data.meshes)):
            msh_file_name = data.meshes[i].file_name
            try:
                shutil.copyfile(
                    msh_file_name,
                    os.path.join(
                        command.output_path,
                        os.path.basename(os.path.normpath(msh_file_name)),
                    ),
                )
            except:
                logger.warning(f"No {msh_file_name} to copy to output folder")

    # Write command line arguments to output folder
    with open(
        os.path.join(command.output_path, command.run_name + "_args.json"), "w"
    ) as f:
        json.dump(args, f, indent=2)

    # Write all major variables to .pkl file in output folder
    if bool(command.pickle_save):
        with open(
            os.path.join(command.output_path, command.run_name + ".pkl"), "wb"
        ) as f:
            pickle.dump([command, index, data, operators, estimation, assembly], f)


########################################################
#                                                      #
#                                                      #
#  88888888ba   88           ,ad8888ba,  888888888888  #
#  88      "8b  88          d8"'    `"8b      88       #
#  88      ,8P  88         d8'        `8b     88       #
#  88aaaaaa8P'  88         88          88     88       #
#  88""""""'    88         88          88     88       #
#  88           88         Y8,        ,8P     88       #
#  88           88          Y8a.    .a8P      88       #
#  88           88888888888  `"Y8888Y"'       88       #
#                                                      #
#                                                      #
########################################################


def plot_block_labels(segment, block, station, closure):
    plt.figure()
    plt.title("West and east labels")
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )

    for i in range(len(segment)):
        plt.text(
            segment.mid_lon_plate_carree.values[i],
            segment.mid_lat_plate_carree.values[i],
            str(segment["west_labels"][i]) + "," + str(segment["east_labels"][i]),
            fontsize=8,
            color="m",
            horizontalalignment="center",
            verticalalignment="center",
        )

    for i in range(len(station)):
        plt.text(
            station.lon.values[i],
            station.lat.values[i],
            str(station.block_label[i]),
            fontsize=8,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )

    for i in range(len(block)):
        plt.text(
            block.interior_lon.values[i],
            block.interior_lat.values[i],
            str(block.block_label[i]),
            fontsize=8,
            color="g",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.gca().set_aspect("equal")
    plt.show()


def plot_input_summary(
    command: Dict,
    segment: pd.DataFrame,
    station: pd.DataFrame,
    block: pd.DataFrame,
    meshes: List,
    mogi: pd.DataFrame,
    sar: pd.DataFrame,
    lon_range: Tuple,
    lat_range: Tuple,
    quiver_scale: float,
):
    """Plot overview figures showing observed and modeled velocities as well
    as velocity decomposition and estimates slip rates.

    Args:
        segment (pd.DataFrame): Fault segments
        station (pd.DataFrame): GPS observations
        block (pd.DataFrame): Block interior points and priors
        meshes (List): Mesh geometries and properties
        mogi (pd.DataFrame): Mogi sources
        sar (pd.DataFrame): SAR observations
        lon_range (Tuple): Latitude range (min, max)
        lat_range (Tuple): Latitude range (min, max)
        quiver_scale (float): Scaling for velocity arrows
    """

    def common_plot_elements(segment: pd.DataFrame, lon_range: Tuple, lat_range: Tuple):
        """Elements common to all subplots

        Args:
            segment (pd.DataFrame): Fault segments
            lon_range (Tuple): Longitude range (min, max)
            lat_range (Tuple): Latitude range (min, max)
        """
        for i in range(len(segment)):
            if segment.dip[i] == 90.0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-k",
                    linewidth=0.5,
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=0.5,
                )

        plt.xlim([lon_range[0], lon_range[1]])
        plt.ylim([lat_range[0], lat_range[1]])
        plt.gca().set_aspect("equal", adjustable="box")

    max_sigma_cutoff = 99.0
    n_subplot_rows = 4
    n_subplot_cols = 3
    subplot_index = 0

    plt.figure(figsize=(12, 16))

    subplot_index += 1
    ax1 = plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
    plt.title("observed velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        station.east_vel,
        station.north_vel,
        scale=quiver_scale,
        scale_units="inches",
        color="red",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("LOS")
    common_plot_elements(segment, lon_range, lat_range)

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("blocks\nprior rotation")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(block)):
        if block.rotation_flag.values[i] > 0:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "r+")
            plt.text(
                block.interior_lon[i],
                block.interior_lat[i],
                f"lon = {block.euler_lon[i]:.3f}\nlat = {block.euler_lat[i]:.3f}\nrate = {block.rotation_rate[i]:.3f}",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "bx")

    # Plot blocks that can have interior strain
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("blocks\ninterior strain allowed")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(block)):
        if block.strain_rate_flag[i] > 0:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "r+")
            plt.text(
                block.interior_lon[i],
                block.interior_lat[i],
                f"interior\nstrain allowed",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.plot(block.interior_lon[i], block.interior_lat[i], "bx")

    # Plot mogi sources
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("Mogi sources")
    common_plot_elements(segment, lon_range, lat_range)
    if len(mogi.lon > 0):
        plt.plot(mogi.lon, mogi.lat, "r+")

    # Skip a subplot
    subplot_index += 1

    # Plot a priori slip rate constraints
    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("strike-slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if segment.ss_rate_flag[i] == 1:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{segment.ss_rate[i]:.1f}({segment.ss_rate_sig[i]:.1f})",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("dip-slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if segment.ds_rate_flag[i] == 1:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{segment.ds_rate[i]:.1f}({segment.ds_rate_sig[i]:.1f})",
                color="blue",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("tensile-slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if segment.ts_rate_flag[i] == 1:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{segment.ts_rate[i]:.1f}({segment.ts_rate_sig[i]:.1f})",
                color="green",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("slip rate constraints")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(meshes)):
        is_constrained_tde = np.zeros(meshes[i].n_tde)
        is_constrained_tde[meshes[i].top_elements] = meshes[i].top_slip_rate_constraint
        is_constrained_tde[meshes[i].bot_elements] = meshes[i].bot_slip_rate_constraint
        is_constrained_tde[meshes[i].side_elements] = meshes[
            i
        ].side_slip_rate_constraint
        is_constrained_tde[meshes[i].ss_slip_constraint_idx] = 3
        is_constrained_tde[meshes[i].ds_slip_constraint_idx] = 3
        is_constrained_tde[meshes[i].coupling_constraint_idx] = 2
        x_coords = meshes[i].points[:, 0]
        y_coords = meshes[i].points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)
        ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", linewidth=0.25, cmap="Oranges"
        )
        pc.set_array(is_constrained_tde)
        ax.add_collection(pc)
        # ax.autoscale()

    plt.suptitle("inputs")
    plt.show(block=False)
    plt.savefig(command.output_path + "/" + "plot_input_summary.png", dpi=300)
    plt.savefig(command.output_path + "/" + "plot_input_summary.pdf")
    logger.success(
        "Wrote figures" + command.output_path + "/" + "plot_input_summary.(pdf, png)"
    )


def plot_estimation_summary(
    command: Dict,
    segment: pd.DataFrame,
    station: pd.DataFrame,
    meshes: List,
    estimation: Dict,
    lon_range: Tuple,
    lat_range: Tuple,
    quiver_scale: float,
):
    """Plot overview figures showing observed and modeled velocities as well
    as velocity decomposition and estimates slip rates.

    Args:
        segment (pd.DataFrame): Fault segments
        station (pd.DataFrame): GPS observations
        meshes (List): List of mesh dictionaries
        estimation (Dict): All estimated values
        lon_range (Tuple): Latitude range (min, max)
        lat_range (Tuple): Latitude range (min, max)
        quiver_scale (float): Scaling for velocity arrows
    """

    def common_plot_elements(segment: pd.DataFrame, lon_range: Tuple, lat_range: Tuple):
        """Elements common to all subplots
        Args:
            segment (pd.DataFrame): Fault segments
            lon_range (Tuple): Longitude range (min, max)
            lat_range (Tuple): Latitude range (min, max)
        """
        for i in range(len(segment)):
            if segment.dip[i] == 90.0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-k",
                    linewidth=0.5,
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=0.5,
                )
        plt.xlim([lon_range[0], lon_range[1]])
        plt.ylim([lat_range[0], lat_range[1]])
        plt.gca().set_aspect("equal", adjustable="box")

    max_sigma_cutoff = 99.0
    n_subplot_rows = 4
    n_subplot_cols = 3
    subplot_index = 0

    plt.figure(figsize=(12, 16))
    subplot_index += 1
    ax1 = plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
    plt.title("observed velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        station.east_vel,
        station.north_vel,
        scale=quiver_scale,
        scale_units="inches",
        color="red",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("model velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel,
        estimation.north_vel,
        scale=quiver_scale,
        scale_units="inches",
        color="blue",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("residual velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_residual,
        estimation.north_vel_residual,
        scale=quiver_scale,
        scale_units="inches",
        color="green",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("rotation velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_rotation,
        estimation.north_vel_rotation,
        scale=quiver_scale,
        scale_units="inches",
        color="orange",
    )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("elastic segment velocities")
    common_plot_elements(segment, lon_range, lat_range)
    plt.quiver(
        station.lon,
        station.lat,
        estimation.east_vel_elastic_segment,
        estimation.north_vel_elastic_segment,
        scale=quiver_scale,
        scale_units="inches",
        color="magenta",
    )

    if command.solve_type != "dense_no_meshes":
        if len(meshes) > 0:
            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("elastic tde velocities")
            common_plot_elements(segment, lon_range, lat_range)
            plt.quiver(
                station.lon,
                station.lat,
                estimation.east_vel_tde,
                estimation.north_vel_tde,
                scale=quiver_scale,
                scale_units="inches",
                color="black",
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("segment strike-slip \n (negative right-lateral)")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if estimation.strike_slip_rate_sigma[i] < max_sigma_cutoff:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.strike_slip_rates[i]:.1f}({estimation.strike_slip_rate_sigma[i]:.1f})",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.strike_slip_rates[i]:.1f}(*)",
                color="red",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("segment dip-slip \n (positive convergences)")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if estimation.dip_slip_rate_sigma[i] < max_sigma_cutoff:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.dip_slip_rates[i]:.1f}({estimation.dip_slip_rate_sigma[i]:.1f})",
                color="blue",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.dip_slip_rates[i]:.1f}(*)",
                color="blue",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1)
    plt.title("segment tensile-slip \n (negative convergences)")
    common_plot_elements(segment, lon_range, lat_range)
    for i in range(len(segment)):
        if estimation.tensile_slip_rate_sigma[i] < max_sigma_cutoff:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.tensile_slip_rates[i]:.1f}({estimation.tensile_slip_rate_sigma[i]:.1f})",
                color="green",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )
        else:
            plt.text(
                segment.mid_lon_plate_carree[i],
                segment.mid_lat_plate_carree[i],
                f"{estimation.tensile_slip_rates[i]:.1f}(*)",
                color="green",
                clip_on=True,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=7,
            )

    if command.solve_type != "dense_no_meshes":
        if len(meshes) > 0:
            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("TDE slip (strike-slip)")
            common_plot_elements(segment, lon_range, lat_range)
            # plot_meshes(meshes, estimation.tde_strike_slip_rates, plt.gca())
            fill_value = estimation.tde_strike_slip_rates
            fill_value_range = [np.min(fill_value), np.max(fill_value)]
            ax = plt.gca()
            for i in range(len(meshes)):
                x_coords = meshes[i].points[:, 0]
                y_coords = meshes[i].points[:, 1]
                vertex_array = np.asarray(meshes[i].verts)

                xy = np.c_[x_coords, y_coords]
                verts = xy[vertex_array]
                pc = matplotlib.collections.PolyCollection(
                    verts, edgecolor="none", cmap="rainbow"
                )
                if i == 0:
                    tde_slip_component_start = 0
                    tde_slip_component_end = meshes[i].n_tde
                else:
                    tde_slip_component_start = tde_slip_component_end
                    tde_slip_component_end = tde_slip_component_start + meshes[i].n_tde
                pc.set_array(
                    fill_value[tde_slip_component_start:tde_slip_component_end]
                )
                pc.set_clim(fill_value_range)
                ax.add_collection(pc)
                # ax.autoscale()
                if i == len(meshes) - 1:
                    plt.colorbar(pc, label="slip (mm/yr)")

                # Add mesh edge
                x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
                y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
                x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
                y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
                plt.plot(x_edge, y_edge, color="black", linewidth=1)

            subplot_index += 1
            plt.subplot(
                n_subplot_rows, n_subplot_cols, subplot_index, sharex=ax1, sharey=ax1
            )
            plt.title("TDE slip (dip-slip)")
            common_plot_elements(segment, lon_range, lat_range)
            # plot_meshes(meshes, estimation.tde_dip_slip_rates, plt.gca())
            fill_value = estimation.tde_dip_slip_rates
            fill_value_range = [np.min(fill_value), np.max(fill_value)]
            ax = plt.gca()
            for i in range(len(meshes)):
                x_coords = meshes[i].points[:, 0]
                y_coords = meshes[i].points[:, 1]
                vertex_array = np.asarray(meshes[i].verts)

                xy = np.c_[x_coords, y_coords]
                verts = xy[vertex_array]
                pc = matplotlib.collections.PolyCollection(
                    verts, edgecolor="none", cmap="rainbow"
                )
                if i == 0:
                    tde_slip_component_start = 0
                    tde_slip_component_end = meshes[i].n_tde
                else:
                    tde_slip_component_start = tde_slip_component_end
                    tde_slip_component_end = tde_slip_component_start + meshes[i].n_tde
                pc.set_array(
                    fill_value[tde_slip_component_start:tde_slip_component_end]
                )
                pc.set_clim(fill_value_range)
                ax.add_collection(pc)
                # ax.autoscale()
                if i == len(meshes) - 1:
                    plt.colorbar(pc, label="slip (mm/yr)")

                # Add mesh edge
                x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
                y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
                x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
                y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
                plt.plot(x_edge, y_edge, color="black", linewidth=1)

    subplot_index += 1
    plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
    plt.title("Residual velocity histogram")
    residual_velocity_vector = np.concatenate(
        (estimation.east_vel_residual.values, estimation.north_vel_residual.values)
    )
    mean_average_error = np.mean(np.abs(residual_velocity_vector))
    mean_squared_error = (
        np.sum(residual_velocity_vector**2.0) / residual_velocity_vector.size
    )

    # Create histogram of residual velocities
    plt.hist(residual_velocity_vector, 50)
    plt.xlabel("residual velocity (mm/yr)")
    plt.ylabel("N")
    plt.title(
        f"mae = {mean_average_error:.2f} (mm/yr), mse = {mean_squared_error:.2f} (mm/yr)^2"
    )

    plt.show(block=False)
    plt.savefig(command.output_path + "/" + "plot_estimation_summary.png", dpi=300)
    plt.savefig(command.output_path + "/" + "plot_estimation_summary.pdf")
    logger.success(
        "Wrote figures"
        + command.output_path
        + "/"
        + "plot_estimation_summary.(pdf, png)"
    )


def plot_matrix_abs_log(matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log10(np.abs(matrix + EPS)), cmap="plasma_r")
    plt.colorbar()
    plt.show()


def plot_meshes(meshes: List, fill_value: np.array, ax):
    for i in range(len(meshes)):
        x_coords = meshes[i].points[:, 0]
        y_coords = meshes[i].points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)

        if not ax:
            ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", cmap="rainbow"
        )
        if i == 0:
            fill_start = 0
            fill_end = meshes[i].n_tde
        else:
            fill_start = fill_end
            fill_end = fill_start + meshes[i].n_tde
        pc.set_array(fill_value[fill_start:fill_end])
        ax.add_collection(pc)
        ax.autoscale()
        plt.colorbar(pc, label="slip (mm/yr)")

        # Add mesh edge
        x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
        y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
        plt.plot(x_edge, y_edge, color="black", linewidth=1)


def plot_segment_displacements(
    segment,
    station,
    command,
    segment_idx,
    strike_slip,
    dip_slip,
    tensile_slip,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    quiver_scale,
):
    u_east, u_north, u_up = get_okada_displacements(
        segment.lon1.values[segment_idx],
        segment.lat1[segment_idx],
        segment.lon2[segment_idx],
        segment.lat2[segment_idx],
        segment.locking_depth[segment_idx],
        segment.burial_depth[segment_idx],
        segment.dip[segment_idx],
        segment.azimuth[segment_idx],
        command.material_lambda,
        command.material_mu,
        strike_slip,
        dip_slip,
        tensile_slip,
        station.lon,
        station.lat,
    )
    plt.figure()
    plt.plot(
        [segment.lon1[segment_idx], segment.lon2[segment_idx]],
        [segment.lat1[segment_idx], segment.lat2[segment_idx]],
        "-r",
    )
    uplimit = np.max(np.abs(u_up))
    plt.scatter(
        station.lon, station.lat, c=u_up, s=10, cmap="bwr", vmin=-uplimit, vmax=uplimit
    )
    plt.quiver(
        station.lon,
        station.lat,
        u_east,
        u_north,
        scale=quiver_scale,
        scale_units="inches",
    )

    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Okada displacements: longitude and latitude")
    plt.show()


def plot_strain_rate_components_for_block(closure, segment, station, block_idx):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    vel_east, vel_north, vel_up = get_strain_rate_displacements(
        station,
        segment,
        block_idx=block_idx,
        strain_rate_lon_lon=1,
        strain_rate_lat_lat=0,
        strain_rate_lon_lat=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 2)
    vel_east, vel_north, vel_up = get_strain_rate_displacements(
        station,
        segment,
        block_idx=block_idx,
        strain_rate_lon_lon=0,
        strain_rate_lat_lat=1,
        strain_rate_lon_lat=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 3)
    vel_east, vel_north, vel_up = get_strain_rate_displacements(
        station,
        segment,
        block_idx=block_idx,
        strain_rate_lon_lon=0,
        strain_rate_lat_lat=0,
        strain_rate_lon_lat=1,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )
    plt.show()


def plot_rotation_components(closure, station):
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    vel_east, vel_north, vel_up = get_rotation_displacements(
        station.lon.values,
        station.lat.values,
        omega_x=1,
        omega_y=0,
        omega_z=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 2)
    vel_east, vel_north, vel_up = get_rotation_displacements(
        station.lon.values,
        station.lat.values,
        omega_x=0,
        omega_y=1,
        omega_z=0,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )

    plt.subplot(1, 3, 3)
    vel_east, vel_north, vel_up = get_rotation_displacements(
        station.lon.values,
        station.lat.values,
        omega_x=0,
        omega_y=0,
        omega_z=1,
    )
    for i in range(closure.n_polygons()):
        plt.plot(
            closure.polygons[i].vertices[:, 0],
            closure.polygons[i].vertices[:, 1],
            "k-",
            linewidth=0.5,
        )
    plt.quiver(
        station.lon,
        station.lat,
        vel_east,
        vel_north,
        scale=1e7,
        scale_units="inches",
        color="r",
    )
    plt.show()


def get_default_plotting_dict(command, estimation, station):
    """
    Parameters:
    -----------
    estimation : dictionary
    station : dataframe
    command : dictionary

    Returns:
    --------
    p : dictionary

    The returned dictionary includes the following keys and their default values:
        - WORLD_BOUNDARIES: Data loaded from "WorldHiVectors.mat".
        - FIGSIZE_VECTORS: (12, 6) - Default figure size for vector plots.
        - FONTSIZE: 16 - Default font size.
        - LON_RANGE: - Inferred from command.
        - LAT_RANGE: - Inferred from command.
        - LON_TICKS: - Inferred from command.
        - LAT_TICKS: - Inferred from command.
        - SLIP_RATE_MIN: - Inferred from estimation.
        - SLIP_RATE_MAX: - Inferred from estimation.
        - LAND_COLOR: "lightgray" - Color of land areas.
        - LAND_LINEWIDTH: 0.5 - Line width for land boundaries.
        - LAND_ZORDER: 0 - Z-order for land boundaries.
        - KEY_RECTANGLE_ANCHOR: [0, -90] - Anchor point for the key rectangle.
        - KEY_RECTANGLE_WIDTH: 3.0 - Width of the key rectangle.
        - KEY_RECTANGLE_HEIGHT: 1.55 - Height of the key rectangle.
        - KEY_ARROW_LON: 5.0 - Inferred from command.
        - KEY_ARROW_LAT: -85.0 - Inferred from command.
        - KEY_ARROW_MAGNITUDE: - Magnitude for the key arrow.
        - KEY_ARROW_TEXT: - Text for the key arrow.
        - KEY_ARROW_COLOR: "k" - Color for the key arrow.
        - KEY_BACKGROUND_COLOR: "white" - Background color for the key.
        - KEY_LINEWIDTH: 1.0 - Line width for the key.
        - KEY_EDGECOLOR: "k" - Edge color for the key.
        - ARROW_MAGNITUDE_MIN: 0.0 - Minimum arrow magnitude.
        - ARROW_MAGNITUDE_MAX: - Maximum arrow magnitude (inferred from command).
        - ARROW_COLORMAP: cm.plasma - Colormap for arrows.
        - ARROW_SCALE: - Inferred from command.
        - ARROW_WIDTH: 0.0025 - Width for arrows.
        - ARROW_LINEWIDTH: 0.5 - Line width for arrows.
        - ARROW_EDGECOLOR: "k" - Edge color for arrows.
        - SEGMENT_LINE_WIDTH_OUTER: 2.0 - Outer line width for segments.
        - SEGMENT_LINE_WIDTH_INNER: 1.0 - Inner line width for segments.
        - SEGMENT_LINE_COLOR_OUTER: "k" - Outer line color for segments.
        - SEGMENT_LINE_COLOR_INNER: "w" - Inner line color for segments.

    """

    slip_rate_scale = 0.5 * np.max(
        (
            np.abs(estimation.strike_slip_rates),
            np.abs(estimation.dip_slip_rates),
            np.abs(estimation.tensile_slip_rates),
        )
    )

    vel_scale = np.round(
        0.6
        * np.max(
            (
                np.abs(station.east_vel),
                np.abs(station.north_vel),
            )
        )
    )
    vel_scale = round(vel_scale / 5) * 5

    p = addict.Dict()
    p.FIGSIZE_VECTORS = (12, 6)
    p.FONTSIZE = 16
    p.LON_RANGE = command.lon_range
    p.LAT_RANGE = command.lat_range
    p.LON_TICKS = np.linspace(command.lon_range[0], command.lon_range[1], 3)
    p.LAT_TICKS = np.linspace(command.lat_range[0], command.lat_range[1], 3)
    p.SLIP_RATE_MIN = -slip_rate_scale
    p.SLIP_RATE_MAX = slip_rate_scale
    p.LAND_COLOR = "lightgray"
    p.LAND_LINEWIDTH = 0.5
    p.LAND_ZORDER = 0
    p.KEY_RECTANGLE_ANCHOR = [0, -90]
    p.KEY_RECTANGLE_WIDTH = 3.0
    p.KEY_RECTANGLE_HEIGHT = 1.55
    p.KEY_ARROW_LON = np.mean(command.lon_range)
    p.KEY_ARROW_LAT = np.min(command.lat_range) + 0.05 * (
        command.lat_range[1] - command.lat_range[0]
    )
    p.KEY_ARROW_MAGNITUDE = vel_scale
    p.KEY_ARROW_TEXT = f"{vel_scale:d} mm/yr"
    p.KEY_ARROW_COLOR = "k"
    p.KEY_BACKGROUND_COLOR = "white"
    p.KEY_LINEWIDTH = 1.0
    p.KEY_EDGECOLOR = "k"
    p.ARROW_MAGNITUDE_MIN = 0.0
    p.ARROW_MAGNITUDE_MAX = 0.35 * vel_scale
    p.ARROW_COLORMAP = cm.plasma
    p.ARROW_SCALE = vel_scale
    p.ARROW_WIDTH = 0.0025
    p.ARROW_LINEWIDTH = 0.5
    p.ARROW_EDGECOLOR = "k"
    p.SEGMENT_LINE_WIDTH_OUTER = 2.0
    p.SEGMENT_LINE_WIDTH_INNER = 1.0
    p.SEGMENT_LINE_COLOR_OUTER = "k"
    p.SEGMENT_LINE_COLOR_INNER = "w"

    # Read coastlines and trim to within map boundaries

    WORLD_BOUNDARIES = sio.loadmat("WorldHiVectors.mat")

    # Buffer around map frame
    shift = 5
    # Use matplotlib path tool to make a rectangle
    maprect = matplotlib.path.Path(
        [
            (p.LON_RANGE[0] - shift, p.LAT_RANGE[0] - shift),
            (p.LON_RANGE[1] + shift, p.LAT_RANGE[0] - shift),
            (p.LON_RANGE[1] + shift, p.LAT_RANGE[1] + shift),
            (p.LON_RANGE[0] - shift, p.LAT_RANGE[1] + shift),
        ]
    )
    lon = WORLD_BOUNDARIES["lon"]
    lat = WORLD_BOUNDARIES["lat"]
    coastpoints = np.array((lon[:, 0], lat[:, 0])).T
    # Find coast points within rectangle
    coast_idx = maprect.contains_points(coastpoints)
    # Make sure NaNs that separate land bodies are intact
    coast_idx[np.isnan(lon[:, 0])] = True
    # Add coordinates to dict
    p.coastlon = coastpoints[coast_idx, 0]
    p.coastlat = coastpoints[coast_idx, 1]
    return p


def plot_common_elements(p, segment, lon_range, lat_range):
    """
    Plots common map elements such as segments and axis settings.

    This function plots map segments as lines and sets the longitude and latitude
    ranges and ticks for the plot. It also adjusts the aspect ratio and sets labels
    and tick parameters.

    Parameters:
    p (dictionary): Plotting parameters
    segment (DataFrame): A DataFrame containing segment information with columns
                         'lon1', 'lon2', 'lat1', and 'lat2' representing the
                         starting and ending coordinates of each segment.
    lon_range (tuple): A tuple specifying the longitude range as (min_lon, max_lon).
    lat_range (tuple): A tuple specifying the latitude range as (min_lat, max_lat).

    The function performs the following steps:
    1. Plots the outer segment lines in black with a specified line width.
    2. Plots the inner segment lines in white with a specified line width.
    3. Sets the x and y axis limits to the provided longitude and latitude ranges.
    4. Sets the x and y axis ticks to predefined values.
    5. Adjusts the aspect ratio to be equal.
    6. Sets the x and y axis labels and their font sizes.
    7. Sets the tick parameters, including the label size.
    """
    for i in range(len(segment)):
        plt.plot(
            [segment.lon1[i], segment.lon2[i]],
            [segment.lat1[i], segment.lat2[i]],
            "-k",
            linewidth=p.SEGMENT_LINE_WIDTH_OUTER,
        )
    for i in range(len(segment)):
        plt.plot(
            [segment.lon1[i], segment.lon2[i]],
            [segment.lat1[i], segment.lat2[i]],
            "-w",
            linewidth=p.SEGMENT_LINE_WIDTH_INNER,
        )

    plt.xlim([lon_range[0], lon_range[1]])
    plt.ylim([lat_range[0], lat_range[1]])
    plt.xticks(p.LON_TICKS)
    plt.yticks(p.LAT_TICKS)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("longitude (degrees)", fontsize=p.FONTSIZE)
    plt.ylabel("latitude (degrees)", fontsize=p.FONTSIZE)
    plt.tick_params(labelsize=p.FONTSIZE)


def plot_vel_arrows_elements(p, lon, lat, east_velocity, north_velocity, arrow_scale):
    """
    Plots velocity vectors as arrows on a map along with other map elements.

    This function plots velocity vectors given eastward and northward components
    of velocity, scaling the arrows appropriately. It also draws land boundaries,
    a white background rectangle for the key, and an arrow legend.

    Parameters:
    p (dictionary): Plotting parameters
    lon (ndarray): Array of longitudes.
    lat (ndarray): Array latitudes.
    east_velocity (ndarray): Array of eastward velocity components.
    north_velocity (ndarray): Array of northward velocity components.
    arrow_scale (float): Scaling factor for the arrows.

    The function uses global plotting parameters defined in a dictionary `p`,
    which should include keys for arrow properties, land properties, and key
    rectangle and arrow legend properties.

    The function performs the following steps:
    1. Calculates the magnitude of the velocity vectors.
    2. Normalizes the velocity magnitudes for colormap scaling.
    3. Plots the velocity vectors using `plt.quiver`.
    4. Draws land boundaries using `plt.fill`.
    5. Draws a white background rectangle for the key using `mpatches.Rectangle`.
    6. Adds an arrow legend using `plt.quiverkey`.
    7. Sets the aspect ratio of the plot to be equal.
    8. Displays the plot using `plt.show`.

    Minimal example:
    >>> east_velocity = np.array([1.0, 2.0, 3.0])
    >>> north_velocity = np.array([1.0, 2.0, 3.0])
    >>> arrow_scale = 1.0
    >>> plot_vel_arrows_elements(east_velocity, north_velocity, arrow_scale)
    """

    # Draw velocity vectors
    velocity_magnitude = np.sqrt(east_velocity**2.0 + north_velocity**2.0)
    norm = Normalize()
    norm.autoscale(velocity_magnitude)
    norm.vmin = p.ARROW_MAGNITUDE_MIN
    norm.vmax = p.ARROW_MAGNITUDE_MAX
    colormap = p.ARROW_COLORMAP
    quiver_handle = plt.quiver(
        lon,
        lat,
        east_velocity,
        north_velocity,
        scale=p.ARROW_SCALE * arrow_scale,
        width=p.ARROW_WIDTH,
        scale_units="inches",
        color=colormap(norm(velocity_magnitude)),
        linewidth=p.ARROW_LINEWIDTH,
        edgecolor=p.ARROW_EDGECOLOR,
    )

    # Draw land
    plt.fill(
        p.coastlon,
        p.coastlat,
        color=p.LAND_COLOR,
        linewidth=p.LAND_LINEWIDTH,
        zorder=p.LAND_ZORDER,
    )

    # Draw white background rectangle
    rect = mpatches.Rectangle(
        p.KEY_RECTANGLE_ANCHOR,
        p.KEY_RECTANGLE_WIDTH,
        p.KEY_RECTANGLE_HEIGHT,
        fill=True,
        color=p.KEY_BACKGROUND_COLOR,
        linewidth=p.KEY_LINEWIDTH,
        ec=p.KEY_EDGECOLOR,
    )
    plt.gca().add_patch(rect)

    # Draw arrow legend
    plt.quiverkey(
        quiver_handle,
        p.KEY_ARROW_LON,
        p.KEY_ARROW_LAT,
        p.KEY_ARROW_MAGNITUDE,
        p.KEY_ARROW_TEXT,
        coordinates="data",
        color=p.KEY_ARROW_COLOR,
        fontproperties={"size": p.FONTSIZE},
    )

    plt.gca().set_aspect("equal")
    plt.show()


def plot_vels(
    p, segment, lon, lat, east_vel, north_vel, arrow_scale, title_string="velocities"
):
    """
    Plots a map of velocity vectors with common map elements.

    This function creates a plot with a specified title, plotting segments and
    velocity vectors on a map. The plot includes common map elements such as
    axis settings and segment lines.

    Parameters:
    p (addict.Dict): A dictionary containing plotting parameters.
    segment (DataFrame): A DataFrame containing segment information with columns
                         'lon1', 'lon2', 'lat1', and 'lat2' representing the
                         starting and ending coordinates of each segment.
    east_vel (ndarray): Array of eastward velocity components.
    north_vel (ndarray): Array of northward velocity components.
    arrow_scale (float): Arrow length scale factor.
    title_string (str): The title of the plot.

    The function performs the following steps:
    1. Creates a figure with a specified size.
    2. Sets the plot title with the specified font size.
    3. Plots common map elements including segment lines and axis settings.
    4. Plots velocity vectors as arrows with scaling and color mapping.
    """
    plt.figure(figsize=p.FIGSIZE_VECTORS)
    plt.title(title_string, fontsize=p.FONTSIZE)
    plot_common_elements(p, segment, p.LON_RANGE, p.LAT_RANGE)
    plot_vel_arrows_elements(p, lon, lat, east_vel, north_vel, arrow_scale=arrow_scale)


def plot_residuals(p, segment, station):
    """
    Plots the residuals of east and north velocity estimates as scatter plots with
    mean absolute error (MAE) and mean squared error (MSE) for a given station.

    The function creates two plots:
    1. A scatter plot of the station locations colored by MAE.
    2. A scatter plot of the station locations colored by MSE.

    Parameters:
    -----------
    p : object
        An object containing plot configurations such as figure size, longitude range,
        and latitude range.
    segment : object
        An object representing a specific segment of data or region to be plotted.
    station : DataFrame
        A pandas DataFrame containing the station data with columns 'lon' and 'lat' for
        longitude and latitude respectively.

    Returns:
    --------
    None
    """

    mae = np.abs(station.model_east_vel_residual.values) + np.abs(
        station.model_north_vel_residual.values
    )
    mse = np.sqrt(
        station.model_east_vel_residual.values**2.0
        + station.model_north_vel_residual.values**2.0
    )

    plt.figure(figsize=p.FIGSIZE_VECTORS)
    plt.scatter(
        station.lon,
        station.lat,
        s=25,
        edgecolors="k",
        c=mae,
        cmap="YlOrRd",
        linewidths=0.1,
    )
    plot_common_elements(p, segment, p.LON_RANGE, p.LAT_RANGE)
    plt.clim(0, 10)
    plt.title("mean average error")
    plt.show()

    plt.figure(figsize=p.FIGSIZE_VECTORS)
    plt.scatter(
        station.lon,
        station.lat,
        s=25,
        edgecolors="k",
        c=mse,
        cmap="YlOrRd",
        linewidths=0.1,
    )
    plot_common_elements(p, segment, p.LON_RANGE, p.LAT_RANGE)
    plt.clim(0, 10)
    plt.title("mean squared error")
    plt.show()


def plot_segment_rates(p, segment, estimation, rate_type, rate_scale=1):
    """
    Plots the slip rates (strike-slip, dip-slip, tensile-slip, or a combination of dip and tensile slip)
    for given segments on a map.

    The function creates a plot with segments color-coded and line width scaled based on the slip rates.
    The colors represent:
    - Red: Negative rates (right-lateral for strike-slip, normal for dip-slip, convergence for tensile-slip, convergence for combination)
    - Blue: Positive rates (left-lateral for strike-slip, reverse for dip-slip, extension for tensile-slip, extension for combination)

    A legend is added to distinguish the negative and positive rates.

    Parameters:
    -----------
    p : object
        An object containing plot configurations such as figure size, fonts, colors, and map boundaries.
    segment : DataFrame
        A pandas DataFrame containing segment data with columns 'lon1', 'lon2', 'lat1', and 'lat2' for
        the start and end coordinates of each segment.
    rate_type : str
        A string indicating the type of slip rate to plot. Can be one of:
        - 'ss'   : strike-slip rates
        - 'ds'   : dip-slip rates
        - 'ts'   : tensile-slip rates
        - 'dsts' : combination of dip-slip and tensile-slip rates
    rate_scale : float, optional
        A scaling factor for the slip rate line widths, default is 1.

    Returns:
    --------
    None
    """
    plt.figure(figsize=p.FIGSIZE_VECTORS)

    if rate_type == "ss":
        plt.title("strike-slip rates", fontsize=p.FONTSIZE)
        label_text_negative = "right-lateral"
        label_text_positive = "left-lateral"
    elif rate_type == "ds":
        plt.title("dip-slip rates", fontsize=p.FONTSIZE)
        label_text_negative = "normal"
        label_text_positive = "reverse"
    elif rate_type == "ts":
        plt.title("tensile-slip rates", fontsize=p.FONTSIZE)
        label_text_negative = "convergence"
        label_text_positive = "extension"
    elif rate_type == "dsts":
        plt.title("dip+tensile-slip rates", fontsize=p.FONTSIZE)
        label_text_negative = "convergence"
        label_text_positive = "extension"

    plot_common_elements(p, segment, p.LON_RANGE, p.LAT_RANGE)

    plt.fill(
        p.coastlon,
        p.coastlat,
        color=p.LAND_COLOR,
        linewidth=p.LAND_LINEWIDTH,
        zorder=p.LAND_ZORDER,
    )

    for i in range(len(segment)):
        if rate_type == "ss":
            if estimation.strike_slip_rates[i] < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale * estimation.strike_slip_rates[i],
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale * estimation.strike_slip_rates[i],
                )

        if rate_type == "ds":
            if estimation.dip_slip_rates[i] < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale * estimation.dip_slip_rates[i],
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale * estimation.dip_slip_rates[i],
                )

        if rate_type == "ts":
            if estimation.tensile_slip_rates[i] < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale * estimation.tensile_slip_rates[i],
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale * estimation.tensile_slip_rates[i],
                )

        if rate_type == "dsts":
            if (estimation.dip_slip_rates[i] - estimation.tensile_slip_rates[i]) < 0:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-b",
                    linewidth=rate_scale
                    * (estimation.dip_slip_rates[i] - estimation.tensile_slip_rates[i]),
                )
            else:
                plt.plot(
                    [segment.lon1[i], segment.lon2[i]],
                    [segment.lat1[i], segment.lat2[i]],
                    "-r",
                    linewidth=rate_scale
                    * (estimation.dip_slip_rates[i] - estimation.tensile_slip_rates[i]),
                )

    # Legend
    black_segments = mlines.Line2D(
        [],
        [],
        color="red",
        marker="s",
        linestyle="None",
        markersize=10,
        label=label_text_negative,
    )
    red_segments = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="s",
        linestyle="None",
        markersize=10,
        label=label_text_positive,
    )
    plt.legend(
        handles=[black_segments, red_segments],
        loc="lower right",
        fontsize=p.FONTSIZE,
        framealpha=1.0,
        edgecolor="k",
    ).get_frame().set_boxstyle("Square")


def plot_fault_geometry(p, segment, meshes):
    """
    Plots the fault geometry (segments and triangular dislocation element meshes) on a map.

    The function creates a plot with segments color-coded and line width scaled based on the slip rates.
    The colors represent:
    - Black : Standard segments
    - Red : Segments replaced by triangular dislocation element meshes

    Parameters:
    -----------
    p : object
        An object containing plot configurations such as figure size, fonts, colors, and map boundaries.
    segment : DataFrame
        A pandas DataFrame containing segment data with columns 'lon1', 'lon2', 'lat1', and 'lat2' for
        the start and end coordinates of each segment.
    meshes : Dict

    Returns:
    --------
    None
    """
    plt.figure(figsize=p.FIGSIZE_VECTORS)

    plot_common_elements(p, segment, p.LON_RANGE, p.LAT_RANGE)

    plt.fill(
        p.coastlon,
        p.coastlat,
        color=p.LAND_COLOR,
        linewidth=p.LAND_LINEWIDTH,
        zorder=p.LAND_ZORDER,
    )

    for i in range(len(meshes)):
        x_coords = meshes[i].points[:, 0]
        y_coords = meshes[i].points[:, 1]
        vertex_array = np.asarray(meshes[i].verts)

        ax = plt.gca()
        xy = np.c_[x_coords, y_coords]
        verts = xy[vertex_array]
        pc = matplotlib.collections.PolyCollection(
            verts, edgecolor="none", alpha=0.2, facecolor="red"
        )
        ax.add_collection(pc)

        # Add mesh edge
        x_edge = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        y_edge = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        x_edge = np.append(x_edge, x_coords[meshes[0].ordered_edge_nodes[0, 0]])
        y_edge = np.append(y_edge, y_coords[meshes[0].ordered_edge_nodes[0, 0]])
        plt.plot(x_edge, y_edge, color="red", linewidth=1, linestyle="--")

    for i in range(len(segment)):
        if segment.patch_file_name[i] == -1:
            plt.plot(
                [segment.lon1[i], segment.lon2[i]],
                [segment.lat1[i], segment.lat2[i]],
                "-k",
                linewidth=1,
            )
        else:
            plt.plot(
                [segment.lon1[i], segment.lon2[i]],
                [segment.lat1[i], segment.lat2[i]],
                "-r",
                linewidth=1,
            )


def plot_mesh_mode(meshes, eigenvectors_to_tde_slip, mesh_idx, mode_idx, start_idx=0):
    plt.figure()
    plt.scatter(
        meshes[mesh_idx].lon_centroid,
        meshes[mesh_idx].lat_centroid,
        s=10,
        c=eigenvectors_to_tde_slip[mesh_idx][start_idx::2, mode_idx],
    )
    plt.plot(meshes[mesh_idx].x_perimeter, meshes[mesh_idx].y_perimeter, "-k")
    plt.gca().set_aspect("equal")
    plt.title(f"{meshes[0].file_name}, {mesh_idx=}, {mode_idx=}")
    plt.show()


def plot_tde_boundary_condition_labels(meshes, mesh_idx):
    top_indices = np.asarray(np.where(meshes[mesh_idx].top_elements))
    bot_indices = np.asarray(np.where(meshes[mesh_idx].bot_elements))
    side_indices = np.asarray(np.where(meshes[mesh_idx].side_elements))
    print(f"{top_indices.size=}")
    print(f"{bot_indices.size=}")
    print(f"{side_indices.size=}")

    plt.figure()
    plt.scatter(
        meshes[mesh_idx].lon_centroid,
        meshes[mesh_idx].lat_centroid,
        s=0.1,
        c="k",
    )
    plt.plot(
        meshes[mesh_idx].lon_centroid[top_indices].flatten(),
        meshes[mesh_idx].lat_centroid[top_indices].flatten(),
        "rx",
        label="top",
    )
    plt.plot(
        meshes[mesh_idx].lon_centroid[bot_indices].flatten(),
        meshes[mesh_idx].lat_centroid[bot_indices].flatten(),
        "gs",
        label="bot",
    )
    plt.plot(
        meshes[mesh_idx].lon_centroid[side_indices].flatten(),
        meshes[mesh_idx].lat_centroid[side_indices].flatten(),
        "c^",
        label="side",
    )

    plt.plot(meshes[mesh_idx].x_perimeter, meshes[mesh_idx].y_perimeter, "-k")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.title(f"{meshes[mesh_idx].file_name}, {mesh_idx=}")
    plt.show()


################################################################################################
#                                                                                              #
#                                                                                              #
#  88        88  888888888888  88  88           88  888888888888  88  88888888888  ad88888ba   #
#  88        88       88       88  88           88       88       88  88          d8"     "8b  #
#  88        88       88       88  88           88       88       88  88          Y8,          #
#  88        88       88       88  88           88       88       88  88aaaaa     `Y8aaaaa,    #
#  88        88       88       88  88           88       88       88  88"""""       `"""""8b,  #
#  88        88       88       88  88           88       88       88  88                  `8b  #
#  Y8a.    .a8P       88       88  88           88       88       88  88          Y8a     a8P  #
#   `"Y8888Y"'        88       88  88888888888  88       88       88  88888888888  "Y88888P"   #
#                                                                                              #
#                                                                                              #
################################################################################################
def get_logger(command):
    # Create logger
    logger.remove()  # Remove any existing loggers includeing default stderr

    # Define the custom format
    log_format = (
        "<level>{level}</level>: "
        "<level>{message}</level> - "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    )

    # Configure the logger with the custom format
    # logger.remove()  # Remove default logger configuration
    logger.add(lambda msg: print(msg, end=""), format=log_format, colorize=True)

    # logger.add(
    #     sys.stdout,
    #     # format="[{level}] {message}",
    #     # format="<cyan>[{level}]</cyan> <green>{message}</green>",
    #     colorize=True,
    # )
    # logger.add(command.run_name + ".log")
    logger.add(command.output_path + "/" + command.run_name + ".log")
    logger.info(f"Read: {command.file_name}")
    logger.info("RUN_NAME: " + command.run_name)
    logger.info(f"Write log file: {command.output_path}/{command.run_name}.log")
    return logger


def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


def make_default_segment(length):
    """
    Create a default segment Dict of specified length
    """
    default_segment = pd.DataFrame(
        columns=[
            "name",
            "lon1",
            "lat1",
            "lon2",
            "lat2",
            "dip",
            "res",
            "other3",
            "other6",
            "other7",
            "other8",
            "other9",
            "other10",
            "other11",
            "other12",
            "locking_depth",
            "locking_depth_sig",
            "locking_depth_flag",
            "dip_sig",
            "dip_flag",
            "ss_rate",
            "ss_rate_sig",
            "ss_rate_flag",
            "ds_rate",
            "ds_rate_sig",
            "ds_rate_flag",
            "ts_rate",
            "ts_rate_sig",
            "ts_rate_flag",
            "burial_depth",
            "burial_depth_sig",
            "burial_depth_flag",
            "resolution_override",
            "resolution_other",
            "patch_file_name",
            "patch_flag",
            "patch_slip_file",
            "patch_slip_flag",
        ]
    )
    # Set everything to zeros, then we'll fill in a few specific values
    length_vec = range(length)
    for key, value in default_segment.items():
        default_segment[key] = np.zeros_like(length_vec)
    default_segment.locking_depth = +15
    default_segment.dip = +90
    for i in range(len(default_segment.name)):
        default_segment.name[i] = "segment_" + str(i)

    return default_segment


def polygon_area(x, y):
    """
    From: https://newbedev.com/calculate-area-of-polygon-given-x-y-coordinates
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def great_circle_latitude_find(lon1, lat1, lon2, lat2, lon):
    """
    Determines latitude as a function of longitude along a great circle.
    LAT = gclatfind(LON1, LAT1, LON2, LAT2, LON) finds the latitudes of points of
    specified LON that lie along the great circle defined by endpoints LON1, LAT1
    and LON2, LAT2. Angles should be passed as degrees.
    """
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    lon = np.deg2rad(lon)
    lat = np.arctan(
        np.tan(lat1) * np.sin(lon - lon2) / np.sin(lon1 - lon2)
        - np.tan(lat2) * np.sin(lon - lon1) / np.sin(lon1 - lon2)
    )
    return lat


def get_cross_partials(vector):
    """
    Returns a linear operator R that when multiplied by
    vector a gives the cross product a cross b
    """
    return np.array(
        [
            [0, vector[2], -vector[1]],
            [-vector[2], 0, vector[0]],
            [vector[1], -vector[0], 0],
        ]
    )


def cartesian_vector_to_spherical_vector(vel_x, vel_y, vel_z, lon, lat):
    """
    This function transforms vectors from Cartesian to spherical components.
    Arguments:
        vel_x: array of x components of velocity
        vel_y: array of y components of velocity
        vel_z: array of z components of velocity
        lon: array of station longitudes
        lat: array of station latitudes
    Returned variables:
        vel_north: array of north components of velocity
        vel_east: array of east components of velocity
        vel_up: array of up components of velocity
    """
    projection_matrix = np.array(
        [
            [
                -np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                -np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                np.cos(np.deg2rad(lat)),
            ],
            [-np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)), 0],
            [
                -np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                -np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                -np.sin(np.deg2rad(lat)),
            ],
        ]
    )
    vel_north, vel_east, vel_up = np.dot(
        projection_matrix, np.array([vel_x, vel_y, vel_z])
    )
    return vel_north, vel_east, vel_up


def get_segment_oblique_projection(lon1, lat1, lon2, lat2, skew=True):
    """
    Use pyproj oblique mercator: https://proj.org/operations/projections/omerc.html

    According to: https://proj.org/operations/projections/omerc.html
    This is this already rotated by the fault strike but the rotation can be undone with +no_rot
    > +no_rot
    > No rectification (not “no rotation” as one may well assume).
    > Do not take the last step from the skew uv-plane to the map XY plane.
    > Note: This option is probably only marginally useful,
    > but remains for (mostly) historical reasons.

    The version with north still pointing "up" appears to be called the
    Rectified skew orthomorphic projection or Hotine oblique Mercator projection
    https://pro.arcgis.com/en/pro-app/latest/help/mapping/properties/rectified-skew-orthomorphic.htm
    """
    if lon1 > 180.0:
        lon1 = lon1 - 360
    if lon2 > 180.0:
        lon2 = lon2 - 360

    # Check if latitudes are too close to identical
    # If lat1 and lat2 are the same at the 5 decimal place proj with fail
    # Perturb lat2 slightly to avoid this.
    if np.isclose(lat1, lat2):
        latitude_offset = 0.001
        lat2 += latitude_offset

    projection_string = (
        "+proj=omerc "
        + "+lon_1="
        + str(lon1)
        + " "
        + "+lat_1="
        + str(lat1)
        + " "
        + "+lon_2="
        + str(lon2)
        + " "
        + "+lat_2="
        + str(lat2)
        + " "
        + "+ellps=WGS84"
    )
    if not skew:
        projection_string += " +no_rot"
    projection = pyproj.Proj(pyproj.CRS.from_proj4(projection_string))
    return projection


def latitude_to_colatitude(lat):
    """
    Convert from latitude to colatitude
    NOTE: Not sure why I need to treat the scalar case differently but I do.
    """
    if lat.size == 1:  # Deal with the scalar case
        if lat >= 0:
            colat = 90.0 - lat
        elif lat < 0:
            colat = -90.0 - lat
    else:  # Deal with the array case
        colat = np.zeros_like(lat)
        colat[np.where(lat >= 0)[0]] = 90.0 - lat[np.where(lat >= 0)[0]]
        colat[np.where(lat < 0)[0]] = -90.0 - lat[np.where(lat < 0)[0]]
    return colat


def get_transverse_projection(lon0, lat0):
    """
    Use pyproj oblique mercator: https://proj.org/operations/projections/tmerc.html
    """
    if lon0 > 180.0:
        lon0 = lon0 - 360
    projection_string = (
        "+proj=tmerc "
        + "+lon_0="
        + str(lon0)
        + " "
        + "+lat_0="
        + str(lat0)
        + " "
        + "+ellps=WGS84"
    )
    projection = pyproj.Proj(pyproj.CRS.from_proj4(projection_string))
    return projection


def get_keep_index_12(length_of_array: int):
    """Calculate an indexing array that given and array:
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    returns
    [1, 2, 4, 5, 7, 8]
    This is useful for selecting only indices associated with
    horizontal motions

    Args:
        length_of_array (int): Length of initial array.  Should be divisible by 3

    Returns:
        idx (np.array): Array of indices to return
    """
    idx = np.delete(np.arange(0, length_of_array), np.arange(2, length_of_array, 3))
    return idx


def interleave2(array_1, array_2):
    """Interleaves two arrays, with alternating entries.
    Given array_1 = [0, 2, 4, 6] and array_2 = [1, 3, 5, 7]
    returns
    [0, 1, 2, 3, 4, 5, 6, 7]
    This is useful for assembling velocity/slip components into a combined array.

    Args:
        array_1, array_2 (np.array): Arrays to interleave. Should be equal length

    Returns:
        interleaved_array (np.array): Interleaved array
    """
    interleaved_array = np.empty((array_1.size + array_2.size), dtype=array_1.dtype)
    interleaved_array[0::2] = array_1
    interleaved_array[1::2] = array_2
    return interleaved_array


def interleave3(array_1, array_2, array_3):
    """Interleaves three arrays, with alternating entries.
    Given array_1 = [0, 3, 6, 9], array_2 = [1, 4, 7, 10], and array_3 = [2, 5, 8, 11]
    returns
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    This is useful for assembling velocity/slip components into a combined array.

    Args:
        array_1, array_2, array_3 (np.array): Arrays to interleave. Should be equal length

    Returns:
        interleaved_array (np.array): Interleaved array
    """
    interleaved_array = np.empty(
        (array_1.size + array_2.size + array_3.size), dtype=array_1.dtype
    )
    interleaved_array[0::3] = array_1
    interleaved_array[1::3] = array_2
    interleaved_array[2::3] = array_3
    return interleaved_array


def get_2component_index(indices: np.array):
    """Returns indices into 2-component array, where each entry of input array
    corresponds to two entries in the 2-component array
    Given indices = [0, 2, 10, 6]
    returns
    [0, 1, 4, 5, 20, 21, 12, 13]
    This is useful for referencing velocity/slip components corresponding to a set
    of stations/faults.

    Args:
        indices (np.array): Element index array

    Returns:
        idx (np.array): Component index array (2 * length of indices)
    """
    idx = np.sort(np.append(2 * (indices + 1) - 2, 2 * (indices + 1) - 1))
    return idx


def get_3component_index(indices: np.array):
    """Returns indices into 3-component array, where each entry of input array
    corresponds to three entries in the 3-component array
    Given indices = [0, 2, 10, 6]
    returns
    [0, 1, 2, 6, 7, 8, 27, 28, 29, 15, 16, 17]
    This is useful for referencing velocity/slip components corresponding to a set
    of stations/faults.

    Args:
        indices (np.array): Element index array

    Returns:
        idx (np.array): Component index array (3 * length of indices)
    """
    idx = np.sort(
        np.append(3 * (indices + 1) - 3, (3 * (indices + 1) - 2, 3 * (indices + 1) - 1))
    )
    return idx


def align_velocities(df_1, df_2, distance_threshold):
    # Add block_label to dataframes if it's not there
    if not "block_label" in df_1.columns:
        df_1["block_label"] = 0

    if not "block_label" in df_2.columns:
        df_2["block_label"] = 0

    # Find approximate distances between all station pairs between data sets
    station_to_station_distances = cdist(
        np.array([df_1.lon, df_1.lat]).T, np.array([df_2.lon, df_2.lat]).T
    )

    # For each  velocity find the closest distance and check if it's less than a distance_threshold (approximate) away
    match_index_1 = []
    match_index_2 = []

    for i in range(len(df_1)):
        if np.min(station_to_station_distances[i, :]) < distance_threshold:
            min_index = np.argmin(station_to_station_distances[i, :])
            match_index_1.append(i)
            match_index_2.append(min_index)

    # Create smaller data frames for the matching locations
    df_1_match = df_1.iloc[match_index_1].copy()
    df_2_match = df_2.iloc[match_index_2].copy()

    # Build the linear operator
    operator_rotation = get_global_float_block_rotation_partials(df_1_match)
    keep_idx = get_keep_index_12(operator_rotation.shape[0])
    operator_rotation = operator_rotation[keep_idx, :]
    differential_velocties_match = interleave2(
        df_2_match.east_vel.values - df_1_match.east_vel.values,
        df_2_match.north_vel.values - df_1_match.north_vel.values,
    )

    # Solve for rotation vector that best aligns the velocities in the two velocity fields
    covariance_matrix = np.linalg.inv(operator_rotation.T @ operator_rotation)
    rotation_vector_align = (
        covariance_matrix @ operator_rotation.T @ differential_velocties_match
    )

    # Rotate the stations in the 1 data set that are not collocated with Weiss data into the Weiss reference frame
    operator_rotation = get_global_float_block_rotation_partials(df_1)
    keep_idx = get_keep_index_12(operator_rotation.shape[0])
    operator_rotation = operator_rotation[keep_idx, :]

    # Rotate subset of gbm velocites into weiss reference frame
    df_1_aligned = copy.deepcopy(df_1)
    rotated_vels_match = operator_rotation @ rotation_vector_align
    df_1_aligned["east_vel"] = df_1_aligned["east_vel"] + rotated_vels_match[0::2]
    df_1_aligned["north_vel"] = df_1_aligned["north_vel"] + rotated_vels_match[1::2]

    return df_1_aligned


def get_newest_run_folder(rewind=0):
    """
    Generate a new folder name based on existing numeric folder names.

    This function scans the current directory for folders with numeric names,
    identifies the highest number, and returns a new folder name that is one
    greater than the highest number, formatted as a zero-padded 10-digit string.

    Returns:
        str: A new folder name as a zero-padded 10-digit string.

    Raises:
        ValueError: If no numeric folder names are found in the current directory.

    Example:
        If the current directory contains folders named "0000000001", "0000000002",
        and "0000000003", the function will return "0000000004".
    """
    # Get all folder names
    folder_names = glob.glob("./../runs/*/")

    # Remove trailing slashes
    folder_names = [folder_name.rstrip(os.sep) for folder_name in folder_names]

    # Remove anything before numerical folder name
    folder_names = [folder_name[-10:] for folder_name in folder_names]

    # Check to see if the folder name is a native run number
    folder_names_runs = list()
    for folder_name in folder_names:
        try:
            folder_names_runs.append(int(folder_name))
        except ValueError as ex:
            pass

    # Get new folder name
    newest_folder_number = np.max(folder_names_runs) - rewind
    newest_folder_name = f"./../runs/{newest_folder_number:010d}"

    return newest_folder_name


def read_run(folder_name):
    """
    Reads and loads data from a specified folder containing a pickled file.

    This function opens a pickled file located in the directory `./../runs/{folder_name}/`,
    where `{folder_name}` is provided as an argument. The pickled file is expected to contain
    a list of six elements, which are extracted and returned as separate variables.

    Parameters:
    folder_name (str): The name of the folder containing the pickled file.

    Returns:
    tuple: A tuple containing the following elements in order:
        - command: command dictionary.
        - estimation: estimation command dictionary.
        - station: station dataframe.
        - segment: segment dataframe.
        - block: block dataframe.
        - meshes: meshes list.

    Example:
    >>> command, estimation, station, segment, block, meshes = read_run('example_folder')
    """
    pickle_file = open(f"{folder_name}/output.pkl", "rb")
    pickle_data = pickle.load(pickle_file)
    command = pickle_data[0]
    estimation = pickle_data[1]
    station = pickle_data[2]
    segment = pickle_data[3]
    block = pickle_data[4]
    meshes = pickle_data[5]
    del pickle_data
    return command, estimation, station, segment, block, meshes


def diagnose_matrix(mat):
    """
    Visualizes and diagnoses a matrix for rank deficiency.

    This function generates visualizations of the input matrix and scans for
    the first rank deficient column. It creates two plots: one showing the
    logarithm of the absolute values of the matrix elements, and another
    showing the sparsity pattern of the matrix. It then iterates through the
    columns of the matrix to identify the first column where the rank is
    deficient.

    Parameters:
    mat (numpy.ndarray): The matrix to be diagnosed.

    Example:
    diagnose_matrix(operators.eigen * np.sqrt(weighting_vector_eigen[:, None]))

    Visualizations:
    - Logarithm of the absolute values of the matrix elements.
    - Sparsity pattern of the matrix.

    Prints:
    - The index of the first rank deficient column and its rank.
    """
    """
    Example call for QP solve operator
    diagnose_matrix(operators.eigen * np.sqrt(weighting_vector_eigen[:, None]))
    """
    plt.figure(figsize=(20, 4))
    plt.imshow(np.log10(np.abs(mat)), aspect="auto")
    plt.show()

    plt.figure(figsize=(20, 4))
    plt.spy(mat, aspect="auto")
    plt.show()

    print("Scanning for first rank deficient column")
    for col in range(1, mat.shape[1]):
        print(f"Column: {col}")
        rank = np.linalg.matrix_rank(mat[:, 0:col])
        if rank < col:
            print(f"RANK DEFICIENT: {col=}, {rank=}")
            break
