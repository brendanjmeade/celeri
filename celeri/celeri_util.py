from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from loguru import logger
from pydantic import (
    BaseModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)
from rich.console import Console
from rich.table import Table
from rich.text import Text
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from celeri.config import Config


def sph2cart(lon, lat, radius):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def triangle_area(triangles):
    # The norm of the cross product of two sides is twice the area
    # https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
    return np.linalg.norm(triangle_normal(triangles), axis=1) / 2.0


def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


def triangle_normal(triangles):
    # The cross product of two sides is a normal vector
    # https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
    return np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1
    )


def get_logger(config: Config):
    # Create logger
    logger.remove()  # Remove any existing loggers including default stderr

    # Initialize Rich console
    console = Console()

    def rich_sink(message):
        """Custom sink for loguru that uses rich for formatting with proper indentation"""
        record = message.record

        # Level colors
        level_styles = {
            "TRACE": "dim white",
            "DEBUG": "white",
            "INFO": "blue",
            "SUCCESS": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold white on red",
        }

        level = record["level"].name
        style = level_styles.get(level, "white")

        # Create the level badge (exactly 10 chars wide, left-aligned)
        level_text = f"{level:<10}"

        # Create the message content
        time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        location = f"{record['name']}:{record['function']}:{record['line']}"
        message_str = str(
            record["message"]
        ).rstrip()  # Remove any trailing whitespace/newlines

        # Format the location/timestamp line
        location_line = f"- {location} - {time_str}"

        # For ERROR and CRITICAL levels, include exception info if available
        if level in ["ERROR", "CRITICAL"] and record["exception"]:
            exc_type = record["exception"].type
            exc_value = record["exception"].value
            exc_traceback = record["exception"].traceback

            # Build the content with styled parts
            content = Text(message_str, style="white")
            content.append("\n")
            content.append(location_line, style="dim white")

            # Format the exception with full traceback if available
            if exc_traceback:
                import traceback as tb

                tb_lines = "".join(
                    tb.format_exception(exc_type, exc_value, exc_traceback)
                )
                content.append("\n")
                content.append(tb_lines, style="dim white")
            else:
                content.append("\n")
                content.append(
                    f"{exc_type.__name__ if exc_type else 'Unknown'}: {exc_value}",
                    style="dim white",
                )
        else:
            # Build normal message with white message and gray location/timestamp
            content = Text(message_str, style="white")
            content.append("\n")
            content.append(location_line, style="dim white")

        # Create text with level prefix
        prefix = Text(level_text, style=style)

        # Print with hanging indent using a grid table
        table = Table.grid(padding=0, expand=True)
        table.add_column(width=11, no_wrap=True)  # 10 chars + 1 space
        table.add_column(ratio=1)

        table.add_row(prefix + " ", content)

        # Use file handle and suppress automatic newline
        console.file = sys.stderr  # Ensure we're writing to stderr like default logger
        console.print(table, crop=False, end="")

    # Add the rich sink for console output with backtrace for errors
    logger.add(rich_sink, colorize=True, backtrace=True, diagnose=True)

    # Add file logging with simple format (no colors in file)
    file_format = "{level: <10} | {time:YYYY-MM-DD HH:mm:ss} | {name}:{function}:{line} | {message}"
    logger.add(
        (config.output_path / config.run_name).with_suffix(".log"),
        format=file_format,
        colorize=False,  # No color codes in log file
        backtrace=True,  # Include traceback in log file
        diagnose=True,  # Include variable values in errors
    )

    logger.info("RUN_NAME: " + config.run_name)
    logger.info(f"Write log file: {config.output_path}/{config.run_name}.log")
    return logger


def polygon_area(x, y):
    """From: https://newbedev.com/calculate-area-of-polygon-given-x-y-coordinates."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def great_circle_latitude_find(lon1, lat1, lon2, lat2, lon):
    """Determines latitude as a function of longitude along a great circle.
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
    """Returns a linear operator R that when multiplied by
    vector a gives the cross product a cross b.
    """
    return np.array(
        [
            [0, vector[2], -vector[1]],
            [-vector[2], 0, vector[0]],
            [vector[1], -vector[0], 0],
        ]
    )


def cartesian_vector_to_spherical_vector(vel_x, vel_y, vel_z, lon, lat):
    """This function transforms vectors from Cartesian to spherical components.

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
    """Use pyproj oblique mercator: https://proj.org/operations/projections/omerc.html.

    According to: https://proj.org/operations/projections/omerc.html
    This is this already rotated by the fault strike but the rotation can be undone with +no_rot
    > +no_rot
    > No rectification (not "no rotation" as one may well assume).
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


def latitude_to_colatitude(lat: np.ndarray) -> np.ndarray:
    """Convert from latitude to colatitude."""
    colat = np.zeros_like(lat)
    colat[np.where(lat >= 0)[0]] = 90.0 - lat[np.where(lat >= 0)[0]]
    colat[np.where(lat < 0)[0]] = -90.0 - lat[np.where(lat < 0)[0]]
    return colat


def get_transverse_projection(lon0, lat0):
    """Use pyproj oblique mercator: https://proj.org/operations/projections/tmerc.html."""
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
    [1, 2, 3, 4, 5, 6, 7, 8, 9].

    Returns:
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
    Given array_1 = [0, 2, 4, 6] and array_2 = [1, 3, 5, 7].

    Returns:
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
    Given array_1 = [0, 3, 6, 9], array_2 = [1, 4, 7, 10], and array_3 = [2, 5, 8, 11].

    Returns:
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


def get_2component_index(indices: np.ndarray):
    """Returns indices into 2-component array, where each entry of input array
    corresponds to two entries in the 2-component array
    Given indices = [0, 2, 10, 6].

    Returns:
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


def get_3component_index(indices: np.ndarray):
    """Returns indices into 3-component array, where each entry of input array
    corresponds to three entries in the 3-component array
    Given indices = [0, 2, 10, 6].

    Returns:
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
    if "block_label" not in df_1.columns:
        df_1["block_label"] = 0

    if "block_label" not in df_2.columns:
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

    # TODO fix late import
    from celeri.operators import get_global_float_block_rotation_partials

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


def get_newest_run_folder(*, base: Path | None = None, rewind=0) -> Path:
    """Generate a new folder name based on existing numeric folder names.

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
    if base is None:
        base = Path("./../runs/")
    folder_names = base.iterdir()

    # Remove trailing slashes
    folder_names = [folder_name.name for folder_name in folder_names]

    # Remove anything before numerical folder name
    folder_names = [folder_name[-10:] for folder_name in folder_names]

    # Check to see if the folder name is a native run number
    folder_names_runs = []
    for folder_name in folder_names:
        try:
            folder_names_runs.append(int(folder_name))
        except ValueError:
            pass

    # Get new folder name
    newest_folder_number = max(folder_names_runs) - rewind
    newest_folder_name = base / f"{newest_folder_number:010d}"

    return newest_folder_name


def diagnose_matrix(mat):
    """Visualizes and diagnoses a matrix for rank deficiency.

    This function generates visualizations of the input matrix and scans for
    the first rank deficient column. It creates two plots: one showing the
    logarithm of the absolute values of the matrix elements, and another
    showing the sparsity pattern of the matrix. It then iterates through the
    columns of the matrix to identify the first column where the rank is
    deficient.

    Parameters
    ----------
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


def euler_pole_covariance_to_rotation_vector_covariance(
    omega_x, omega_y, omega_z, euler_pole_covariance_all
):
    """This function takes the model parameter covariance matrix
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


class RelativePathSerializerMixin(BaseModel):
    """Mixin to convert paths to relative paths depending on context."""

    @model_serializer(mode="wrap")
    def _relative_path_context(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, object]:
        """Convert paths to relative paths depending on context."""
        ctx = info.context or {}
        relative_to = ctx.get("paths_relative_to", None)
        if relative_to is None:
            return handler(self)

        assert isinstance(relative_to, Path)
        if not relative_to.is_absolute():
            raise ValueError(
                f"Context 'relative_paths' must be an absolute path, got {relative_to}"
            )

        data = handler(self)
        for name in type(self).model_fields:
            is_path = isinstance(getattr(self, name), Path)
            if is_path:
                relative_path = Path(data[name]).relative_to(relative_to, walk_up=True)
                if isinstance(data[name], str):
                    data[name] = str(relative_path)
                else:
                    data[name] = relative_path
        return data
