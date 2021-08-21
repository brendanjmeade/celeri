import copy
import json
import meshio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial
from addict import Dict
from numpy.lib.shape_base import split
from pyproj import Geod
from matplotlib import path

import celeri
import celeri_closure


GEOID = Geod(ellps="WGS84")
KM2M = 1.0e3
RADIUS_EARTH = np.float64((GEOID.a + GEOID.b) / 2)


def read_data(command_file_name):
    # Read command data
    with open(command_file_name, "r") as f:
        command = json.load(f)

    # Read segment data
    segment = pd.read_csv(command["segment_file_name"])
    segment = segment.loc[:, ~segment.columns.str.match("Unnamed")]

    # Read block data
    block = pd.read_csv(command["block_file_name"])
    block = block.loc[:, ~block.columns.str.match("Unnamed")]

    # Read mesh data
    with open(command["mesh_param_file_name"], "r") as f:
        mesh_param = json.load(f)
    meshes = {}
    for i in range(len(mesh_param)):
        meshes[i] = meshio.read(mesh_param[i]["mesh_filename"])
        meshes[i].verts = meshes[i].get_cells_type("triangle")
        # Expand mesh coordinates
        meshes[i].lon1 = meshes[i].points[meshes[i].verts[:, 0], 0]
        meshes[i].lon2 = meshes[i].points[meshes[i].verts[:, 1], 0]
        meshes[i].lon3 = meshes[i].points[meshes[i].verts[:, 2], 0]
        meshes[i].lat1 = meshes[i].points[meshes[i].verts[:, 0], 1]
        meshes[i].lat2 = meshes[i].points[meshes[i].verts[:, 1], 1]
        meshes[i].lat3 = meshes[i].points[meshes[i].verts[:, 2], 1]
        meshes[i].dep1 = meshes[i].points[meshes[i].verts[:, 0], 2]
        meshes[i].dep2 = meshes[i].points[meshes[i].verts[:, 1], 2]
        meshes[i].dep3 = meshes[i].points[meshes[i].verts[:, 2], 2]
        meshes[i].centroids = np.mean(meshes[i].points[meshes[i].verts, :], axis=1)
        # Cartesian coordinates in meters
        meshes[i].x1, meshes[i].y1, meshes[i].z1 = celeri.sph2cart(
            meshes[i].lon1,
            meshes[i].lat1,
            celeri.RADIUS_EARTH + KM2M * meshes[i].dep1,
        )
        meshes[i].x2, meshes[i].y2, meshes[i].z2 = celeri.sph2cart(
            meshes[i].lon2,
            meshes[i].lat2,
            celeri.RADIUS_EARTH + KM2M * meshes[i].dep2,
        )
        meshes[i].x3, meshes[i].y3, meshes[i].z3 = celeri.sph2cart(
            meshes[i].lon3,
            meshes[i].lat3,
            celeri.RADIUS_EARTH + KM2M * meshes[i].dep3,
        )
        # Cross products for orientations
        tri_leg1 = np.transpose(
            [
                np.deg2rad(meshes[i].lon2 - meshes[i].lon1),
                np.deg2rad(meshes[i].lat2 - meshes[i].lat1),
                (1 + KM2M * meshes[i].dep2 / celeri.RADIUS_EARTH)
                - (1 + KM2M * meshes[i].dep1 / celeri.RADIUS_EARTH),
            ]
        )
        tri_leg2 = np.transpose(
            [
                np.deg2rad(meshes[i].lon3 - meshes[i].lon1),
                np.deg2rad(meshes[i].lat3 - meshes[i].lat1),
                (1 + KM2M * meshes[i].dep3 / celeri.RADIUS_EARTH)
                - (1 + KM2M * meshes[i].dep1 / celeri.RADIUS_EARTH),
            ]
        )
        meshes[i].nv = np.cross(tri_leg1, tri_leg2)
        azimuth, elevation, r = celeri.cart2sph(
            meshes[i].nv[:, 0], meshes[i].nv[:, 1], meshes[i].nv[:, 2]
        )
        meshes[i].strike = celeri.wrap2360(-np.rad2deg(azimuth))
        meshes[i].dip = 90 - np.rad2deg(elevation)
        meshes[i].dip_flag = meshes[i].dip != 90

    # Read station data
    if (
        not command.__contains__("station_file_name")
        or len(command["station_file_name"]) == 0
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
    else:
        station = pd.read_csv(command["station_file_name"])
        station = station.loc[:, ~station.columns.str.match("Unnamed")]

    # Read Mogi source data
    if (
        not command.__contains__("mogi_file_name")
        or len(command["mogi_file_name"]) == 0
    ):
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
    else:
        mogi = pd.read_csv(command["mogi_file_name"])
        mogi = mogi.loc[:, ~mogi.columns.str.match("Unnamed")]

    # Read SAR data
    if not command.__contains__("sar_file_name") or len(command["sar_file_name"]) == 0:
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

    else:
        sar = pd.read_csv(command["sar_file_name"])
        sar = sar.loc[:, ~sar.columns.str.match("Unnamed")]
    return command, segment, block, meshes, station, mogi, sar


def sph2cart(lon, lat, radius):
    x = radius * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = radius * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = radius * np.sin(np.deg2rad(lat))
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return azimuth, elevation, r


def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


def process_station(station, command):
    if command["unit_sigmas"] == "yes":  # Assign unit uncertainties, if requested
        station.east_sig = np.ones_like(station.east_sig)
        station.north_sig = np.ones_like(station.north_sig)
        station.up_sig = np.ones_like(station.up_sig)

    station["depth"] = np.zeros_like(station.lon)
    station["x"], station["y"], station["z"] = celeri.sph2cart(
        station.lon, station.lat, celeri.RADIUS_EARTH
    )
    station = station.drop(np.where(station.flag == 0)[0])
    station = station.reset_index(drop=True)
    return station


def locking_depth_manager(segment, command):
    """
    This function assigns the locking depths given in the command file to any
    segment that has the same locking depth flag.  Segments with flag =
    0, 1 are untouched.
    """
    segment = segment.copy(deep=True)
    segment.locking_depth.values[segment.locking_depth_flag == 2] = command[
        "locking_depth_flag2"
    ]
    segment.locking_depth.values[segment.locking_depth_flag == 3] = command[
        "locking_depth_flag3"
    ]
    segment.locking_depth.values[segment.locking_depth_flag == 4] = command[
        "locking_depth_flag4"
    ]
    segment.locking_depth.values[segment.locking_depth_flag == 5] = command[
        "locking_depth_flag5"
    ]

    if command["locking_depth_override_flag"] == "yes":
        segment.locking_depth.values = command["locking_depth_override_value"]
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
    togg_off = np.where(
        (segment.patch_flag != 0)
        & (segment.patch_file_name != 0)
        & (segment.patch_file_name <= len(meshes))
    )[0]
    segment.locking_depth.values[togg_off] = 0
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
    swap_endpoint_idx = np.where(cross_product[:,2] < 0)
    segment_copy.lon1.values[swap_endpoint_idx] = segment.lon2.values[swap_endpoint_idx]
    segment_copy.lat1.values[swap_endpoint_idx] = segment.lat2.values[swap_endpoint_idx]
    segment_copy.lon2.values[swap_endpoint_idx] = segment.lon1.values[swap_endpoint_idx]
    segment_copy.lat2.values[swap_endpoint_idx] = segment.lat1.values[swap_endpoint_idx]
    return segment_copy
    # for i in range(x1.size):
    #     cross_product = np.cross([x1[i], y1[i], z1[i]], [x2[i], y2[i], z2[i]])
    #     if cross_product[2] <= 0:
    #         print(i, "Reordering endpoints")
    #         segment_copy.lon1.values[i] = segment.lon2.values[i]
    #         segment_copy.lat1.values[i] = segment.lat2.values[i]
    #         segment_copy.lon2.values[i] = segment.lon1.values[i]
    #         segment_copy.lat2.values[i] = segment.lat1.values[i]

    # segment_copy = copy.deepcopy(segment)
    # for i in range(len(segment)):
    #     if segment.lon1[i] > segment.lon2[i]:
    #         # print(i, "Reordering endpoints for segment ", segment.name[i].strip())
    #         segment_copy.lon1.values[i] = segment.lon2.values[i]
    #         segment_copy.lat1.values[i] = segment.lat2.values[i]
    #         segment_copy.lon2.values[i] = segment.lon1.values[i]
    #         segment_copy.lat2.values[i] = segment.lat1.values[i]
    # return segment_copy


def segment_centroids(segment):
    """Calculate segment centroids."""
    segment["centroid_x"] = np.zeros_like(segment.lon1)
    segment["centroid_y"] = np.zeros_like(segment.lon1)
    segment["centroid_z"] = np.zeros_like(segment.lon1)
    segment["centroid_lon"] = np.zeros_like(segment.lon1)
    segment["centroid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment_forward_azimuth, _, _ = celeri.GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )
        segment_down_dip_azimuth = segment_forward_azimuth + 90.0 * np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        )
        azx = (segment.y2[i] - segment.y1[i]) / (segment.x2[i] - segment.x1[i])
        azx = np.arctan(-1.0 / azx)  # TODO: MAKE THIS VARIABLE NAME DESCRIPTIVE
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
        ) = celeri.GEOID.fwd(
            segment.mid_lon[i],
            segment.mid_lat[i],
            segment_down_dip_azimuth,
            segment_down_dip_distance,
        )
        segment.centroid_x.values[i] = segment.mid_x[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.cos(azx)
        segment.centroid_y.values[i] = segment.mid_y[i] + np.sign(
            np.cos(np.deg2rad(segment.dip[i]))
        ) * segment_down_dip_distance * np.sin(azx)
    segment.centroid_lon.values[segment.centroid_lon < 0.0] += 360.0
    return segment


def process_segment(segment, command, meshes):

    segment["x1"], segment["y1"], segment["z1"] = celeri.sph2cart(
        segment.lon1, segment.lat1, celeri.RADIUS_EARTH
    )
    segment["x2"], segment["y2"], segment["z2"] = celeri.sph2cart(
        segment.lon2, segment.lat2, celeri.RADIUS_EARTH
    )

    segment = celeri.order_endpoints_sphere(segment)

    # This calculation needs to account for the periodic nature of longitude.
    # Calculate the periodic longitudinal separation.
    # @BJM: Is this better done with GEIOD?
    sep = segment.lon2 - segment.lon1
    periodic_lon_separation = np.where(
        sep > 180,
        sep - 360,
        np.where(sep < -180, sep + 360, sep)
    )
    segment["mid_lon_plate_carree"] = segment.lon1.values + periodic_lon_separation / 2.0

    # No worries for latitude because there's no periodicity.
    segment["mid_lat_plate_carree"] = (segment.lat1.values + segment.lat2.values) / 2.0
    segment["mid_lon"] = np.zeros_like(segment.lon1)
    segment["mid_lat"] = np.zeros_like(segment.lon1)

    for i in range(len(segment)):
        segment.mid_lon.values[i], segment.mid_lat.values[i] = celeri.GEOID.npts(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i], 1
        )[0]
    segment.mid_lon.values[segment.mid_lon < 0.0] += 360.0

    segment["mid_x"], segment["mid_y"], segment["mid_z"] = celeri.sph2cart(
        segment.mid_lon, segment.mid_lat, celeri.RADIUS_EARTH
    )
    segment = celeri.locking_depth_manager(segment, command)
    segment = celeri.zero_mesh_segment_locking_depth(segment, meshes)
    # segment.locking_depth.values = PatchLDtoggle(segment.locking_depth, segment.patch_file_name, segment.patch_flag, Command.patchFileNames) % Set locking depth to zero on segments that are associated with patches # TODO: Write this after patches are read in.
    segment = celeri.segment_centroids(segment)
    return segment

def polygon_area(x, y):
    """
    From: https://newbedev.com/calculate-area-of-polygon-given-x-y-coordinates
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def assign_block_labels(segment, station, block, mogi, sar):
    """
    Ben Thompson's implementation of the half edge approach to the
    block labeling problem and east/west assignment.
    """
    #segment = split_segments_crossing_meridian(segment)
    
    np_segments = np.zeros((len(segment), 2, 2))
    np_segments[:, 0, 0] = segment.lon1.to_numpy()
    np_segments[:, 1, 0] = segment.lon2.to_numpy()
    np_segments[:, 0, 1] = segment.lat1.to_numpy()
    np_segments[:, 1, 1] = segment.lat2.to_numpy()

    closure = celeri_closure.run_block_closure(np_segments)
    labels = celeri_closure.get_segment_labels(closure)

    segment["west_labels"] = labels[:,0]
    segment["east_labels"] = labels[:,1]

    # Check for unprocessed indices
    unprocessed_indices = np.union1d(
        np.where(segment["east_labels"] < 0),
        np.where(segment["west_labels"] < 0),
    )
    if len(unprocessed_indices) > 0:
        print("Found unproccessed indices")

    # Find relative areas of each block to identify an external block
    block["area_plate_carree"] = -1 * np.ones(len(block))
    for i in range(closure.n_polygons()):
        vs = closure.polygon_vertices[i]
        block.area_plate_carree.values[i] = celeri.polygon_area(vs[:, 0], vs[:, 1])
    external_block_idx = np.argmax(block.area_plate_carree)

    # Assign block labels points to block interior points
    block["block_label"] = closure.assign_points(block.interior_lon.to_numpy(), block.interior_lat.to_numpy())

    # Assign block labels to GPS stations
    if not station.empty:
        station['block_label'] = closure.assign_points(
            station.lon.to_numpy(), 
            station.lat.to_numpy()
        )

    # Assign block labels to SAR locations
    if not sar.empty:
        sar['block_label'] = closure.assign_points(sar.lon.to_numpy(), sar.lat.to_numpy())

    # Assign block labels to Mogi sources
    if not mogi.empty:
        mogi["block_label"] = closure.assign_points(mogi.lon.to_numpy(), mogi.lat.to_numpy())

    return segment, closure

def split_segments_crossing_meridian(segment):
    """
    Splits segments along the prime meridian with one endpoint on the
    prime meridian. All other segment properties are taken from
    the original segment.
    """
    segment.lon1 = celeri.wrap2360(segment.lon1.values)
    segment.lon2 = celeri.wrap2360(segment.lon2.values)

    # Get longitude differences
    prime_meridian_cross = np.abs(segment.lon1 - segment.lon2) > 180
    split_idx = np.where(prime_meridian_cross)

    if any(prime_meridian_cross):
        #  Split those segments crossing the meridian
        split_lat = great_circle_latitude_find(
            segment.lon1.values[split_idx],
            segment.lat1.values[split_idx],
            segment.lon2.values[split_idx],
            segment.lat2.values[split_idx],
            360.0 * np.ones(len(split_idx)),
        )
        split_lat = np.rad2deg(split_lat)
        # Replicate split segment properties and assign new endpoints
        segment_whole = copy.copy(segment[~prime_meridian_cross])
        segment_split = copy.copy(segment[prime_meridian_cross])
        segment_split = pd.concat([segment_split, segment_split])
        # Insert the split coordinates
        segment_split.lon2.values[0 : len(split_lat)] = 0.0
        segment_split.lat2.values[0 : len(split_lat)] = split_lat
        segment_split.lon1.values[len(split_lat):] = 0.0
        segment_split.lat1.values[len(split_lat):] = split_lat
        # [segment_split.midLon, segment_split.midLat] = segmentmidpoint(split.lon1, split.lat1, split.lon2, split.lat2);
        segment_split.x1, segment_split.y1, segment_split.z1 = celeri.sph2cart(
            segment_split.lon1.values, segment_split.lat1.values, RADIUS_EARTH
        )
        segment_split.x2, segment_split.y2, segment_split.z2 = celeri.sph2cart(
            segment_split.lon2.values, segment_split.lat2.values, RADIUS_EARTH
        )
        segment = pd.concat([segment_split, segment_whole])
        segment = order_endpoints_sphere(segment)  # TODO: WHY IS THIS FAILING???
    return segment

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
            command["sar_weight"]
        )
        sar["x"], sar["y"], sar["z"] = celeri.sph2cart(
            np.deg2rad(sar.lon), np.deg2rad(sar.lat), celeri.RADIUS_EARTH
        )
        sar["block_label"] = -1 * np.ones_like(sar.x)
    else:
        sar["dep"] = []
        sar["x"] = []
        sar["y"] = []
        sar["x"] = []
        sar["block_label"] = []
    return sar


# def create_assembly_dictionary():
#     assembly = {}
#     assembly["data"] = {}
#     assembly["sigma"] = {}
#     assembly["index"] = {}
#     return assembly


# def merge_geodetic_data(assembly, station, sar):
#     """
#     Merge GPS and InSAR data to a single assembly dictionary
#     """
#     assembly["data"]["n_stations"] = len(station)
#     assembly["data"]["n_sar"] = len(sar)
#     assembly["data"]["east_vel"] = station.east_vel.to_numpy()
#     assembly["sigma"]["east_sig"] = station.east_sig.to_numpy()
#     assembly["data"]["north_vel"] = station.north_vel.to_numpy()
#     assembly["sigma"]["north_sig"] = station.north_sig.to_numpy()
#     assembly["data"]["up_vel"] = station.up_vel.to_numpy()
#     assembly["sigma"]["up_sig"] = station.up_sig.to_numpy()
#     assembly["data"][
#         "sar_line_of_sight_change_val"
#     ] = sar.line_of_sight_change_val.to_numpy()
#     assembly["sigma"][
#         "sar_line_of_sight_change_sig"
#     ] = sar.line_of_sight_change_sig.to_numpy()
#     assembly["data"]["lon"] = np.concatenate(
#         (station.lon.to_numpy(), sar.lon.to_numpy())
#     )
#     assembly["data"]["lat"] = np.concatenate(
#         (station.lat.to_numpy(), sar.lat.to_numpy())
#     )
#     assembly["data"]["depth"] = np.concatenate(
#         (station.depth.to_numpy(), sar.depth.to_numpy())
#     )
#     assembly["data"]["x"] = np.concatenate((station.x.to_numpy(), sar.x.to_numpy()))
#     assembly["data"]["y"] = np.concatenate((station.y.to_numpy(), sar.y.to_numpy()))
#     assembly["data"]["z"] = np.concatenate((station.z.to_numpy(), sar.z.to_numpy()))
#     assembly["data"]["block_label"] = np.concatenate(
#         (station.block_label.to_numpy(), sar.block_label.to_numpy())
#     )
#     assembly["index"]["sar_coordinate_idx"] = np.arange(
#         len(station), len(station) + len(sar)
#     )  # TODO: Not sure this is correct
#     return assembly


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
            dlat_dx = (
                -z / (x ** 2 + y ** 2) ** (3 / 2) / (1 + z ** 2 / (x ** 2 + y ** 2)) * x
            )
            dlat_dy = (
                -z / (x ** 2 + y ** 2) ** (3 / 2) / (1 + z ** 2 / (x ** 2 + y ** 2)) * y
            )
            dlat_dz = (
                1 / (x ** 2 + y ** 2) ** (1 / 2) / (1 + z ** 2 / (x ** 2 + y ** 2))
            )
            dlon_dx = -y / x ** 2 / (1 + (y / x) ** 2)
            dlon_dy = 1 / x / (1 + (y / x) ** 2)
            dlon_dz = 0
            dmag_dx = x / np.sqrt(x ** 2 + y ** 2 + z ** 2)
            dmag_dy = y / np.sqrt(x ** 2 + y ** 2 + z ** 2)
            dmag_dz = z / np.sqrt(x ** 2 + y ** 2 + z ** 2)
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


def get_block_constraint_partials(block):
    """
    Partials for a priori block motion constraints.
    Essentially a set of eye(3) matrices
    """
    apriori_block_idx = np.where(block.apriori_flag.to_numpy() == 1)[0]
    operator = np.zeros((3 * len(apriori_block_idx), 3 * len(block)))
    for i in range(len(apriori_block_idx)):
        start_row = 3 * i
        start_column = 3 * apriori_block_idx[i]
        operator[start_row : start_row + 3, start_column : start_column + 3] = np.eye(3)
    return operator


def block_constraints(assembly, block, command):
    """
    Applying a priori block motion constraints
    """
    block_constraint_partials = get_block_constraint_partials(block)
    assembly.index.block_constraints_idx = np.where(block.apriori_flag == 1)[0]
    assembly.data.n_block_constraints = 3 * len(assembly.index.block_constraints_idx)
    assembly.data.block_constraints = np.zeros(block_constraint_partials.shape[0])
    assembly.sigma.block_constraints = np.zeros(block_constraint_partials.shape[0])
    if assembly.data.n_block_constraints > 0:
        (
            assembly.data.block_constraints[0::3],
            assembly.data.block_constraints[1::3],
            assembly.data.block_constraints[2::3],
        ) = celeri.sph2cart(
            np.deg2rad(block.euler_lon[assembly.index.block_constraints_idx]),
            np.deg2rad(block.euler_lat[assembly.index.block_constraints_idx]),
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

    assembly.sigma.block_constraint_weight = command["block_constraint_weight"]
    return assembly, block_constraint_partials


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


def get_fault_slip_rate_partials(segment, block):
    """
    Calculate partial derivatives for slip rate constraints
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
        # Projection on to fault strike
        segment_azimuth, _, _ = celeri.GEOID.inv(
            segment.lon1[i], segment.lat1[i], segment.lon2[i], segment.lat2[i]
        )  # TODO: Need to check this vs. matlab azimuth for consistency
        unit_x_parallel = np.cos(np.deg2rad(90 - segment_azimuth))
        unit_y_parallel = np.sin(np.deg2rad(90 - segment_azimuth))
        unit_x_perpendicular = np.sin(np.deg2rad(segment_azimuth - 90))
        unit_y_perpendicular = np.cos(np.deg2rad(segment_azimuth - 90))

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
            scale_factor = -1
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


def slip_rate_constraints(assembly, segment, block, command):
    print("Isolating slip rate constraints")
    for i in range(len(segment.lon1)):
        if segment.ss_rate_flag[i] == 1:
            "{:.2f}".format(segment.ss_rate[i])
            print(
                "- Strike-slip rate constraint on "
                + segment.name[i].strip()
                + ": rate = "
                + "{:.2f}".format(segment.ss_rate[i])
                + " (mm/yr), 1-sigma uncertainty = +/-"
                + "{:.2f}".format(segment.ss_rate_sig[i])
                + " (mm/yr)"
            )
        if segment.ds_rate_flag[i] == 1:
            print(
                "- Dip-slip rate constraint on "
                + segment.name[i].strip()
                + ": rate = "
                + "{:.2f}".format(segment.ds_rate[i])
                + " (mm/yr), 1-sigma uncertainty = +/-"
                + "{:.2f}".format(segment.ds_rate_sig[i])
                + " (mm/yr)"
            )
        if segment.ts_rate_flag[i] == 1:
            print(
                "- Tensile-slip rate constraint on "
                + segment.name[i].strip()
                + ": rate = "
                + "{:.2f}".format(segment.ts_rate[i])
                + " (mm/yr), 1-sigma uncertainty = +/-"
                + "{:.2f}".format(segment.ts_rate_sig[i])
                + " (mm/yr)"
            )

    slip_rate_constraint_partials = get_fault_slip_rate_partials(segment, block)
    slip_rate_constraint_flag = np.concatenate(
        (segment.ss_rate_flag, segment.ds_rate_flag, segment.ts_rate_flag)
    )
    assembly.index.slip_rate_constraints = np.where(slip_rate_constraint_flag == 1)[0]
    assembly.data.n_slip_rate_constraints = len(assembly.index.slip_rate_constraints)
    assembly.data.slip_rate_constraints = np.concatenate(
        (segment.ss_rate, segment.ds_rate, segment.ts_rate)
    )
    assembly.data.slip_rate_constraints = assembly.data.slip_rate_constraints[
        assembly.index.slip_rate_constraints
    ]
    assembly.sigma.slip_rate_constraints = np.concatenate(
        (segment.ss_rate_sig, segment.ds_rate_sig, segment.ts_rate_sig)
    )
    assembly.sigma.slip_rate_constraints = assembly.sigma.slip_rate_constraints[
        assembly.index.slip_rate_constraints
    ]
    slip_rate_constraint_partials = slip_rate_constraint_partials[
        assembly.index.slip_rate_constraints, :
    ]
    assembly.sigma.slip_rate_constraint_weight = command["slip_constraint_weight"]
    return assembly, slip_rate_constraint_partials


# def sphere_azimuth(lon1, lat1, lon2, lat2):
#     """
#     Calculates azimuth between sets of points on a sphere.
#     AZ = sphereazimuth(LON1, LAT1, LON2, LAT2) calculates the azimuth, AZ,
#     between points defined by coordinates (LON1, LAT1) and (LON2, LAT2).
#     The coordinate arrays must all be the same size.
#     TODO: Replace with pyproj???
#     """
#     num = np.sin(np.deg2rad(lon2 - lon1))
#     den = np.cos(np.deg2rad(lat1)) * np.tan(np.deg2rad(lat2)) - np.sin(
#         np.deg2rad(lat1)
#     ) * np.cos(np.deg2rad(lon2 - lon1))
#     az = np.rad2deg(np.arctan2(num, den))
#     return az


# def great_circle_point(lon, lat, az, dist):
#     """
#     Finds coordinates of a point along a great circle.
#     Determines the coordinates LON2, LAT2 of a point lying
#     along a great circle originating at point LON1, LAT1.
#     The azimuth of the great circle is given as AZ and the
#     angular distance between the two points is DIST. All input
#     arguments should be given in degrees, including the distance.
#     TODO: Distance in degrees???
#     TODO: Replace with pyproj???
#     """
#     # plat = asind(sind(lat).*cosd(dist) + cosd(lat).*sind(dist).*cosd(az));
#     # a = sind(dist).*sind(az).*cosd(lat);
#     # b = cosd(dist) - sind(lat).*sind(plat);

#     # if verLessThan('matlab', '8.0')
#     #     plon = lon + rad2deg(atan2(a, b));
#     # else
#     #     plon = lon + atan2d(a, b);
#     # end
#     # plon(b == 0) = lon(b == 0);
#     # plon(b == 0 & plat == 0) = lon(b == 0 & plat == 0) + 90;

#     plat = np.arcsin(
#         np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(dist))
#         + np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(dist)) * np.cos(np.deg2rad(az))
#     )
#     a = np.sin(np.deg2rad(dist)) * np.sin(np.deg2rad(az)) * np.cos(np.deg2rad(lat))
#     b = np.cos(np.deg2rad(dist)) - np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(plat))
#     plon = lon + np.rad2deg(np.arctan2(a, b))
#     plon[b == 0] = lon[b == 0]
#     plon[b == 0 and plat == 0] = lon[b == 0 and plat == 0] + 90.0
#     return plon, plat


def plot_block_labels(segment, block, station):
    plt.figure()
    plt.title("West and east labels")
    for i in range(len(segment)):
        plt.plot(
            [segment.lon1.values[i], segment.lon2.values[i]],
            [segment.lat1.values[i], segment.lat2.values[i]],
            "-k",
            linewidth=0.5,
        )
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

def test_end2end():
    """
    This doesn't actually check for correctness much at all,
    but just tests to make sure that a full block model run executes without errors.
    """
    command_file_name = './data/global/global_command.json'
    command, segment, block, meshes, station, mogi, sar = celeri.read_data(command_file_name)
    station = celeri.process_station(station, command)
    segment = celeri.process_segment(segment, command, meshes)
    sar = celeri.process_sar(sar, command)
    closure = celeri.assign_block_labels(segment, station, block, mogi, sar)
    assert(len(closure.polygon_vertices) == 306)

    assembly = Dict()
    operators = Dict()
    assembly = celeri.merge_geodetic_data(assembly, station, sar)
    assembly, operators.block_motion_constraints = celeri.block_constraints(assembly, block, command)
    assembly, operators.slip_rate_constraints = celeri.slip_rate_constraints(assembly, segment, block, command)