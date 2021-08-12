import copy
import numpy as np
import pandas as pd
from numpy.lib.shape_base import split
from pyproj import Geod
from matplotlib import path

import celeri


RADIUS_EARTH = np.float64(6371e3)  # m
GEOID = Geod(ellps="WGS84")


def sph2cart(lon, lat, radius):
    x = radius * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = radius * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = radius * np.sin(np.deg2rad(lat))
    return x, y, z


def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


def process_station(station, command):
    if command["unit_sigmas"] == "yes":  # Assign unit uncertainties, if requested
        station.east_sig = np.ones_like(station.east_sig)
        station.north_sig = np.ones_like(station.north_sig)
        station.up_sig = np.ones_like(station.up_sig)

    station["dep"] = np.zeros_like(
        station.lon
    )  # Add a "dep" field of all zeros, to be used with project_tri_coords
    station["x"], station["y"], station["z"] = celeri.sph2cart(
        station.lon, station.lat, celeri.RADIUS_EARTH
    )
    station = station[station.tog == True]  # Keep only the stations that are toggled on
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
    # segment_copy = copy.deepcopy(segment)
    # x1, y1, z1 = celeri.sph2cart(segment.lon1, segment.lat1, 1)
    # x2, y2, z2 = celeri.sph2cart(segment.lon1, segment.lat2, 1)
    # for i in range(x1.size):
    #     cross_product = np.cross([x1[i], y1[i], z1[i]], [x2[i], y2[i], z2[i]])
    #     if cross_product[2] <= 0:
    #         print(i, "Reordering endpoints")
    #         segment_copy.lon1.values[i] = segment.lon2.values[i]
    #         segment_copy.lat1.values[i] = segment.lat2.values[i]
    #         segment_copy.lon2.values[i] = segment.lon1.values[i]
    #         segment_copy.lat2.values[i] = segment.lat1.values[i]
    # return segment_copy
    segment_copy = copy.deepcopy(segment)
    for i in range(len(segment)):
        if segment.lon1[i] > segment.lon2[i]:
            print(i, "Reordering endpoints for segment ", segment.name[i].strip())
            segment_copy.lon1.values[i] = segment.lon2.values[i]
            segment_copy.lat1.values[i] = segment.lat2.values[i]
            segment_copy.lon2.values[i] = segment.lon1.values[i]
            segment_copy.lat2.values[i] = segment.lat1.values[i]
    return segment_copy


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
        azx = np.arctan(-1.0 / azx)  # TODO: FIX THIS VARIABLE NAME
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


def process_segment(segment, command):
    segment = celeri.order_endpoints_sphere(segment)
    segment["x1"], segment["y1"], segment["z1"] = celeri.sph2cart(
        segment.lon1, segment.lat1, celeri.RADIUS_EARTH
    )
    segment["x2"], segment["y2"], segment["z2"] = celeri.sph2cart(
        segment.lon2, segment.lat2, celeri.RADIUS_EARTH
    )
    segment["mid_lon_plate_carree"] = (segment.lon1.values + segment.lon2.values) / 2.0
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
    # segment.locking_depth.values = PatchLDtoggle(segment.locking_depth, segment.patch_file_name, segment.patch_flag, Command.patchFileNames) % Set locking depth to zero on segments that are associated with patches # TODO: Write this after patches are read in.
    segment = celeri.segment_centroids(segment)
    return segment


def inpolygon(xq, yq, xv, yv):
    """From:
    https://stackoverflow.com/questions/31542843/inpolygon-for-python-examples-of-matplotlib-path-path-contains-points-method
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


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

        # Replicate split segment properties and assign new endpoints
        segment_whole = copy.copy(segment[~prime_meridian_cross])
        segment_split = copy.copy(segment[prime_meridian_cross])
        segment_split = pd.concat([segment_split, segment_split])

        # Insert the split coordinates
        segment_split.lon2.values[0 : len(split_idx)] = 360.0
        segment_split.lat2.values[0 : len(split_idx)] = split_lat
        segment_split.lon1.values[len(split_idx) + 1 : -1] = 0.0
        segment_split.lat1.values[len(split_idx) + 1 : -1] = split_lat
        # [segment_split.midLon, segment_split.midLat] = segmentmidpoint(split.lon1, split.lat1, split.lon2, split.lat2);
        segment_split.x1, segment_split.y1, segment_split.z1 = celeri.sph2cart(
            segment_split.lon1.values, segment_split.lat1.values, RADIUS_EARTH
        )
        segment_split.x2, segment_split.y2, segment_split.z2 = celeri.sph2cart(
            segment_split.lon2.values, segment_split.lat2.values, RADIUS_EARTH
        )
        segment = pd.concat([segment_split, segment_whole])
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
    lat = np.atan(
        np.tan(lat1) * np.sin(lon - lon2) / np.sin(lon1 - lon2)
        - np.tan(lat2) * np.sin(lon - lon1) / np.sin(lon1 - lon2)
    )
    return


def sphere_azimuth(lon1, lat1, lon2, lat2):
    """
    Calculates azimuth between sets of points on a sphere.
    AZ = sphereazimuth(LON1, LAT1, LON2, LAT2) calculates the azimuth, AZ,
    between points defined by coordinates (LON1, LAT1) and (LON2, LAT2).
    The coordinate arrays must all be the same size.
    """
    num = np.sin(np.deg2rad(lon2 - lon1))
    den = np.cos(np.deg2rad(lat1)) * np.tan(np.deg2rad(lat2)) - np.sin(
        np.deg2rad(lat1)
    ) * np.cos(np.deg2rad(lon2 - lon1))
    az = np.rad2deg(np.arctan2(num, den))
    return az
