import copy
import numpy as np
import pandas as pd
import scipy.spatial
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
    station = station[
        station.flag == True
    ]  # Keep only the stations that are toggled on
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
    """
    From:
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


def polygon_area(x, y):
    """
    From: https://newbedev.com/calculate-area-of-polygon-given-x-y-coordinates
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def assign_block_labels(segment, station, block, sar):
    """
    Ben Thompson's implementation of the half edge approach to the
    block labeling problem and east/west assignment.
    """
    np_segments = np.zeros((len(segment), 2, 2))
    np_segments[:, 0, 0] = segment.lon1.to_numpy()
    np_segments[:, 1, 0] = segment.lon2.to_numpy()
    np_segments[:, 0, 1] = segment.lat1.to_numpy()
    np_segments[:, 1, 1] = segment.lat2.to_numpy()

    # De-duplicate the vertices and build an ptr_edge array
    all_vertices = np_segments.reshape((-1, 2))
    tree = scipy.spatial.KDTree(all_vertices, leafsize=1000)
    duplicates = tree.query_ball_point(all_vertices, 1e-8)

    edge_idx_to_vertex_idx = []
    vertex_idx_to_edge_idx = []
    dedup_vertices = []
    original_to_new = dict()
    for i in range(np_segments.shape[0]):
        v1_idx = duplicates[2 * i][0]
        v2_idx = duplicates[2 * i + 1][0]

        if v1_idx == 2 * i:
            original_to_new[2 * i] = len(dedup_vertices)
            dedup_vertices.append(np_segments[i][0])
            vertex_idx_to_edge_idx.append([])

        if v2_idx == 2 * i + 1:
            original_to_new[2 * i + 1] = len(dedup_vertices)
            dedup_vertices.append(np_segments[i][1])
            vertex_idx_to_edge_idx.append([])

        edge_idx = len(edge_idx_to_vertex_idx)
        edge_idx_to_vertex_idx.append(
            (original_to_new[v1_idx], original_to_new[v2_idx])
        )
        vertex_idx_to_edge_idx[original_to_new[v1_idx]].append(edge_idx)
        vertex_idx_to_edge_idx[original_to_new[v2_idx]].append(edge_idx)

    n_edges = len(edge_idx_to_vertex_idx)
    n_vertices = len(dedup_vertices)

    edge_idx_to_vertex_idx = np.array(edge_idx_to_vertex_idx, dtype=int)
    # reverse_edge_idx_to_vertex_idx = np.fliplr(edge_idx_to_vertex_idx)
    vertex_idx_to_edge_idx = np.array(vertex_idx_to_edge_idx, dtype=object)
    dedup_vertices = np.array(dedup_vertices)

    # Check that the vertices are unique up to 1e-8 now.
    new_tree = scipy.spatial.KDTree(dedup_vertices)
    assert np.all(
        [
            v[0] == i
            for i, v in enumerate(new_tree.query_ball_point(dedup_vertices, 1e-8))
        ]
    )

    def angle(v1, v2, v3):
        """
        Compute the angle between the vector (v2, v3) and (v1, v2)
        The angle is constrained to lie in [-np.pi, np.pi]
        No turn will result in an angle of 0.
        A left turn will produce a positive angle.
        A right turn will produce a negative angle.
        """
        A1 = np.arctan2(v3[1] - v2[1], v3[0] - v2[0])
        A2 = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])

        angle = A1 - A2
        if angle < -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def identify_rightward_half_edge(v1_idx, v2_idx, edge_idx):
        v1, v2 = dedup_vertices[[v1_idx, v2_idx]]

        # All the edges connected to v2 except for the edge (v1,v2)
        possible_edges = [
            e_i for e_i in vertex_idx_to_edge_idx[v2_idx] if e_i != edge_idx
        ]

        # Identify the angle for each potential edge.
        angles = []
        edge_direction = []
        for e in possible_edges:
            possible_vs = edge_idx_to_vertex_idx[e]
            direction = 1 if possible_vs[0] == v2_idx else 0
            v3 = dedup_vertices[possible_vs[direction]]
            angles.append(angle(v1, v2, v3))
            edge_direction.append(direction)

        # The right turn will have the smallest angle.
        right_idx = np.argmin(angles)

        # Return a half edge index instead of an edge index.
        return 2 * possible_edges[right_idx] + edge_direction[right_idx]

    """
    Introducing... half edges! 
    Now, the edge from v1_idx --> v2_idx will be different from the edge from v2_idx --> v1_idx.
    half edge idx 2*edge_idx+0 refers to the edge (v2_idx, v1_idx0)
    half edge idx 2*edge_idx+1 refers to the edge (v1_idx, v2_idx0)
    Thus every edge corresponds to two oppositely ordered half edges.

    Then, for each half edge, identify the next half edge that is "to the right"
    in the direction that the half edge points. That is, following the vector
    from the first vertex to the second vertex along that half edge, which half 
    edge from the connected edges is the one that turns most sharply to the right.

    From this right_half_edge data structure, it will be straightforward to follow
    the next rightwards half edge around a polygon and identify each individual 
    polygon. 
    """

    right_half_edge = np.empty(n_edges * 2, dtype=int)
    for edge_idx in range(n_edges):
        v1_idx, v2_idx = edge_idx_to_vertex_idx[edge_idx]
        right_half_edge[2 * edge_idx + 0] = identify_rightward_half_edge(
            v2_idx, v1_idx, edge_idx
        )
        right_half_edge[2 * edge_idx + 1] = identify_rightward_half_edge(
            v1_idx, v2_idx, edge_idx
        )

    def get_half_edge_vertices(half_edge_idx):
        """
        A helper function to get the vertex indices corresponding to a half edge.
        """
        v1_idx, v2_idx = edge_idx_to_vertex_idx[half_edge_idx // 2]
        if half_edge_idx % 2 == 0:
            v2_idx, v1_idx = v1_idx, v2_idx
        return v1_idx, v2_idx

    # Which polygon lies to the right of the half edge.
    right_polygon = np.full(n_edges * 2, -1, dtype=int)

    # Lists specifying which half edges lie in each polygon.
    polygons = []

    for half_edge_idx in range(2 * n_edges):
        # If this half edge is already in a polygon, skip it.
        if right_polygon[half_edge_idx] >= 0:
            continue

        # Follow a new polygon around its loop by indexing the right_half_edge array.
        polygon_idx = len(polygons)
        polygons.append([half_edge_idx])
        next_idx = right_half_edge[half_edge_idx]
        while next_idx != half_edge_idx:
            # Step 1) Check that we don't have errors!
            if next_idx in polygons[-1]:
                raise Exception(
                    "Geometry problem: unexpected loop found in polygon traversal."
                )
            if right_polygon[next_idx] != -1:
                raise Exception("Geometry problem: write a better error message here")

            # Step 2) Record the half edge
            polygons[-1].append(next_idx)

            # Step 3)
            next_idx = right_half_edge[next_idx]

        right_polygon[polygons[-1]] = polygon_idx

    # Convert the polygons array which lists half edges into a
    # polygon_vertices array which lists the vertices for each polygon.
    polygon_vertices = []
    for p in polygons:
        polygon_vertices.append([])
        for half_edge_idx in p:
            v1_idx, v2_idx = get_half_edge_vertices(half_edge_idx)
            polygon_vertices[-1].append(v2_idx)

    unprocessed_value = (
        -1
    )  # use negative number here so that it never accidentally collides with a real block label
    segment["east_labels"] = unprocessed_value
    segment["west_labels"] = unprocessed_value

    # Identify east and west labels based on the blocks assigned to each half edge.
    for current_block_label, p in enumerate(polygons):
        for half_edge_idx in p:
            v1_idx, v2_idx = get_half_edge_vertices(half_edge_idx)
            v1 = dedup_vertices[v1_idx]
            v2 = dedup_vertices[v2_idx]
            edge_vector = v2 - v1
            edge_right_normal = [edge_vector[1], -edge_vector[0]]

            # East side because right-hand normal points east
            # And west side if not!
            # Remember, because we're dealing with half edges,
            # we need to do integer division by two to get the normal edge index
            if edge_right_normal[0] > 0:
                segment["east_labels"].values[half_edge_idx // 2] = current_block_label
            else:
                segment["west_labels"].values[half_edge_idx // 2] = current_block_label

    # Check for unprocessed indices
    unprocessed_indices = np.union1d(
        np.where(segment["east_labels"] == unprocessed_value),
        np.where(segment["west_labels"] == unprocessed_value),
    )
    if len(unprocessed_indices) > 0:
        print("Found unproccessed indices")

    # Find relative areas of each block to identify an external block
    block["area_plate_carree"] = -1 * np.ones(len(block))
    for i in range(len(polygon_vertices)):
        p = polygon_vertices[i]
        vs = np.concatenate([dedup_vertices[p], dedup_vertices[p[0]][None, :]])
        block.area_plate_carree.values[i] = celeri.polygon_area(vs[:, 0], vs[:, 1])
    external_block_idx = np.argmax(block.area_plate_carree)

    # Assign block labels points to block interior points
    block["block_label"] = -1 * np.ones(len(block))
    for i in range(len(polygon_vertices)):
        if i != external_block_idx:
            p = polygon_vertices[i]
            vs = np.concatenate([dedup_vertices[p], dedup_vertices[p[0]][None, :]])
            block_polygon = celeri.inpolygon(
                block.interior_lon.to_numpy(),
                block.interior_lat.to_numpy(),
                vs[:, 0],
                vs[:, 1],
            )
            block_polygon_idx = np.where(block_polygon == True)[0]
            block.block_label.values[block_polygon_idx] = i
    # Any unassigned interior point should be put on the external block
    block.block_label.values[block.block_label == -1] = external_block_idx
    block.block_label = block.block_label.astype(int)

    # Assign block labels to GPS stations
    if not station.empty:
        station["block_label"] = -1 * np.ones(len(station))
        for i in range(len(polygon_vertices)):
            if i != external_block_idx:
                p = polygon_vertices[i]
                vs = np.concatenate([dedup_vertices[p], dedup_vertices[p[0]][None, :]])
                stations_polygon = celeri.inpolygon(
                    station.lon.to_numpy(), station.lat.to_numpy(), vs[:, 0], vs[:, 1]
                )
                stations_polygon_idx = np.where(stations_polygon == True)[0]
                station.block_label.values[stations_polygon_idx] = i
        # Any unassigned stations should be put on the external block
        station.block_label.values[station.block_label == -1] = external_block_idx
        station.block_label = station.block_label.astype(int)

    # Assign block labels to SAR locations
    if not sar.empty:
        sar["block_label"] = -1 * np.ones(len(sar))
        for i in range(len(polygon_vertices)):
            if i != external_block_idx:
                p = polygon_vertices[i]
                vs = np.concatenate([dedup_vertices[p], dedup_vertices[p[0]][None, :]])
                sar_polygon = celeri.inpolygon(
                    sar.lon.to_numpy(), sar.lat.to_numpy(), vs[:, 0], vs[:, 1]
                )
                sar_polygon_idx = np.where(sar_polygon == True)[0]
                sar.block_label.values[sar_polygon_idx] = i

        # Any unassigned stations should be put on the external block
        sar.block_label.values[sar.block_label == -1] = external_block_idx
        sar.block_label = sar.block_label.astype(int)

    return segment, station, block, sar


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
