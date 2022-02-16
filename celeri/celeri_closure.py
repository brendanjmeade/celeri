from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from spherical_geometry.polygon import SingleSphericalPolygon
from spherical_geometry.great_circle_arc import intersection

from .celeri_util import sph2cart


def angle_between_vectors(v1, v2, v3):
    """
    Compute the angle between the vector (v2, v3) and (v1, v2)
    The angle is constrained to lie in [-np.pi, np.pi]

    No turn will result in an angle of 0.
    A left turn will produce a positive angle.
    A right turn will produce a negative angle.

    The function is designed for units in (longitude degrees, latitude degrees)
    and will handle the meridian. If you have units in another coordinate system,
    this function will cause problems.
    """

    # *Meridian handling*
    # The furthest longitudinal distance between any two points is 180 degrees,
    # so if the distance is greater than that, then we subtract 360 degrees.
    # Note that this solution should work well regardless of whether the longitude
    # coordinate range is [0,360) or [-180,180)
    A1x = v3[0] - v2[0]
    if A1x > 180:
        A1x -= 360
    if A1x < -180:
        A1x += 360

    A2x = v2[0] - v1[0]
    if A2x > 180:
        A2x -= 360
    if A2x < -180:
        A2x += 360

    A1 = np.arctan2(v3[1] - v2[1], A1x)
    A2 = np.arctan2(v2[1] - v1[1], A2x)

    angle = A1 - A2
    if angle < -np.pi:
        angle += 2 * np.pi
    elif angle > np.pi:
        angle -= 2 * np.pi
    return angle


@dataclass()
class BoundingBox:
    """
    A bounding box on a sphere can be defined by the minimum and maximum latitude
    and longitude.

    *Inverse longitude*:
    In the case where the box crosses the meridian, we specify the inverse region
    of longitude. As an example, suppose the box spans from 355 deg longitude to
    5 degrees longitude, we instead store the [5,355] range of longitude and when
    we do queries to identify if a point is inside the bounding box we exclude rather
    than include values between min_lon and max_lon.
    """

    min_lon: float
    max_lon: float
    inverse_lon: bool
    min_lat: float
    max_lat: float

    @classmethod
    def from_polygon(cls, vertices):
        lon_interval, inverse = find_longitude_interval(vertices[:, 0])
        return BoundingBox(
            min_lon=lon_interval[0],
            max_lon=lon_interval[1],
            inverse_lon=inverse,
            min_lat=np.min(vertices[:, 1]),
            max_lat=np.max(vertices[:, 1]),
        )

    def contains(self, lon, lat):
        in_lat = (self.min_lat <= lat) & (lat <= self.max_lat)
        if not self.inverse_lon:
            # If the polygon spans more than 180 degrees, don't trust the
            # bounding box.  The possible failure mode here is that a
            # circumpolar block can exclude points that are south of its
            # southernmost edge.
            if self.max_lon - self.min_lon > 180:
                return np.ones_like(lon, dtype=bool)
            in_lon = (self.min_lon <= lon) & (lon <= self.max_lon)
        else:
            # Same as above, but for an inverse min/max lon range, having
            # max-min longitude < 180 is equivalent to having the true extent of
            # the block greater than 180 degrees.
            if self.max_lon - self.min_lon < 180:
                return np.ones_like(lon, dtype=bool)
            in_lon = (self.min_lon >= lon) | (lon >= self.max_lon)
        return in_lat & in_lon


def find_longitude_interval(lon):
    """
    Given a list of polygon longitude values, we want to identify the maximum
    and minimum longitude for that polygon. On its face, that seems like a
    simple (min, max), but the meridian means that the problem is not quite
    that simple. First, we need to split all the intervals across the meridian.
    Then, we combine the resulting intervals.

    After combining the intervals, there should be either one or two intervals.

     - If there is one interval, that polygon does not cross the meridian and
       we return the actual (min, max) longitude.

     - If there are two intervals, the polygon crosses the meridian and there
       is a (X, 360) interval and a (0, Y) interval. As a result, we instead
       return the inverse longitude interval of (Y, X) and specify the inverse
       = True return value.
    """

    # Step 1) Split intervals.
    intervals = []
    for i in range(lon.shape[0] - 1):
        s1 = lon[i]
        s2 = lon[i + 1]

        # If the longitudes are separated by more than 180 degrees, then
        # this is a meridian crossing segment where s1 is <180 and s2 is
        # >180 degrees. So use (s1, 0) and (360, s2).
        if s2 - s1 > 180:
            intervals.append((0, s1))
            intervals.append((s2, 360))
        # Similiarly, separated by -180 degrees suggests that s1 is >180
        # and s2 is <180. So use (s1,360), (0,s2)
        elif s2 - s1 < -180:
            intervals.append((s1, 360))
            intervals.append((0, s2))
        else:
            intervals.append((s1, s2))
    intervals = np.array([(s1, s2) if s1 < s2 else (s2, s1) for s1, s2 in intervals])

    # Step 2) Combine intervals
    # Fun classic intro algorithms problem: how to combine intervals in O(n log(n))?
    # Sort them by the first value, and then combine adjacent intervals.
    sorted_intervals = intervals[intervals[:, 0].argsort()]
    combined_intervals = []
    cur_interval = sorted_intervals[0]
    for next_interval in sorted_intervals[1:]:
        if next_interval[0] > cur_interval[1]:
            combined_intervals.append(cur_interval)
            cur_interval = next_interval
        else:
            cur_interval[1] = max(cur_interval[1], next_interval[1])
    combined_intervals.append(cur_interval)

    # Step 3) Determine if we want the specified interval or the inverse of the
    # meridian-split interval.
    if len(combined_intervals) == 1:
        final_interval = combined_intervals[0]
        inverse = False
    elif len(combined_intervals) == 2:
        if combined_intervals[0][0] == 0:
            final_interval = [combined_intervals[0][1], combined_intervals[1][0]]
        else:
            final_interval = [combined_intervals[1][0], combined_intervals[0][1]]
        inverse = True
    else:
        raise Exception(
            "More than two longitude intervals identified in "
            "find_longitude_intervals. This is an unexpected and "
            "surprising error that suggests a malformed polygon."
        )
    return final_interval, inverse


@dataclass()
class Polygon:
    # The half edges on the boundary, in rightwards order.
    edge_idxs: np.ndarray

    # The vertex indices on the boundary, in rightwards order.
    vertex_idxs: np.ndarray

    # The actual vertices
    vertices: np.ndarray

    # The actual vertices in unit sphere x,y,z coordinates
    vertices_xyz: np.ndarray = None

    # an arbitrary point that is interior to the polygon

    interior_xyz: np.ndarray = None

    # A bounding box defining the minimum and maximum lon/lat used for
    # a fast shortcircuiting of the in-polygon test.
    bounds: BoundingBox = None

    # The spherical_geometry has some useful tools so we store a reference to
    # a spherical_geometry.polygon.SingleSphericalPolygon in case we need
    # those methods.
    _sg_polygon: SingleSphericalPolygon = None

    # Angle in steradians. A full sphere is 4*pi, half sphere is 2*pi.
    area_steradians: float = None

    def __init__(self, edge_idxs, vertex_idxs, vs):
        self.edge_idxs = edge_idxs
        self.vertex_idxs = vertex_idxs
        self.vertices = vs

        x, y, z = sph2cart(vs[:, 0], vs[:, 1], 1.0)
        xyz = np.hstack((x[:, None], y[:, None], z[:, None]))

        for i in range(vs.shape[0] - 1):
            dx = vs[i + 1, 0] - vs[i, 0]

            # Make sure we skip any meridian crossing edges.
            if -180 < dx < 180:
                midpt = (vs[i + 1, :] + vs[i, :]) / 2
                edge_vector = vs[i + 1, :] - vs[i, :]
                edge_right_normal = np.array([edge_vector[1], -edge_vector[0]])

                # Offset into the interior. Note that edge_right_normal is not
                # normalized so this scales the distance with respect to the
                # length of the current edge.
                interior_pt = midpt + edge_right_normal / 4

                # Check to make sure the arc from midpt to interior_pt does not
                # intersect any other edges of this polygon
                has_intersection = False
                for j in range(vs.shape[0] - 1):
                    if i == j:
                        continue
                    Ipt = intersection(
                        sph2cart(*midpt, 1.0),
                        sph2cart(*interior_pt, 1.0),
                        sph2cart(*vs[j], 1.0),
                        sph2cart(*vs[j + 1], 1.0),
                    )
                    # intersection returns nans if the two great circles do not
                    # intersect
                    if not np.all(np.isnan(Ipt)):
                        has_intersection = True

                if has_intersection:
                    # Look at the next potential interior point.
                    continue
                else:
                    # Stop after we've found an acceptable interior point.
                    if i == vs.shape[0] - 2:
                        raise ValueError("Failed to find a valid interior point for this polygon.")
                    break

        x, y, z = sph2cart(interior_pt[0], interior_pt[1], 1.0)

        self.vertices = vs
        self.interior = interior_pt
        self.vertices_xyz = xyz
        self.interior_xyz = np.array([x, y, z])
        self.bounds = BoundingBox.from_polygon(self.vertices)
        self._sg_polygon = SingleSphericalPolygon(self.vertices_xyz, self.interior_xyz)
        self.area_steradians = self._sg_polygon.area()

    def contains_point(self, lon, lat):
        """
        Returns whether each point specified by (lon, lat) is within the
        spherical polygon defined by polygon_idx.

        The intermediate calculation uses a great circle intersection test.
        An explanation of this calculation is copied from. The source code is
        modified from the same source. The primary modification is to vectorize
        over a list of points rather than a single test point.

        https://github.com/spacetelescope/spherical_geometry/blob/e00f4ef619eb2871b305eded2a537a95c858b001/spherical_geometry/great_circle_arc.py#L91

        A, B : (*x*, *y*, *z*) Nx3 arrays of triples
            Endpoints of the first great circle arcs.

        C, D : (*x*, *y*, *z*) Nx3 arrays of triples
            Endpoints of the second great circle arcs.

        Notes
        -----
        The basic intersection is computed using linear algebra as follows
        [1]_:
        .. math::
            T = \\lVert(A × B) × (C × D)\rVert
        To determine the correct sign (i.e. hemisphere) of the
        intersection, the following four values are computed:
        .. math::

            s_1 = ((A × B) × A) \\cdot T

            s_2 = (B × (A × B)) \\cdot T

            s_3 = ((C × D) × C) \\cdot T

            s_4 = (D × (C × D)) \\cdot T

        For :math:`s_n`, if all positive :math:`T` is returned as-is.  If
        all negative, :math:`T` is multiplied by :math:`-1`.  Otherwise
        the intersection does not exist and is undefined.

        References
        ----------

        .. [1] Method explained in an `e-mail
            <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_
            by Roger Stafford.

        http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271
        """

        # TODO: Currently, the bounding box test is turned off!
        # TODO: Currently, the bounding box test is turned off!
        # TODO: Currently, the bounding box test is turned off!
        # Start by throwing out points that aren't in the polygon's bounding box.
        # This is purely an optimization and is not necessary for correctness.
        # The bounding box approximation is only valid for a spherical polygon
        # that takes up less than half the sphere.
        # if self.area_steradians < 2 * np.pi:
        #     is_in_bounds = self.bounds.contains(lon, lat)
        # else:
        #     is_in_bounds = np.ones(lon.shape[0], dtype=bool)
        is_in_bounds = np.ones(lon.shape[0], dtype=bool)
        in_bounds_lon = lon[is_in_bounds]
        in_bounds_lat = lat[is_in_bounds]

        A = self.vertices_xyz[:-1, :]
        B = self.vertices_xyz[1:, :]
        x, y, z = sph2cart(in_bounds_lon, in_bounds_lat, 1.0)
        C = np.hstack((x[:, None], y[:, None], z[:, None]))
        D = self.interior_xyz[None, :]

        ABX = np.cross(A, B)
        CDX = np.cross(C, D)

        T = np.cross(ABX[:, None, :], CDX[None, :, :], axis=2)
        T /= np.linalg.norm(T, axis=2)[:, :, None]
        s = np.zeros(T.shape[:2])

        s += np.sign(np.sum(np.cross(ABX, A)[:, None, :] * T, axis=2))
        s += np.sign(np.sum(np.cross(B, ABX)[:, None, :] * T, axis=2))
        s += np.sign(np.sum(np.cross(CDX, C)[None, :, :] * T, axis=2))
        s += np.sign(np.sum(np.cross(D, CDX)[None, :, :] * T, axis=2))
        s3d = s[:, :, None]

        cross = np.where(s3d == -4, -T, np.where(s3d == 4, T, np.nan))

        equals = (
            np.all(A[:, None] == C[None, :], axis=2)
            | np.all(A[:, None] == D[None, :], axis=-1)
            | np.all(B[:, None] == C[None, :], axis=-1)
            | np.all(B[:, None] == D[None, :], axis=-1)
        )

        intersection = np.where(equals[:, :, None], np.nan, cross)

        crossings = np.isfinite(intersection[..., 0])
        n_crossings = np.sum(crossings, axis=0)

        # The final result is a combination of the result from the bounds test
        # and the more precise test.
        contained = np.zeros(lat.shape[0], dtype=bool)
        contained[is_in_bounds] = (n_crossings % 2) == 0
        return contained


@dataclass()
class BlockClosureResult:
    # The vertices of the block geometry
    vertices: np.ndarray = None

    # An array mapping from edges to the vertices that compose the edge.
    edge_idx_to_vertex_idx: np.ndarray = None

    # An array mapping from vertices to edges. The reverse of edge_idx_to_vertex_idx
    vertex_idx_to_edge_idx: np.ndarray = None

    # The polygon blocks!
    polygons: List[Polygon] = None

    def n_edges(self):
        return self.edge_idx_to_vertex_idx.shape[0]

    def n_vertices(self):
        return self.vertices.shape[0]

    def n_polygons(self):
        return len(self.polygons)

    def identify_rightward_half_edge(self, v1_idx, v2_idx, edge_idx):
        v1, v2 = self.vertices[[v1_idx, v2_idx]]

        # All the edges connected to v2 except for the edge (v1,v2)
        possible_edges = [
            e_i for e_i in self.vertex_idx_to_edge_idx[v2_idx] if e_i != edge_idx
        ]

        # Identify the angle for each potential edge.
        angles = []
        edge_direction = []
        for e in possible_edges:
            possible_vs = self.edge_idx_to_vertex_idx[e]
            direction = 1 if possible_vs[0] == v2_idx else 0
            v3 = self.vertices[possible_vs[direction]]
            angles.append(angle_between_vectors(v1, v2, v3))
            edge_direction.append(direction)

        # The right turn will have the smallest angle.
        right_idx = np.argmin(angles)

        # Return a half edge index instead of an edge index.
        return 2 * possible_edges[right_idx] + edge_direction[right_idx]

    def get_half_edge_vertices(self, half_edge_idx):
        v1_idx, v2_idx = self.edge_idx_to_vertex_idx[half_edge_idx // 2]
        if half_edge_idx % 2 == 0:
            v2_idx, v1_idx = v1_idx, v2_idx
        return v1_idx, v2_idx

    def assign_points(self, lon, lat):
        block_assignments = np.full(lat.shape[0], -1)
        for i in range(self.n_polygons()):
            block_assignments[self.polygons[i].contains_point(lon, lat)] = i
        return block_assignments


def run_block_closure(np_segments):
    """
    Ben Thompson's implementation of the half edge approach to the
    block labeling problem and east/west assignment.

    np_segments is expected to be a (N, 2, 2) shaped array specifying the
    two (lon, lat) endpoints for each of N segments.
    """

    # STEP 1) Build a graph! De-duplicate vertices and build an array relating edge
    #         indices to vertex indices and vice-versa.
    closure = decompose_segments_into_graph(np_segments)

    # Introducing... half edges!
    # Now, the edge from v1_idx --> v2_idx will be different from the edge from
    # v2_idx --> v1_idx.
    # half edge idx 2*edge_idx+0 refers to the edge (v2_idx, v1_idx0)
    # half edge idx 2*edge_idx+1 refers to the edge (v1_idx, v2_idx0)
    # Thus every edge corresponds to two oppositely ordered half edges.
    #
    # Then, for each half edge, identify the next half edge that is "to the right"
    # in the direction that the half edge points. That is, following the vector
    # from the first vertex to the second vertex along that half edge, which half
    # edge from the connected edges is the one that turns most sharply to the right.
    #
    # From this right_half_edge data structure, it will be straightforward to follow
    # the next rightwards half edge around a polygon and identify each individual
    # polygon.
    right_half_edge = np.empty(closure.n_edges() * 2, dtype=int)
    for edge_idx in range(closure.n_edges()):
        v1_idx, v2_idx = closure.edge_idx_to_vertex_idx[edge_idx]
        right_half_edge[2 * edge_idx + 0] = closure.identify_rightward_half_edge(
            v2_idx, v1_idx, edge_idx
        )
        right_half_edge[2 * edge_idx + 1] = closure.identify_rightward_half_edge(
            v1_idx, v2_idx, edge_idx
        )

    # Lists specifying which half edges lie in each polygon.
    closure.polygons = traverse_polygons(closure, right_half_edge)

    return closure


def decompose_segments_into_graph(np_segments):
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

    edge_idx_to_vertex_idx = np.array(edge_idx_to_vertex_idx, dtype=int)
    vertex_idx_to_edge_idx = np.array(vertex_idx_to_edge_idx, dtype=object)
    dedup_vertices = np.array(dedup_vertices)

    return BlockClosureResult(
        vertices=dedup_vertices,
        edge_idx_to_vertex_idx=edge_idx_to_vertex_idx,
        vertex_idx_to_edge_idx=vertex_idx_to_edge_idx,
    )


def traverse_polygons(closure, right_half_edge):

    # Which polygon lies to the right of the half edge.
    right_polygon = np.full(closure.n_edges() * 2, -1, dtype=int)

    polygon_edge_idxs = []

    for half_edge_idx in range(2 * closure.n_edges()):
        # If this half edge is already in a polygon, skip it.
        if right_polygon[half_edge_idx] >= 0:
            continue

        # Follow a new polygon around its loop by indexing the right_half_edge array.
        polygon_idx = len(polygon_edge_idxs)
        polygon_edge_idxs.append([half_edge_idx])

        next_idx = right_half_edge[half_edge_idx]
        while next_idx != half_edge_idx:
            # Step 1) Check that we don't have errors!
            if next_idx in polygon_edge_idxs[-1]:
                raise Exception(
                    "Geometry problem: unexpected loop found in polygon traversal."
                )
            if right_polygon[next_idx] != -1:
                raise Exception("Geometry problem: write a better error message here")

            # Step 2) Record the half edge
            polygon_edge_idxs[-1].append(next_idx)

            # Step 3)
            next_idx = right_half_edge[next_idx]

        right_polygon[polygon_edge_idxs[-1]] = polygon_idx

    polygon_vertex_idxs = []
    for p in polygon_edge_idxs:
        polygon_vertex_idxs.append([])
        for half_edge_idx in p:
            v1_idx, v2_idx = closure.edge_idx_to_vertex_idx[half_edge_idx // 2]
            if half_edge_idx % 2 == 0:
                v2_idx, v1_idx = v1_idx, v2_idx
            polygon_vertex_idxs[-1].append(v2_idx)

    polygons = []
    for i in range(len(polygon_edge_idxs)):
        vs = np.concatenate(
            [
                closure.vertices[polygon_vertex_idxs[i]],
                closure.vertices[polygon_vertex_idxs[i][0]][None, :],
            ]
        )
        polygons.append(Polygon(polygon_edge_idxs[i], polygon_vertex_idxs[i], vs))

    return polygons


def get_right_normal(p1, p2):
    dx = p2[0] - p1[0]
    if dx > 180:
        dx -= 360
    elif dx < -180:
        dx += 360

    return [p2[1] - p1[1], -dx]


def get_segment_labels(closure):
    # use negative number as the default value so that it never accidentally
    # collides with a real block label which will all have values >= 0
    segment_labels = np.full((closure.n_edges(), 2), -1)

    # Identify east and west labels based on the blocks assigned to each half edge.
    for current_block_label, p in enumerate(closure.polygons):
        for half_edge_idx in p.edge_idxs:
            p1_idx, p2_idx = closure.get_half_edge_vertices(half_edge_idx)
            edge_right_normal = get_right_normal(
                closure.vertices[p1_idx], closure.vertices[p2_idx]
            )

            # East side because right-hand normal points east
            # And west side if not!
            # Remember, because we're dealing with half edges,
            # we need to do integer division by two to get the normal edge index
            if edge_right_normal[0] > 0:
                segment_labels[half_edge_idx // 2, 1] = current_block_label
            elif edge_right_normal[0] == 0:
                raise ValueError(
                    "Segments lying precisely on lines of latitude are not yet "
                    "supported."
                )
            else:
                segment_labels[half_edge_idx // 2, 0] = current_block_label

    return segment_labels


def plot_segment_labels(segments, labels):
    plt.figure()
    for i in range(segments.shape[0]):
        middle = np.mean(segments[i], axis=0)
        plt.plot(
            segments[i, :, 0],
            segments[i, :, 1],
            "-k",
            linewidth=0.5,
        )
        plt.text(
            middle[0],
            middle[1],
            str(labels[i, 0]) + "," + str(labels[i, 1]),
            fontsize=8,
            color="m",
            horizontalalignment="center",
            verticalalignment="center",
        )
    plt.show()
