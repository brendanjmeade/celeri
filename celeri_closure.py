from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

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
    # so if the distance is greater than that, then we subtract 360 degrees
    # from the larger of the two values. Note that this solution should work well
    # regardless of whether the longitude coordinate range is [0,360) or [-180,180)
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
class BlockClosureResult:
    # The vertices of the block geometry
    vertices: np.ndarray = None

    # An array mapping from edges to the vertices that compose the edge.
    edge_idx_to_vertex_idx: np.ndarray = None

    # An array mapping from vertices to edges. The reverse of edge_idx_to_vertex_idx
    vertex_idx_to_edge_idx: np.ndarray = None

    # For each polygon, a list of the half edges on its boundary, in order.
    polygon_edge_idxs: List[List[int]] = None

    # For each polygon, a list of the vertex indices on its boundary, in order.
    polygon_vertex_idxs: List[List[int]] = None

    # For each polygon, an array with the actual vertices
    polygon_vertices: List[np.ndarray] = None
    
    # For each polygon, an array with the actual vertices in unit sphere x,y,z coordinates 
    polygon_vertices_xyz: List[np.ndarray] = None
        
    # For each polygon, an arbitrary point that is interior to the polygon.
    polygon_interior_xyz: List[np.ndarray] = None

    def n_edges(self):
        return self.edge_idx_to_vertex_idx.shape[0]

    def n_vertices(self):
        return self.vertices.shape[0]

    def n_polygons(self):
        return len(self.polygon_edge_idxs)

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
    
    def in_polygon(self, polygon_idx, lon, lat):
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
            T = \lVert(A × B) × (C × D)\rVert
        To determine the correct sign (i.e. hemisphere) of the
        intersection, the following four values are computed:
        .. math::
        
            s_1 = ((A × B) × A) \cdot T
            
            s_2 = (B × (A × B)) \cdot T
            
            s_3 = ((C × D) × C) \cdot T
            
            s_4 = (D × (C × D)) \cdot T
            
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
        A = self.polygon_vertices_xyz[polygon_idx][:-1,:]
        B = self.polygon_vertices_xyz[polygon_idx][1:,:]
        x, y, z = unit_sph2cart(lon, lat)
        C = np.hstack((x[:,None],y[:,None],z[:,None]))
        D = self.polygon_interior_xyz[polygon_idx][None,:]

        ABX = np.cross(A, B)
        CDX = np.cross(C, D)

        T = np.cross(ABX[:, None, :], CDX[None,:, :], axis=2)
        T /= np.linalg.norm(T, axis=2)[:,:,None]
        s = np.zeros(T.shape[:2])

        s += np.sign(np.sum(np.cross(ABX, A)[:,None,:] * T, axis=2))
        s += np.sign(np.sum(np.cross(B, ABX)[:,None,:] * T, axis=2))
        s += np.sign(np.sum(np.cross(CDX, C)[None,:,:] * T, axis=2))
        s += np.sign(np.sum(np.cross(D, CDX)[None,:,:] * T, axis=2))
        s3d = s[:,:, None]

        cross = np.where(s3d == -4, -T, np.where(s3d == 4, T, np.nan))

        equals = (np.all(A[:,None] == C[None,:], axis=2) |
                  np.all(A[:,None] == D[None,:], axis=-1) |
                  np.all(B[:,None] == C[None,:], axis=-1) |
                  np.all(B[:,None] == D[None,:], axis=-1))

        intersection = np.where(equals[:,:,None], np.nan, cross)

        crossings = np.isfinite(intersection[...,0])
        n_crossings = np.sum(crossings, axis=0)

        return (n_crossings % 2) == 0
    
    def assign_points(self, lon, lat):
        # TODO: this is super slow because it's fundamentally an O(N^2) operation.
        # This can be mitigated because most points will be nowhere near a given 
        # block so we can do a bounding box test to short-circuit the in_polygon 
        # test. An explanation of how to do this is provided here:
        # https://gis.stackexchange.com/questions/17788/how-to-compute-the-bounding-box-of-multiple-layers-in-lat-long
        block_assignments = np.full(lat.shape[0], -1)
        for i in range(self.n_polygons()):
            block_assignments[self.in_polygon(i, lon, lat)] = i
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
    # Now, the edge from v1_idx --> v2_idx will be different from the edge from v2_idx --> v1_idx.
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
    closure.polygon_edge_idxs, closure.polygon_vertex_idxs = traverse_polygons(
        closure, right_half_edge
    )

    closure.polygon_vertices = []
    closure.polygon_vertices_xyz = []
    for i in range(closure.n_polygons()):
        p = closure.polygon_vertex_idxs[i]
        vs = np.concatenate([closure.vertices[p], closure.vertices[p[0]][None, :]])
        closure.polygon_vertices.append(vs)
        
        x, y, z = unit_sph2cart(vs[:,0], vs[:,1])
        xyz = np.hstack((x[:,None],y[:,None],z[:,None]))
        closure.polygon_vertices_xyz.append(xyz)
        
    closure.polygon_interior_xyz = []
    for p in closure.polygon_vertices:
        for i in range(p.shape[0]):
            dx = p[i+1,0] - p[i+1,0]
            
            # Make sure we skip any meridian crossing edges. 
            if -180 < dx < 180:
                midpt = (p[i+1, :] + p[i, :]) / 2
                edge_vector = p[i+1, :] - p[i, :]
                edge_right_normal = np.array([edge_vector[1], -edge_vector[0]])
                
                # Offset only a small amount into the interior to avoid stepping 
                # back across a different edge into the exterior.
                interior_pt = midpt + edge_right_normal * 0.01
                # Stop after we've found an acceptable interior point.
                break
                
        x, y, z = unit_sph2cart(interior_pt[0], interior_pt[1])
        closure.polygon_interior_xyz.append(np.array([x, y, z]))

    return closure

def unit_sph2cart(lon, lat):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = np.cos(lon_rad) * np.cos(lat_rad)
    y = np.sin(lon_rad) * np.cos(lat_rad)
    z = np.sin(lat_rad)
    return x, y, z
    
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
    
    # A spherical polygon defines two enclosed regions. We specify which is the "interior"
    # via this point. 
    polygon_interior_pts = []

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

    return polygon_edge_idxs, polygon_vertex_idxs


def get_segment_labels(closure):
    # use negative number as the default value so that it never accidentally collides with
    # a real block label which will all have values >= 0
    segment_labels = np.full((closure.n_edges(), 2), -1)

    # Identify east and west labels based on the blocks assigned to each half edge.
    for current_block_label, p in enumerate(closure.polygon_edge_idxs):
        for half_edge_idx in p:
            v1_idx, v2_idx = closure.get_half_edge_vertices(half_edge_idx)
            v1 = closure.vertices[v1_idx]
            v2 = closure.vertices[v2_idx]
            edge_vector = v2 - v1
            edge_right_normal = [edge_vector[1], -edge_vector[0]]

            # East side because right-hand normal points east
            # And west side if not!
            # Remember, because we're dealing with half edges,
            # we need to do integer division by two to get the normal edge index
            if edge_right_normal[0] > 0:
                segment_labels[half_edge_idx // 2, 1] = current_block_label
            elif edge_right_normal[0] == 0:
                raise ValueError(
                    "Segments lying precisely on lines of latitude are not yet supported."
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


def test_closure():
    # First test a simple two triangle geometry.
    np_segments = np.array(
        [
            [[0, 0], [9, 1]],
            [[9, 1], [1, 10]],
            [[9, 1], [10, 11]],
            [[10, 11], [1, 10]],
            [[1, 10], [0, 0]],
        ],
        dtype=np.float64,
    )

    closure = run_block_closure(np_segments)
    labels = get_segment_labels(closure)

    correct_labels = np.array([[0, 1], [0, 2], [2, 1], [1, 2], [1, 0]])
    np.testing.assert_array_equal(labels, correct_labels)

    # Then shift one of the points to lie on the other side of the meridian.
    # Instead of (0,0), use (359,0). The labels should be the same because
    # this change in position doesn't change the topology of the blocks
    np_segments_meridian = np_segments.copy()
    np_segments[0, 0] = [359.9, 0]
    np_segments[4, 1] = [359.9, 0]

    closure_meridian = run_block_closure(np_segments_meridian)
    labels_meridian = get_segment_labels(closure_meridian)
    np.testing.assert_array_equal(labels_meridian, labels)

