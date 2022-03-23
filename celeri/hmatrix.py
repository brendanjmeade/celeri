from dataclasses import dataclass
from re import T
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass()
class TreeNode:
    idx_start: int
    idx_end: int
    center: np.ndarray
    radius: float
    is_leaf: bool
    left: Optional["TreeNode"]
    right: Optional["TreeNode"]


@dataclass()
class Tree:
    """
    The tree construction re-orders the original inputs so that the points
    within each TreeNode are contained in a contiguous block of indices.
    `ordered_idxs` is the mapping from the original indices to the
    """

    ordered_idxs: np.ndarray
    pts: np.ndarray
    radii: np.ndarray
    root: TreeNode


def build_tree(pts, radii, min_pts_per_box=10):
    """
    Construct a sphere tree where each internal node of the tree represents a
    sphere containing all its child entities. The tree construction process
    receives three parameters:

    pts: the center of each element.

    radii: the radius of each element. Remember that we're dealing with spherical
           approximations to elements here instead of the triangular elements
           themselves.

    min_pts_per_box: this determines when we'll stop splitting. If a box has more
                     than min_pts_per_box elements, we keep splitting.
    """

    # We'll start with the element indices in the order that they were given to this function.
    # build_tree_node will re-order these indices at each step to enforce the rule that
    # left child indices must be less than right child indices.
    ordered_idxs = np.arange(pts.shape[0])
    # The rest of the tree construction process will be handled by the recursive function:
    # build_tree_node. The last two parameters are idx_start and idx_end. For the root of the
    # tree, we pass the full set of elements: (0, pts.shape[0])
    root = build_tree_node(pts, radii, min_pts_per_box, ordered_idxs, 0, pts.shape[0])
    return Tree(ordered_idxs, pts, radii, root)


def build_tree_node(
    all_pts, all_radii, min_pts_per_box, ordered_idxs, idx_start, idx_end
):
    # 1) Collect the relevant element data.
    # A view into the ordered_idxs array for the elements we're working on here.
    idx_view = ordered_idxs[idx_start:idx_end]
    # And the center and radius of each element.
    pts = all_pts[idx_view]
    radii = all_radii[idx_view]

    # 2) Define the bounding box.
    box_center = np.mean(pts, axis=0)
    sep = pts - box_center[None, :]
    box_axis_length = np.max(sep, axis=0)
    box_radius = np.max(np.linalg.norm(sep, axis=1) + radii)

    # 3) Build the node
    # To start with, the left and right child are absent and is_leaf=True.
    # If the node is not a leaf, we'll overwrite these below.
    node = TreeNode(
        idx_start, idx_end, box_center, box_radius, is_leaf=True, left=None, right=None
    )

    # 4) Return if the node is a leaf node.
    # If there are fewer than min_pts_per_box elements in this node, then we do not split.
    if idx_end - idx_start <= min_pts_per_box:
        return node

    # 5) If the node is not a leaf, split!
    # First, find which axis of the box is longest
    split_d = np.argmax(box_axis_length)

    # Then identify which elements are on the left hand side of the box along that axis.
    split_val = np.median(pts[:, split_d])
    is_left = pts[:, split_d] < split_val

    # 6) Re-arrange indices.
    # Since we're going to re-arrange indices, we need to save the relevant indices first.
    left_idxs = idx_view[np.where(is_left)[0]].copy()
    right_idxs = idx_view[np.where(~is_left)[0]].copy()
    n_left = left_idxs.shape[0]
    # Then assign the left side indices to the beginning of our index block
    idx_view[:n_left] = left_idxs
    # And assign the right side indices to the end of our index block.
    idx_view[n_left:] = right_idxs

    # 7) Create children!
    idx_split = idx_start + n_left
    node.is_leaf = False

    # We recursively call build_tree_node here. The key difference between the left and right
    # sides is that the left receives the index block [idx_start, idx_split) and the right
    # receives the index block [idx_split, idx_end). Thus, we've created a smaller, equivalent
    # problem.
    node.left = build_tree_node(
        all_pts, all_radii, min_pts_per_box, ordered_idxs, idx_start, idx_split
    )
    node.right = build_tree_node(
        all_pts, all_radii, min_pts_per_box, ordered_idxs, idx_split, idx_end
    )

    return node


def _traverse(obs_node, src_node, min_separation, direct_list, approx_list):
    dist = np.linalg.norm(obs_node.center - src_node.center)
    if dist > min_separation * (obs_node.radius + src_node.radius):
        # We're far away, use an approximate interaction
        approx_list.append((obs_node, src_node))
    elif obs_node.is_leaf and src_node.is_leaf:
        # If we get here, then we can't split the nodes anymore but they are
        # still close. That means we need to use a exact interaction.
        direct_list.append((obs_node, src_node))
    else:
        # We're close by, so we should recurse and use the child tree nodes.
        # But which node should we recurse with?
        split_src = (
            (obs_node.radius < src_node.radius) and not src_node.is_leaf
        ) or obs_node.is_leaf

        if split_src:
            _traverse(obs_node, src_node.left, min_separation, direct_list, approx_list)
            _traverse(
                obs_node, src_node.right, min_separation, direct_list, approx_list
            )
        else:
            _traverse(obs_node.left, src_node, min_separation, direct_list, approx_list)
            _traverse(
                obs_node.right, src_node, min_separation, direct_list, approx_list
            )


def traverse(obs_node, src_node, min_separation=1.5):
    """
    This function constructs two lists of node pairs by performing a dual tree
    traversal. This is useful for constructing an HMatrix. The first return
    value, `direct_list` contains those pairs of nodes representing blocks of a
    matrix that should not be approximated. The second return value,
    `approx_list` contains those pairs of nodes representing blocks of a matrix
    that *should* be approximated.

    The basic algorithm takes two tree nodes and:
    - if the nodes are far away, we construct an approximate matrix block.
    - if the nodes are both leaves and are close together, we construct a direct matrix block
    - if the nodes are not leaves and are close together, we recurse and check those nodes children.
    """
    direct_list = []
    approx_list = []
    _traverse(obs_node, src_node, min_separation, direct_list, approx_list)
    return direct_list, approx_list


def _check_tree(pts, radii, tree, node):
    """
    This function traverses a tree and checks to make sure that all the entities
    in each tree node and fully contained with the spherical bounds of that tree
    node.
    """
    if node is None:
        return True
    idxs = tree.ordered_idxs[node.idx_start : node.idx_end]
    dist = np.linalg.norm(pts[idxs] - node.center, axis=1) + radii[idxs]
    if np.any(dist > node.radius):
        return False
    else:
        return _check_tree(pts, radii, tree, node.left) and _check_tree(
            pts, radii, tree, node.right
        )


def plot_tree_level(node, depth, **kwargs):
    if depth == 0:
        circle = plt.Circle(tuple(node.center[:2]), node.radius, fill=False, **kwargs)
        plt.gca().add_patch(circle)
    if node.left is None or depth == 0:
        return
    else:
        plot_tree_level(node.left, depth - 1, **kwargs)
        plot_tree_level(node.right, depth - 1, **kwargs)


def plot_tree(tree):
    """
    Plots circles representing all the nodes in the tree. Each level is given a separate subplot.
    """
    plt.figure(figsize=(9, 9))
    for depth in range(9):
        plt.subplot(3, 3, 1 + depth)
        plt.title(f"level = {depth}")
        plot_tree_level(tree.root, depth, color="b", linewidth=0.5)
        plt.xlim(
            [
                tree.root.center[0] - tree.root.radius,
                tree.root.center[0] + tree.root.radius,
            ]
        )
        plt.ylim(
            [
                tree.root.center[1] - tree.root.radius,
                tree.root.center[1] + tree.root.radius,
            ]
        )
    plt.tight_layout()
    plt.show()


@dataclass()
class TempSurface:
    # ideally, these arrays are all views into other arrays without copying.
    pts: np.ndarray
    normals: np.ndarray
    quad_wts: np.ndarray
    jacobians: np.ndarray


def build_temp_surface(surf, s, e):
    return TempSurface(
        surf.pts[s:e],
        surf.normals[s:e],
        surf.quad_wts[s:e],
        surf.jacobians[s:e],
    )


"""
This is an HMatrix implementation focused on getting going quickly. 

First, terminology!!
In the HMatrix context, the term "block" refers to a contiguous rectangular
subset of a matrix. The blocks referred to here have absolutely no
connection to the tectonic "blocks" that celeri models. This is an
unfortunate collision in terminology, but I think it's better to just call
them blocks in both settings because it will be very clear from the local
context which type of block is under consideration.

Normally, HMatrices combine two core ideas:
1. **Low rank approximation** of matrix blocks to reduce memory usage and
computation.
2. **Adaptive cross approximation** (ACA) as a method to construct those low rank
blocks without constructing the entire original matrix block first. 

This implementation currently does not use ACA to construct the matrix
blocks. This is simple expediency.
"""


@dataclass
class HMatrix:
    # The prespecified tolerance. Note that the tolerance is an absolute tolerance.
    tol: float

    # The estimated frobenius norm of the matrix. This is computed during construction.
    frob_est: float

    # The obs and source sphere trees.
    obs_tree: Tree
    src_tree: Tree

    # The pairs of tree nodes that should not be approximated.
    direct_pairs: List[Tuple[TreeNode, TreeNode]]
    # The pairs of tree nodes that should be approximated.
    approx_pairs: List[Tuple[TreeNode, TreeNode]]

    # The actual matrix entries corresponding to each direct matrix block.
    direct_blocks: List[np.ndarray]
    # The approximate matrix blocks. This are tuples (U, V) containing the
    # factorized matrix representation.
    approx_blocks: List[Tuple[np.ndarray, np.ndarray]]

    # The shape of the matrix.
    shape: List[int]

    def report_compression_ratio(self):
        """
        Returns a fraction that indicates how much less memory is used than the
        corresponding original dense matrix.
        """
        simple_entries = np.prod(self.shape)

        h_entries = 0
        for (U, V) in self.approx_blocks:
            h_entries += U.size + V.size

        direct_entries = 0
        for D in self.direct_blocks:
            direct_entries += D.size

        return (h_entries + direct_entries) / simple_entries

    def dot(self, x):
        """
        Perform a matrix-vector product with the vector `x`
        """
        n_obs = self.shape[0] // 2
        y_tree = np.zeros((n_obs, 2))

        # Step 1) transform from the input index ordering to src_tree index ordering.
        x_tree = x.reshape((-1, 2))[self.src_tree.ordered_idxs]

        # Step 2) Multiply the direct blocks.
        for i, (obs_node, src_node) in enumerate(self.direct_pairs):
            x_block = x_tree[src_node.idx_start : src_node.idx_end]
            y_tree[obs_node.idx_start : obs_node.idx_end] += (
                self.direct_blocks[i].dot(x_block.ravel()).reshape((-1, 2))
            )

        # Step 3) Multiply the approx blocks.
        for i, (obs_node, src_node) in enumerate(self.approx_pairs):
            x_block = x_tree[src_node.idx_start : src_node.idx_end]
            U, V = self.approx_blocks[i]
            y_tree[obs_node.idx_start : obs_node.idx_end] += U.dot(
                V.dot(x_block.ravel())
            ).reshape((-1, 2))

        # Transform back to the original output index ordering from the obs_tree
        # index ordering.
        y_h = np.zeros((n_obs, 2))
        y_h[self.obs_tree.ordered_idxs] = y_tree
        return y_h.ravel()

    def transpose_dot(self, y):
        # TODO: Better to name rmatvec and matvec for consistency with scipy.sparse?
        # TODO: Better to name rmatvec and matvec for consistency with scipy.sparse?
        n_src = self.shape[1] // 2
        x_tree = np.zeros((n_src, 2))

        # Step 1) transform from the input index ordering to obs_tree index ordering.
        y_tree = y.reshape((-1, 2))[self.obs_tree.ordered_idxs]

        # Step 2) Multiply the direct blocks.
        for i, (obs_node, src_node) in enumerate(self.direct_pairs):
            y_block = y_tree[obs_node.idx_start : obs_node.idx_end].ravel()
            x_tree[src_node.idx_start : src_node.idx_end] += (
                y_block.ravel() @ self.direct_blocks[i]
            ).reshape((-1, 2))

        # Step 3) Multiply the approx blocks.
        for i, (obs_node, src_node) in enumerate(self.approx_pairs):
            y_block = y_tree[obs_node.idx_start : obs_node.idx_end]
            U, V = self.approx_blocks[i]
            x_tree[src_node.idx_start : src_node.idx_end] += (
                (y_block.ravel() @ U) @ V
            ).reshape((-1, 2))

        # Step 4) Transform back to the original output index ordering from the src_tree
        # index ordering.
        x_h = np.zeros((n_src, 2))
        x_h[self.src_tree.ordered_idxs] = x_tree
        return x_h.ravel()


def build_hmatrix(
    orig_matrix,
    obs_pts,
    src_pts,
    src_radii,
    tol,
    min_separation=1.5,
    min_pts_per_box=20,
):
    # Step 1) Construct the observation and source spatial trees.
    obs_tree = build_tree(
        obs_pts, np.zeros(obs_pts.shape[0]), min_pts_per_box=min_pts_per_box
    )
    # print(src_pts.shape, src_radii.shape)
    src_tree = build_tree(src_pts, src_radii, min_pts_per_box=min_pts_per_box)

    # Step 2) Traverse the trees to identify which matrix (pairs of source and
    # observation nodes) should be approximated and which should be treated
    # directly (not approximated).
    direct_pairs, approx_pairs = traverse(
        obs_tree.root, src_tree.root, min_separation=min_separation
    )

    # Step 3) Construct the approximate matrix blocks. We will:
    # - loop over the node pairs
    # - extract the corresponding block of the original matrix
    # - compute the SVD for the block.
    # - Identify how many singular values need to be retained to meet the
    #   tolerance requirements.
    # - store the factorized matrix block as a tuple (U, V)

    reshaped = orig_matrix.reshape((obs_pts.shape[0], 2, src_pts.shape[0], 2))

    # We will need the frobenius norm of the full original matrix.
    frob_est = np.linalg.norm(orig_matrix, "fro")
    approx_blocks = []
    for obs_node, src_node in approx_pairs:
        obs_idxs = obs_tree.ordered_idxs[obs_node.idx_start : obs_node.idx_end]
        src_idxs = src_tree.ordered_idxs[src_node.idx_start : src_node.idx_end]
        h_block = reshaped[obs_idxs, :][:, :, src_idxs]
        h_block = h_block.reshape((h_block.shape[0] * 2, -1))
        U, S, V = np.linalg.svd(h_block, full_matrices=False)
        block_tol = tol * (h_block.size / reshaped.size) * frob_est
        frob_K = np.sqrt(np.cumsum(S[::-1] ** 2))[::-1]

        # np.argmax finds the first entry for which the tolerance is greater
        # than the remaining frobenius norm sum. but it fails if the correct
        # matrix rank is full rank. so we first check that condition.
        if frob_K[-1] > block_tol:
            appx_rank = S.shape[0]
        else:
            appx_rank = np.argmax(frob_K < block_tol)

        approx_blocks.append((U[:, :appx_rank] * S[None, :appx_rank], V[:appx_rank]))

    # Step 4) Extract the non-approximated "direct" blocks of the original matrix.
    direct_blocks = []
    for obs_node, src_node in direct_pairs:
        obs_idxs = obs_tree.ordered_idxs[obs_node.idx_start : obs_node.idx_end]
        src_idxs = src_tree.ordered_idxs[src_node.idx_start : src_node.idx_end]
        d_block = reshaped[obs_idxs, :][:, :, src_idxs]
        direct_blocks.append(
            d_block.reshape(
                (
                    d_block.shape[0] * d_block.shape[1],
                    d_block.shape[2] * d_block.shape[3],
                )
            )
        )

    # Step 5) Build the HMatrix data object and return it.
    return HMatrix(
        tol,
        frob_est,
        obs_tree,
        src_tree,
        direct_pairs,
        approx_pairs,
        direct_blocks,
        approx_blocks,
        orig_matrix.shape,
    )


# def build_hmatrix_from_mesh_tdes(
#     mesh, station, operators_mesh, tol, min_separation=1.5, min_pts_per_box=20
# ):
#     """
#     This function translates the TDE mesh to station matrix problem into a
#     problem using more typical H-matrix inputs.

#     That is, we convert from:
#     - a celeri mesh
#     - a celeri station object

#     To:
#     - a list of observation points in numpy array with shape (N, 3)
#     - a list of source centers with shape (M, 3) and source "radii" with shape (M)

#     Note: why do the sources have "radii"? The H-matrix algorithms are much
#     simpler when described in terms of spheres rather than triangles. Since all
#     we care about is minimum distances, instead of working with triangles, we
#     can convert to bounding spheres for those triangles.
#     """
#     # This is the original dense matrix that we're going to approximate
#     M = operators_mesh.tde_to_velocities
#     # Delete vertical components.
#     M = np.delete(M, np.arange(2, M.shape[0], 3), axis=0)
#     M = np.delete(M, np.arange(2, M.shape[1], 3), axis=1)

#     # Step 1) Identify the centroid of each triangle
#     tri_centers = mesh.centroids.copy()
#     tri_pts = np.transpose(
#         np.array(
#             [
#                 [mesh.lon1, mesh.lon2, mesh.lon3],
#                 [mesh.lat1, mesh.lat2, mesh.lat3],
#                 [mesh.dep1, mesh.dep2, mesh.dep3],
#             ]
#         ),
#         (2, 1, 0),
#     )

#     # Step 2) construct an (N, 3) array with observation points (for the z
#     # coordinate, we assume z=0).
#     obs_pts = np.array([station.lon, station.lat, 0 * station.lat]).T.copy()

#     # Step 3) Convert all units to kilometers. In order to build the H-matrix
#     # distance tree, we need to have the distance units approximately the same
#     # in each dimension. This is an approximation conversion from lon/lat to km.
#     #
#     # TODO: this conversion should adapt to the location somehow. It is not
#     # important to have an exact distance conversion since these locations are
#     # only used to define the "farfield" and "nearfield" and having some slack
#     # in those definitions is fine. Ideas:
#     # - a single number per mesh. This will be approximately correct
#     # - Just use 3D distances?
#     for arr in [tri_centers, tri_pts, obs_pts]:
#         arr[..., 0] *= 85  # appx at 40 degrees north
#         arr[..., 1] *= 111  # appx at 40 degrees north

#     # Step 4) Calculate the radius of the bounding sphere for each triangle.
#     tri_radii = np.min(
#         np.linalg.norm(tri_pts - tri_centers[:, None, :], axis=2), axis=1
#     )

#     # Step 5) Construct the H-matrix now that all the locations have been converted!
#     return build_hmatrix(
#         M,
#         obs_pts,
#         tri_centers,
#         tri_radii,
#         tol,
#         min_separation=min_separation,
#         min_pts_per_box=min_pts_per_box,
#     )


def build_hmatrix_from_mesh_tdes(
    mesh, station, M, tol, min_separation=1.5, min_pts_per_box=20
):
    """
    This function translates the TDE mesh to station matrix problem into a
    problem using more typical H-matrix inputs.

    That is, we convert from:
    - a celeri mesh
    - a celeri station object

    To:
    - a list of observation points in numpy array with shape (N, 3)
    - a list of source centers with shape (M, 3) and source "radii" with shape (M)

    Note: why do the sources have "radii"? The H-matrix algorithms are much
    simpler when described in terms of spheres rather than triangles. Since all
    we care about is minimum distances, instead of working with triangles, we
    can convert to bounding spheres for those triangles.
    """
    # This is the original dense matrix that we're going to approximate
    # M = operators_mesh.tde_to_velocities
    # Delete vertical components.
    M = np.delete(M, np.arange(2, M.shape[0], 3), axis=0)
    M = np.delete(M, np.arange(2, M.shape[1], 3), axis=1)

    # Step 1) Identify the centroid of each triangle
    tri_centers = mesh.centroids.copy()
    tri_pts = np.transpose(
        np.array(
            [
                [mesh.lon1, mesh.lon2, mesh.lon3],
                [mesh.lat1, mesh.lat2, mesh.lat3],
                [mesh.dep1, mesh.dep2, mesh.dep3],
            ]
        ),
        (2, 1, 0),
    )

    # Step 2) construct an (N, 3) array with observation points (for the z
    # coordinate, we assume z=0).
    obs_pts = np.array([station.lon, station.lat, 0 * station.lat]).T.copy()

    # Step 3) Convert all units to kilometers. In order to build the H-matrix
    # distance tree, we need to have the distance units approximately the same
    # in each dimension. This is an approximation conversion from lon/lat to km.
    #
    # TODO: this conversion should adapt to the location somehow. It is not
    # important to have an exact distance conversion since these locations are
    # only used to define the "farfield" and "nearfield" and having some slack
    # in those definitions is fine. Ideas:
    # - a single number per mesh. This will be approximately correct
    # - Just use 3D distances?
    for arr in [tri_centers, tri_pts, obs_pts]:
        arr[..., 0] *= 85  # appx at 40 degrees north
        arr[..., 1] *= 111  # appx at 40 degrees north

    # Step 4) Calculate the radius of the bounding sphere for each triangle.
    tri_radii = np.min(
        np.linalg.norm(tri_pts - tri_centers[:, None, :], axis=2), axis=1
    )

    # Step 5) Construct the H-matrix now that all the locations have been converted!
    return build_hmatrix(
        M,
        obs_pts,
        tri_centers,
        tri_radii,
        tol,
        min_separation=min_separation,
        min_pts_per_box=min_pts_per_box,
    )
