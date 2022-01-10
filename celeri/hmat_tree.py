from dataclasses import dataclass
from typing import Optional

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
    ordered_idxs: np.ndarray
    pts: np.ndarray
    radii: np.ndarray
    root: TreeNode


def build_tree(pts, radii, min_pts_per_box=10):
    # The tree construction process receives three parameters:
    #
    # pts: the center of each element.
    #
    # radii: the radius of each element. Remember that we're dealing with spherical
    #        approximations to elements here instead of the triangular elements
    #        themselves.
    #
    # min_pts_per_box: this determines when we'll stop splitting. If a box has more
    #                  than min_pts_per_box elements, we keep splitting.

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


def _traverse(obs_node, src_node, safety_factor, direct_list, approx_list):
    dist = np.linalg.norm(obs_node.center - src_node.center)
    if dist > safety_factor * (obs_node.radius + src_node.radius):
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
            _traverse(obs_node, src_node.left, safety_factor, direct_list, approx_list)
            _traverse(obs_node, src_node.right, safety_factor, direct_list, approx_list)
        else:
            _traverse(obs_node.left, src_node, safety_factor, direct_list, approx_list)
            _traverse(obs_node.right, src_node, safety_factor, direct_list, approx_list)


def traverse(obs_node, src_node, safety_factor=1.5):
    direct_list = []
    approx_list = []
    _traverse(obs_node, src_node, safety_factor, direct_list, approx_list)
    return direct_list, approx_list


def check_tree(pts, radii, tree, node):
    if node is None:
        return True
    idxs = tree.ordered_idxs[node.idx_start : node.idx_end]
    dist = np.linalg.norm(pts[idxs] - node.center, axis=1) + radii[idxs]
    if np.any(dist > node.radius):
        return False
    else:
        return check_tree(pts, radii, tree, node.left) and check_tree(
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
