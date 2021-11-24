def get_mesh_edge_elements(meshes):
    # Find indices of elements lining top, bottom, and sides of each mesh

    from ismember import ismember

    celeri.get_ordered_edge_nodes(meshes)

    for i in range(meshes):
        coords = meshes[i].meshio_object.points
        vertices = meshes[i].verts
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
        meshes[i].bot_elements = bots

        # Side elements are a set difference between all edges and tops, bottoms
        sides = np.full(len(vertices), False, dtype=bool)
        sides[side_1_in_edge_idx] = True
        sides[side_2_in_edge_idx] = True
        sides[side_3_in_edge_idx] = True
        sides[np.where(tops != 0)] = False
        sides[np.where(bots != 0)] = False
        meshes[i].side_elements = sides
