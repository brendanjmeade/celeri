def ordered_mesh_edges(meshes):
    import numpy as np
    
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
            idx = np.where((edge_nodes==meshes[i].ordered_edge_nodes[j-1, 1])) # Edge node indices the same as previous row, second column
            next_idx = np.where(idx[0][:] != last_row) # One of those indices is the last row itself. Find the other row index
            next_row = idx[0][next_idx] # Index of the next ordered row
            next_col = idx[1][next_idx] # Index of the next ordered column (1 or 2)
            if next_col == 1:
                next_col_ord = [1, 0] # Flip edge ordering
            else:
                next_col_ord = [0, 1]
            meshes[i].ordered_edge_nodes[j, :] = edge_nodes[next_row, next_col_ord] 
            last_row = next_row # Update last_row so that it's excluded in the next iteration
    return meshes