def plot_mesh(meshes, fill_value, ax):
    import matplotlib.collections
	for i in range(meshes):
		x_coords = meshes[i].meshio_object.points[:,0]
		y_coords = meshes[i].meshio_object.points[:,1]
		vertex_array = np.asarray(meshes[i].verts)

		if not ax: ax=plt.gca()
		xy = np.c_[x_coords, y_coords]
		verts= xy[vertex_array]
		pc = matplotlib.collections.PolyCollection(verts, edgecolor="none", cmap="rainbow")
		pc.set_array(fill_value)
		ax.add_collection(pc)
		ax.autoscale()

		# Add mesh edge        
		ax.plot(x_coords[meshes[i].ordered_edge_nodes[:,0]], y_coords[meshes[i].ordered_edge_nodes[:,0]], color="black")
		ax.set_aspect('equal')
		plt.show()