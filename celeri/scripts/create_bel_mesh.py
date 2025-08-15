#!/usr/bin/env python3
import argparse
from pathlib import Path

import gmsh
import meshio
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj


def main():
    parser = argparse.ArgumentParser(
        description="Create a horizontal mesh from an existing .msh file with buffered convex hull"
    )
    parser.add_argument("msh_file", help="Input MSH file path", type=Path)
    parser.add_argument(
        "msh_buffer_length_scale",
        type=float,
        help="Buffer length scale in kilometers",
    )
    parser.add_argument(
        "msh_depth",
        type=float,
        help="Depth in kilometers (will be made negative)",
    )
    parser.add_argument(
        "msh_element_length_scale",
        type=float,
        help="Target mesh element length scale in kilometers",
    )
    args = parser.parse_args()

    # Ensure depth is negative
    depth = -abs(args.msh_depth)
    
    # Read the input mesh file
    print(f"Reading mesh file: {args.msh_file}")
    mesh = meshio.read(args.msh_file)
    
    # Extract lon/lat coordinates from mesh points
    points = mesh.points
    lons = points[:, 0]
    lats = points[:, 1]
    
    # Create convex hull around the lon/lat points
    print("Computing convex hull...")
    coords = np.column_stack((lons, lats))
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]
    
    # Create a Shapely polygon from the convex hull
    polygon = Polygon(hull_points)
    
    # Buffer the polygon by the specified distance in kilometers
    print(f"Buffering convex hull by {args.msh_buffer_length_scale} km...")
    
    # Calculate the center of the polygon for projection
    center_lon = polygon.centroid.x
    center_lat = polygon.centroid.y
    
    # Create a local projection centered on the mesh
    proj_string = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"
    transformer_to_local = pyproj.Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
    transformer_to_lonlat = pyproj.Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
    
    # Transform to local coordinates, buffer, then transform back
    polygon_local = transform(transformer_to_local.transform, polygon)
    buffer_distance_m = args.msh_buffer_length_scale * 1000  # Convert km to m
    buffered_polygon_local = polygon_local.buffer(buffer_distance_m)
    buffered_polygon = transform(transformer_to_lonlat.transform, buffered_polygon_local)
    
    # Extract the boundary coordinates and resample uniformly
    boundary = buffered_polygon.exterior
    
    # Calculate the perimeter in the local coordinate system for accurate distance
    boundary_local = transform(transformer_to_local.transform, boundary)
    perimeter_length = boundary_local.length
    
    # Calculate number of points based on desired element size
    # Use smaller spacing on boundary to ensure good mesh quality
    boundary_spacing = args.msh_element_length_scale * 1000 * 0.5  # Half the element size in meters
    num_points = max(int(perimeter_length / boundary_spacing), 20)  # At least 20 points
    
    print(f"Resampling boundary with {num_points} uniformly spaced points...")
    
    # Resample the boundary with uniform spacing
    resampled_points = []
    for i in range(num_points):
        # Get point at uniform distance along the boundary
        point = boundary_local.interpolate(i * perimeter_length / num_points)
        # Transform back to lon/lat
        lon, lat = transformer_to_lonlat.transform(point.x, point.y)
        resampled_points.append((lon, lat))
    
    # Create a new mesh using gmsh
    print(f"Creating new mesh with element size {args.msh_element_length_scale} km...")
    gmsh.initialize()
    gmsh.model.add("horizontal_mesh")
    
    # Add points to gmsh
    point_tags = []
    for lon, lat in resampled_points:
        tag = gmsh.model.geo.addPoint(lon, lat, depth, args.msh_element_length_scale)
        point_tags.append(tag)
    
    # Create lines connecting the points
    line_tags = []
    for i in range(len(point_tags)):
        j = (i + 1) % len(point_tags)
        tag = gmsh.model.geo.addLine(point_tags[i], point_tags[j])
        line_tags.append(tag)
    
    # Create a curve loop and surface
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # Synchronize and generate the mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)  # 2D mesh
    
    # Set all node depths to the specified value
    print(f"Setting all node depths to {depth} km...")
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(node_coords).reshape(-1, 3)
    coords[:, 2] = depth  # Set all z-values to the specified depth
    
    for i, tag in enumerate(node_tags):
        x, y, z = coords[i]
        gmsh.model.mesh.setNode(int(tag), [x, y, z], [])
    
    # Generate output filename
    output_file = Path(f"{args.msh_file.stem}_{args.msh_buffer_length_scale}_{abs(depth)}_{args.msh_element_length_scale}.msh")
    
    # Write the mesh to file
    print(f"Writing output mesh to: {output_file}")
    gmsh.write(str(output_file))
    gmsh.finalize()
    
    print(f"Successfully created horizontal mesh with {len(coords)} nodes at depth {depth} km")


if __name__ == "__main__":
    main()