#!/usr/bin/env python3
import gmsh
import numpy as np
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert STL file to MSH with vertical scaling')
    parser.add_argument('input_file', help='Input STL file path')
    parser.add_argument('--vertical_scale_factor', type=float, default=0.01, 
                        help='Scale factor to reverse Z coordinates scaling (default: 0.01)')
    args = parser.parse_args()
    
    # Get input and generate output filenames
    input_file = args.input_file
    output_file = os.path.splitext(input_file)[0] + ".msh"
    
    gmsh.initialize()
    gmsh.open(input_file)
    
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(nodeCoords).reshape(-1, 3)
    
    # Reverse the Z scaling by dividing by the scale factor
    coords[:, 2] /= args.vertical_scale_factor
    
    for i, tag in enumerate(nodeTags):
        x, y, z = coords[i]
        gmsh.model.mesh.setNode(int(tag), [x, y, z], [])
    
    gmsh.write(output_file)
    gmsh.finalize()
    
    print(f"Converted {input_file} to {output_file} with vertical scale factor {args.vertical_scale_factor}")

if __name__ == "__main__":
    main()