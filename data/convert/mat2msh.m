function mat2msh(matfile)
% Writes a Gmsh 2.0.8 .msh file from a .mat file

% Load mesh geometry
p = ReadPatches(matfile);

% Initialize msh file
[~, fname] = fileparts(matfile);
fid = fopen([fname '.msh'], 'w');
fprintf(fid, '$MeshFormat\n2 0 8\n$EndMeshFormat\n$Nodes\n');
% Write nodes
fprintf(fid, '%g\n', p.nc);
fprintf(fid, '%g %g %g %g\n', [1:p.nc; p.c']);
fprintf(fid, '$EndNodes\n$Elements\n');
% Write elements (triangles only)
fprintf(fid, '%g\n', p.nEl);
fprintf(fid, '%g %g %g %g %g %g %g %g %g\n', [1:p.nEl; repmat([2 3 0 1 0]', 1, p.nEl); p.v']);
fprintf(fid, '$EndElements');
fclose(fid);
fprintf(1, 'Wrote %s\n', [fname '.msh'])