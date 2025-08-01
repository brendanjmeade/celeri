function command2json(command_file_name)

% Read blocks style .command file
% command_file_name = "basic.command";
command = ReadCommand(command_file_name);
% Convert struct field names to celeri style

command.file_name = command.fileName;
command.reuse_elastic = command.reuseElastic;
command.reuse_elastic_file = command.reuseElasticFile;
command.save_kernels = command.saveKernels;
command.material_lambda = 3e10;
command.material_mu = 3e10;
command.unit_sigmas = command.unitSigmas;
command.locking_depth_flag2 = command.ldTog2;
command.locking_depth_flag3 = command.ldTog3;
command.locking_depth_flag4 = command.ldTog4;
command.locking_depth_flag5 = command.ldTog5;
command.locking_depth_override_flag = command.ldOvTog;
command.locking_depth_overide_value = command.ldOvValue;
% command.mesh_file_names = command.patchFileNames; % Not currently needed
command.tri_smooth = command.triSmooth;
command.smooth_type = command.smoothType;
command.n_iterations = command.nIter;
command.tri_edge = command.triEdge;
command.tri_depth_tolerance = command.triDepthTol;
command.tri_con_weight = command.triConWgt;
command.strain_method = command.strainMethod;
command.sar_file_name = command.sarFileName;
command.tri_slip_constraint_type = command.triSlipConstraintType;
command.save_all = command.dumpall;
command.mogi_file_name = command.mogiFileName;
command.solution_method = command.solutionMethod;
command.ridge_param = command.ridgeParam;
command.tri_full_coupling = command.triFullCoup;
command.tri_slip_sign = command.trislipsign;
command.n_eigs = command.neigs;
command.segment_file_name = command.segFileName;
command.station_file_name = command.staFileName;
command.block_file_name = command.blockFileName;
command.mesh_parameters_file_name = "mesh_parameters.json"; % A sensible default?
command.station_data_weight = command.stationDataWgt;
command.station_data_weight_min = command.stationDataWgtMin;
command.station_data_weight_max = command.stationDataWgtMax;
command.station_data_weight_steps = command.stationDataWgtSteps;
command.slip_constraint_weight = command.slipConWgt;
command.slip_constraint_weight_min = command.slipConWgtMin;
command.slip_constraint_weight_max = command.slipConWgtMin;
command.slip_constraint_weight_steps = command.slipConWgtSteps;
command.block_constraint_weight = command.blockConWgt;
command.block_constraint_weight_min = command.blockConWgtMin;
command.block_constraint_weight_max = command.blockConWgtMax;
command.block_constraint_weight_steps = command.blockConWgtSteps;
command.slip_file_names = command.slipFileNames;

% Delete blocks style field names
command = rmfield(command, "fileName");
command = rmfield(command, "reuseElastic");
command = rmfield(command, "reuseElasticFile");
command = rmfield(command, "saveKernels");
command = rmfield(command, "poissonsRatio");
command = rmfield(command, "unitSigmas");
command = rmfield(command, "ldTog2");
command = rmfield(command, "ldTog3");
command = rmfield(command, "ldTog4");
command = rmfield(command, "ldTog5");
command = rmfield(command, "ldOvTog");
command = rmfield(command, "ldOvValue");
command = rmfield(command, "aprioriBlockName");
command = rmfield(command, "patchFileNames");
command = rmfield(command, "triSmooth");
command = rmfield(command, "pmagTriSmooth");
command = rmfield(command, "smoothType");
command = rmfield(command, "nIter");
command = rmfield(command, "triEdge");
command = rmfield(command, "triDepthTol");
command = rmfield(command, "triConWgt");
command = rmfield(command, "strainMethod");
command = rmfield(command, "sarFileName");
command = rmfield(command, "sarRamp");
command = rmfield(command, "sarWgt");
command = rmfield(command, "triSlipConstraintType");
command = rmfield(command, "inversionType");
command = rmfield(command, "inversionParam01");
command = rmfield(command, "inversionParam02");
command = rmfield(command, "inversionParam03");
command = rmfield(command, "inversionParam04");
command = rmfield(command, "inversionParam05");
command = rmfield(command, "dumpall");
command = rmfield(command, "mogiFileName");
command = rmfield(command, "solutionMethod");
command = rmfield(command, "ridgeParam");
command = rmfield(command, "triFullCoup");
command = rmfield(command, "tvrlambda");
command = rmfield(command, "trislipsign");
command = rmfield(command, "neigs");
command = rmfield(command, "segFileName");
command = rmfield(command, "staFileName");
command = rmfield(command, "blockFileName");
command = rmfield(command, "faultRes");
command = rmfield(command, "stationDataWgt");
command = rmfield(command, "stationDataWgtMin");
command = rmfield(command, "stationDataWgtMax");
command = rmfield(command, "stationDataWgtSteps");
command = rmfield(command, "slipConWgt");
command = rmfield(command, "slipConWgtMin");
command = rmfield(command, "slipConWgtMax");
command = rmfield(command, "slipConWgtSteps");
command = rmfield(command, "blockConWgt");
command = rmfield(command, "blockConWgtMin");
command = rmfield(command, "blockConWgtMax");
command = rmfield(command, "blockConWgtSteps");
command = rmfield(command, "slipFileNames");

% Encode as JSON and save to file
json_file_name = append(strrep(command_file_name, ".", "_"), ".json");
fprintf(fopen(json_file_name, "w"), jsonencode(command));
fprintf(1, "Wrote %s \n", json_file_name);
