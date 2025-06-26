function segment2csv(segment_file_name)

% km2m = 1e3; % Convert kilometers to meters

% Read file from old .segment format
segment = ReadSegmentTri(segment_file_name);
sizeseg = size(segment.lon1);

% Create new field names
segment.locking_depth = segment.lDep;
segment.locking_depth_sig = segment.lDepSig;
segment.locking_depth_flag = segment.lDepTog;
segment.dip_sig = segment.dipSig;
segment.dip_flag = segment.dipTog;
segment.ss_rate = segment.ssRate;
segment.ss_rate_sig = segment.ssRateSig;
segment.ss_rate_flag = segment.ssRateTog;
segment.ds_rate = segment.dsRate;
segment.ds_rate_sig = segment.dsRateSig;
segment.ds_rate_flag = segment.dsRateTog;
segment.ts_rate = segment.tsRate;
segment.ts_rate_sig = segment.tsRateSig;
segment.ts_rate_flag = segment.tsRateTog;
segment.resolution_override = segment.resOver;
segment.resolution_other = segment.resOther;
segment.patch_file_name = segment.patchFile - 1; % Make all 0 into -1
segment.patch_flag = segment.patchTog;
segment.patch_slip_file = segment.patchSlipFile;
segment.patch_slip_flag = segment.patchSlipTog;
segment.ss_rate_bound_flag = zeros(sizeseg);
segment.ss_rate_bound_min = -1.0*ones(sizeseg);
segment.ss_rate_bound_max = 1.0*ones(sizeseg);
segment.ds_rate_bound_flag = zeros(sizeseg)
segment.ds_rate_bound_min = -1.0*ones(sizeseg);
segment.ds_rate_bound_max = 1.0*ones(sizeseg);
segment.ts_rate_bound_flag = zeros(sizeseg)
segment.ts_rate_bound_min = -1.0*ones(sizeseg);
segment.ts_rate_bound_max = 1.0*ones(sizeseg);

% Delete old field names
segment = rmfield(segment, "lDep");
segment = rmfield(segment, "lDepSig");
segment = rmfield(segment, "lDepTog");
segment = rmfield(segment, "dipSig");
segment = rmfield(segment, "dipTog");
segment = rmfield(segment, "ssRate");
segment = rmfield(segment, "ssRateSig");
segment = rmfield(segment, "ssRateTog");
segment = rmfield(segment, "dsRate");
segment = rmfield(segment, "dsRateSig");
segment = rmfield(segment, "dsRateTog");
segment = rmfield(segment, "tsRate");
segment = rmfield(segment, "tsRateSig");
segment = rmfield(segment, "tsRateTog");
segment = rmfield(segment, "bDep");
segment = rmfield(segment, "bDepSig");
segment = rmfield(segment, "bDepTog");
segment = rmfield(segment, "resOver");
segment = rmfield(segment, "resOther");
segment = rmfield(segment, "patchFile");
segment = rmfield(segment, "patchTog");
segment = rmfield(segment, "patchSlipFile");
segment = rmfield(segment, "patchSlipTog");

% Save as .csv
csv_file_name = append(strrep(segment_file_name, ".", "_"), ".csv");
struct2csv(segment, csv_file_name);
fprintf(1, "Wrote %s \n", csv_file_name);
