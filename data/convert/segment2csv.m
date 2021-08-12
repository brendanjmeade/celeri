% function segment2csv(segment_file_name)

% Read file from old .segment format
segment_file_name = "Reference_GBM.segment";
segment = ReadSegmentTri(segment_file_name);

% Create new field names
segment.locking_depth = segment.lDep;
segment.locking_depth_sig = segment.lDepSig;
segment.locking_depth_flag = segment.lDepTog;
segment.dip_sig = segment.dipSig;
segment.dip_flag = segment.dipTog;

segment.dip_flag = ssRate;
segment.dip_flag = ssRateSig;
segment.dip_flag = ssRateTog;
segment.dip_flag = dsRate;
segment.dip_flag = dsRateSig;
segment.dip_flag = dsRateTog;
segment.dip_flag = tsRate;
segment.dip_flag = tsRateSig;
segment.dip_flag = tsRateTog;
segment.dip_flag = bDep;
segment.dip_flag = bDepSig;
segment.dip_flag =bDepTog;


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


% Save as .csv
segment
csv_file_name = append(strrep(segment_file_name, ".", "_"), ".csv");
% struct2csv(segment, csv_file_name);