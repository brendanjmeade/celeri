function block2csv(block_file_name)

% Read file from old .block format
block = ReadBlock(block_file_name);

% Create new field names
block.interior_lon = block.interiorLon;
block.interior_lat = block.interiorLat;
block.euler_lon = block.eulerLon;
block.euler_lon_sig = block.eulerLonSig;
block.euler_lat = block.eulerLat;
block.euler_lat_sig = block.eulerLatSig;
block.rotation_rate = block.rotationRate;
block.rotation_rate_sig = block.rotationRateSig;
block.rotation_flag = block.rotationInfo;
block.apriori_flag = block.aprioriTog;
block.strain_rate = zeros(numel(block.interior_lon), 1);
block.strain_rate_sig = zeros(numel(block.interior_lon), 1);
block.strain_rate_flag = zeros(numel(block.interior_lon), 1);

% Delete old field names
block = rmfield(block, "interiorLon");
block = rmfield(block, "interiorLat");
block = rmfield(block, "eulerLon");
block = rmfield(block, "eulerLonSig");
block = rmfield(block, "eulerLat");
block = rmfield(block, "eulerLatSig");
block = rmfield(block, "rotationRate");
block = rmfield(block, "rotationRateSig");
block = rmfield(block, "rotationInfo");
block = rmfield(block, "aprioriTog");

% Save as .csv
csv_file_name = append(strrep(block_file_name, ".", "_"), ".csv");
struct2csv(block, csv_file_name);
fprintf(1, "Wrote %s \n", csv_file_name);
