function stadata2csv(stadata_file_name)

% Read file from old .station format
station = ReadStation(stadata_file_name);

% Create new field names
station.east_vel = station.eastVel;
station.north_vel = station.northVel;
station.east_sig = station.eastSig;
station.north_sig = station.northSig;
station.flag = station.tog;
station.up_vel = station.upVel;
station.up_sig = station.upSig;
station.east_adjust = station.eastAdj;
station.north_adjust = station.northAdj;
station.up_adjust = station.upAdj;

% Delete old field names
station = rmfield(station, "eastVel");
station = rmfield(station, "northVel");
station = rmfield(station, "eastSig");
station = rmfield(station, "northSig");
station = rmfield(station, "tog");
station = rmfield(station, "upVel");
station = rmfield(station, "upSig");
station = rmfield(station, "eastAdj");
station = rmfield(station, "northAdj");
station = rmfield(station, "upAdj");

% Save as .csv
csv_file_name = append(strrep(stadata_file_name, ".", "_"), ".csv");
struct2csv(station, csv_file_name);