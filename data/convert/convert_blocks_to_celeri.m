function convert_blocks_to_celeri(command_file_name)
command = ReadCommand(command_file_name);
command2json(command_file_name);
segment2csv(command.segFileName);
block2csv(command.blockFileName);
stadata2csv(command.staFileName);