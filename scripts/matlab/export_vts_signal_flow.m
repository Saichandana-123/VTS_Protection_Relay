
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
model = 'P642_VTS_Reference';
modelFile = fullfile(projectRoot, 'models', 'Simulink', [model '.slx']);
load_system(modelFile);

blocks = find_system(model, ...
    'SearchDepth', 1, ...
    'Type', 'Block');

rows = {};

for k = 1:numel(blocks)
    block = blocks{k};

    % Read block information
    blockType = get_param(block, 'BlockType');

    try
        ports = get_param(block, 'PortHandles');
    catch
        continue;
    end

    % Record incoming signal connections
    if isfield(ports, 'Inport')
        for p = 1:numel(ports.Inport)
            portHandle = ports.Inport(p);
            lineHandle = get_param(portHandle, 'Line');

            if lineHandle == -1
                source = '(unconnected)';
            else
                sourcePort = get_param(lineHandle, 'SrcPortHandle');

                if sourcePort == -1
                    source = '(unknown source)';
                else
                    sourceBlock = get_param(sourcePort, 'Parent');
                    source = get_param(sourceBlock, 'Name');
                end
            end

            rows(end+1, :) = { ...
                block, blockType, 'INPUT', p, source}; %#ok<SAGROW>
        end
    end

    % Record outgoing signal connections
    if isfield(ports, 'Outport')
        for p = 1:numel(ports.Outport)
            portHandle = ports.Outport(p);
            lineHandle = get_param(portHandle, 'Line');

            if lineHandle == -1
                destination = '(unconnected)';
                rows(end+1, :) = { ...
                    block, blockType, 'OUTPUT', p, destination}; %#ok<SAGROW>
            else
                destinationPorts = get_param(lineHandle, 'DstPortHandle');

                for q = 1:numel(destinationPorts)
                    destinationPort = destinationPorts(q);

                    if destinationPort == -1
                        continue;
                    end

                    destinationBlock = ...
                        get_param(destinationPort, 'Parent');

                    destination = get_param(destinationBlock, 'Name');

                    rows(end+1, :) = { ...
                        block, blockType, 'OUTPUT', p, destination}; %#ok<SAGROW>
                end
            end
        end
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'Block', 'BlockType', 'Direction', 'Port', 'ConnectedBlock'});

outputFile = fullfile(projectRoot, 'docs', 'vts_signal_flow.csv');
writetable(T, outputFile);

fprintf('Exported %d connection records to %s\n', height(T), outputFile);
