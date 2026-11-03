
clear;
clc;

%% 1. Locate the project and files
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));

inputDir = fullfile(projectRoot, ...
    'data', 'simulink_dataset', 'inputs');

outputDir = fullfile(projectRoot, ...
    'data', 'simulink_dataset', 'outputs');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

model = 'P642_VTS_Reference';
modelFile = fullfile(scriptDir, [model '.slx']);

assert(isfile(modelFile), 'Simulink model not found.');
assert(isfile(fullfile(inputDir, 'manifest.csv')), ...
    'Run the Python input generator first.');

%% 2. Read the list of input cases
manifest = readtable(fullfile(inputDir, 'manifest.csv'), ...
    'TextType', 'string');

load_system(modelFile);

%% 3. Run one case first
% After case_001 succeeds, change this to height(manifest).
numberOfCasesToRun = height(manifest);

for k = 1:min(numberOfCasesToRun, height(manifest))

    caseId = manifest.case_id(k);
    inputFile = fullfile(inputDir, manifest.filename(k));

    fprintf('\nRunning %s: %s\n', ...
        caseId, manifest.scenario(k));

    %% Read the Python-generated CSV
    T = readtable(inputFile);

    % Column 1 = time
    % Columns 2-23 = 22 model inputs
    input_data = T{:,:};

    assert(size(input_data, 2) == 23, ...
        'Expected time + 22 input signals.');

    assert(all(isfinite(input_data), 'all'), ...
        'Input contains NaN or Inf values.');

    %% Pass input_data to Simulink
    simIn = Simulink.SimulationInput(model);
    simIn = simIn.setVariable('input_data', input_data);

    % The model should use the time column and input signals
    % through its existing From Workspace/input interface.
    % Match the simulation duration to the input sequence.
    simIn = simIn.setModelParameter( ...
        'StopTime', num2str(T.time(end), '%.6f'));

    simOut = sim(simIn);
    disp(simOut.logsout.getElementNames())

    %% Extract outputs using the existing model output order
    slow = simOut.yout{1}.Values;
    fast = simOut.yout{2}.Values;
    ind  = simOut.yout{3}.Values;

    %% Check the output sample counts
    assert(numel(slow.Data) == height(T), ...
        'SLOW_BLOCK output length mismatch.');

    assert(numel(fast.Data) == height(T), ...
        'FAST_BLOCK output length mismatch.');

    assert(numel(ind.Data) == height(T), ...
        'INDICATION output length mismatch.');

    %% Combine inputs and Simulink-generated labels
    R = T;
    R.SLOW_BLOCK = double(slow.Data(:));
    R.FAST_BLOCK = double(fast.Data(:));
    R.INDICATION = double(ind.Data(:));

    R.case_id = repmat(caseId, height(R), 1);
    R.scenario = repmat(manifest.scenario(k), height(R), 1);
    
    % Extract Stage 1 intermediate signals by their actual names
    logs = simOut.logsout;

    R.("1_OR_OUT")  = logs.getElement('1_OR_OUT').Values.Data(:);
    R.("2_OR_OUT")  = logs.getElement('2_OR_OUT').Values.Data(:);
    R.("3_OR_OUT")  = logs.getElement('3_OR_OUT').Values.Data(:);
    R.("1_AND_OUT") = logs.getElement('1_AND_OUT').Values.Data(:);
    R.("2_AND_OUT") = logs.getElement('2_AND_OUT').Values.Data(:);
    %% Save the output
    outputFile = fullfile(outputDir, ...
        char(caseId + "_output.csv"));

    writetable(R, outputFile);

    fprintf('Saved output: %s\n', outputFile);
    fprintf('Rows: %d\n', height(R));

end

fprintf('\nInitial test completed.\n');
fprintf('Output folder: %s\n', outputDir);

close_system(model, 0);
