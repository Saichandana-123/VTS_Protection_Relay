
clear;
clc;

%% 1. Locate project
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
    'Input manifest not found.');

%% 2. Complete signal inventory

% Detector outputs
detectorSignals = {
    'IA_OK'
    'IB_OK'
    'IC_OK'
    'VAB_OK'
    'VBC_OK'
    'V2_OK'
    'I2_OK'
    'DELTA_IA_OK'
    'DELTA_IB_OK'
    'DELTA_IC_OK'
};

% Stage 1
stage1Signals = {
    '1_OR_OUT'
    '2_OR_OUT'
    '3_OR_OUT'
    '1_AND_OUT'
    '2_AND_OUT'
};

% Timers
timerSignals = {
    '240_TIMER_OUT'
    '20_TIMER_OUT'
    'VTS_TIMER_OUT'
};

% Stage 2
stage2Signals = {
    '3_AND_OUT'
    '4_AND_OUT'
    '4_OR_OUT'
    '5_AND_OUT'
};

% Flip-flops
flipFlopSignals = {
    'SR_1_OUT'
    'SR_2_OUT'
    'SR_3_OUT'
    'SR_4_OUT'
};

% Stage 3
stage3Signals = {
    '6_AND_OUT'
    '7_AND_OUT'
    '8_AND_OUT'
    '9_AND_OUT'
};

% Stage 4
stage4Signals = {
    '10_AND_OUT'
};

% Stage 5
stage5Signals = {
    '5_OR_OUT'
    '6_OR_OUT'
    '7_OR_OUT'
};

% Stage 6 is extracted from the existing yout outputs.

allSignals = [
    detectorSignals
    stage1Signals
    timerSignals
    stage2Signals
    flipFlopSignals
    stage3Signals
    stage4Signals
    stage5Signals
];

%% 3. Load model and manifest

manifest = readtable( ...
    fullfile(inputDir, 'manifest.csv'), ...
    'TextType', 'string');

load_system(modelFile);

numberOfCasesToRun = height(manifest);

%% 4. Run every case

for k = 1:numberOfCasesToRun

    caseId = manifest.case_id(k);
    inputFile = fullfile(inputDir, manifest.filename(k));

    fprintf('\nRunning %s (%d/%d)\n', ...
        caseId, k, numberOfCasesToRun);

    %% Read input data
    T = readtable(inputFile);

    input_data = T{:,:};

    assert(size(input_data, 2) == 23, ...
        'Expected time + 22 input signals.');

    assert(all(isfinite(input_data), 'all'), ...
        'Input contains NaN or Inf.');

    %% Run Simulink
    simIn = Simulink.SimulationInput(model);

    simIn = simIn.setVariable('input_data', input_data);

    simIn = simIn.setModelParameter( ...
        'StopTime', num2str(T.time(end), '%.6f'));

    simOut = sim(simIn);

    %% Extract final outputs
    slow = simOut.yout{1}.Values;
    fast = simOut.yout{2}.Values;
    ind  = simOut.yout{3}.Values;

    % Check that all final outputs have the expected number of samples
    assert(numel(slow.Data) == height(T), ...
        'SLOW_BLOCK sample count mismatch in case %d', k);

    assert(numel(fast.Data) == height(T), ...
        'FAST_BLOCK sample count mismatch in case %d', k);

    assert(numel(ind.Data) == height(T), ...
        'INDICATION sample count mismatch in case %d', k);

    R = T;

    R.SLOW_BLOCK = double(slow.Data(:));
    R.FAST_BLOCK = double(fast.Data(:));
    R.INDICATION = double(ind.Data(:));

    R.case_id = repmat(caseId, height(R), 1);
    R.scenario = repmat(manifest.scenario(k), height(R), 1);

    %% Extract every intermediate signal

    logs = simOut.logsout;

    loggedNames = logs.getElementNames();

    missingSignals = {};

    for j = 1:numel(allSignals)

        signalName = allSignals{j};

        if ~any(strcmp(loggedNames, signalName))
            missingSignals{end+1} = signalName;
            continue;
        end

        signalElement = logs.getElement(signalName);
        signalValues = signalElement.Values.Data(:);

        assert(numel(signalValues) == height(T), ...
            'Sample count mismatch for signal: %s', signalName);

        R.(signalName) = double(signalValues);

    end

    %% Stop if required signals are missing
    if ~isempty(missingSignals)

        fprintf('\nMissing logged signals:\n');

        for j = 1:numel(missingSignals)
            fprintf('  - %s\n', missingSignals{j});
        end

        error(['Complete signal logging in Simulink before ' ...
               'running the full dataset.']);
    end

    %% Save complete output
    outputFile = fullfile(outputDir, ...
        char(caseId + "_output.csv"));

    writetable(R, outputFile);

    fprintf('Saved: %s | Rows: %d | Signals: %d\n', ...
        outputFile, height(R), numel(allSignals));

end

fprintf('\nAll %d cases completed successfully.\n', ...
    numberOfCasesToRun);

fprintf('Output folder: %s\n', outputDir);

close_system(model, 0);
