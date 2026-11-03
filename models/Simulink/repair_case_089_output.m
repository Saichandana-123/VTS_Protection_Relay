clear;
clc;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));

inputDir = fullfile(projectRoot, ...
    'data', 'simulink_dataset', 'inputs');

outputDir = fullfile(projectRoot, ...
    'data', 'simulink_dataset', 'outputs');

model = 'P642_VTS_Reference';
modelFile = fullfile(scriptDir, [model '.slx']);

assert(isfile(modelFile), 'Simulink model not found.');

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

stage1Signals = {
    '1_OR_OUT'
    '2_OR_OUT'
    '3_OR_OUT'
    '1_AND_OUT'
    '2_AND_OUT'
};

timerSignals = {
    '240_TIMER_OUT'
    '20_TIMER_OUT'
    'VTS_TIMER_OUT'
};

stage2Signals = {
    '3_AND_OUT'
    '4_AND_OUT'
    '4_OR_OUT'
    '5_AND_OUT'
};

flipFlopSignals = {
    'SR_1_OUT'
    'SR_2_OUT'
    'SR_3_OUT'
    'SR_4_OUT'
};

stage3Signals = {
    '6_AND_OUT'
    '7_AND_OUT'
    '8_AND_OUT'
    '9_AND_OUT'
};

stage4Signals = {
    '10_AND_OUT'
};

stage5Signals = {
    '5_OR_OUT'
    '6_OR_OUT'
    '7_OR_OUT'
};

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

assert(numel(allSignals) == 34, ...
    'Expected exactly 34 intermediate signals.');

manifest = readtable( ...
    fullfile(inputDir, 'manifest.csv'), ...
    'TextType', 'string', ...
    'VariableNamingRule', 'preserve');

caseIndex = find(manifest.case_id == "case_089", 1);
assert(~isempty(caseIndex), 'case_089 was not found in manifest.csv.');

caseId = manifest.case_id(caseIndex);
inputFile = fullfile(inputDir, manifest.filename(caseIndex));

load_system(modelFile);

T = readtable(inputFile, 'VariableNamingRule', 'preserve');
input_data = T{:,:};

assert(size(input_data, 2) == 23, ...
    'Expected time + 22 input signals.');
assert(height(T) == 13001, ...
    'Expected 13,001 input rows.');
assert(all(isfinite(input_data), 'all'), ...
    'Input contains NaN or Inf.');

simIn = Simulink.SimulationInput(model);
simIn = simIn.setVariable('input_data', input_data);
simIn = simIn.setModelParameter( ...
    'StopTime', num2str(T.time(end), '%.6f'));
simOut = sim(simIn);

slow = simOut.yout{1}.Values;
fast = simOut.yout{2}.Values;
ind = simOut.yout{3}.Values;

assert(numel(slow.Data) == height(T), ...
    'SLOW_BLOCK sample count mismatch.');
assert(numel(fast.Data) == height(T), ...
    'FAST_BLOCK sample count mismatch.');
assert(numel(ind.Data) == height(T), ...
    'INDICATION sample count mismatch.');

R = T;
R.SLOW_BLOCK = double(slow.Data(:));
R.FAST_BLOCK = double(fast.Data(:));
R.INDICATION = double(ind.Data(:));
R.case_id = repmat(caseId, height(R), 1);
R.scenario = repmat(manifest.scenario(caseIndex), height(R), 1);

logs = simOut.logsout;
loggedNames = logs.getElementNames();
missingSignals = {};

for j = 1:numel(allSignals)
    signalName = allSignals{j};

    if ~any(strcmp(loggedNames, signalName))
        missingSignals{end + 1} = signalName;
        continue;
    end

    signalElement = logs.getElement(signalName);
    signalValues = signalElement.Values.Data(:);

    assert(numel(signalValues) == height(T), ...
        'Sample count mismatch for signal: %s', signalName);

    R.(signalName) = double(signalValues);
end

if ~isempty(missingSignals)
    fprintf('\nMissing logged signals:\n');
    for j = 1:numel(missingSignals)
        fprintf('  - %s\n', missingSignals{j});
    end
    error(['Complete signal logging in Simulink before ' ...
        'running the full dataset.']);
end

referenceTable = readtable( ...
    fullfile(outputDir, 'case_088_output.csv'), ...
    'VariableNamingRule', 'preserve');

assert(width(R) == 62, 'Expected exactly 62 output columns.');
assert(isequal(R.Properties.VariableNames, ...
    referenceTable.Properties.VariableNames), ...
    'Output names or ordering do not match case_088_output.csv.');

outputFile = fullfile(outputDir, 'case_089_output.csv');
writetable(R, outputFile);

fprintf('Saved: %s | Rows: %d | Signals: %d\n', ...
    outputFile, height(R), numel(allSignals));

close_system(model, 0);
