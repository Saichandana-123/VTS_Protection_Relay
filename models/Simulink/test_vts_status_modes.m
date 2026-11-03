
clear;
clc;

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
load_system(fullfile(scriptDir, [model '.slx']));

% Use one existing case so the waveform is identical in all tests.
% Use a voltage-loss scenario instead of the healthy case.
T = readtable(fullfile(inputDir, 'case_027.csv'));
baseInput = T{:,:};

% Column 1 = time.
% Column 13 = VTS_BLOCKING.
% Column 14 = VTS_AUTORESET.
% Column 15 = VTS_MANRESET.
%
% These columns follow the current Python SIGNALS order.

testNames = ["Disabled", "Indication", "Blocking"];

% IMPORTANT:
% This tests the current binary VTS_BLOCKING interface.
% It does not prove Disabled and Indication are distinct modes.
blockingValues = [0, 0, 1];

for k = 1:3
    input_data = baseInput;

    input_data(:,13) = blockingValues(k);

    simIn = Simulink.SimulationInput(model);
    simIn = simIn.setVariable('input_data', input_data);
    simIn = simIn.setModelParameter( ...
        'StopTime', num2str(input_data(end,1), '%.6f'));

    simOut = sim(simIn);

    slow = simOut.yout{1}.Values;
    fast = simOut.yout{2}.Values;
    ind  = simOut.yout{3}.Values;

    R = array2table(input_data, ...
        'VariableNames', T.Properties.VariableNames);

    R.SLOW_BLOCK = double(slow.Data(:));
    R.FAST_BLOCK = double(fast.Data(:));
    R.INDICATION = double(ind.Data(:));

    filename = fullfile(outputDir, ...
        char("status_test_" + testNames(k) + ".csv"));
    writetable(R, filename);

    fprintf('%s test saved: %s\n', testNames(k), filename);
end

close_system(model, 0);
