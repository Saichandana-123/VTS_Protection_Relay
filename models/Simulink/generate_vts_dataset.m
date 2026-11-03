
%% P642 VTS - Initial Automated Dataset Generation
clear;
clc;

model = 'P642_VTS_Reference';

% Simulation settings
Ts = 0.001;
t = (0:Ts:5)';
N = length(t);

% Define four test scenarios
scenario_names = {
    'Healthy'
    'VoltageDip_1s'
    'VoltageDip_2s'
    'VoltageDip_Restoration'
};

num_scenarios = length(scenario_names);
results = cell(num_scenarios, 1);

for k = 1:num_scenarios

    % Default healthy conditions
    VAB = 110 * ones(N,1);
    VBC = 110 * ones(N,1);
    V2 = zeros(N,1);
    I2 = zeros(N,1);
    I2_SETTING = 0.05 * ones(N,1);

    delta_IA = zeros(N,1);
    delta_IB = zeros(N,1);
    delta_IC = zeros(N,1);
    In = ones(N,1);

    ANY_POLE_DEAD = zeros(N,1);
    MCB_VTS = zeros(N,1);
    VTS_BLOCKING = ones(N,1);
    VTS_AUTORESET = ones(N,1);
    VTS_MANRESET = zeros(N,1);
    ANY_VOLTAGE_DEPENDENT_FUNCTION = zeros(N,1);
    ACCELERATE_IND = zeros(N,1);
    tVTS = ones(N,1);

    IA = 0.5 * ones(N,1);
    IB = 0.5 * ones(N,1);
    IC = 0.5 * ones(N,1);
    I_SETTING = ones(N,1);
    ALL_POLE_DEAD = zeros(N,1);

    % Scenario-specific voltage disturbances
    switch k
        case 1
            % Healthy voltage throughout

        case 2
            % Voltage dip: 1 to 2 seconds
            VAB(t >= 1 & t < 2) = 50;
            VBC(t >= 1 & t < 2) = 50;

        case 3
            % Longer voltage dip: 1 to 3.5 seconds
            VAB(t >= 1 & t < 3.5) = 50;
            VBC(t >= 1 & t < 3.5) = 50;


        case 4
            % Voltage dip followed by restoration
            % Dip from 1 s to 2 s, then voltage recovers
            VAB(t >= 1 & t < 2) = 50;
            VBC(t >= 1 & t < 2) = 50;

            % Explicit restoration to nominal voltage at 2 s
            VAB(t >= 2) = 110;
            VBC(t >= 2) = 110;

    end

    % Input columns must match the Simulink Inport order
    input_data = [ ...
        t, VAB, VBC, V2, I2, I2_SETTING, ...
        delta_IA, delta_IB, delta_IC, In, ...
        ANY_POLE_DEAD, MCB_VTS, VTS_BLOCKING, ...
        VTS_AUTORESET, VTS_MANRESET, ...
        ANY_VOLTAGE_DEPENDENT_FUNCTION, ACCELERATE_IND, ...
        tVTS, IA, IB, IC, I_SETTING, ALL_POLE_DEAD ];

    % Run the model using this scenario's inputs
    in = Simulink.SimulationInput(model);
    in = in.setExternalInput(input_data);
    out = sim(in);

    % Extract outputs
    slow = out.yout{1}.Values;
    fast = out.yout{2}.Values;
    ind  = out.yout{3}.Values;

    % Save each scenario's inputs and outputs
    results{k} = struct( ...
        'name', scenario_names{k}, ...
        'input_data', input_data, ...
        'slow', slow, ...
        'fast', fast, ...
        'indication', ind);

    fprintf('Completed %d/%d: %s\n', ...
        k, num_scenarios, scenario_names{k});
end

save('P642_VTS_initial_dataset.mat', ...
    'results', 'scenario_names', 'Ts');

disp('Initial dataset saved successfully.');
