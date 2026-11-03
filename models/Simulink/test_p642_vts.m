
%% P642 VTS - Controlled Test 1: Reset Disabled
clear;
clc;

% Simulation time
t = (0:0.001:5)';

% Number of samples
N = length(t);

%% Inputs

% 1. VAB
VAB = 110 * ones(N,1);
VAB(t >= 1 & t < 3.5) = 50;

% 2. VBC
VBC = 110 * ones(N,1);
VBC(t >= 1 & t < 3.5) = 50;

% 3. V2
V2 = zeros(N,1);

% 4. I2
I2 = zeros(N,1);

% 5. I2_SETTING
I2_SETTING = 0.05 * ones(N,1);

% 6-8. Superimposed currents
delta_IA = zeros(N,1);
delta_IB = zeros(N,1);
delta_IC = zeros(N,1);

% 9. Nominal current
In = ones(N,1);

% 10. ANY_POLE_DEAD
ANY_POLE_DEAD = zeros(N,1);

% 11. MCB_VTS
MCB_VTS = zeros(N,1);

% 12. VTS_BLOCKING
VTS_BLOCKING = ones(N,1);

% 13. VTS_AUTORESET - ENABLED
VTS_AUTORESET = ones(N,1);

% 14. VTS_MANRESET - DISABLED
VTS_MANRESET = zeros(N,1);

% 15. ANY_VOLTAGE_DEPENDENT_FUNCTION
ANY_VOLTAGE_DEPENDENT_FUNCTION = zeros(N,1);

% 16. ACCELERATE_IND
ACCELERATE_IND = zeros(N,1);

% 17. tVTS
tVTS = ones(N,1);

% 18. IA
IA = 0.5 * ones(N,1);

% 19. IB
IB = 0.5 * ones(N,1);

% 20. IC
IC = 0.5 * ones(N,1);

% 21. I_SETTING
I_SETTING = ones(N,1);

% 22. ALL_POLE_DEAD
ALL_POLE_DEAD = zeros(N,1);

disp('AUTORESET values:');
disp(unique(VTS_AUTORESET));

disp('MANRESET values:');
disp(unique(VTS_MANRESET));

%% Create input matrix
% Column 1 is time.
% Columns 2-23 correspond to Simulink input ports 1-22.

input_data = [ ...
    t, ...
    VAB, ...
    VBC, ...
    V2, ...
    I2, ...
    I2_SETTING, ...
    delta_IA, ...
    delta_IB, ...
    delta_IC, ...
    In, ...
    ANY_POLE_DEAD, ...
    MCB_VTS, ...
    VTS_BLOCKING, ...
    VTS_AUTORESET, ...
    VTS_MANRESET, ...
    ANY_VOLTAGE_DEPENDENT_FUNCTION, ...
    ACCELERATE_IND, ...
    tVTS, ...
    IA, ...
    IB, ...
    IC, ...
    I_SETTING, ...
    ALL_POLE_DEAD ...
];

%% Save input for Simulink
save('p642_test_input.mat', 'input_data');

disp('Test input created successfully.');
disp(size(input_data));
disp('Automatic reset enabled: 1');
disp('Manual reset disabled: 0');

%% Run simulation
out = sim('P642_VTS_Reference');

%% Extract outputs
slow = out.yout{1}.Values;
fast = out.yout{2}.Values;
ind  = out.yout{3}.Values;

%% Display output transitions
outputs = {
    'SLOW_BLOCK', slow;
    'FAST_BLOCK', fast;
    'INDICATION', ind
    };

for k = 1:size(outputs,1)
    name = outputs{k,1};
    signal = outputs{k,2};

    values = double(signal.Data(:));
    times = signal.Time(:);

    idx = [1; find(diff(values) ~= 0) + 1];

    fprintf('\n%s transitions:\n', name);
    disp(table(times(idx), values(idx), ...
        'VariableNames', {'Time','Value'}));
end