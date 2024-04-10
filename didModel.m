%   Script for presenting Data-Driven Inverse Dynamics approach to error
%   learning of feedforward controllers based on neural networks for 2-DOF
%   flexure manipulator.
%   
%   This code illustrates methods from M. Pikulinski*, P. Malczyk, R. Aarts,
%   Data-Driven Inverse Dynamics Modeling Using Neural-Networks and
%   Regression-Based Techniques, 2024. If you use it, please refer to the
%   paper.
% 
%   *Corresponding author e-mail: maciej.pikulinski.dokt@pw.edu.pl

%% Settings
% Data settings
s.data.name = "random_1"; % random_1, random_2, spiral
s.data.Ts = 0.005; % Operational sampling time [s]

% DID settings
s.did.rho = 0.650;  % Weighting factor rho [-]
s.did.rth = 1e-3;   % Weighting factor rho threshold [-]
s.did.wt = 0.150;   % Window time [s]
s.did.rt = 0.200;   % Recomputation interval [s]
s.did.pt = 0.025;   % Prediction horizon [s]
s.did.alpha = 1e-7; % Ridge regression coefficient [s]

% Set the updating method from the following list: @did.smWeighted,
% @did.smWindowed, @did.chWeighted, @did.chWindowed, @did.qrWeighted,
% @did.wWindowed.
s.did.update = @did.smWeighted;

%% Load and prepare data
% Load given dataset from a file
data = load(fullfile("data/", sprintf("%s.mat", s.data.name)));

% Data structure map:
% - data.x - measured state
% - data.r - desired trajectory
% - data.f - actuators' force
%    - data.f.fb - feedback force (tracking data.r results in data.x)
%    - data.f.ident.r - ident.-based feedforward (FF) on data.r
%    - data.f.nn - neural-network-based FF
%       - data.f.nn.lnn - LNN-based FF
%          - data.f.nn.lnn.x - LNN-based FF values on data.x
%          - data.f.nn.lnn.r - LNN-based FF values on data.r
%       - data.f.nn.fnn - FNN-based FF
%          - data.f.nn.fnn.x - FNN-based FF values on data.x
%          - data.f.nn.fnn.r - FNN-based FF values on data.r

% Prepare training dataset. Use measured state as input data and values of
% controllers also taken for the measured state (indirect approach).
tData.input.q   = data.x(1:2, :);
tData.input.dq  = data.x(3:4, :);
tData.input.ddq = data.x(5:6, :);
tData.target = data.f.fb - (data.f.nn.lnn.x + data.f.nn.fnn.x);

%% Preprocess data
% Data is saved at 1000 Hz, downsample it to 1 / Ts (defaults 200 Hz)
tData = downsampleData(tData, 1000 * s.data.Ts);

%% Train models and predict values
% First, a sequence of updated models is generated as if done online,
% retrieving a data stream. Then, predictions are computed from the
% sequence of models. Although a real-life implementation would update
% and predict in a single step, the coded approach does not limit the
% generality of the results. CAUTION must be taken when interpreting the
% results, as the state sent to the DID controller would differ due to
% the controller's previous actions during the real-life work.
[D, wEff] = s.did.update(s, tData);
[e, t, ids] = did.predict(s, tData, D, wEff);

%% Utility functions
function dData = downsampleData(data, factor)
    fn ={'q', 'dq', 'ddq'};
    for i = 1:numel(fn)
        dData.input.(fn{i}) = data.input.(fn{i})(:, 1:factor:end);
    end
    dData.target = data.target(:, 1:factor:end);
end