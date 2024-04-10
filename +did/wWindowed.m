function [D, wEff] = wWindowed(s, tData)
%WWINDOWED Windowed DID based on Woodbury formula
%   Input:
%      - s - Settings structure
%      - tData - Training data
%   Return:
%      - D - Sequence of models D
%      - wEff - Effective window (related to weighted algorithm)
% 
%   This code illustrates methods from M. Pikulinski*, P. Malczyk, R. Aarts,
%   (2024), Data-Driven Inverse Dynamics Modeling Using Neural-Networks and
%   Regression-Based Techniques, Multibody System Dynamics (under review).
%   If you use it, please refer to the paper.
% 
%   *Corresponding author e-mail: maciej.pikulinski.dokt@pw.edu.pl

% Transform time-based settings into sample-based
wEff = floor(s.did.wt / s.data.Ts);
r    = s.did.rt / s.data.Ts;

% Internals uses sigma among rho
sigma = sqrt(s.did.rho);

% Create single state matrix and build initial data
X = [tData.input.q; tData.input.dq];

EInit = [tData.target(:, 1:wEff)];
ZInit = [X(:, 1:wEff); X(:, 2:(wEff + 1))];

E = [tData.target(:, (wEff + 1):(end - 1))];
Z = [X(:, (wEff + 1):(end - 1)); X(:, (wEff + 2):end)];

% Create matrices storing current data of the model
Ew = zeros(size(E, 1), wEff);
Zw = zeros(size(Z, 1), wEff);
for i = 1:wEff
    Ew(:, i) = EInit(:, i) * sigma^(wEff - i);
    Zw(:, i) = ZInit(:, i) * sigma^(wEff - i);
end

% Compute ridge regression regularization term
ridgeReg = s.did.alpha * eye(size(Zw, 1));

% Initialize model and updating matrices
P = pinv(Zw * Zw' + ridgeReg);
D(:, :, 1) = Ew * Zw' * P; P = P / s.did.rho;
CInv = [-1 0; 0 1];

% Compoute models by updating
for i = 2:(size(E, 2) + 1)
    zOld  = Zw(:, 1);
    eOld  = Ew(:, 1);
    zNext = Z(:, i - 1);
    eNext = E(:, i - 1);
    
    % Maintain matrices storing current data of the model
    Zw = [Zw(:, 2:end) * sigma zNext];
    Ew = [Ew(:, 2:end) * sigma eNext];

    % Recomputation
    if rem(i - 1, r) == 0
        P = pinv(Zw * Zw' + ridgeReg);
        D(:, :, i) = Ew * Zw' * P; P = P / s.did.rho;
        continue;
    end

    model = D(:, :, i - 1);

    U = [zOld * sigma zNext];
    V = [eOld * sigma eNext];

    PkU = P * U;

    gamma = inv(CInv + U'*(PkU));

    model = model + ...
        (V - model * U) * (gamma) * (PkU');

    P = (P - PkU * (gamma) * (PkU')) / s.did.rho;
    P = (P + P') / 2;

    D(:, :, i) = model;
end

end
