function [D, wEff] = chWeighted(s, tData)
%CHWEIGHTED Weighted DID based on Cholesky decomposition
%   Input:
%      - s - Settings structure
%      - tData - Training data
%   Return:
%      - D - Sequence of models D
%      - wEff - Effective window (related to weighted algorithm)
% 
%   This code illustrates methods from M. Pikulinski*, P. Malczyk, R. Aarts,
%   Data-Driven Inverse Dynamics Modeling Using Neural-Networks and
%   Regression-Based Techniques, 2024. If you use it, please refer to the
%   paper.
% 
%   *Corresponding author e-mail: maciej.pikulinski.dokt@pw.edu.pl

% Transform time-based settings into sample-based
wEff = floor(log(s.did.rth) / log(s.did.rho));
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
R = chol((Zw * Zw' + ridgeReg)); EZw = Ew * Zw';
D(:, :, 1) = (R \ (R' \ EZw'))';
R = R * sigma;

% Compoute models by updating
for i = 2:(size(E, 2) + 1)
    zNext = Z(:, i - 1);
    eNext = E(:, i - 1);
    
    % Maintain matrices storing current data of the model
    Zw = [Zw(:, 2:end) * sigma zNext];
    Ew = [Ew(:, 2:end) * sigma eNext];

    % Recomputation
    if rem(i - 1, r) == 0
        R = chol((Zw * Zw' + ridgeReg)); EZw = Ew * Zw';
        D(:, :, i) = (R \ (R' \ EZw'))';
        R = R * sigma;
        continue;
    end

    % Add (update) the latest measurement
    R = cholupdate(R, zNext);
    EZw = EZw * s.did.rho + eNext * zNext';
    model = (R \ (R' \ EZw'))';
    
    R = R * sigma;

    D(:, :, i) = model;
end

end
