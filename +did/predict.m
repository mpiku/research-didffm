function [e, t, ids] = predict(s, tData, D, wEff)
%PREDICT Predict error with DID method
%   Input:
%      - s - Settings structure
%      - tData - Training data
%      - D - Sequence of models
%      - wEff - Effective window (related to weighted algorithm)
%   Return:
%      - e - Predicted error values
%      - t - Time points of prediction
%      - ids - Indices of predicted signals related to original data
% 
%   This code illustrates methods from M. Pikulinski*, P. Malczyk, R. Aarts,
%   Data-Driven Inverse Dynamics Modeling Using Neural-Networks and
%   Regression-Based Techniques, 2024. If you use it, please refer to the
%   paper.
% 
%   *Corresponding author e-mail: maciej.pikulinski.dokt@pw.edu.pl

% Transform time-based settings into sample-based
p = s.did.pt / s.data.Ts;

% Build input matrix
X = [tData.input.q; tData.input.dq]; ids = (wEff + 1):(size(X, 2) - 1);
Z = [X(:, ids); X(:, (ids + 1))];

% Predict
pCounter = 0;
for i = 1:(size(Z, 2))
    if mod(pCounter, p) == 0
        pCounter = 0;
        model = D(:, :, i);
    end

    e(:, i) = model * Z(:, i);
    t(i) = (i + wEff) * s.data.Ts; 

    pCounter = pCounter + 1;
end

end

