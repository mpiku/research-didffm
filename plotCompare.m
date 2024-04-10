%   Script for plotting comparison of different control structures
%   including feedforward controller based on classical identification,
%   Lagrangian Neural Networks, Feedforward Neural Networks and their
%   combinations assisted by Data-Driven Inverse Dynamics controller.
%
%   Before plotting, please run the main script generating data: didModel.m
%   
%   This code illustrates methods from M. Pikulinski*, P. Malczyk, R. Aarts,
%   (2024), Data-Driven Inverse Dynamics Modeling Using Neural-Networks and
%   Regression-Based Techniques, Multibody System Dynamics (under review).
%   If you use it, please refer to the paper.
% 
%   *Corresponding author e-mail: maciej.pikulinski.dokt@pw.edu.pl

%% Settings
plotAct = 2; % Actuator to plot data (1 - 'Actuator Y', 2 - 'Actuator X')

%% Compute errors of different control structures
idsDownsampled = 1 + (ids - 1) * 1000 * s.data.Ts;
gTruth = data.f.fb(:, idsDownsampled);

errIdx = 0;

% Error of ident.-based FF (no data for spiral traj.)
if ~strcmp(s.data.name, "spiral")
    errIdx = errIdx + 1;
    errValues{errIdx} = gTruth - data.f.ident.r(:, idsDownsampled);
    labels{errIdx} = 'Ident.';
    colors{errIdx} = [0 0.4470 0.7410];
end

% Error of LNN-based FF
errIdx = errIdx + 1;
errValues{errIdx} = gTruth - data.f.nn.lnn.r(:, idsDownsampled);
labels{errIdx} = 'LNN';
colors{errIdx} = [0.8500 0.3250 0.0980];

% Error of LNN-FNN-based FF
errIdx = errIdx + 1;
errValues{errIdx} = gTruth - (data.f.nn.lnn.r(:, idsDownsampled) + ...
    data.f.nn.fnn.r(:, idsDownsampled));
labels{errIdx} = 'LNN, FNN';
colors{errIdx} = [0.9290 0.6940 0.1250];

% Error of LNN-FNN-based FF assisted by DID
errIdx = errIdx + 1;
errValues{errIdx} = gTruth - (data.f.nn.lnn.r(:, idsDownsampled) + ...
    data.f.nn.fnn.r(:, idsDownsampled) + e);
labels{errIdx} = 'LNN, FNN, DID';
colors{errIdx} = [0.4940 0.1840 0.5560];

% Find mean, std dev. and filter for presentation purposes
for i = 1:errIdx
    errMean{i} = mean(errValues{i}, 2);
    errStd{i} = std(errValues{i}, 0, 2);

    % Median filter
    errValues{i} = medfilt1(errValues{i}', 3, [], 1)';
end

%% Draw figure
h = figure(); set(h, 'units', 'centimeters', 'pos', [0 0 21 6]);

% Error time history plot
subplot(1, 2, 1);

for i = 1:errIdx
    plot(t, errValues{i}(plotAct, :), ...
        'LineWidth', 1.5, ...
        'Color', colors{i}, ...
        'DisplayName', sprintf('%s ($\\mu = %.3f$, $\\sigma = %.3f$)', ...
            labels{i}, errMean{i}(plotAct), errStd{i}(plotAct)));
    hold on;
end
line(xlim(), [0 0], ...
    'Color', [.2 .2 .2], ...
    'LineStyle', '--', ...
    'LineWidth', 1.5, ...
    'HandleVisibility', 'off');
hold off;

ylim([-0.5 0.5]);

ax = gca;
grid on; grid minor;
ax.GridColor = 'black';
ax.MinorGridColor = 'black';
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.2;

set(gca,'fontname','CMU Serif');
set(0,'defaulttextInterpreter','latex');

xlabel('Time [s]');
ylabel('Feedforward error [N]');
legend('Interpreter', 'latex');

% Error statistics plot
subplot(1, 2, 2);

line([0 (numel(errValues) + 1)], [0 0], ...
    'Color', [.2 .2 .2], 'LineStyle', '--', 'LineWidth', 1.5, ...
    'HandleVisibility', 'off'); hold on;
for i = 1:errIdx
    errorbar(i, errMean{i}(plotAct), ...
        errStd{i}(plotAct), ...
        'LineStyle', 'none', ...
        'Marker', 'x', ...
        'MarkerSize', 10, ...
        'CapSize', 10, ...
        'LineWidth', 1.5, ...
        'Color', colors{i}); hold on;
end
hold off;

xlim([0 (errIdx + 1)]);
ylim([-0.5 0.5]);

ax = gca;
grid on; grid minor;
ax.GridColor = 'black';
ax.MinorGridColor = 'black';
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.2;

set(gca,'fontname','CMU Serif');
set(0,'defaulttextInterpreter','latex');

ylabel('Error statistics [N]');
xticks(1:errIdx);
xticklabels(labels);
xAxisProperties= get(gca, 'XAxis');
xAxisProperties.TickLabelInterpreter = 'latex';