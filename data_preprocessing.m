%% =========================================================
% PERFORMANCE METRICS COMPARISON
% Proposed vs w/o CNN-GRU vs w/o Dragonfly
%% =========================================================

clc;
clear;
close all;

%% =========================================================
% METRIC VALUES
%% =========================================================

metrics = {'Throughput', ...
           'PDR', ...
           'Spectral Efficiency', ...
           'Latency', ...
           'Energy', ...
           'Interference'};

% Proposed Model
proposed = [532.5  99.86  5.32  5.06  2.92  0.0660];

% Without CNN-GRU
without_cnn = [395.7  81.89  3.90  6.11  4.36  0.1412];

% Without Dragonfly Optimization
without_da = [446.5  87.88  4.45  6.06  3.67  0.1097];

%% =========================================================
% CREATE FIGURES
%% =========================================================

figure('Color','white');

%% ---------------------------------------------------------
% 1. THROUGHPUT COMPARISON
%% ---------------------------------------------------------

subplot(2,3,1);

bar([proposed(1), without_cnn(1), without_da(1)]);

set(gca,'XTickLabel',{'Proposed','w/o CNN-GRU','w/o DA'});

ylabel('Mbps');

title('Throughput Comparison');

grid on;

%% ---------------------------------------------------------
% 2. PDR COMPARISON
%% ---------------------------------------------------------

subplot(2,3,2);

bar([proposed(2), without_cnn(2), without_da(2)]);

set(gca,'XTickLabel',{'Proposed','w/o CNN-GRU','w/o DA'});

ylabel('%');

title('PDR Comparison');

grid on;

%% ---------------------------------------------------------
% 3. SPECTRAL EFFICIENCY
%% ---------------------------------------------------------

subplot(2,3,3);

bar([proposed(3), without_cnn(3), without_da(3)]);

set(gca,'XTickLabel',{'Proposed','w/o CNN-GRU','w/o DA'});

ylabel('bps/Hz');

title('Spectral Efficiency');

grid on;

%% ---------------------------------------------------------
% 4. LATENCY COMPARISON
%% ---------------------------------------------------------

subplot(2,3,4);

bar([proposed(4), without_cnn(4), without_da(4)]);

set(gca,'XTickLabel',{'Proposed','w/o CNN-GRU','w/o DA'});

ylabel('ms');

title('Latency Comparison');

grid on;

%% ---------------------------------------------------------
% 5. ENERGY COMPARISON
%% ---------------------------------------------------------

subplot(2,3,5);

bar([proposed(5), without_cnn(5), without_da(5)]);

set(gca,'XTickLabel',{'Proposed','w/o CNN-GRU','w/o DA'});

ylabel('W');

title('Energy Consumption');

grid on;

%% ---------------------------------------------------------
% 6. INTERFERENCE COMPARISON
%% ---------------------------------------------------------

subplot(2,3,6);

bar([proposed(6), without_cnn(6), without_da(6)]);

set(gca,'XTickLabel',{'Proposed','w/o CNN-GRU','w/o DA'});

ylabel('Normalized');

title('Interference Comparison');

grid on;

%% =========================================================
% OVERALL TITLE
%% =========================================================

sgtitle('Hybrid CNN-GRU + DQN + Dragonfly Optimization');

disp('===================================================');
disp(' PERFORMANCE METRICS COMPARISON COMPLETED ');
disp('===================================================');