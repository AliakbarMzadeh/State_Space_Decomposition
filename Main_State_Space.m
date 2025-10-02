%% Re-run with filtering to create new result files
clear; close all; clc;

load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part2.mat');

dataCell     = Filtered_200_Part2;
fs           = 1000;
MAX_OSC      = 6;
MAX_AR       = 2*MAX_OSC;
startIdx     = 4;
endIdx       = 5;
outputFolder = 'OSC_Results';

%  Amplitude threshold
AMP_THRESHOLD = 0.01;  % Adjust this!

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for idx = startIdx:endIdx
    y = dataCell{idx};
    
    [osc_param, osc_AIC, osc_mean, ~, ~] = ...
        osc_decomp_uni(y, fs, MAX_OSC, MAX_AR);
    
    [~, K] = min(osc_AIC);
    osc_f = osc_param(K, K+1 : 2*K);
    
    M   = squeeze(osc_mean(:,:,K));
    T_i = size(M, 2);
    
    raw_ts = zeros(K, T_i);
    for k = 1:K
        raw_ts(k,:) = M(2*k-1, :);
    end
    
    p2p_raw   = max(raw_ts,[],2) - min(raw_ts,[],2);
    order_raw = floor(log10(p2p_raw));
    
    % Filter by amplitude
    validIdx = p2p_raw >= AMP_THRESHOLD;
    
    osc_f_valid     = osc_f(validIdx);
    p2p_raw_valid   = p2p_raw(validIdx);
    order_raw_valid = order_raw(validIdx);
    raw_ts_valid    = raw_ts(validIdx, :);
    K_valid         = sum(validIdx);
    
    if K_valid > 0
        y_hat  = sum(raw_ts_valid, 1);
        mseVal = mean((y - y_hat).^2);
    else
        y_hat  = zeros(size(y));
        mseVal = mean(y.^2);
    end
    
    % Save with ALL fields
    result.numComponents    = K_valid;
    result.numComponentsOrig = K;
    result.frequencies      = osc_f_valid;
    result.p2pRaw          = p2p_raw_valid;
    result.orderRaw        = order_raw_valid;
    result.mse             = mseVal;
    result.ampThreshold    = AMP_THRESHOLD;
    result.removedCount    = K - K_valid;
    
    fileName = sprintf('cell_%03d_results.mat', idx);
    save(fullfile(outputFolder, fileName), ...
         'result', 'osc_param', 'osc_f_valid');
    
    fprintf('Cell %03d: %d/%d components valid\n', idx, K_valid, K);
end

%% 


%% Summary Table - Simplified Style (Mean Values)

files = dir(fullfile('OSC_Results','cell_*_results.mat'));
N = numel(files);

% Preallocate containers
cellIdx       = zeros(N,1);
nCompsOrig    = zeros(N,1);
nCompsValid   = zeros(N,1);
removedCount  = zeros(N,1);
meanFreq      = zeros(N,1);
meanP2P       = zeros(N,1);
meanOrder     = zeros(N,1);
mseAll        = zeros(N,1);
ampThreshold  = zeros(N,1);

% Loop & extract
for i = 1:N
    % Parse index from filename
    name = files(i).name;
    idx  = sscanf(name,'cell_%d_results.mat');
    
    % Load
    S = load(fullfile(files(i).folder, name));
    
    % Fill values
    cellIdx(i) = idx;
    
    % Check if new format (with filtering fields)
    if isfield(S.result, 'numComponentsOrig')
        nCompsOrig(i)   = S.result.numComponentsOrig;
        nCompsValid(i)  = S.result.numComponents;
        removedCount(i) = S.result.removedCount;
        ampThreshold(i) = S.result.ampThreshold;
    else
        nCompsOrig(i)   = S.result.numComponents;
        nCompsValid(i)  = S.result.numComponents;
        removedCount(i) = 0;
        ampThreshold(i) = NaN;
    end
    
    % Calculate means (handle empty case)
    if nCompsValid(i) > 0
        meanFreq(i)  = mean(S.result.frequencies);
        meanP2P(i)   = mean(S.result.p2pRaw);
        meanOrder(i) = mean(S.result.orderRaw);
    else
        meanFreq(i)  = NaN;
        meanP2P(i)   = NaN;
        meanOrder(i) = NaN;
    end
    
    mseAll(i) = S.result.mse;
end

% Build and display table
T = table(cellIdx, nCompsOrig, nCompsValid, removedCount, ...
          meanFreq, meanP2P, meanOrder, mseAll, ampThreshold, ...
    'VariableNames', {
      'CellIndex','OrigOsc','ValidOsc','Removed', ...
      'MeanFreq','MeanP2P','MeanOrder','MSE','Threshold'});

disp(T);

% Summary statistics
fprintf('\n========== SUMMARY STATISTICS ==========\n');
fprintf('Total cells processed: %d\n', N);
fprintf('Cells with valid components: %d (%.1f%%)\n', ...
        sum(nCompsValid > 0), 100*sum(nCompsValid > 0)/N);
fprintf('Cells with NO valid components: %d (%.1f%%)\n', ...
        sum(nCompsValid == 0), 100*sum(nCompsValid == 0)/N);
fprintf('Average valid components per cell: %.2f\n', mean(nCompsValid));
fprintf('Total components removed: %d\n', sum(removedCount));
if ~isnan(ampThreshold(1))
    fprintf('Amplitude threshold used: %.4f\n', ampThreshold(1));
else
    fprintf('⚠️  Old files detected - no filtering applied\n');
end
fprintf('========================================\n');
%%
%% Display Valid Frequencies for All Cells

files = dir(fullfile('OSC_Results','cell_*_results.mat'));

fprintf('\n========== VALID FREQUENCIES [Hz] ==========\n');
for i = 1:numel(files)
    name = files(i).name;
    idx  = sscanf(name,'cell_%d_results.mat');
    S    = load(fullfile(files(i).folder, name));
    
    fprintf('Cell %03d: ', idx);
    if S.result.numComponents > 0
        fprintf('%s\n', mat2str(S.result.frequencies, 2));
    else
        fprintf('No valid components\n');
    end
end
fprintf('=============================================\n');
%%
%% Plot Valid Components for Specific Cell

cellNum = 4;  % ⭐ Change this to the cell you want to plot

% Load original data
load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part2.mat');
y_original = Filtered_200_Part2{cellNum};

% Load results
fname = fullfile('OSC_Results', sprintf('cell_%03d_results.mat', cellNum));
S = load(fname);

% Reconstruct individual components from osc_param
K = S.result.numComponents;
if K == 0
    warning('Cell %d has no valid components!', cellNum);
    return;
end

% Get full osc_mean for reconstruction
[osc_param, osc_AIC, osc_mean, ~, ~] = ...
    osc_decomp_uni(y_original, 1000, 6, 12);

[~, K_orig] = min(osc_AIC);
M = squeeze(osc_mean(:,:,K_orig));
T_i = size(M, 2);

% Extract all components (including invalid ones)
raw_ts_all = zeros(K_orig, T_i);
for k = 1:K_orig
    raw_ts_all(k,:) = M(2*k-1, :);
end

% Calculate p2p and filter
p2p_all = max(raw_ts_all,[],2) - min(raw_ts_all,[],2);
validIdx = p2p_all >= S.result.ampThreshold;
raw_ts_valid = raw_ts_all(validIdx, :);

% Time vector
t = (0:T_i-1) / 1000;  % time in seconds

% Create figure
figure('Position', [100 100 1200 800]);

% Plot 1: Original Signal
subplot(K+2, 1, 1);
plot(t, y_original, 'k', 'LineWidth', 1);
title(sprintf('Cell %03d - Original Signal', cellNum), 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Amplitude');
grid on;

% Plot 2-K+1: Individual Valid Components
for k = 1:K
    subplot(K+2, 1, k+1);
    plot(t, raw_ts_valid(k,:), 'LineWidth', 1.5);
    title(sprintf('Component %d: %.2f Hz (p2p=%.4f)', ...
          k, S.result.frequencies(k), S.result.p2pRaw(k)), 'FontSize', 10);
    ylabel('Amplitude');
    grid on;
end

% Plot K+2: Reconstructed Signal
subplot(K+2, 1, K+2);
y_reconstructed = sum(raw_ts_valid, 1);
plot(t, y_original, 'k', 'LineWidth', 1, 'DisplayName', 'Original'); hold on;
plot(t, y_reconstructed, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Reconstructed');
title(sprintf('Reconstruction (MSE=%.6f)', S.result.mse), 'FontSize', 12);
xlabel('Time (s)');
ylabel('Amplitude');
legend('Location', 'best');
grid on;

sgtitle(sprintf('Cell %03d - Valid Components (Threshold=%.4f)', cellNum, S.result.ampThreshold), ...
        'FontSize', 14, 'FontWeight', 'bold');