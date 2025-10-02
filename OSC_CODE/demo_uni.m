% Define paths
%%
%% Synthetic dataset (N = 250) + OSC decomposition + full reporting + extra OSC plots
% Guarantees:
%   - Each signal has K_true ∈ {2,3,4}
%   - Oscillator freqs ∈ [7, 20] Hz
%   - fs = 250 Hz, N = 250 samples
%   - Saves per-signal truth, model outputs (osc_param-derived), MSE
%   - Plots saved per signal:
%       components_orig.png  : osc components, non-osc, noise, original
%       recon.png            : original vs reconstruction + est amp vs freq
%       oscplot.png          : osc_plot(osc_mean, osc_cov, fs, K_est)
%       phaseplot.png        : osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K_est)
%       spectrum.png         : osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, est_tau2_vec)

clear; close all; clc;

% ----------------------- Fixed parameters (exact) -----------------------
rng(123,'twister');

NUM_SIGNALS       = 10;
fs                = 250;         % Hz
N                 = 250;         % samples
t                 = (0:N-1)/fs;

% Oscillatory truth constraints
K_min             = 2;
K_max             = 4;
f_min             = 5;           % Hz
f_max             = 40;          % Hz
A_min             = 0.10;
A_max             = 1.00;

% Non-oscillatory (trend + exponential decay)
poly_c0_range     = [-0.25, 0.25];
poly_c1_range     = [-0.20, 0.20];
poly_c2_range     = [-0.05, 0.05];
exp_amp_range     = [0.00, 0.60];
exp_tau_range     = [0.08, 0.60]; % seconds

% Noise: AR(1) + white Gaussian
ar1_phi_range     = [0.00, 0.90];
ar1_sigma_range   = [0.00, 0.20];
white_sigma_range = [0.00, 0.20];

% Decomposition controls
MAX_OSC           = 7;
MAX_AR            = 2*MAX_OSC;

% I/O
datasetMatPath    = 'Synthetic_OSC_250.mat';
outputFolder      = 'OSC_Results_250';
plotsFolder       = fullfile(outputFolder,'plots');

% --------------------------- Output directories -------------------------
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
if ~exist(plotsFolder,'dir'); mkdir(plotsFolder); end

% ---------------------- Dataset generation (+ plots) --------------------
dataCell = cell(NUM_SIGNALS,1);
truth(NUM_SIGNALS,1) = struct( ...
    'K',0,'freqs',[],'amps',[],'phases',[], ...
    'poly_c',[0 0 0],'exp_amp',0,'exp_tau',0, ...
    'ar1_phi',0,'ar1_sigma',0,'white_sigma',0);

for i = 1:NUM_SIGNALS
    % ----- Oscillatory components (truth) -----
    K_true      = randi([K_min, K_max],1,1);
    freqs_true  = f_min + (f_max - f_min)*rand(1,K_true);
    amps_true   = A_min + (A_max - A_min)*rand(1,K_true);
    phases_true = 2*pi*rand(1,K_true);

    osc_components = zeros(K_true, N);
    for k = 1:K_true
        osc_components(k,:) = amps_true(k) * sin(2*pi*freqs_true(k)*t + phases_true(k));
    end
    osc_sum = sum(osc_components, 1);

    % ----- Non-oscillatory (trend + exponential) -----
    c0 = poly_c0_range(1) + diff(poly_c0_range)*rand;
    c1 = poly_c1_range(1) + diff(poly_c1_range)*rand;
    c2 = poly_c2_range(1) + diff(poly_c2_range)*rand;
    trend = c0 + c1*t + c2*(t.^2);

    b_exp = exp_amp_range(1) + diff(exp_amp_range)*rand;
    tau   = exp_tau_range(1) + diff(exp_tau_range)*rand;
    decay = b_exp * exp(-t./tau);

    nonosc = trend + decay;

    % ----- Noise (AR(1) + white) -----
    phi     = ar1_phi_range(1)   + diff(ar1_phi_range)*rand;
    sig_ar1 = ar1_sigma_range(1) + diff(ar1_sigma_range)*rand;
    sig_wn  = white_sigma_range(1)+ diff(white_sigma_range)*rand;

    e  = sig_ar1*randn(1,N);
    ar = zeros(1,N);
    for n = 2:N
        ar(n) = phi*ar(n-1) + e(n);
    end
    wn    = sig_wn*randn(1,N);
    noise = ar + wn;

    % ----- Final signal (original) -----
    y_total = osc_sum + nonosc + noise;

    % ----- Store truth and signal -----
    dataCell{i}        = y_total;
    truth(i).K         = K_true;
    truth(i).freqs     = freqs_true;
    truth(i).amps      = amps_true;
    truth(i).phases    = phases_true;
    truth(i).poly_c    = [c0 c1 c2];
    truth(i).exp_amp   = b_exp;
    truth(i).exp_tau   = tau;
    truth(i).ar1_phi   = phi;
    truth(i).ar1_sigma = sig_ar1;
    truth(i).white_sigma = sig_wn;

    % ----- Plot: osc components, non-osc, noise, original -----
    fig = figure('Visible','off','Units','pixels','Position',[80 80 1250 900]);
    tiledlayout(4,1,'Padding','compact','TileSpacing','compact');

    % Oscillator components
    nexttile;
    hold on;
    for k = 1:K_true
        plot(t, osc_components(k,:), 'LineWidth', 1.0);
    end
    hold off; grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    lgd = compose('f=%.2f Hz', freqs_true);
    legend(lgd, 'Location','northeastoutside');
    title(sprintf('Signal %d — Oscillatory Components (K_{true}=%d)', i, K_true));

    % Non-oscillatory
    nexttile;
    plot(t, nonosc, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Non-oscillatory (trend + exponential)');

    % Noise
    nexttile;
    plot(t, noise, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Noise (AR(1) + white)');

    % Original
    nexttile;
    plot(t, y_total, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Original Signal = Oscillatory + Non-oscillatory + Noise');

    exportgraphics(fig, fullfile(plotsFolder, sprintf('cell_%03d_components_orig.png',i)), 'Resolution', 150);
    close(fig);
end

% Persist dataset
save(datasetMatPath,'dataCell','fs','t','truth','-v7.3');

% ------------------ Decomposition + per-signal reporting ----------------
summary(NUM_SIGNALS,1) = struct( ...
    'idx',0, ...
    'trueK',0, ...
    'trueFreqs','', ...
    'trueAmps','', ...
    'estK',0, ...
    'estFreqs','', ...
    'mse',0.0 );

txtReportPath = fullfile(outputFolder,'summary_all.txt');
fid_txt = fopen(txtReportPath,'w');
fprintf(fid_txt,'idx,trueK,trueFreqs,estK,estFreqs,MSE\n');

for idx = 1:NUM_SIGNALS
    y = dataCell{idx};

    % --- Run OSC model (5 outputs expected) ---
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp_uni(y, fs, MAX_OSC, MAX_AR);

    % --- Select K and extract parameters from row K_est ---
    [~, K_est] = min(osc_AIC);

    if K_est > 0
        rowK        = osc_param(K_est, :);
        est_a       = rowK(1:K_est);                   % per-component 'a'
        est_freqs   = rowK(K_est+1 : 2*K_est);         % per-component freq [Hz]

        remainder   = rowK(2*K_est+1 : end);

        % Per-component sigma2 and tau2 extraction (robust to multiple layouts)
        if numel(remainder) >= K_est + 1
            % [sigma2_1..sigma2_K, tau2_scalar]
            est_sigma2_vec = remainder(1:K_est);
            tau2_scalar    = remainder(K_est+1);
            est_tau2_vec   = repmat(tau2_scalar, 1, K_est);
        elseif numel(remainder) >= 2*K_est
            % [sigma2_1..sigma2_K, tau2_1..tau2_K]
            est_sigma2_vec = remainder(1:K_est);
            est_tau2_vec   = remainder(K_est+1 : 2*K_est);
        elseif numel(remainder) >= 2
            % [sigma2_scalar, tau2_scalar] -> replicate
            est_sigma2_vec = repmat(remainder(1), 1, K_est);
            est_tau2_vec   = repmat(remainder(2), 1, K_est);
        elseif numel(remainder) == 1
            est_sigma2_vec = repmat(remainder(1), 1, K_est);
            est_tau2_vec   = zeros(1, K_est);
        else
            est_sigma2_vec = zeros(1, K_est);
            est_tau2_vec   = zeros(1, K_est);
        end

        % --- Smoothed states for selected K and reconstruction ---
        M_sel   = squeeze(osc_mean(:,:,K_est));        % (2*K_est) x N
        raw_ts  = zeros(K_est, N);
        for k = 1:K_est
            raw_ts(k,:) = M_sel(2*k-1, :);             % first coord per oscillator
        end
        y_hat  = sum(raw_ts,1);
        p2pRaw = max(raw_ts,[],2) - min(raw_ts,[],2);
        ordRaw = floor(log10(max(p2pRaw, realmin)));
    else
        est_a = [];
        est_freqs = [];
        est_sigma2_vec = [];
        est_tau2_vec   = [];
        M_sel  = zeros(0,N);
        raw_ts = zeros(0,N);
        y_hat  = zeros(1,N);
        p2pRaw = [];
        ordRaw = [];
    end

    % --- MSE of the reconstruction from predicted components ---
    mseVal = mean((y - y_hat).^2);

    % --- Save per-signal MAT output (includes full osc_param + derived) ---
    result.numComponents = K_est;
    result.frequencies   = est_freqs(:).';
    result.osc_a         = est_a(:).';
    result.p2pRaw        = p2pRaw(:).';
    result.orderRaw      = ordRaw(:).';
    result.sigma2        = est_sigma2_vec(:).';
    result.tau2          = est_tau2_vec(:).';
    result.mse           = mseVal;

    save(fullfile(outputFolder, sprintf('cell_%03d_results.mat',idx)), ...
         'result','osc_param','osc_AIC','osc_mean','osc_cov','osc_phase', ...
         'est_freqs','est_a','M_sel');

    % --- Plot: Original vs Reconstruction + est amplitudes vs frequency ---
    fig1 = figure('Visible','off','Units','pixels','Position',[80 80 1150 520]);
    tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    nexttile;
    plot(t, y, 'LineWidth', 1.0); hold on;
    plot(t, y_hat, 'LineWidth', 1.0); hold off; grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title(sprintf('Signal %d — Original vs Reconstruction (K_{est}=%d, MSE=%.4g)', idx, K_est, mseVal));
    legend({'Original','Sum of predicted components'},'Location','best');

    nexttile;
    if K_est > 0
        [f_sorted, ord] = sort(est_freqs(:).');
        p2p_sorted = p2pRaw(ord);
        bar(f_sorted, p2p_sorted); grid on;
        xlabel('Frequency [Hz]'); ylabel('Peak-to-peak (raw)');
        title('Estimated Component Amplitudes vs Frequency');
    else
        axis off;
    end

    exportgraphics(fig1, fullfile(plotsFolder, sprintf('cell_%03d_recon.png',idx)), 'Resolution', 150);
    close(fig1);

    % --- Extra requested plots (robust capture of the figure handle) ---
    if K_est > 0
        % 1) osc_plot(osc_mean, osc_cov, fs, K_est)
        figs_before = findobj('Type','figure');
        osc_plot(osc_mean, osc_cov, fs, K_est);
        drawnow;
        figs_after  = findobj('Type','figure');
        newFigs     = setdiff(figs_after, figs_before, 'stable');
        if isempty(newFigs)
            h = get(0,'CurrentFigure');
        else
            h = newFigs(end);
        end
        if ~isempty(h) && ishghandle(h)
            set(h,'Visible','off'); drawnow;
            exportgraphics(h, fullfile(plotsFolder, sprintf('cell_%03d_oscplot.png',idx)), 'Resolution', 150);
            close(h);
        end

        % 2) osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K_est)
        figs_before = findobj('Type','figure');
        osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K_est);
        drawnow;
        figs_after  = findobj('Type','figure');
        newFigs     = setdiff(figs_after, figs_before, 'stable');
        if isempty(newFigs)
            h = get(0,'CurrentFigure');
        else
            h = newFigs(end);
        end
        if ~isempty(h) && ishghandle(h)
            set(h,'Visible','off'); drawnow;
            exportgraphics(h, fullfile(plotsFolder, sprintf('cell_%03d_phaseplot.png',idx)), 'Resolution', 150);
            close(h);
        end

        % 3) osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, est_tau2_vec)
        figs_before = findobj('Type','figure');
        try
            osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, est_tau2_vec);
        catch
            tau2_scalar = 0;
            if ~isempty(est_tau2_vec), tau2_scalar = est_tau2_vec(1); end
            osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, tau2_scalar);
        end
        drawnow;
        figs_after  = findobj('Type','figure');
        newFigs     = setdiff(figs_after, figs_before, 'stable');
        if isempty(newFigs)
            h = get(0,'CurrentFigure');
        else
            h = newFigs(end);
        end
        if ~isempty(h) && ishghandle(h)
            set(h,'Visible','off'); drawnow;
            exportgraphics(h, fullfile(plotsFolder, sprintf('cell_%03d_spectrum.png',idx)), 'Resolution', 150);
            close(h);
        end
    end

    % --- Update summary (true vs estimated) ---
    tf_cells = arrayfun(@(x) sprintf('%.4f', x), truth(idx).freqs, 'UniformOutput', false);
    ta_cells = arrayfun(@(x) sprintf('%.4f', x), truth(idx).amps,  'UniformOutput', false);
    ef_cells = arrayfun(@(x) sprintf('%.4f', x), est_freqs,        'UniformOutput', false);
    trueFreqs_str = ['[', strjoin(tf_cells, ', '), ']'];
    trueAmps_str  = ['[', strjoin(ta_cells, ', '), ']'];
    estFreqs_str  = ['[', strjoin(ef_cells, ', '), ']'];

    summary(idx).idx       = idx;
    summary(idx).trueK     = truth(idx).K;
    summary(idx).trueFreqs = trueFreqs_str;
    summary(idx).trueAmps  = trueAmps_str;
    summary(idx).estK      = K_est;
    summary(idx).estFreqs  = estFreqs_str;
    summary(idx).mse       = mseVal;

    fprintf(fid_txt,'%d,%d,%s,%d,%s,%.10g\n', ...
        idx, truth(idx).K, trueFreqs_str, K_est, estFreqs_str, mseVal);
end
fclose(fid_txt);

% Aggregate summary
summaryTable = struct2table(summary);
save(fullfile(outputFolder,'summary_all.mat'),'summary','summaryTable');
writetable(summaryTable, fullfile(outputFolder,'summary_all.csv'));

% ------------------------------ Console log -----------------------------
fprintf('DONE\n');
fprintf('Dataset: %s\n', datasetMatPath);
fprintf('Per-signal results: %s\n', outputFolder);
fprintf('Plots: %s\n', plotsFolder);
fprintf('Summary: %s and %s\n', ...
    fullfile(outputFolder,'summary_all.mat'), fullfile(outputFolder,'summary_all.csv'));
fprintf('TXT report: %s\n', fullfile(outputFolder,'summary_all.txt'));

















































%%






clear; close all; clc;
%%
%% Synthetic dataset (N = 250) + OSC decomposition + full reporting + extra OSC plots
% Exact guarantees:
%   - Each signal has K_true ∈ {2,3,4}
%   - Oscillator freqs ∈ [7, 20] Hz
%   - fs = 250 Hz, N = 250 samples
%   - Saves per-signal truth, model outputs (osc_param, osc_f via extraction), MSE
%   - Plots saved per signal:
%       1) components_orig.png  : osc components, non-osc, noise, original
%       2) recon.png            : original vs reconstruction + est amp vs freq
%       3) oscplot.png          : osc_plot(osc_mean, osc_cov, fs, K_est)
%       4) phaseplot.png        : osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K_est)
%       5) spectrum.png         : osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, est_tau2_vec)

clear; close all; clc;

% ----------------------- Fixed parameters (exact) -----------------------
rng(123,'twister');

NUM_SIGNALS       = 1;
fs                = 250;         % Hz
N                 = 250;         % samples
t                 = (0:N-1)/fs;

% Oscillatory truth constraints
K_min             = 2;
K_max             = 4;
f_min             = 7;           % Hz
f_max             = 20;          % Hz
A_min             = 0.10;
A_max             = 1.00;

% Non-oscillatory (trend + exponential decay)
poly_c0_range     = [-0.25, 0.25];
poly_c1_range     = [-0.20, 0.20];
poly_c2_range     = [-0.05, 0.05];
exp_amp_range     = [0.00, 0.60];
exp_tau_range     = [0.08, 0.60]; % seconds

% Noise: AR(1) + white Gaussian
ar1_phi_range     = [0.00, 0.90];
ar1_sigma_range   = [0.00, 0.20];
white_sigma_range = [0.00, 0.20];

% Decomposition controls
MAX_OSC           = 7;
MAX_AR            = 2*MAX_OSC;

% I/O
datasetMatPath    = 'Synthetic_OSC_250.mat';
outputFolder      = 'OSC_Results_250';
plotsFolder       = fullfile(outputFolder,'plots');

% --------------------------- Output directories -------------------------
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
if ~exist(plotsFolder,'dir'); mkdir(plotsFolder); end

% ---------------------- Dataset generation (+ plots) --------------------
dataCell = cell(NUM_SIGNALS,1);
truth(NUM_SIGNALS,1) = struct( ...
    'K',0,'freqs',[],'amps',[],'phases',[], ...
    'poly_c',[0 0 0],'exp_amp',0,'exp_tau',0, ...
    'ar1_phi',0,'ar1_sigma',0,'white_sigma',0);

for i = 1:NUM_SIGNALS
    % ----- Oscillatory components (truth) -----
    K_true      = randi([K_min, K_max],1,1);
    freqs_true  = f_min + (f_max - f_min)*rand(1,K_true);
    amps_true   = A_min + (A_max - A_min)*rand(1,K_true);
    phases_true = 2*pi*rand(1,K_true);

    osc_components = zeros(K_true, N);
    for k = 1:K_true
        osc_components(k,:) = amps_true(k) * sin(2*pi*freqs_true(k)*t + phases_true(k));
    end
    osc_sum = sum(osc_components, 1);

    % ----- Non-oscillatory (trend + exponential) -----
    c0 = poly_c0_range(1) + diff(poly_c0_range)*rand;
    c1 = poly_c1_range(1) + diff(poly_c1_range)*rand;
    c2 = poly_c2_range(1) + diff(poly_c2_range)*rand;
    trend = c0 + c1*t + c2*(t.^2);

    b_exp = exp_amp_range(1) + diff(exp_amp_range)*rand;
    tau   = exp_tau_range(1) + diff(exp_tau_range)*rand;
    decay = b_exp * exp(-t./tau);

    nonosc = trend + decay;

    % ----- Noise (AR(1) + white) -----
    phi     = ar1_phi_range(1)   + diff(ar1_phi_range)*rand;
    sig_ar1 = ar1_sigma_range(1) + diff(ar1_sigma_range)*rand;
    sig_wn  = white_sigma_range(1)+ diff(white_sigma_range)*rand;

    e  = sig_ar1*randn(1,N);
    ar = zeros(1,N);
    for n = 2:N
        ar(n) = phi*ar(n-1) + e(n);
    end
    wn    = sig_wn*randn(1,N);
    noise = ar + wn;

    % ----- Final signal (original) -----
    y_total = osc_sum + nonosc + noise;

    % ----- Store truth and signal -----
    dataCell{i}        = y_total;
    truth(i).K         = K_true;
    truth(i).freqs     = freqs_true;
    truth(i).amps      = amps_true;
    truth(i).phases    = phases_true;
    truth(i).poly_c    = [c0 c1 c2];
    truth(i).exp_amp   = b_exp;
    truth(i).exp_tau   = tau;
    truth(i).ar1_phi   = phi;
    truth(i).ar1_sigma = sig_ar1;
    truth(i).white_sigma = sig_wn;

    % ----- Plot: osc components, non-osc, noise, original -----
    fig = figure('Visible','off','Units','pixels','Position',[80 80 1250 900]);
    tiledlayout(4,1,'Padding','compact','TileSpacing','compact');

    % Oscillator components
    nexttile;
    hold on;
    for k = 1:K_true
        plot(t, osc_components(k,:), 'LineWidth', 1.0);
    end
    hold off; grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    lgd = compose('f=%.2f Hz', freqs_true);
    legend(lgd, 'Location','northeastoutside');
    title(sprintf('Signal %d — Oscillatory Components (K_{true}=%d)', i, K_true));

    % Non-oscillatory
    nexttile;
    plot(t, nonosc, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Non-oscillatory (trend + exponential)');

    % Noise
    nexttile;
    plot(t, noise, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Noise (AR(1) + white)');

    % Original
    nexttile;
    plot(t, y_total, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Original Signal = Oscillatory + Non-oscillatory + Noise');

    exportgraphics(fig, fullfile(plotsFolder, sprintf('cell_%03d_components_orig.png',i)), 'Resolution', 150);
    close(fig);
end

% Persist dataset
save(datasetMatPath,'dataCell','fs','t','truth','-v7.3');

% ------------------ Decomposition + per-signal reporting ----------------
summary(NUM_SIGNALS,1) = struct( ...
    'idx',0, ...
    'trueK',0, ...
    'trueFreqs','', ...
    'trueAmps','', ...
    'estK',0, ...
    'estFreqs','', ...
    'mse',0.0 );

txtReportPath = fullfile(outputFolder,'summary_all.txt');
fid_txt = fopen(txtReportPath,'w');
fprintf(fid_txt,'idx,trueK,trueFreqs,estK,estFreqs,MSE\n');

for idx = 1:NUM_SIGNALS
    y = dataCell{idx};

    % --- Run OSC model (5 outputs expected) ---
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp_uni(y, fs, MAX_OSC, MAX_AR);

    % --- Select K and extract parameters from row K_est ---
    [~, K_est] = min(osc_AIC);

    if K_est > 0
        rowK        = osc_param(K_est, :);
        est_a       = rowK(1:K_est);                   % per-component 'a'
        est_freqs   = rowK(K_est+1 : 2*K_est);         % per-component freq [Hz]

        remainder   = rowK(2*K_est+1 : end);

        % Per-component sigma2 and tau2 extraction (robust to multiple layouts)
        if numel(remainder) >= K_est + 1
            % [sigma2_1..sigma2_K, tau2_scalar]
            est_sigma2_vec = remainder(1:K_est);
            tau2_scalar    = remainder(K_est+1);
            est_tau2_vec   = repmat(tau2_scalar, 1, K_est);
        elseif numel(remainder) >= 2*K_est
            % [sigma2_1..sigma2_K, tau2_1..tau2_K]
            est_sigma2_vec = remainder(1:K_est);
            est_tau2_vec   = remainder(K_est+1 : 2*K_est);
        elseif numel(remainder) >= 2
            % [sigma2_scalar, tau2_scalar] -> replicate
            est_sigma2_vec = repmat(remainder(1), 1, K_est);
            est_tau2_vec   = repmat(remainder(2), 1, K_est);
        elseif numel(remainder) == 1
            est_sigma2_vec = repmat(remainder(1), 1, K_est);
            est_tau2_vec   = zeros(1, K_est);
        else
            est_sigma2_vec = zeros(1, K_est);
            est_tau2_vec   = zeros(1, K_est);
        end

        % --- Smoothed states for selected K and reconstruction ---
        M_sel   = squeeze(osc_mean(:,:,K_est));        % (2*K_est) x N
        raw_ts  = zeros(K_est, N);
        for k = 1:K_est
            raw_ts(k,:) = M_sel(2*k-1, :);             % first coord per oscillator
        end
        y_hat  = sum(raw_ts,1);
        p2pRaw = max(raw_ts,[],2) - min(raw_ts,[],2);
        ordRaw = floor(log10(max(p2pRaw, realmin)));
    else
        est_a = [];
        est_freqs = [];
        est_sigma2_vec = [];
        est_tau2_vec   = [];
        M_sel  = zeros(0,N);
        raw_ts = zeros(0,N);
        y_hat  = zeros(1,N);
        p2pRaw = [];
        ordRaw = [];
    end

    % --- MSE of the reconstruction from predicted components ---
    mseVal = mean((y - y_hat).^2);

    % --- Save per-signal MAT output (includes full osc_param + derived) ---
    result.numComponents = K_est;
    result.frequencies   = est_freqs(:).';
    result.osc_a         = est_a(:).';
    result.p2pRaw        = p2pRaw(:).';
    result.orderRaw      = ordRaw(:).';
    result.sigma2        = est_sigma2_vec(:).';        % per-component
    result.tau2          = est_tau2_vec(:).';          % per-component (replicated if scalar)
    result.mse           = mseVal;

    save(fullfile(outputFolder, sprintf('cell_%03d_results.mat',idx)), ...
         'result','osc_param','osc_AIC','osc_mean','osc_cov','osc_phase', ...
         'est_freqs','est_a','M_sel');

    % --- Plot: Original vs Reconstruction + est amplitudes vs frequency ---
    fig1 = figure('Visible','off','Units','pixels','Position',[80 80 1150 520]);
    tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    nexttile;
    plot(t, y, 'LineWidth', 1.0); hold on;
    plot(t, y_hat, 'LineWidth', 1.0); hold off; grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title(sprintf('Signal %d — Original vs Reconstruction (K_{est}=%d, MSE=%.4g)', idx, K_est, mseVal));
    legend({'Original','Sum of predicted components'},'Location','best');

    nexttile;
    if K_est > 0
        [f_sorted, ord] = sort(est_freqs(:).');
        p2p_sorted = p2pRaw(ord);
        bar(f_sorted, p2p_sorted); grid on;
        xlabel('Frequency [Hz]'); ylabel('Peak-to-peak (raw)');
        title('Estimated Component Amplitudes vs Frequency');
    else
        axis off;
    end

    exportgraphics(fig1, fullfile(plotsFolder, sprintf('cell_%03d_recon.png',idx)), 'Resolution', 150);
    close(fig1);

    % --- Extra requested plots (saved to files) ---
    if K_est > 0
        % 1) osc_plot(osc_mean, osc_cov, fs, K_est)
        f_osc = figure('Visible','off');
        osc_plot(osc_mean, osc_cov, fs, K_est);
        exportgraphics(f_osc, fullfile(plotsFolder, sprintf('cell_%03d_oscplot.png',idx)), 'Resolution', 150);
        close(f_osc);

        % 2) osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K_est)
        f_phase = figure('Visible','off');
        osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K_est);
        exportgraphics(f_phase, fullfile(plotsFolder, sprintf('cell_%03d_phaseplot.png',idx)), 'Resolution', 150);
        close(f_phase);

        % 3) osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, est_tau2_vec)
        f_spec = figure('Visible','off');
        try
            osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, est_tau2_vec);
        catch
            % If implementation expects scalar tau2, pass first element as scalar
            tau2_scalar = 0;
            if ~isempty(est_tau2_vec)
                tau2_scalar = est_tau2_vec(1);
            end
            osc_spectrum_plot(y, fs, est_a, est_freqs, est_sigma2_vec, tau2_scalar);
        end
        exportgraphics(f_spec, fullfile(plotsFolder, sprintf('cell_%03d_spectrum.png',idx)), 'Resolution', 150);
        close(f_spec);
    end

    % --- Update summary (true vs estimated) ---
    tf_cells = arrayfun(@(x) sprintf('%.4f', x), truth(idx).freqs, 'UniformOutput', false);
    ta_cells = arrayfun(@(x) sprintf('%.4f', x), truth(idx).amps,  'UniformOutput', false);
    ef_cells = arrayfun(@(x) sprintf('%.4f', x), est_freqs,        'UniformOutput', false);
    trueFreqs_str = ['[', strjoin(tf_cells, ', '), ']'];
    trueAmps_str  = ['[', strjoin(ta_cells, ', '), ']'];
    estFreqs_str  = ['[', strjoin(ef_cells, ', '), ']'];

    summary(idx).idx       = idx;
    summary(idx).trueK     = truth(idx).K;
    summary(idx).trueFreqs = trueFreqs_str;
    summary(idx).trueAmps  = trueAmps_str;
    summary(idx).estK      = K_est;
    summary(idx).estFreqs  = estFreqs_str;
    summary(idx).mse       = mseVal;

    fprintf(fid_txt,'%d,%d,%s,%d,%s,%.10g\n', ...
        idx, truth(idx).K, trueFreqs_str, K_est, estFreqs_str, mseVal);
end
fclose(fid_txt);

% Aggregate summary
summaryTable = struct2table(summary);
save(fullfile(outputFolder,'summary_all.mat'),'summary','summaryTable');
writetable(summaryTable, fullfile(outputFolder,'summary_all.csv'));

% ------------------------------ Console log -----------------------------
fprintf('DONE\n');
fprintf('Dataset: %s\n', datasetMatPath);
fprintf('Per-signal results: %s\n', outputFolder);
fprintf('Plots: %s\n', plotsFolder);
fprintf('Summary: %s and %s\n', ...
    fullfile(outputFolder,'summary_all.mat'), fullfile(outputFolder,'summary_all.csv'));
fprintf('TXT report: %s\n', fullfile(outputFolder,'summary_all.txt'));



%%















load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/New_Session_Part_Filtered/Filtered_200_Part2.mat')


%%
%% Synthetic dataset (N = 250 samples) + OSC decomposition + full reporting
% - Each signal has K_true ∈ {2,3,4} oscillators
% - Oscillator frequencies uniform in [7, 20] Hz
% - All signals have exactly N = 250 samples at fs = 250 Hz
% - For every signal, save:
%     • Ground-truth oscillatory freqs/amps/phases
%     • Non-oscillatory parameters (poly + exponential)
%     • Noise parameters (AR(1) + white)
%     • Model outputs: osc_param, osc_f, AIC-selected K, MSE
% - For every signal, produce two figures:
%     • components_orig.png  (oscillators, non-osc, noise, original)
%     • recon.png            (original vs reconstruction + est amplitudes vs freq)
% - Save per-signal .mat files and an aggregated summary as .mat and .csv

clear; close all; clc;


% ----------------------- Fixed parameters (exact) -----------------------
rng(123,'twister');

NUM_SIGNALS       = 1;         % total number of synthetic signals
fs                = 250;         % sampling frequency [Hz]
N                 = 250;         % samples (exact)
t                 = (0:N-1)/fs;  % time vector [s]

% Oscillatory truth constraints
K_min             = 2;           % minimum oscillators
K_max             = 4;           % maximum oscillators
f_min             = 7;           % Hz
f_max             = 50;          % Hz
A_min             = 0.10;        % amplitude min
A_max             = 1.00;        % amplitude max

% Non-oscillatory (trend + exponential decay)
poly_c0_range     = [-0.25, 0.25];
poly_c1_range     = [-0.20, 0.20];
poly_c2_range     = [-0.05, 0.05];
exp_amp_range     = [0.00, 0.60];
exp_tau_range     = [0.08, 0.60]; % seconds

% Noise: AR(1) + white Gaussian
ar1_phi_range     = [0.00, 0.90];
ar1_sigma_range   = [0.00, 0.20];
white_sigma_range = [0.00, 0.20];

% Decomposition controls
MAX_OSC           = 7;
MAX_AR            = 2*MAX_OSC;

% I/O
datasetMatPath    = 'Synthetic_OSC_250.mat';
outputFolder      = 'OSC_Results_250';
plotsFolder       = fullfile(outputFolder,'plots');

% --------------------------- Output directories -------------------------
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
if ~exist(plotsFolder,'dir'); mkdir(plotsFolder); end

% ---------------------- Dataset generation (+ plots) --------------------
dataCell = cell(NUM_SIGNALS,1);
truth(NUM_SIGNALS,1) = struct( ...
    'K',0, ...
    'freqs',[], 'amps',[], 'phases',[], ...
    'poly_c',[0 0 0], 'exp_amp',0, 'exp_tau',0, ...
    'ar1_phi',0, 'ar1_sigma',0, 'white_sigma',0 ...
);

for i = 1:NUM_SIGNALS
    % ----- Oscillatory components (truth) -----
    K_true      = randi([K_min, K_max],1,1);
    freqs_true  = f_min + (f_max - f_min)*rand(1,K_true);
    amps_true   = A_min + (A_max - A_min)*rand(1,K_true);
    phases_true = 2*pi*rand(1,K_true);

    osc_components = zeros(K_true, N);
    for k = 1:K_true
        osc_components(k,:) = amps_true(k) * sin(2*pi*freqs_true(k)*t + phases_true(k));
    end
    osc_sum = sum(osc_components, 1);

    % ----- Non-oscillatory (trend + exponential) -----
    c0 = poly_c0_range(1) + diff(poly_c0_range)*rand;
    c1 = poly_c1_range(1) + diff(poly_c1_range)*rand;
    c2 = poly_c2_range(1) + diff(poly_c2_range)*rand;
    trend = c0 + c1*t + c2*(t.^2);

    b_exp = exp_amp_range(1) + diff(exp_amp_range)*rand;
    tau   = exp_tau_range(1) + diff(exp_tau_range)*rand;
    decay = b_exp * exp(-t./tau);

    nonosc = trend + decay;

    % ----- Noise (AR(1) + white) -----
    phi     = ar1_phi_range(1)   + diff(ar1_phi_range)*rand;
    sig_ar1 = ar1_sigma_range(1) + diff(ar1_sigma_range)*rand;
    sig_wn  = white_sigma_range(1)+ diff(white_sigma_range)*rand;

    e  = sig_ar1*randn(1,N);
    ar = zeros(1,N);
    for n = 2:N
        ar(n) = phi*ar(n-1) + e(n);
    end
    wn    = sig_wn*randn(1,N);
    noise = ar + wn;

    % ----- Final signal (original) -----
    y_total = osc_sum + nonosc + noise;

    % ----- Store truth and signal -----
    dataCell{i}     = y_total;
    truth(i).K      = K_true;
    truth(i).freqs  = freqs_true;
    truth(i).amps   = amps_true;
    truth(i).phases = phases_true;
    truth(i).poly_c = [c0 c1 c2];
    truth(i).exp_amp    = b_exp;
    truth(i).exp_tau    = tau;
    truth(i).ar1_phi    = phi;
    truth(i).ar1_sigma  = sig_ar1;
    truth(i).white_sigma= sig_wn;

    % ----- Plot: oscillators, non-osc, noise, original (one figure) -----
    fig = figure('Visible','off','Units','pixels','Position',[80 80 1250 900]);
    tiledlayout(4,1,'Padding','compact','TileSpacing','compact');

    % Oscillator components (each component shown + legend with freqs)
    nexttile;
    hold on;
    for k = 1:K_true
        plot(t, osc_components(k,:), 'LineWidth', 1.0);
    end
    hold off; grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    lgd = compose('f=%.2f Hz', freqs_true);
    legend(lgd, 'Location','northeastoutside');
    title(sprintf('Signal %d — Oscillatory Components (K_{true}=%d)', i, K_true));

    % Non-oscillatory (trend + exponential)
    nexttile;
    plot(t, nonosc, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Non-oscillatory (trend + exponential)');

    % Noise
    nexttile;
    plot(t, noise, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Noise (AR(1) + white)');

    % Original = Osc + Non-osc + Noise
    nexttile;
    plot(t, y_total, 'LineWidth', 1.0); grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title('Original Signal = Oscillatory + Non-oscillatory + Noise');

    exportgraphics(fig, fullfile(plotsFolder, sprintf('cell_%03d_components_orig.png',i)), 'Resolution', 150);
    close(fig);
end

% Persist dataset
save(datasetMatPath,'dataCell','fs','t','truth','-v7.3');

% ------------------ Decomposition + per-signal reporting ----------------
summary(NUM_SIGNALS,1) = struct( ...
    'idx',0, ...
    'trueK',0, ...
    'trueFreqs',"", ...
    'trueAmps',"", ...
    'estK',0, ...
    'estFreqs',"", ...
    'mse',0.0 );

% Initialize a simple TXT report too
txtReportPath = fullfile(outputFolder,'summary_all.txt');
fid_txt = fopen(txtReportPath,'w');
fprintf(fid_txt,'idx,trueK,trueFreqs,estK,estFreqs,MSE\n');

for idx = 1:NUM_SIGNALS
    y = dataCell{idx};

    % --- Run OSC model ---
    [osc_param, osc_AIC, osc_mean, ~, ~] = osc_decomp_uni(y, fs, MAX_OSC, MAX_AR);
    [~, K_est] = min(osc_AIC);

    if K_est > 0
        est_freqs = osc_param(K_est, K_est+1:2*K_est);
        M         = squeeze(osc_mean(:,:,K_est));     % (2*K_est) × N
        raw_ts    = zeros(K_est, N);
        for k = 1:K_est
            raw_ts(k,:) = M(2*k-1, :);
        end
        y_hat  = sum(raw_ts,1);
        p2pRaw = max(raw_ts,[],2) - min(raw_ts,[],2);
        ordRaw = floor(log10(max(p2pRaw, realmin)));
    else
        est_freqs = [];
        raw_ts    = zeros(0, N);
        y_hat     = zeros(1, N);
        p2pRaw    = [];
        ordRaw    = [];
    end

    mseVal = mean((y - y_hat).^2);

    % --- Save per-signal MAT output as requested ---
    result.numComponents = K_est;
    result.frequencies   = est_freqs(:).';
    result.p2pRaw        = p2pRaw(:).';
    result.orderRaw      = ordRaw(:).';
    result.mse           = mseVal;

    save(fullfile(outputFolder, sprintf('cell_%03d_results.mat',idx)), ...
         'result', 'osc_param', 'osc_AIC', 'est_freqs');

    % --- Plot: original vs reconstruction + est amplitudes vs frequency ---
    fig1 = figure('Visible','off','Units','pixels','Position',[80 80 1150 520]);
    tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    nexttile;
    plot(t, y, 'LineWidth', 1.0); hold on;
    plot(t, y_hat, 'LineWidth', 1.0); hold off; grid on; xlim([t(1) t(end)]);
    xlabel('Time [s]'); ylabel('Amplitude');
    title(sprintf('Signal %d — Original vs Reconstruction (K_{est}=%d, MSE=%.4g)', idx, K_est, mseVal));
    legend({'Original','Reconstruction'},'Location','best');

    nexttile;
    if K_est > 0
        [f_sorted, ord] = sort(est_freqs(:).');
        p2p_sorted = p2pRaw(ord);
        bar(f_sorted, p2p_sorted); grid on;
        xlabel('Frequency [Hz]'); ylabel('Peak-to-peak (raw)');
        title('Estimated Component Amplitudes vs Frequency');
    else
        axis off;
    end

    exportgraphics(fig1, fullfile(plotsFolder, sprintf('cell_%03d_recon.png',idx)), 'Resolution', 150);
    close(fig1);

    % --- Update summary (true vs estimated) ---
    trueFreqs_str = "[" + strjoin(string(round(truth(idx).freqs,4)), ", ") + "]";
    trueAmps_str  = "[" + strjoin(string(round(truth(idx).amps,4)),  ", ") + "]";
    estFreqs_str  = "[" + strjoin(string(round(est_freqs,4)),        ", ") + "]";

    summary(idx).idx       = idx;
    summary(idx).trueK     = truth(idx).K;
    summary(idx).trueFreqs = trueFreqs_str;
    summary(idx).trueAmps  = trueAmps_str;
    summary(idx).estK      = K_est;
    summary(idx).estFreqs  = estFreqs_str;
    summary(idx).mse       = mseVal;

    % TXT line (no brace indexing anywhere)
    fprintf(fid_txt,'%d,%d,%s,%d,%s,%.10g\n', ...
        idx, truth(idx).K, trueFreqs_str, K_est, estFreqs_str, mseVal);
end
fclose(fid_txt);

% --- Aggregate summary saves (no brace-indexing used) ---
summaryTable = struct2table(summary);
save(fullfile(outputFolder,'summary_all.mat'),'summary','summaryTable');

csvPath = fullfile(outputFolder,'summary_all.csv');
writetable(summaryTable, csvPath);

%% ------------------------------ Console log -----------------------------
fprintf('DONE\n');
fprintf('Dataset: %s\n', datasetMatPath);
fprintf('Per-signal results: %s\n', outputFolder);
fprintf('Plots: %s\n', plotsFolder);
fprintf('Summary: %s and %s\n', fullfile(outputFolder,'summary_all.mat'), csvPath);
fprintf('TXT report: %s\n', fullfile(outputFolder,'summary_all.txt'));


%% Batch OSC Decomposition for Cell Array of Time Series (Raw Amplitude Magnitude)
% Runs osc_decomp_uni on each element of a cell array and
% saves, for each series:
%   • Number of oscillators (K)
%   • Frequencies of each component (osc_f)
%   • Peak‐to‐peak amplitude (raw) & its 10^n order
%   • Reconstruction MSE
%   • Full osc_param matrix
%
% User‐configurable parameters:

clear; close all; clc;



load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/New_Session_Part_Filtered/Filtered_200_Part4.mat')


% 1) Your data: an N×1 cell array, each cell = 1×T_i double
dataCell     = Filtered_200_Part4;    % e.g. workspace variable (503×1 cell)
fs           = 1000;                % Sampling frequency [Hz]
MAX_OSC      = 7;                  % Maximum number of oscillators
MAX_AR       = 2*MAX_OSC;           % Maximum AR order
startIdx     = 1;                   % First cell index to process
endIdx       = 500;     % Last cell index to process -- numel(dataCell)
outputFolder = 'OSC_Results';       % Folder to save results

% 2) Create output directory if needed
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 3) Main processing loop
for idx = startIdx:endIdx
    %--- Extract current time series
    y = dataCell{idx};             % 1×T_i vector
    
    %--- Run univariate OSC decomposition
    [osc_param, osc_AIC, osc_mean, ~, ~] = ...
        osc_decomp_uni(y, fs, MAX_OSC, MAX_AR);
    
    %--- Select optimal number of oscillators
    [~, K] = min(osc_AIC);         % scalar
    
    %--- Extract oscillator frequencies
    osc_f = osc_param(K, K+1 : 2*K);   % 1×K
    
    %--- Get smoothed‐state coordinates for K‐oscillator fit
    M   = squeeze( osc_mean(:,:,K) );  % (2*K)×T_i
    T_i = size(M, 2);
    
    %--- Reconstruct raw signals & compute envelopes
    raw_ts = zeros(K, T_i);
    for k = 1:K
        raw_ts(k,:) = M(2*k-1, :);     % first coordinate
    end
    
    %--- 2) Compute peak‐to‐peak on raw series and orders
    p2p_raw   = max(raw_ts,[],2) - min(raw_ts,[],2);  % K×1
    order_raw = floor( log10( p2p_raw ) );            % K×1
    
    %--- Reconstruct full signal and compute MSE
    y_hat  = sum(raw_ts, 1);
    mseVal = mean( (y - y_hat).^2 );
    
    %--- Package results into a struct
    result.numComponents = K;
    result.frequencies   = osc_f;
    result.p2pRaw        = p2p_raw;
    result.orderRaw      = order_raw;
    result.mse           = mseVal;
    
    %--- Save .mat with result struct, osc_param, and osc_f
    fileName = sprintf('cell_%03d_results.mat', idx);
    save(fullfile(outputFolder, fileName), ...
         'result', 'osc_param', 'osc_f');
end

%% 


% 1) Build the full filename
idx = 2;
fname = fullfile('OSC_Results', sprintf('cell_%03d_results.mat', idx));

% 2) Load it
S = load(fname);     % loads S.result, S.osc_param, S.osc_f

% 3) View the main struct
disp(S.result);

% 4) Drill into fields
fprintf('Cell %d → K = %d oscillators\n', idx, S.result.numComponents);
fprintf('Frequencies: %s\n', mat2str(S.result.frequencies,4));
fprintf('Raw p2p amplitudes: %s\n', mat2str(S.result.p2pRaw,4));
fprintf('Orders (10^n): %s\n', mat2str(S.result.orderRaw));
fprintf('Reconstruction MSE: %g\n', S.result.mse);
%% 


% 1) List result files
files = dir(fullfile('OSC_Results','cell_*_results.mat'));

% 2) Preallocate containers
N = numel(files);
cellIdx    = zeros(N,1);
nComps     = zeros(N,1);
meanFreq   = zeros(N,1);
meanP2P    = zeros(N,1);
meanOrder  = zeros(N,1);
mseAll     = zeros(N,1);

% 3) Loop & extract
for i = 1:N
    % parse index from filename
    name = files(i).name;                        % e.g. 'cell_042_results.mat'
    idx  = sscanf(name,'cell_%d_results.mat');
    
    % load
    S    = load(fullfile(files(i).folder, name));
    
    % fill
    cellIdx(i)   = idx;
    nComps(i)    = S.result.numComponents;
    meanFreq(i)  = mean(S.result.frequencies);
    meanP2P(i)   = mean(S.result.p2pRaw);
    meanOrder(i) = mean(S.result.orderRaw);
    mseAll(i)    = S.result.mse;
end

% 4) Build and display table
T = table(cellIdx, nComps, meanFreq, meanP2P, meanOrder, mseAll, ...
    'VariableNames', {
      'CellIndex','NumOsc','MeanFreq','MeanP2P','MeanOrder','MSE' });
disp(T);













%%
load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part2.mat');


%%

clear; close all; clc;

load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part2.mat');

% Extract trials at specified indices
y = Filtered_200_Part2{10};

fs = 1000;

MAX_OSC = 8;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(y,fs,MAX_OSC);
[minAIC,K] = min(osc_AIC);
osc_a = osc_param(K,1:K);
osc_f = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_tau2 = osc_param(K,3*K+1);
[hess,grad,mll] = osc_ll_hess(y,fs,osc_param(K,1:3*K+1));
cov_est = inv(hess);
fprintf('The number of oscillators is K=%d.\n',K);
fprintf('The periods of K oscillators are:\n');
for k=1:K
    fprintf(' %.2f (95%% CI: [%.2f %.2f]) years\n',1./osc_f(k),1./(osc_f(k)+1.96*sqrt(cov_est(K+k,K+k))),1./(osc_f(k)-1.96*sqrt(cov_est(K+k,K+k))));
end
osc_plot(osc_mean,osc_cov,fs,K)
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K)
osc_spectrum_plot(y,fs,osc_a,osc_f,osc_sigma2,osc_tau2)






%%

% M is 2K×T from squeeze(osc_mean(:,:,K))
T    = size(M,2);
raw  = zeros(K, T);

% 1) Reconstruct raw coordinate for each oscillator
for k = 1:K
    raw(k,:) = M(2*k-1, :);   % first Cartesian coordinate = the oscillator time-series
end

% 2) Compute peak-to-peak on raw series
p2p_raw   = max(raw,[],2) - min(raw,[],2);
order_raw = floor(log10(p2p_raw));

% 3) Print a comparison table
fprintf('Osc   Raw p2p Amp   Order  ||  Env p2p Amp   Order\n');
for k = 1:K
    fprintf('%2d   %12.5g   1e%2d   ||   %12.5g   1e%2d\n', ...
            k, p2p_raw(k), order_raw(k), p2p_env(k), order_env(k));
end


%%
% 1) Build each oscillator’s reconstructed time-series (first coordinate)
T      = size(M,2);
raw_ts = zeros(K, T);
for k = 1:K
    raw_ts(k,:) = M(2*k-1, :);
end

% 2) Sum components → reconstructed signal y_hat
y_hat = sum(raw_ts, 1);

% 3) Time vector (seconds)
t = (0:T-1)/fs;

% 4) Plot sum of oscillators
figure;
plot(t, y_hat, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Sum of Oscillator Components');

% 5) Plot error: original minus reconstruction
err = y - y_hat;
figure;
plot(t, err, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Error');
title('Difference: Input – Sum of Components');

% 6) Compute & report MSE
mse_val = mean(err.^2);
fprintf('Mean squared error (MSE) = %g\n', mse_val);

%%
%=== Overlay all individual oscillators, their sum, and the original signal ===

% Assume you have:
%   raw_ts (K×T)    : each row k is oscillator k’s reconstructed time-series
%   y_hat (1×T)     : sum(raw_ts,1)
%   y     (1×T)     : original input
%   t     (1×T)     : time vector, e.g. (0:T-1)/fs

figure;
hold on;

% Generate a set of distinct colors
cols = lines(K+2);

% 1) Plot each oscillator
for k = 1:K
    plot(t, raw_ts(k,:), 'Color', cols(k,:), 'LineWidth', 1.2);
end

% 2) Plot sum of oscillators
plot(t, y_hat, 'Color', cols(K+1,:), 'LineWidth', 2);

% 3) Plot original data with a dashed black line
plot(t, y,      '--',       'Color', [0 0 0],     'LineWidth', 2);

hold off;

% 4) Labels, title, grid
xlabel('Time (s)');
ylabel('Amplitude');
title('Individual Oscillators, Their Sum, and Original Signal');
grid on;

% 5) Legend entries
leg = cell(K+2,1);
for k = 1:K
    leg{k} = sprintf('Oscillator %d', k);
end
leg{K+1} = 'Sum of Oscillators';
leg{K+2} = 'Original y';
legend(leg, 'Location', 'best');






%%

osc_plot(osc_mean,osc_cov,fs,K)










%%

%% Complete OSC Decomposition & Analysis Script

clear; close all; clc;

% 1) Load data and set parameters
load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part2.mat');
y      = Filtered_200_Part2{146};   % 1×T
fs     = 1000;                      % sampling frequency
MAX_OSC = 12;                       
MAX_AR  = 2*MAX_OSC;                % default AR order

% 2) Run univariate decomposition
[osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = ...
    osc_decomp_uni(y, fs, MAX_OSC, MAX_AR);

% Select optimal number of oscillators
[~, K] = min(osc_AIC);  

% 3) Extract smoothed states for K-oscillator model
M = squeeze(osc_mean(:,:,K));      % (2*K)×T
T = size(M,2);
t = (0:T-1)/fs;

% 4) Reconstruct individual oscillator time series (first coordinate)
raw_ts = zeros(K, T);
for k = 1:K
    raw_ts(k,:) = M(2*k-1, :);
end

% 5) Compute instantaneous amplitude envelopes
env = zeros(K, T);
for k = 1:K
    x1 = M(2*k-1, :);
    x2 = M(2*k,   :);
    env(k,:) = sqrt(x1.^2 + x2.^2);
end

% 6) Compute peak-to-peak ranges and orders
p2p_raw   = max(raw_ts,[],2) - min(raw_ts,[],2);
order_raw = floor(log10(p2p_raw));

p2p_env   = max(env,[],2) - min(env,[],2);
order_env = floor(log10(p2p_env));

% 7) Model amplitudes and orders
a       = osc_param(K,1:K);
order_a = floor(log10(a));

% 8) Display all magnitudes in a table
fprintf('Osc\t a_k\t    order(a_k)\t p2p_raw\t order(raw)\t p2p_env\t order(env)\n');
for k = 1:K
    fprintf('%2d\t %10.5g\t 1e%2d\t %10.5g\t 1e%2d\t %10.5g\t 1e%2d\n', ...
        k, a(k), order_a(k), p2p_raw(k), order_raw(k), p2p_env(k), order_env(k));
end

% 9) Reconstruct full signal and compute MSE
y_hat = sum(raw_ts,1);
err   = y - y_hat;
mse   = mean(err.^2);

fprintf('\nReconstruction MSE: %g\n\n', mse);

% 10) Plot sum of oscillators
figure;
plot(t, y_hat, 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Amplitude');
title('Sum of Oscillator Components');
grid on;

% 11) Plot reconstruction error
figure;
plot(t, err, 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('Error');
title('Reconstruction Error (y - y\_hat)');
grid on;

% 12) Overlay original vs. reconstructed
figure;
plot(t, y,     'LineWidth',1.5); hold on;
plot(t, y_hat, 'LineWidth',1.5);
hold off;
xlabel('Time (s)'); ylabel('Amplitude');
title('Original Signal vs. Reconstruction');
legend('Original y','Reconstructed y\_hat','Location','best');
grid on;

% 13) Overlay all components, their sum, and original
figure; hold on;
cols = lines(K+2);
for k = 1:K
    plot(t, raw_ts(k,:), 'Color', cols(k,:), 'LineWidth',1.2);
end
plot(t, y_hat, 'Color', cols(K+1,:), 'LineWidth',2);
plot(t, y,     '--', 'Color', [0 0 0],     'LineWidth',2);
hold off;
xlabel('Time (s)'); ylabel('Amplitude');
title('Individual Components, Sum, and Original');
legend_entries = [ ...
    arrayfun(@(k)sprintf('Oscillator %d',k), 1:K, 'UniformOutput', false), ...
    {'Sum of Oscillators','Original y'} ];
legend(legend_entries,'Location','best');
grid on;













%% 
%% Dynamic saving for different cells
clear; close all; clc;

%% Dynamic saving for different cells
cellIdx = 146;  % change this index as needed

% Load and decompose
load('Filtered_200_Part2.mat');
y = Filtered_200_Part2{cellIdx};
fs = 1000;
MAX_OSC = 6;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(y,fs,MAX_OSC);
[~,K] = min(osc_AIC);
osc_a      = osc_param(K,1:K);
osc_f      = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_tau2   = osc_param(K,3*K+1);

% Compute Hessian and covariance
[hess,~,~] = osc_ll_hess(y,fs,osc_param(K,1:3*K+1));
cov_est = inv(hess);

% Display
fprintf('Cell %d: K = %d oscillators\n', cellIdx, K);

% Plotting and dynamic filenames and saving plots properly
% Component plot
osc_plot(osc_mean,osc_cov,fs,K);
h1 = gcf;  % handle of the figure created by osc_plot
fn1 = sprintf('osc_components_cell%03d.png', cellIdx);
drawnow; saveas(h1, fn1);
close(h1);

% Phase plot
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K);
h2 = gcf;
fn2 = sprintf('osc_phases_cell%03d.png', cellIdx);
drawnow; saveas(h2, fn2);
close(h2);

% Spectrum plot
t = figure;
osc_spectrum_plot(y,fs,osc_a,osc_f,osc_sigma2,osc_tau2);
h3 = gcf;
fn3 = sprintf('osc_spectrum_cell%03d.png', cellIdx);
drawnow; saveas(h3, fn3);
close(h3);

% Save parameters to .mat
matName = sprintf('osc_results_cell%03d.mat', cellIdx);
save(matName, 'osc_param', 'osc_f');

% Export raw parameters and frequencies to CSV without altering format
paramCsv = sprintf('osc_param_cell%03d.csv', cellIdx);
writematrix(osc_param(K,:), paramCsv);

freqCsv = sprintf('osc_f_cell%03d.csv', cellIdx);
writematrix(osc_f(:)', freqCsv);

% Export Amplitude and Frequency as table to CSV for readability
tbl = table((1:K)', osc_a(:), osc_f(:), 'VariableNames', {'OscID','Amplitude','Frequency_Hz'});
csvName = sprintf('osc_results_cell%03d.csv', cellIdx);
writetable(tbl, csvName);
matName = sprintf('osc_results_cell%03d.mat', cellIdx);
save(matName, 'osc_param', 'osc_f');

% Export to CSV
tbl = table((1:K)', osc_a(:), osc_f(:), 'VariableNames', {'OscID','Amplitude','Frequency_Hz'});
csvName = sprintf('osc_results_cell%03d.csv', cellIdx);
writetable(tbl, csvName);

% When completed, files:
% fn1, fn2, fn3, matName, csvName




%% Dynamic saving for different cells
clear; close all; clc;

load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part2.mat');
cellIdx = 146;  % change this index as needed

% Load and decompose
load('Filtered_200_Part2.mat');
y = Filtered_200_Part2{cellIdx};
fs = 1000;
MAX_OSC = 6;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(y,fs,MAX_OSC);
[~,K] = min(osc_AIC);
osc_a      = osc_param(K,1:K);
osc_f      = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_tau2   = osc_param(K,3*K+1);

% Compute Hessian and covariance
[hess,~,~] = osc_ll_hess(y,fs,osc_param(K,1:3*K+1));
cov_est = inv(hess);

% Display
fprintf('Cell %d: K = %d oscillators\n', cellIdx, K);

% Plotting and dynamic filenames
fig1 = figure; osc_plot(osc_mean,osc_cov,fs,K);
fn1 = sprintf('osc_components_cell%03d.png', cellIdx);
saveas(fig1, fn1);

fig2 = figure; osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K);
fn2 = sprintf('osc_phases_cell%03d.png', cellIdx);
saveas(fig2, fn2);

fig3 = figure; osc_spectrum_plot(y,fs,osc_a,osc_f,osc_sigma2,osc_tau2);
fn3 = sprintf('osc_spectrum_cell%03d.png', cellIdx);
saveas(fig3, fn3);

% Save parameters
matName = sprintf('osc_results_cell%03d.mat', cellIdx);
save(matName, 'osc_param', 'osc_f');

% Export to CSV
tbl = table((1:K)', osc_a(:), osc_f(:), 'VariableNames', {'OscID','Amplitude','Frequency_Hz'});
csvName = sprintf('osc_results_cell%03d.csv', cellIdx);
writetable(tbl, csvName);

% When completed, files:O
% fn1, fn2, fn3, matName, csvName



%% 
clear; close all; clc;

% Load the cell array containing the filtered data for Bandpass 200, Part 1.
% This file must contain the variable 'Filtered_200_Part1' (a cell array with 503 trials).
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part4.mat');  

fs = 1000;      % Sampling frequency
MAX_OSC = 4;    % Maximum number of oscillators to use in the model
numTrials = 10;  % Process the first 100 trials

% Preallocate a cell array to store the frequency vector (osc_param(K, K+1:2*K)) for each trial.
freqResults = cell(numTrials, 1);

for trial = 1:numTrials
    % Extract the trial data (each trial may have a different length)
    y = Filtered_200_Part4{trial};
    
    % Run the oscillation decomposition model on the trial data.
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
    
    % Determine the optimal model order K based on minimum AIC.
    [~, K] = min(osc_AIC);
    
    % Extract the frequency parameters from osc_param:
    % For the optimal order K, extract the elements from column (K+1) to (2*K).
    freq_vector = osc_param(K, K+1:2*K);
    
    % Store the frequency vector for the current trial.
    freqResults{trial} = freq_vector;
end

% Save the frequency results for the first 100 trials into a .mat file.
% The file will be named 'Filtered_200_Part1_F.mat'.
save('Filtered_200_Part14_F_10.mat', 'freqResults_F200_P1_10');

%%


clear; close all; clc;

% Load the cell array containing the filtered data for Bandpass 200, Part 1.
% This file should contain the variable 'Filtered_200_Part1' (a cell array with 503 trials).
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/4Parts_CleanTrials_Histogram_Task/Filtered_200_Part1.mat');  

fs = 1000;      % Sampling frequency
MAX_OSC = 5;    % Maximum number of oscillators to consider in the model


% Process the second 100 trials: trials 101 to 200.
startTrial = 451;
endTrial = 500;
numTrials = endTrial - startTrial + 1;  % 100 trials

% Preallocate cell array for frequency results and vector for optimal K values.
freqResults_F_200_Part1_500 = cell(numTrials, 1);
K_F_200_Part4_110 = zeros(numTrials, 1);

for i = 1:numTrials
    trialIdx = startTrial + i - 1;
    y = Filtered_200_Part1{trialIdx};  % Get the data for the current trial
    
    % Run the oscillation decomposition model on the trial data.
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
    
    % Determine the optimal model order K based on the minimum AIC.
    [~, K] = min(osc_AIC);
    K_F_200_Part4_110(i) = K;  % Save the optimal K value for this trial
    
    % Extract the frequency parameters from osc_param for the optimal K.
    % The frequency vector is located in columns (K+1) to (2*K).
    freq_vector = osc_param(K, K+1:2*K);
    
    % Save the frequency vector for the current trial.
    freqResults_F_200_Part1_500{i} = freq_vector;
end

% Save the frequency results to a .mat file with the specified variable name.
save('Filtered_200_Part1_F_500.mat', 'freqResults_F_200_Part1_500');

% Save the optimal K values for each trial to a separate .mat file.
%save('Filtered_200_Part3_K_51U70.mat', 'K_F_200_Part3_51U70');








%%

%load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_15_1.mat')
% Make sure the .mat file and variable name are correct
% Define file paths for y_1 to y_4 and raw data for y_5
y_1_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/generated-Shuffle/Generated/SumTimeSeries.mat'


y_2_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Parts/200_Parts/Cue1LFP_Filtered_200_Trial_313_Part_2.mat' 



y_3_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Parts/200_Parts/Cue1LFP_Filtered_200_Trial_313_Part_3.mat'

y_4_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Parts/200_Parts/Cue1LFP_Filtered_200_Trial_313_Part_4.mat'
%y_5_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/same_Variable_Name/Cue1LFP_Filtered_400.mat';

% Load data for y_1 to y_4
y_1 = load(y_1_path);
% y_2 = load(y_2_path);
% y_3 = load(y_3_path);
% y_4 = load(y_4_path);
%y_5 = load(y_5_path);
% Ensure proper field extraction
y_1 = get_single_variable(y_1);
% y_2 = get_single_variable(y_2);
% y_3 = get_single_variable(y_3);
% y_4 = get_single_variable(y_4);
%y_5 = y_5.Cue1LFP_Filtered(313,:);
% Raw data for y_5
%y_5 = Cue1LFP(313, :); % Replace 'Cue1LFP' with the actual variable containing raw data

% Collect all datasets into a cell array
datasets = {y_1};
dataset_names = {'y_1'};

% Define save path for results
save_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Shuffle';
mkdir(save_path); % Create directory if it doesn't exist

% Sampling frequency and maximum oscillators
fs = 1000;       % Sampling frequency
MAX_OSC = 5;     % Maximum number of oscillators
indices_to_process =  [1];

% Loop through each dataset
for i = indices_to_process
    y = datasets{i};          % Current dataset
    dataset_name = dataset_names{i}; % Dataset name for saving files

    fprintf('Processing dataset: %s\n', dataset_name);

    % Perform osc decomposition
    [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
    [minAIC, K] = min(osc_AIC);
    osc_a = osc_param(K, 1:K);
    osc_f = osc_param(K, K+1:2*K);
    osc_sigma2 = osc_param(K, 2*K+1:3*K);
    osc_tau2 = osc_param(K, 3*K+1);

    % Save plots
    osc_plot(osc_mean, osc_cov, fs, K);
    saveas(gcf, sprintf('%s/%s_Oscillation.png', save_path, dataset_name));
    close;

    osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K);
    saveas(gcf, sprintf('%s/%s_Phase.png', save_path, dataset_name));
    close;

    osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2);
    saveas(gcf, sprintf('%s/%s_Spectrum.png', save_path, dataset_name));
    close;

    % Save osc_param as .mat file
    save(sprintf('%s/%s_osc_param.mat', save_path, dataset_name), 'osc_param');
end

fprintf('Processing complete. Results saved to %s.\n', save_path);



%%

load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_15_1.mat')
% Make sure the .mat file and variable name are correct
base_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Filtered_Data_Windowed_BandPass 50,...400';
save_path = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Analysis_Results';
mkdir(save_path); % Create save directory if it doesn't exist

% Define parameters
fs = 1000;       % Sampling frequency
MAX_OSC = 5;     % Maximum number of oscillators
selected_trials = [313]; % List of selected trials

% Get all subfolders in the base path
subfolders = dir(base_path);
subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

% Loop through each subfolder
for i = 1:length(subfolders)
    folder_name = subfolders(i).name;
    folder_path = fullfile(base_path, folder_name);

    % Parse trial and filter specification from folder name
    tokens = regexp(folder_name, 'Trials_(\d+)_filtered_(\d+)', 'tokens');
    if isempty(tokens)
        warning('Folder name "%s" does not match the expected pattern. Skipping.', folder_name);
        continue;
    end
    trial_num = str2double(tokens{1}{1});
    filter_spec = tokens{1}{2};

    fprintf('Processing folder: %s (Trial: %d, Filter: %s)\n', folder_name, trial_num, filter_spec);

    % Load all 4 components in the folder
    component_files = dir(fullfile(folder_path, '*.mat'));
    components = cell(1, 4);
    for j = 1:4
        component_file = fullfile(folder_path, sprintf('part_%d.mat', j));
        if isfile(component_file)
            data = load(component_file);
            field_names = fieldnames(data);
            components{j} = data.(field_names{1}); % Assuming the .mat file has one variable
        else
            warning('File not found: %s', component_file);
            components{j} = [];
        end
    end

    % Run the code for each component
    for j = 1:4
        if isempty(components{j})
            continue;
        end
        y = components{j};

        % Perform osc decomposition
        [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
        [minAIC, K] = min(osc_AIC);
        osc_a = osc_param(K, 1:K);
        osc_f = osc_param(K, K+1:2*K);
        osc_sigma2 = osc_param(K, 2*K+1:3*K);
        osc_tau2 = osc_param(K, 3*K+1);

        % Save plots
        osc_plot(osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/%s_Component_%d_Oscillation.png', save_path, folder_name, j));
        close;

        osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/%s_Component_%d_Phase.png', save_path, folder_name, j));
        close;

        osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2);
        saveas(gcf, sprintf('%s/%s_Component_%d_Spectrum.png', save_path, folder_name, j));
        close;

        % Save osc_param as .mat file
        save(sprintf('%s/%s_Component_%d_osc_param.mat', save_path, folder_name, j), 'osc_param');
    end

    % Process the raw trial data for the selected trial
    if ismember(trial_num, selected_trials)
        fprintf('Processing raw data for Trial %d...\n', trial_num);
        y = Cue1LFP(trial_num, :);

        % Perform osc decomposition
        [osc_param, osc_AIC, osc_mean, osc_cov, osc_phase] = osc_decomp(y, fs, MAX_OSC);
        [minAIC, K] = min(osc_AIC);
        osc_a = osc_param(K, 1:K);
        osc_f = osc_param(K, K+1:2*K);
        osc_sigma2 = osc_param(K, 2*K+1:3*K);
        osc_tau2 = osc_param(K, 3*K+1);

        % Save plots
        osc_plot(osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/Trial_%d_Raw_Oscillation.png', save_path, trial_num));
        close;

        osc_phase_plot(osc_phase, osc_mean, osc_cov, fs, K);
        saveas(gcf, sprintf('%s/Trial_%d_Raw_Phase.png', save_path, trial_num));
        close;

        osc_spectrum_plot(y, fs, osc_a, osc_f, osc_sigma2, osc_tau2);
        saveas(gcf, sprintf('%s/Trial_%d_Raw_Spectrum.png', save_path, trial_num));
        close;

        % Save osc_param as .mat file
        save(sprintf('%s/Trial_%d_Raw_osc_param.mat', save_path, trial_num), 'osc_param');
    end
end

fprintf('Processing complete. Results saved to %s.\n', save_path);

%%
clear all;
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/generated-Shuffle/Generated/TimeSeries5.mat');


y = ts;
% y_1 = average_part1;
%  load('CanadianLynxData.mat');
%  y_2 = lynx;
% y = y_1(1:114).*10000000 + y_2;
%  y = log(y_2.*100);
% 
 %y = (y-mean(y))/1000;
%y = log(y);
%y = channelData(1:1800);




fs = 1000;
MAX_OSC = 5;
[osc_param,osc_AIC,osc_mean,osc_cov,osc_phase] = osc_decomp(y,fs,MAX_OSC);
[minAIC,K] = min(osc_AIC);
osc_a = osc_param(K,1:K);
osc_f = osc_param(K,K+1:2*K);
osc_sigma2 = osc_param(K,2*K+1:3*K);
osc_tau2 = osc_param(K,3*K+1);
[hess,grad,mll] = osc_ll_hess(y,fs,osc_param(K,1:3*K+1));
cov_est = inv(hess);
fprintf('The number of oscillators is K=%d.\n',K);
fprintf('The periods of K oscillators are:\n');
for k=1:K
    fprintf(' %.2f (95%% CI: [%.2f %.2f]) years\n',1./osc_f(k),1./(osc_f(k)+1.96*sqrt(cov_est(K+k,K+k))),1./(osc_f(k)-1.96*sqrt(cov_est(K+k,K+k))));
end
osc_plot(osc_mean,osc_cov,fs,K)
osc_phase_plot(osc_phase,osc_mean,osc_cov,fs,K)
osc_spectrum_plot(y,fs,osc_a,osc_f,osc_sigma2,osc_tau2)

%%


% Compute the sum of all oscillator mean components across K
% Assuming osc_mean has dimensions [2*K-1 x T x K]
% Adjust indexing based on your actual data structure

% Initialize the sum signal
sum_osc_mean = zeros(1, size(osc_mean, 2));
T = size(osc_mean,2);
% Sum across all K oscillators
for k = 1:K
    % Assuming that the mean of each oscillator is stored in osc_mean(2*k-1, :, k)
    sum_osc_mean = sum_osc_mean + osc_mean(2*k-1, :, k);
end
% Plot the sum of all oscillator components
figure;
plot((1:T)/fs, sum_osc_mean, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Sum of All Oscillator Components');
grid on;
set(gca, 'FontSize', 12);
figure;
hold on;
plot((1:T)/fs, y, 'k-', 'DisplayName', 'Original Signal');
%plot((1:T)/fs, sum_osc_mean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Sum of Oscillators');
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal vs. Sum of Oscillator Components');
legend;
grid on;
set(gca, 'FontSize', 12);
hold off;
%%
figure;

plot((1:T)/fs, y - sum_osc_mean, 'b-', 'LineWidth', 1.5);
%%
% Initialize arrays to store the mean phases
mean_phase = zeros(1, T);

% Compute the mean phase using circular statistics
for t = 1:T
    % Extract phases of all K oscillators at time t
    phases = osc_phase(:, t, K); % Adjust indexing based on your data structure
    
    % Convert phases to unit vectors
    unit_vectors = exp(1i * phases);
    
    % Compute the mean resultant vector
    mean_vector = mean(unit_vectors);
    
    % Compute the mean phase angle
    mean_phase(t) = angle(mean_vector);
end
% Initialize arrays to store the summed phase vectors
sum_phase_vector = zeros(1, T);

% Compute the sum of phase vectors
for t = 1:T
    % Extract phases of all K oscillators at time t
    phases = osc_phase(:, t, K); % Adjust indexing based on your data structure
    
    % Convert phases to unit vectors and sum them
    sum_vector = sum(exp(1i * phases));
    
    % Compute the angle of the summed vector
    sum_phase_vector(t) = angle(sum_vector);
end

% Plot the summed phase vectors
figure;
plot((1:T)/fs, sum_phase_vector, 'g-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Summed Phase (radians)');
title('Summed Phase Across All Oscillators');
grid on;
set(gca, 'FontSize', 12);
ylim([-pi, pi]);
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});
%%

% Hilbert for Phase
% Assuming y is your time series and fs is the sampling frequency
y = y(:); % Convert to column vector if it's not already
% Compute the analytic signal
analytic_signal = hilbert(y); % Use y if not filtering
% Calculate instantaneous phase in radians
inst_phase = angle(analytic_signal);
% Unwrap the instantaneous phase
inst_phase_unwrapped = unwrap(inst_phase);
% Define time vector based on sampling frequency
T = length(y);
t = (0:T-1)' / fs; % Time in seconds

% Create a figure with two subplots
figure;

% Plot the original or filtered signal
subplot(2,1,1);
plot(t, y, 'b'); % Use y instead of y_filtered if not filtering
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal');
grid on;
set(gca, 'FontSize', 12);

% Plot the instantaneous phase
subplot(2,1,2);
plot(t, inst_phase_unwrapped, 'r');
xlabel('Time (s)');
ylabel('Phase (radians)');
title('Instantaneous Phase (Unwrapped)');
grid on;
set(gca, 'FontSize', 12);
ylim([-pi, pi]);
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});

%%
% Helper function to extract the single variable from the loaded .mat file
function data = get_single_variable(loaded_data)
    fields = fieldnames(loaded_data);
    if length(fields) == 1
        data = loaded_data.(fields{1});
    else
        error('Loaded .mat file contains multiple variables. Ensure the file has a single variable.');
    end
end
