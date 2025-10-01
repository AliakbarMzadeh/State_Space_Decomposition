%% 1. Load the Trimmed LFP Data
clear; close all; clc;

load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Clean_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/saccade_onset.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Valid_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/raise_times.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_15_1.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_50_Corrected.mat');      % 1-50 Hz filtered data
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_100_Corrected.mat');     % 1-100 Hz filtered data
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_200_Corrected.mat');     % 1-200 Hz filtered data
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/FIltered_Cue1LFP_Brekka_Data/Cue1LFP_Filtered_400_Corrected.mat');     % 1-400 Hz filtered datal
%%
% Load necessary data
clc;
clear all; 
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_16_1.mat')
% Make sure the .mat file and variable name are correct
%load('Cue1_LFP.mat');  % Suppose it contains a variable 'lfp_Trim'
% If the loaded variable is named something else, adjust accordingly:
% lfp_Trim = data.lfp_Trim;
lfp_trimmed = Cue1LFP;
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Clean_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/saccade_onset.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Valid_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/raise_times.mat')
Cue1LFP_Filtered_50 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Cue1LFP_Filtered_50.mat');      % 1-50 Hz filtered data
Cue1LFP_Filtered_100 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Cue1LFP_Filtered_100.mat');     % 1-100 Hz filtered data
Cue1LFP_Filtered_200 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Cue1LFP_Filtered_200.mat');     % 1-200 Hz filtered data
Cue1LFP_Filtered_400 = load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Cue1LFP_Filtered_400.mat');     % 1-400 Hz filtered data


fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Filtered data sets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Trial sets
trial_sets = {
    Clean_Trials, 'Clean Trials';
    Valid_Trials, 'Valid Trials';
    other_trials, 'Trials Not in Both';
    all_trials, 'All Trials';
};

% Time vector
time = (0:size(Cue1LFP, 2) - 1) / fs; % In seconds
%%
% Plot raw and filtered data
for t = 1:length(trial_sets)
    trial_indices = trial_sets{t, 1};
    trial_name = trial_sets{t, 2};

    % Plot raw data
    figure('Name', ['Raw Data - ', trial_name], 'NumberTitle', 'off');
    plot(time, Cue1LFP(trial_indices, :)', 'Color', [0.8, 0.8, 0.8]);
    hold on;
    plot(time, mean(Cue1LFP(trial_indices, :), 1), 'k', 'LineWidth', 2, 'DisplayName', 'Average');
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['Raw Data - ', trial_name]);
    legend('Individual Trials', 'Average');
    grid on;
    saveas(gcf, ['Raw_Data_', strrep(trial_name, ' ', '_'), '.png']);

    % Plot filtered data
    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        figure('Name', [filter_name, ' - ', trial_name], 'NumberTitle', 'off');
        plot(time, filtered_data(trial_indices, :)', 'Color', [0.8, 0.8, 0.8]);
        hold on;
        plot(time, mean(filtered_data(trial_indices, :), 1), 'r', 'LineWidth', 2, 'DisplayName', 'Average');
        hold off;
        xlabel('Time (s)');
        ylabel('Amplitude');
        title([filter_name, ' - ', trial_name]);
        legend('Individual Trials', 'Average');
        grid on;
        saveas(gcf, [strrep(filter_name, ' ', '_'), '_', strrep(trial_name, ' ', '_'), '.png']);
    end
end
%%

fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Define trial sets
trial_sets = {
    Clean_Trials, 'Clean Trials';
    Valid_Trials, 'Valid Trials';
    other_trials, 'Trials Not in Both';
    all_trials, 'All Trials';
};

% Define filtered data
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters
num_random_trials = 5; % Number of random trials to select
time = (0:size(Cue1LFP, 2) - 1) / fs; % Time vector in seconds

% Loop through trial sets and filtered data
for t = 1:size(trial_sets, 1)
    trial_indices = trial_sets{t, 1}; % Get trial indices
    trial_name = trial_sets{t, 2};    % Get trial set name

    % Randomly select trials from the current trial set
    if length(trial_indices) > num_random_trials
        selected_trials = trial_indices(randperm(length(trial_indices), num_random_trials));
    else
        selected_trials = trial_indices; % Use all trials if fewer than num_random_trials
    end

    % Compute the average signal for the entire trial set
    average_signal_raw = mean(Cue1LFP(trial_indices, :), 1);

    % Plot raw data
    figure('Name', ['Raw Data - ', trial_name], 'NumberTitle', 'off');
    hold on;
    for i = 1:length(selected_trials)
        plot(time, Cue1LFP(selected_trials(i), :), 'LineWidth', 1.2, 'DisplayName', sprintf('Trial %d', selected_trials(i)));
    end
    % Add the average signal to the plot
    plot(time, average_signal_raw, 'k--', 'LineWidth', 2, 'DisplayName', 'Average');
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(['Raw Data - ', trial_name]);
    legend('show', 'Location', 'best');
    grid on;
    saveas(gcf, ['Raw_Data_', strrep(trial_name, ' ', '_'), '.png']);
    close;

    % Plot filtered data
    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        % Compute the average signal for the filtered data
        average_signal_filtered = mean(filtered_data(trial_indices, :), 1);

        figure('Name', [filter_name, ' - ', trial_name], 'NumberTitle', 'off');
        hold on;
        for i = 1:length(selected_trials)
            plot(time, filtered_data(selected_trials(i), :), 'LineWidth', 1.2, 'DisplayName', sprintf('Trial %d', selected_trials(i)));
        end
        % Add the average signal to the plot
        plot(time, average_signal_filtered, 'k--', 'LineWidth', 2, 'DisplayName', 'Average');
        hold off;
        xlabel('Time (s)');
        ylabel('Amplitude');
        title([filter_name, ' - ', trial_name]);
        legend('show', 'Location', 'best');
        grid on;
        saveas(gcf, [strrep(filter_name, ' ', '_'), '_', strrep(trial_name, ' ', '_'), '.png']);
        close;
    end
end
%%
% Load necessary data

fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP_Filtered_50, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Define trial sets
trial_sets = {
    Clean_Trials, 'Clean Trials';
    Valid_Trials, 'Valid Trials';
    other_trials, 'Trials Not in Both';
    all_trials, 'All Trials';
};

% Define filtered data sets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters for pwelch
winSize = 64;    % Window size
overlap = 32;    % Overlap
nfft = 128;      % Number of FFT points
num_random_trials = 5; % Number of random trials to select

% Loop through each trial set and filtered data set
for t = 1:size(trial_sets, 1)
    trial_indices = trial_sets{t, 1}; % Get trial indices
    trial_name = trial_sets{t, 2};    % Get trial set name

    % Randomly select trials from the current trial set
    if length(trial_indices) > num_random_trials
        selected_trials = trial_indices(randperm(length(trial_indices), num_random_trials));
    else
        selected_trials = trial_indices; % Use all trials if fewer than num_random_trials
    end

    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        % Initialize figure
        figure('Name', [filter_name, ' - ', trial_name, ' (Power over Frequency)'], 'NumberTitle', 'off');
        hold on;

        % Compute and plot PSD for each selected trial
        psd_matrix = []; % To store power for averaging
        for i = 1:length(selected_trials)
            [pxx, f] = pwelch(filtered_data(selected_trials(i), :), winSize, overlap, nfft, fs);
            psd_matrix = [psd_matrix; pxx']; % Store power for averaging
            plot(f, 10*log10(pxx), 'LineWidth', 1.2, 'DisplayName', sprintf('Trial %d', selected_trials(i)));
        end

        % Compute and plot the average PSD
        avg_pxx = mean(psd_matrix, 1);
        plot(f, 10*log10(avg_pxx), 'k--', 'LineWidth', 2, 'DisplayName', 'Average');

        % Finalize the plot
        xlabel('Frequency (Hz)');
        ylabel('Power/Frequency (dB/Hz)');
        title([filter_name, ' - ', trial_name]);
        legend('show', 'Location', 'best');
        grid on;
        hold off;

        % Save the plot
        saveas(gcf, [strrep(filter_name, ' ', '_'), '_', strrep(trial_name, ' ', '_'), '_Power_Frequency.png']);
        close;
    end
end
%%


fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Define trial sets
trial_sets = {
    Clean_Trials, 'Clean Trials';
    Valid_Trials, 'Valid Trials';
    other_trials, 'Trials Not in Both';
    all_trials, 'All Trials';
};

% Define filtered data sets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters for pwelch
winSize = 64;    % Window size
overlap = 32;    % Overlap
nfft = 128;      % Number of FFT points

% Loop through each trial set
for t = 1:size(trial_sets, 1)
    trial_indices = trial_sets{t, 1}; % Get trial indices
    trial_name = trial_sets{t, 2};    % Get trial set name

    % Randomly select one trial from the current trial set
    random_trial = trial_indices(randi(length(trial_indices)));

    % Initialize figure
    figure('Name', ['Power Over Frequency - ', trial_name], 'NumberTitle', 'off');
    hold on;

    % Plot power over frequency for all filtered datasets
    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        % Compute PSD using pwelch
        [pxx, freq] = pwelch(filtered_data(random_trial, :), winSize, overlap, nfft, fs);

        % Plot PSD
        plot(freq, 10*log10(pxx), 'LineWidth', 1.5, 'DisplayName', filter_name);
    end
    hold off;

    % Finalize the plot
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    title(['Power Over Frequency for Trial ', num2str(random_trial), ' (', trial_name, ')']);
    legend('show', 'Location', 'best');
    grid on;

    % Save the plot
    saveas(gcf, ['Power_Over_Frequency_', strrep(trial_name, ' ', '_'), '_Trial_', num2str(random_trial), '.png']);
    close;
end
%%
% Load necessary data

fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Define trial sets
trial_sets = {
    Clean_Trials, 'Clean Trials';
    Valid_Trials, 'Valid Trials';
    other_trials, 'Trials Not in Both';
    all_trials, 'All Trials';
};

% Define filtered data sets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Welch parameters
winSize = 64;    % Window size
overlap = 32;    % Overlap
nfft = 128;      % Number of FFT points
maxFreq = 800;   % Maximum frequency for truncation (Hz)
tTiles = 20;     % Number of time segments
numSamples = size(Cue1LFP_Filtered_50, 2); % Number of samples per trial
inds = round(linspace(1, numSamples, tTiles+1)); % Indices for time segments

% Loop through each filtered dataset and each trial set
for f = 1:size(filtered_data_sets, 1)
    filtered_data = filtered_data_sets{f, 1};
    filter_name = filtered_data_sets{f, 2};

    for t = 1:size(trial_sets, 1)
        trial_indices = trial_sets{t, 1};
        trial_name = trial_sets{t, 2};

        % Compute the average signal for the trial set
        avg_signal = mean(filtered_data(trial_indices, :), 1);

        % Initialize variables for spectrogram computation
        psdMat = [];
        timeVec = zeros(1, tTiles);

        % Loop through time segments
        for i = 1:tTiles
            idxStart = inds(i);
            idxEnd = inds(i+1);
            segment = avg_signal(idxStart:idxEnd);

            % Compute PSD for the segment
            [pxx, fVec] = pwelch(segment, winSize, overlap, nfft, fs);

            % Truncate frequency range
            idxMax = find(fVec <= maxFreq, 1, 'last');
            fTrunc = fVec(1:idxMax);
            pxxTrunc = pxx(1:idxMax);

            % Store power spectrum for plotting
            psdMat = [psdMat, pxxTrunc]; %#ok<AGROW>
            timeVec(i) = idxStart / fs;  % Approximate segment start time (seconds)
        end

        % Plot Welch Spectrogram
        figure('Name', ['Spectrogram - ', filter_name, ' - ', trial_name], 'NumberTitle', 'off');
        imagesc(timeVec * 1000, fTrunc, 10*log10(psdMat)); % Time in ms
        set(gca, 'YDir', 'normal');
        colorbar;
        xlabel('Time (ms)');
        ylabel('Frequency (Hz)');
        title(['Spectrogram (', filter_name, ' - ', trial_name, ')']);

        % Save the plot
        saveas(gcf, ['Spectrogram_', strrep(filter_name, ' ', '_'), '_', strrep(trial_name, ' ', '_'), '.png']);
        close;
    end
end
%% ===== NEWWWW ======= 


% PART 8: Compute and Plot Average PSD for Clean Trials (1-200 Hz Filtered)
% Define parameters
fs = 1000;          % Sampling frequency (Hz)
winSize = 64;       % Window size for Welch's method
overlap = 32;       % Overlap for Welch's method
nfft = 128;         % Number of FFT points

% Extract Clean Trials from the 1-200 Hz filtered data
clean_data = Cue1LFP_Filtered_200(Clean_Trials, :);
num_clean = size(clean_data, 1);

% Preallocate PSD matrix using pwelch output from the first trial
[pxx_temp, fVec] = pwelch(clean_data(1, :), winSize, overlap, nfft, fs);
numFreqs = length(fVec);
psd_all = zeros(num_clean, numFreqs);

% Compute PSD for each clean trial using pwelch
for k = 1:num_clean
    [pxx, ~] = pwelch(clean_data(k, :), winSize, overlap, nfft, fs);
    psd_all(k, :) = pxx;
end

% Calculate the average PSD over all Clean Trials
avg_psd = mean(psd_all, 1);

% Plot and save the Average PSD (linear scale)
figure;
plot(fVec, avg_psd, 'b', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
title('Average PSD for Clean Trials (1-200 Hz Filtered)');
grid on;
saveas(gcf, 'Average_PSD_CleanTrials_1-200Hz.png');

% PART 9: Plot Log(Power) over Frequency for Random Clean Trials and Average
num_random = 5;  % Number of random trials to display

% Select random rows from clean_data; use corresponding original trial indices for labels
rand_idx = randperm(num_clean, num_random);
random_trial_labels = Clean_Trials(rand_idx);

figure;
hold on;
% Plot log(Power) for each selected random trial
for k = 1:num_random
    plot(fVec, log10(psd_all(rand_idx(k), :)), 'LineWidth', 1.5);
end
% Overlay the average PSD in log scale
plot(fVec, log10(avg_psd), 'k', 'LineWidth', 2);
hold off;

xlabel('Frequency (Hz)');
ylabel('Log_{10}(Power)');
title('Log Power Spectrum: Random Clean Trials and Average (1-200 Hz Filtered)');
legend([arrayfun(@(x) sprintf('Trial %d', x), random_trial_labels, 'UniformOutput', false), {'Average'}], 'Location', 'Best');
grid on;
saveas(gcf, 'LogPower_Random_Average_CleanTrials_1-200Hz.png');




%% 

load('saccade_onset.mat');            % Saccade onset times
load('raise_times.mat');              % Raise times
fs = 1000;                            % Sampling frequency

% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters
num_random_trials = 1; % Number of random trials to select
time = (0:size(Cue1LFP, 2) - 1) / fs; % Time vector in seconds

% Randomly select trials from Clean_Trials
if length(Clean_Trials) > num_random_trials
    selected_trials = Clean_Trials(randperm(length(Clean_Trials), num_random_trials));
else
    selected_trials = Clean_Trials; % Use all trials if fewer than num_random_trials
end

% Loop through each selected trial
for i = 1:length(selected_trials)
    trial_idx = selected_trials(i);

    % Get the saccade onset and raise time for the trial
    saccade_time = double(saccade_onset(trial_idx, 1)) / fs; % Convert to seconds
    raise_time = double(raise_times(1, trial_idx)) / fs;     % Convert to seconds
    pre_raise_time = raise_time - 0.1;                      % 100 ms before Saccade 2

    % Report Saccade 1 and Saccade 2 times
    fprintf('Trial %d:\n', trial_idx);
    fprintf('  Saccade 1 Time: %.3f s\n', saccade_time);
    fprintf('  Saccade 2 Time: %.3f s\n\n', raise_time);

    % Loop through each filtered dataset
    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        % Initialize figure
        figure('Name', sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx), 'NumberTitle', 'off');
        hold on;

        % Plot raw data with black transparent line
        plot(time, Cue1LFP(trial_idx, :), 'Color', [0 0 0 0.3], 'LineWidth', 1.5, 'DisplayName', 'Raw Data');

        % Plot filtered data with green line
        plot(time, filtered_data(trial_idx, :), 'g', 'LineWidth', 1.2, 'DisplayName', filter_name);

        % Add vertical lines with labels
        xline(saccade_time, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 1');           % Saccade 1
        xline(raise_time, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 2');            % Saccade 2
        xline(pre_raise_time, 'Color', [1 0.5 0], 'LineStyle', '--', 'LineWidth', 1.5, ...
              'DisplayName', '100 ms Before Saccade 2');                                    % 100 ms Before Saccade 2

        % Finalize the plot with more resolution on the x-axis
        xlim([0 max(time)]); % Adjust limits to show the full time range
        xticks(0:0.1:max(time)); % Increase x-axis resolution by adding ticks every 100 ms
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx));
        legend('show', 'Location', 'best');
        grid on;

        % Save the plot
        saveas(gcf, sprintf('%s_Trial_%d.png', strrep(filter_name, ' ', '_'), trial_idx));
        close;
    end
end
%%


load('saccade_onset.mat');            % Saccade onset times
load('raise_times.mat');              % Raise times
fs = 1000;                            % Sampling frequency

% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters
num_random_trials = 1; % Number of random trials to select
time = (0:size(Cue1LFP, 2) - 1) / fs; % Time vector in seconds

% Randomly select trials from Clean_Trials
if length(Clean_Trials) > num_random_trials
    selected_trials = Clean_Trials(randperm(length(Clean_Trials), num_random_trials));
else
    selected_trials = Clean_Trials; % Use all trials if fewer than num_random_trials
end

% Loop through each selected trial
for i = 1:length(selected_trials)
    trial_idx = selected_trials(i);

    % Get the saccade onset, memory onset, and raise time for the trial
    saccade_time = double(saccade_onset(trial_idx, 1)) / fs; % Convert to seconds
    raise_time = double(raise_times(1, trial_idx)) / fs;     % Convert to seconds
    memory_time = raise_time - 0.1;                         % Memory onset (100 ms before Saccade 2)

    % Report Saccade 1 and Saccade 2 times
    fprintf('Trial %d:\n', trial_idx);
    fprintf('  Saccade 1 Time (Onset): %.3f s\n', saccade_time);
    fprintf('  Memory Time (Onset): %.3f s\n', memory_time);
    fprintf('  Saccade 2 Time (Onset): %.3f s\n\n', raise_time);

    % Loop through each filtered dataset
    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        % Initialize figure
        figure('Name', sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx), 'NumberTitle', 'off');
        hold on;

        % Color plot sections
        % Pre Saccadic
        fill([0, saccade_time, saccade_time, 0], [-100, -100, 100, 100], [0.8, 0.8, 0.8], ...
            'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Pre Saccadic');
        % Fixation 1
        fill([saccade_time, memory_time, memory_time, saccade_time], [-100, -100, 100, 100], [0.6, 0.9, 0.6], ...
            'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Fixation 1');
        % Stimulus
        fill([memory_time, raise_time, raise_time, memory_time], [-100, -100, 100, 100], [0.6, 0.6, 0.9], ...
            'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Stimulus');
        % Feedback
        fill([raise_time, max(time), max(time), raise_time], [-100, -100, 100, 100], [0.9, 0.6, 0.6], ...
            'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Feedback');

        % Plot raw data with black transparent line
        plot(time, Cue1LFP(trial_idx, :), 'Color', [0 0 0 0.3], 'LineWidth', 1.5, 'DisplayName', 'Raw Data');

        % Plot filtered data with green line
        plot(time, filtered_data(trial_idx, :), 'g', 'LineWidth', 1.2, 'DisplayName', filter_name);

        % Add vertical lines with labels
        xline(saccade_time, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 1 Onset');           % Saccade 1 Onset
        xline(memory_time, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Memory Onset');              % Memory Onset
        xline(raise_time, 'm--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 2 Onset');            % Saccade 2 Onset

        % Finalize the plot with more resolution on the x-axis
        xlim([0 max(time)]); % Adjust limits to show the full time range
        xticks(0:0.1:max(time)); % Increase x-axis resolution by adding ticks every 100 ms
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx));
        legend('show', 'Location', 'best');
        grid on;

        % Save the plot
        saveas(gcf, sprintf('%s_Trial_%d.png', strrep(filter_name, ' ', '_'), trial_idx));
        close;
    end
end
%%


load('saccade_onset.mat');            % Saccade onset times
load('raise_times.mat');              % Raise times
fs = 1000;                            % Sampling frequency

% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50, 'Cue1LFP_Filtered_50';
    Cue1LFP_Filtered_100, 'Cue1LFP_Filtered_100';
    Cue1LFP_Filtered_200, 'Cue1LFP_Filtered_200';
    Cue1LFP_Filtered_400, 'Cue1LFP_Filtered_400';
};

% Parameters
num_random_trials = 3; % Number of random trials to select
time = (0:size(Cue1LFP, 2) - 1) / fs; % Time vector in seconds

% Randomly select trials from Clean_Trials
if length(Clean_Trials) > num_random_trials
    selected_trials = Clean_Trials(randperm(length(Clean_Trials), num_random_trials));
else
    selected_trials = Clean_Trials; % Use all trials if fewer than num_random_trials
end

% Loop through each selected trial
for i = 1:length(selected_trials)
    trial_idx = selected_trials(i);

    % Get the saccade onset, memory onset, and raise time for the trial
    saccade_time = double(saccade_onset(trial_idx, 1)) / fs; % Convert to seconds
    raise_time = double(raise_times(1, trial_idx)) / fs;     % Convert to seconds
    memory_time = raise_time - 0.1;                         % Memory onset (100 ms before Saccade 2)

    % Compute indices for splitting
    idx_saccade = round(saccade_time * fs);
    idx_memory = round(memory_time * fs);
    idx_raise = round(raise_time * fs);
    idx_end = size(Cue1LFP, 2);

    % Loop through each filtered dataset
    for f = 1:size(filtered_data_sets, 1)
        filtered_data = filtered_data_sets{f, 1};
        filter_name = filtered_data_sets{f, 2};

        % Split data into 4 parts
        part_1 = filtered_data(trial_idx, 1:idx_saccade);            % Pre Saccadic
        part_2 = filtered_data(trial_idx, idx_saccade:idx_memory);   % Fixation 1
        part_3 = filtered_data(trial_idx, idx_memory:idx_raise);     % Stimulus
        part_4 = filtered_data(trial_idx, idx_raise:idx_end);        % Feedback

        % Save the parts as .mat files
        save(sprintf('%s_Trial_%d_Part_1.mat', filter_name, trial_idx), 'part_1');
        save(sprintf('%s_Trial_%d_Part_2.mat', filter_name, trial_idx), 'part_2');
        save(sprintf('%s_Trial_%d_Part_3.mat', filter_name, trial_idx), 'part_3');
        save(sprintf('%s_Trial_%d_Part_4.mat', filter_name, trial_idx), 'part_4');
    end
end

%%
% Load necessary data


fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Filtered data sets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Trial sets
trial_sets = {
    Clean_Trials, 'Clean Trials';
    Valid_Trials, 'Valid Trials';
    other_trials, 'Trials Not in Both';
    all_trials, 'All Trials';
};

% Time vector
time = (0:size(Cue1LFP, 2) - 1) / fs; % In seconds

%% Maybe Best one that is work
% Load necessary data
%load('saccade_onset.mat');            % Saccade onset times
%load('raise_times.mat');              % Raise times
%fs = 1000;                            % Sampling frequency


% Load necessary data
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Raw_Brekka_Data(LFP)_Google_Drive copy/lfp_021422_17_2.mat'); % Load Cue1LFP (1080x2000 matrix)

%load('Cue1LFP.mat');                  % Raw data
%
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Cue1LFP_Filtered_50.mat');      % 1-50 Hz filtered data
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Cue1LFP_Filtered_100.mat');     % 1-100 Hz filtered data
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Cue1LFP_Filtered_200.mat');     % 1-200 Hz filtered data
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Cue1LFP_Filtered_400.mat');     % 1-400 Hz filtered data
%


load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Best_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Clean_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Valid_Trials.mat')
%load('Clean_Trials.mat');             % Clean trials indices
%load('Valid_Trials.mat');             % Valid trials indices
fs = 1000;                            % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices


load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/second_saccade_onset.mat')


% Load necessary data

% Load necessary data

% Load necessary data

fs = 1000;                            % Sampling frequency

% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50,  'Filtered_50';
    Cue1LFP_Filtered_100, 'Filtered_100';
    Cue1LFP_Filtered_200, 'Filtered_200';
    Cue1LFP_Filtered_400, 'Filtered_400';
};

% Time vector
time = (0:size(Cue1LFP, 2) - 1) / fs;

% Loop through each filtered dataset
for f = 1:size(filtered_data_sets, 1)
    filtered_data = filtered_data_sets{f, 1};
    prefix_name   = filtered_data_sets{f, 2};

    % Initialize 4 cell arrays
    Part1 = cell(length(Best_Trials), 1);
    Part2 = cell(length(Best_Trials), 1);
    Part3 = cell(length(Best_Trials), 1);
    Part4 = cell(length(Best_Trials), 1);

    % Loop through each trial in Best_Trials
    for t = 1:length(Best_Trials)
        trial_idx = Best_Trials(t);

        % Get event times
        saccade_time = double(saccade_onset(trial_idx, 1)) / fs;
        raise_time   = double(second_saccade_onset(trial_idx, 1)) / fs;

        % Define time segments
        start_time   = 0;
        split_1_time = max(0, saccade_time - 0.1);
        split_2_time = saccade_time + 0.05;
        split_3_time = raise_time - 0.05;
        end_time     = max(time);

        % Convert to indices
        idx_start   = max(1, round(start_time   * fs));
        idx_split_1 = max(1, round(split_1_time * fs));
        idx_split_2 = max(1, round(split_2_time * fs));
        idx_split_3 = max(1, round(split_3_time * fs));
        idx_end     = min(size(Cue1LFP, 2), round(end_time * fs));

        % Extract trial parts
        Part1{t} = filtered_data(trial_idx, idx_start:idx_split_1);
        Part2{t} = filtered_data(trial_idx, idx_split_1:idx_split_2);
        Part3{t} = filtered_data(trial_idx, idx_split_2:idx_split_3);
        Part4{t} = filtered_data(trial_idx, idx_split_3:idx_end);
    end

    % Save each part as its own .mat file with correct variable name
    var1 = sprintf('%s_Part1', prefix_name);
    var2 = sprintf('%s_Part2', prefix_name);
    var3 = sprintf('%s_Part3', prefix_name);
    var4 = sprintf('%s_Part4', prefix_name);

    eval([var1 ' = Part1;']);
    eval([var2 ' = Part2;']);
    eval([var3 ' = Part3;']);
    eval([var4 ' = Part4;']);

    save([var1 '.mat'], var1);
    save([var2 '.mat'], var2);
    save([var3 '.mat'], var3);
    save([var4 '.mat'], var4);
end

    %%



%%
% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50, 'Cue1LFP_Filtered_50';
    Cue1LFP_Filtered_100, 'Cue1LFP_Filtered_100';
    Cue1LFP_Filtered_200, 'Cue1LFP_Filtered_200';
    Cue1LFP_Filtered_400, 'Cue1LFP_Filtered_400';
};

% Parameters
trial_idx = 313;                       % Specific trial to analyze
time = (0:size(Cue1LFP, 2) - 1) / fs;  % Time vector in seconds

% Raise Time = second saccade

% Get the saccade onset and raise time for the specified trial
saccade_time = double(saccade_onset(trial_idx, 1)) / fs; % Convert to seconds
raise_time = double(raise_times(1, trial_idx)) / fs;     % Convert to seconds

% Compute the key time points
start_time = 0;                              % Initial time point
split_1_time = max(0, saccade_time - 0.1);   % Ensure split_1_time is not negative
split_2_time = saccade_time + 0.05;          % Saccade 1 onset + 50 ms
split_3_time = raise_time - 0.05;            % Saccade 2 onset - 50 ms
end_time = max(time);                        % End time point

% Convert times to indices and ensure valid ranges
idx_start = max(1, round(start_time * fs));  % Start index (minimum 1)
idx_split_1 = max(1, round(split_1_time * fs));
idx_split_2 = max(1, round(split_2_time * fs));
idx_split_3 = max(1, round(split_3_time * fs));
idx_end = min(size(Cue1LFP, 2), round(end_time * fs)); % Ensure within array size

% Loop through each filtered dataset
for f = 1:size(filtered_data_sets, 1)
    filtered_data = filtered_data_sets{f, 1};
    filter_name = filtered_data_sets{f, 2};

    % Split data into 4 parts
    part_1 = filtered_data(trial_idx, idx_start:idx_split_1);       % Initial to Saccade 1 - 100 ms
    part_2 = filtered_data(trial_idx, idx_split_1:idx_split_2);     % Saccade 1 - 100 ms to Saccade 1 + 50 ms
    part_3 = filtered_data(trial_idx, idx_split_2:idx_split_3);     % Saccade 1 + 50 ms to Saccade 2 - 50 ms
    part_4 = filtered_data(trial_idx, idx_split_3:idx_end);         % Saccade 2 - 50 ms to End

    % Save the parts as .mat files
    save(sprintf('%s_Trial_%d_Part_1.mat', filter_name, trial_idx), 'part_1');
    save(sprintf('%s_Trial_%d_Part_2.mat', filter_name, trial_idx), 'part_2');
    save(sprintf('%s_Trial_%d_Part_3.mat', filter_name, trial_idx), 'part_3');
    save(sprintf('%s_Trial_%d_Part_4.mat', filter_name, trial_idx), 'part_4');
end
%%
% Load necessary data
load('saccade_onset.mat');            % Saccade onset times
load('raise_times.mat');              % Raise times
fs = 1000;                            % Sampling frequency

% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters
trial_idx = 313;                       % Specific trial to analyze
time = (0:size(Cue1LFP, 2) - 1) / fs;  % Time vector in seconds

% Get the saccade onset and raise time for the specified trial
saccade_time = double(saccade_onset(trial_idx, 1)) / fs; % Convert to seconds
raise_time = double(raise_times(1, trial_idx)) / fs;     % Convert to seconds

% Compute key time intervals
split_1_time = max(0, saccade_time - 0.1);   % Saccade 1 onset - 100 ms
split_2_time = saccade_time + 0.05;          % Saccade 1 onset + 50 ms
split_3_time = raise_time - 0.05;            % Saccade 2 onset - 50 ms
end_time = max(time);                        % End time point

% Loop through each filtered dataset
for f = 1:size(filtered_data_sets, 1)
    filtered_data = filtered_data_sets{f, 1};
    filter_name = filtered_data_sets{f, 2};

    % Initialize figure
    figure('Name', sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx), 'NumberTitle', 'off');
    hold on;

    % Color plot sections
    % Part 1: Initial to Saccade 1 - 100 ms
    fill([0, split_1_time, split_1_time, 0], [-100, -100, 100, 100], [0.8, 0.8, 0.8], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 1: Pre Saccade');
    % Part 2: Saccade 1 - 100 ms to Saccade 1 + 50 ms
    fill([split_1_time, split_2_time, split_2_time, split_1_time], [-100, -100, 100, 100], [0.6, 0.9, 0.6], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 2: Saccade 1 Zone');
    % Part 3: Saccade 1 + 50 ms to Saccade 2 - 50 ms
    fill([split_2_time, split_3_time, split_3_time, split_2_time], [-100, -100, 100, 100], [0.6, 0.6, 0.9], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 3: Stimulus Zone');
    % Part 4: Saccade 2 - 50 ms to End
    fill([split_3_time, end_time, end_time, split_3_time], [-100, -100, 100, 100], [0.9, 0.6, 0.6], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 4: Feedback Zone');

    % Plot raw data with black transparent line
    plot(time, Cue1LFP(trial_idx, :), 'Color', [0 0 0 0.3], 'LineWidth', 1.5, 'DisplayName', 'Raw Data');

    % Plot filtered data with green line
    plot(time, filtered_data(trial_idx, :), 'g', 'LineWidth', 1.2, 'DisplayName', filter_name);

    % Add vertical lines with labels
    xline(split_1_time, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 1 Start'); % Saccade 1 - 100 ms
    xline(saccade_time, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 1 Onset'); % Saccade 1 Onset
    xline(split_2_time, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 1 End');   % Saccade 1 + 50 ms
    xline(split_3_time, 'm--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 2 Start'); % Saccade 2 - 50 ms
    xline(raise_time, 'c--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 2 Onset');   % Saccade 2 Onset

    % Finalize the plot with more resolution on the x-axis
    xlim([0 end_time]); % Adjust limits to show the full time range
    xticks(0:0.1:end_time); % Increase x-axis resolution by adding ticks every 100 ms
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx));
    legend('show', 'Location', 'best');
    grid on;

    % Save the plot
    saveas(gcf, sprintf('%s_Trial_%d_Zones.png', strrep(filter_name, ' ', '_'), trial_idx));
    close;
end

%%
% Load necessary data
load('saccade_onset.mat');            % Saccade onset times
load('raise_times.mat');              % Raise times
fs = 1000;                            % Sampling frequency

% Define filtered datasets
filtered_data_sets = {
    Cue1LFP_Filtered_50, '1-50 Hz Filtered';
    Cue1LFP_Filtered_100, '1-100 Hz Filtered';
    Cue1LFP_Filtered_200, '1-200 Hz Filtered';
    Cue1LFP_Filtered_400, '1-400 Hz Filtered';
};

% Parameters
trial_idx = 313;                       % Specific trial to analyze
time = (0:size(Cue1LFP, 2) - 1) / fs;  % Time vector in seconds

% Get the saccade onset and raise time for the specified trial
saccade_time = double(saccade_onset(trial_idx, 1)) / fs; % Convert to seconds
raise_time = double(raise_times(1, trial_idx)) / fs;     % Convert to seconds

% Compute key time intervals
split_1_time = max(0, saccade_time - 0.1);   % Saccade 1 onset - 100 ms
split_2_time = saccade_time + 0.05;          % Saccade 1 onset + 50 ms
split_3_time = raise_time - 0.05;            % Saccade 2 onset - 50 ms
end_time = max(time);                        % End time point

% Loop through each filtered dataset
for f = 1:size(filtered_data_sets, 1)
    filtered_data = filtered_data_sets{f, 1};
    filter_name = filtered_data_sets{f, 2};

    % Initialize figure
    figure('Name', sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx), 'NumberTitle', 'off');
    hold on;

    % Color plot sections
    % Part 1: Initial to Saccade 1 - 100 ms
    fill([0, split_1_time, split_1_time, 0], [-100, -100, 100, 100], [0.8, 0.8, 0.8], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 1');
    % Part 2: Saccade 1 - 100 ms to Saccade 1 + 50 ms
    fill([split_1_time, split_2_time, split_2_time, split_1_time], [-100, -100, 100, 100], [0.6, 0.9, 0.6], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 2');
    % Part 3: Saccade 1 + 50 ms to Saccade 2 - 50 ms
    fill([split_2_time, split_3_time, split_3_time, split_2_time], [-100, -100, 100, 100], [0.6, 0.6, 0.9], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 3');
    % Part 4: Saccade 2 - 50 ms to End
    fill([split_3_time, end_time, end_time, split_3_time], [-100, -100, 100, 100], [0.9, 0.6, 0.6], ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Part 4');

    % Plot raw data with black transparent line
    plot(time, Cue1LFP(trial_idx, :), 'Color', [0 0 0 0.3], 'LineWidth', 1.5, 'DisplayName', 'Raw Data');

    % Plot filtered data with green line
    plot(time, filtered_data(trial_idx, :), 'g', 'LineWidth', 1.2, 'DisplayName', filter_name);

    % Add vertical lines with labels
    xline(split_1_time, 'r--', 'LineWidth', 1.5, 'DisplayName', '-100 ms before Saccade 1');
    xline(saccade_time, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 1 Onset');
    xline(split_2_time, 'b--', 'LineWidth', 1.5, 'DisplayName', '+50 ms after Saccade 1');
    xline(split_3_time, 'm--', 'LineWidth', 1.5, 'DisplayName', '-50 ms before Saccade 2');
    xline(raise_time, 'c--', 'LineWidth', 1.5, 'DisplayName', 'Saccade 2 Onset');

    % Finalize the plot
    xlim([0 end_time]); % Adjust limits to show the full time range
    xticks(0:0.1:end_time); % Increase x-axis resolution by adding ticks every 100 ms
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('%s and Raw Data for Trial %d', filter_name, trial_idx));
    legend('show', 'Location', 'best');
    grid on;

    % Save the plot
    saveas(gcf, sprintf('%s_Trial_%d_Zones.png', strrep(filter_name, ' ', '_'), trial_idx));
    close;
end

