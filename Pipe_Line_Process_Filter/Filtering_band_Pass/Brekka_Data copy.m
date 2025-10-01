%% 1. Load the Trimmed LFP Data
clear; close all; clc;

load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Data/Brekka_Data/Brekka_Data_Raw_Google_Drive/lfp_021422_16_1.mat')
% Make sure the .mat file and variable name are correct
%load('Cue1_LFP.mat');  % Suppose it contains a variable 'lfp_Trim'
% If the loaded variable is named something else, adjust accordingly:
% lfp_Trim = data.lfp_Trim;
lfp_trimmed = Cue1LFP;
[numTrials, numSamples] = size(Cue1LFP);
fs = 1000;  % Sampling frequency (Hz)

fprintf('Loaded LFP_Saccade_Trim with size: %d trials x %d samples\n', ...
    numTrials, numSamples);

%%
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Clean_Trials.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/saccade_onset.mat')
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/Brekka_PrePeocessing_Filter_CODE/Valid_Trials.mat')
%%
load('Not_trials.mat'); % Load valid_trials (1x349 array)

% Dimensions of Cue1LFP
[numTrials, numSamples] = size(Cue1LFP);

% Identify trials not in valid_trials
all_trials = 1:numTrials;
Valid_Trials = setdiff(all_trials, valid_trials);


%% 2. Compute the Average Over All Trials
lfp_avg = mean(lfp_trimmed, 1, 'omitnan');  
%% 3. Plot the Average and Some Random Trials
% Create a time vector (in seconds)
lfp_Trim = lfp_trimmed;

time = (0 : numSamples-1) / fs;

% Choose how many random trials to plot
nRand = 6;  
randTrials = randperm(numTrials, nRand);  % pick 5 unique random trial indices

% Plot
figure('Name','Average and Random Trials','NumberTitle','off');
hold on;
% Plot some random trials
for i = 1:nRand
    trialIdx = randTrials(i);
    plot(time, lfp_Trim(trialIdx, :), 'DisplayName', sprintf('Trial %d', trialIdx));
end
% Plot the average
plot(time, lfp_avg, 'k', 'LineWidth', 2, 'DisplayName','Average LFP');
hold off;

xlabel('Time (s)');
ylabel('Amplitude');
title('Random Trials and Mean LFP');
legend('Location','best');
grid on;

% Save figure if desired
saveas(gcf, 'RandomTrials_and_Average.png');
%%

%% 3. Plot the Average and Some Random Trials from Valid_Trials
% Create a time vector (in seconds)
lfp_Trim = lfp_trimmed; % Ensure lfp_trimmed contains your data
time = (0 : numSamples-1) / fs;

% Number of random trials to plot
nRand = 10;  
randValidTrials = Valid_Trials(randperm(length(Valid_Trials), nRand)); % Randomly pick 5 valid trial indices

% Calculate the average of valid trials
lfp_avg_valid = mean(lfp_Trim(Valid_Trials, :), 1);

% Plot
figure('Name','Average and Random Valid Trials','NumberTitle','off');
hold on;
% Plot some random trials from Valid_Trials
for i = 1:nRand
    trialIdx = randValidTrials(i);
    plot(time, lfp_Trim(trialIdx, :), 'DisplayName', sprintf('Trial %d', trialIdx));
end
% Plot the average of valid trials
plot(time, lfp_avg_valid, 'k', 'LineWidth', 2, 'DisplayName', 'Average LFP (Valid Trials)');
hold off;

xlabel('Time (s)');
ylabel('Amplitude');
title('Random Valid Trials and Mean LFP');
legend('Location', 'best');
grid on;

% Save figure if desired
saveas(gcf, 'RandomValidTrials10_and_Average.png');

%%
%% Identify Clean Trials from Valid_Trials

% Threshold values
value_min = -25;
value_max = 25;
sample_threshold = 200;

% Initialize Clean_Trials
Clean_Trials = [];

% Iterate through Valid_Trials to check each trial
for i = 1:length(Valid_Trials)
    trialIdx = Valid_Trials(i); % Current trial index
    trial_data = lfp_trimmed(trialIdx, :); % Extract data for the trial
    
    % Find samples not in the range [-25, 25]
    num_out_of_range = sum(trial_data < value_min | trial_data > value_max);
    
    % Check if the number of out-of-range samples exceeds the threshold
    if num_out_of_range <= sample_threshold
        Clean_Trials = [Clean_Trials, trialIdx]; % Add trial to Clean_Trials
    end
end

% Save the Clean_Trials to a .mat file
save('Clean_Trials.mat', 'Clean_Trials');

% Display the results
disp('Clean trials have been identified and saved to Clean_Trials.mat');
disp(['Number of clean trials: ', num2str(length(Clean_Trials))]);

%%

% Load required data
load('Clean_Trials.mat'); % Load Clean_Trials
%load('Valid_Trials.mat'); % Load valid_trials
% Assuming lfp_trimmed is already loaded (1080x2000 matrix)

% Calculate averages
avg_clean_trials = mean(lfp_trimmed(Clean_Trials, :), 1); % Average of Clean Trials
avg_valid_trials = mean(lfp_trimmed(Valid_Trials, :), 1); % Average of Valid Trials
avg_all_trials = mean(lfp_trimmed, 1);                   % Average of All Trials

time = (0:size(lfp_trimmed, 2) - 1) / fs; % Time vector (in seconds)

%% Plot Clean Trials and their Average
%% Plot Clean Trials and Their Average
figure('Name', 'Clean Trials and Their Average', 'NumberTitle', 'off');
plot(time, lfp_trimmed(Clean_Trials, :)', 'Color', [0.8, 0.8, 0.8]); % Plot individual Clean Trials in light gray
hold on;
plot(time, avg_clean_trials, 'r', 'LineWidth', 2, 'DisplayName', 'Average (Clean Trials)');
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Clean Trials and Their Average');
legend('Clean Trials', 'Average');
grid on;
saveas(gcf, 'Clean_Trials_and_Average.png'); % Save plot as PNG

% Plot Valid Trials and Their Average
figure('Name', 'Valid Trials and Their Average', 'NumberTitle', 'off');
plot(time, lfp_trimmed(Valid_Trials, :)', 'Color', [0.8, 0.8, 0.8]); % Plot individual Valid Trials in light gray
hold on;
plot(time, avg_valid_trials, 'b', 'LineWidth', 2, 'DisplayName', 'Average (Valid Trials)');
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Valid Trials and Their Average');
legend('Valid Trials', 'Average');
grid on;
saveas(gcf, 'Valid_Trials_and_Average.png'); % Save plot as PNG

% Plot All Trials and Their Average
figure('Name', 'All Trials and Their Average', 'NumberTitle', 'off');
plot(time, lfp_trimmed', 'Color', [0.8, 0.8, 0.8]); % Plot all trials in light gray
hold on;
plot(time, avg_all_trials, 'k', 'LineWidth', 2, 'DisplayName', 'Average (All Trials)');
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('All Trials and Their Average');
legend('All Trials', 'Average');
grid on;
saveas(gcf, 'All_Trials_and_Average.png'); % Save plot as PNG

%%
% Load necessary data
load('Clean_Trials.mat'); 
load('raise_times.mat');      % Load raise_time (1x1080 array)

% Parameters
num_samples = size(Cue1LFP, 2); % Number of time points in each trial
num_random_trials = 3;          % Number of random trials to plot
fs = 1000;                      % Sampling frequency (adjust based on your data)

% Select random trials from Clean_Trials
random_trials = Valid_Trials(randperm(length(Valid_Trials), num_random_trials));

% Time vector (in seconds)
time = (0:num_samples - 1) / fs;

% Plot random trials with Saccade 1 and Saccade 2 onsets
for i = 1:num_random_trials
    % Get the current trial index
    trial_idx = random_trials(i);
    
    % Extract trial data
    trial_data = Cue1LFP(trial_idx, :);
    
    % Find Saccade 1 and Saccade 2 times
    saccade1_time = saccade_onset(trial_idx);       % Saccade 1 onset time (in samples)
    saccade2_time = raise_times(1, trial_idx);       % Saccade 2 onset time (in samples)
    
    % Plot the trial
    figure('Name', sprintf('Trial %d', trial_idx), 'NumberTitle', 'off');
    plot(time, trial_data, 'b', 'LineWidth', 1.5); % Plot trial data
    hold on;
    
    % Add vertical red lines for Saccade 1 and Saccade 2
    xline(saccade1_time / fs, 'r', 'LineWidth', 2, 'DisplayName', 'Saccade 1 Onset');
    xline(saccade2_time / fs, 'r--', 'LineWidth', 2, 'DisplayName', 'Saccade 2 Onset');
    
    % Add labels, title, and legend
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(sprintf('Trial %d with Saccade 1 and Saccade 2 Onsets', trial_idx));
    legend('Trial Data', 'Saccade 1', 'Saccade 2');
    grid on;
    hold off;
    
    % Save the plot as a PNG file
    saveas(gcf, sprintf('Trial_%d_Saccades.png', trial_idx));
end

%%

% Load necessary data
% load('Cue1LFP.mat');         % Load Cue1LFP (1080x2000 matrix)
% load('Clean_Trials.mat');    % Load Clean_Trials
% load('Valid_Trials.mat');    % Load Valid_Trials
fs = 1000;                   % Sampling frequency

% Identify trials not in Clean_Trials or Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Randomly select one trial from each set
random_clean_trial = Clean_Trials(randi(length(Clean_Trials)));
random_valid_trial = Valid_Trials(randi(length(Valid_Trials)));
random_other_trial = other_trials(randi(length(other_trials)));

% Parameters for pwelch
winSize = 64;    % Window size
overlap = 32;    % Overlap
nfft = 128;      % Number of FFT points

% Initialize figure
figure('Name', 'PSD for Random Trials', 'NumberTitle', 'off');
hold on;

% Plot PSD for the random Clean Trial
[pxx_clean, f_clean] = pwelch(Cue1LFP(random_clean_trial, :), winSize, overlap, nfft, fs);
plot(f_clean, 10*log10(pxx_clean), 'r', 'LineWidth', 1.5, 'DisplayName', sprintf('Clean Trial %d', random_clean_trial));

% Plot PSD for the random Valid Trial
[pxx_valid, f_valid] = pwelch(Cue1LFP(random_valid_trial, :), winSize, overlap, nfft, fs);
plot(f_valid, 10*log10(pxx_valid), 'b', 'LineWidth', 1.5, 'DisplayName', sprintf('Valid Trial %d', random_valid_trial));

% Plot PSD for the random Other Trial
[pxx_other, f_other] = pwelch(Cue1LFP(random_other_trial, :), winSize, overlap, nfft, fs);
plot(f_other, 10*log10(pxx_other), 'g', 'LineWidth', 1.5, 'DisplayName', sprintf('Other Trial %d', random_other_trial));

% Finalize plot
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density for Random Trials');
legend('Location', 'best');
grid on;
hold off;

% Save the plot
saveas(gcf, 'PSD_Random_Trials15.png');

%%
% Load necessary data

load('Clean_Trials.mat');    % Load Clean_Trials
fs = 1000;                   % Sampling frequency

% Parameters for pwelch
winSize = 64;    % Window size
overlap = 32;    % Overlap
nfft = 128;      % Number of FFT points

% Number of random trials to select
num_random_trials = 5;

% Select random trials from Clean_Trials
random_clean_trials = Clean_Trials(randperm(length(Clean_Trials), num_random_trials));

% Initialize figure
figure('Name', 'PSD for Random Clean Trials', 'NumberTitle', 'off');
hold on;

% Loop through the random trials
for i = 1:num_random_trials
    trial_idx = random_clean_trials(i); % Get the trial index
    
    % Compute the PSD for the trial
    [pxx, f] = pwelch(Cue1LFP(trial_idx, :), winSize, overlap, nfft, fs);
    
    % Plot the PSD
    plot(f, 10*log10(pxx), 'LineWidth', 1.5, 'DisplayName', sprintf('Clean Trial %d', trial_idx));
end

% Finalize the plot
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density for Random Clean Trials');
legend('Location', 'best');
grid on;
hold off;

% Save the plot
saveas(gcf, 'PSD_Clean_Trials1.png');

%%
% Load necessary data
%load('Cue1LFP.mat');         % Load Cue1LFP (1080x2000 matrix)
%load('Clean_Trials.mat');    % Load Clean_Trials
%load('Valid_Trials.mat');    % Load Valid_Trials
fs = 1000;                   % Sampling frequency

% Identify trials not in both Clean_Trials and Valid_Trials
all_trials = 1:size(Cue1LFP, 1); % All trial indices
other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

% Define Welch parameters
winSize = 64;   % Window size
overlap = 32;   % Overlap
nfft = 128;     % Number of FFT points
maxFreq = 800;  % Maximum frequency for truncation (Hz)
tTiles = 20;    % Number of time segments
numSamples = size(Cue1LFP, 2); % Number of samples per trial
inds = round(linspace(1, numSamples, tTiles+1)); % Indices for time segments

% Function to compute spectrogram for a given set of trials
computeSpectrogram = @(trial_set, trial_name) ...
    compute_and_plot_spectrogram(mean(Cue1LFP(trial_set, :), 1), fs, tTiles, inds, maxFreq, winSize, overlap, nfft, trial_name);

% Compute and plot spectrogram for each set
computeSpectrogram(Clean_Trials, 'Clean Trials');
computeSpectrogram(Valid_Trials, 'Valid Trials');
computeSpectrogram(other_trials, 'Trials Not in Both');
computeSpectrogram(all_trials, 'All Trials');





%%
% Load the Cue1LFP data
%load('Cue1LFP.mat'); % Load Cue1LFP (1080x2000 matrix)

% Sampling frequency
fs = 1000; % Hz

% Define frequency ranges for filters
filter_specs = [
    1, 400; % Bandpass 1-400 Hz
    1, 200; % Bandpass 1-200 Hz
    1, 100  % Bandpass 1-100 Hz
];

% Filtered data filenames
output_filenames = {
    'Cue1LFP_Filtered_400.mat',
    'Cue1LFP_Filtered_200.mat',
    'Cue1LFP_Filtered_100.mat'
};

% Plot filenames
plot_filenames = {
    'Filter_Response_1_400Hz.png',
    'Filter_Response_1_200Hz.png',
    'Filter_Response_1_100Hz.png'
};

% Loop through each filter specification
for i = 1:size(filter_specs, 1)
    % Get filter specifications
    low_cutoff = filter_specs(i, 1);
    high_cutoff = filter_specs(i, 2);

    % Design the bandpass filter
    [b, a] = butter(4, [low_cutoff, high_cutoff] / (fs / 2), 'bandpass');

    % Apply the filter to each trial
    Cue1LFP_Filtered = filtfilt(b, a, Cue1LFP')'; % Transpose to filter along columns

    % Save the filtered data to a .mat file
    save(output_filenames{i}, 'Cue1LFP_Filtered');

    % Plot the filter response
    figure('Name', sprintf('Filter Response (%d-%d Hz)', low_cutoff, high_cutoff), 'NumberTitle', 'off');
    freqz(b, a, 2048, fs);
    title(sprintf('Filter Response (%d-%d Hz)', low_cutoff, high_cutoff));

    % Save the filter response plot
    saveas(gcf, plot_filenames{i});

    % Print status
    fprintf('Filtered data for %d-%d Hz saved to %s\n', low_cutoff, high_cutoff, output_filenames{i});
    fprintf('Filter response plot for %d-%d Hz saved to %s\n', low_cutoff, high_cutoff, plot_filenames{i});
end
%%


%%


% Load the Cue1LFP data
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Raw_Brekka_Data(LFP)_Google_Drive copy/lfp_021422_17_2.mat'); % Loads Cue1LFP

% Sampling frequency
fs = 1000; % Hz

% Define frequency ranges for filters
filter_specs = [
    1, 400; % Bandpass 1–400 Hz
    1, 200; % Bandpass 1–200 Hz
    1, 100; % Bandpass 1–100 Hz
    1, 50   % Bandpass 1–50 Hz
];

% Filtered data filenames
output_filenames = {
    'Cue1LFP_Filtered_400.mat', ...
    'Cue1LFP_Filtered_200.mat', ...
    'Cue1LFP_Filtered_100.mat', ...
    'Cue1LFP_Filtered_50.mat'
};

% Plot filenames
plot_filenames = {
    'Filter_Response_1_400Hz.png', ...
    'Filter_Response_1_200Hz.png', ...
    'Filter_Response_1_100Hz.png', ...
    'Filter_Response_1_50Hz.png'
};

for i = 1:size(filter_specs, 1)
    low_cutoff  = filter_specs(i,1);
    high_cutoff = filter_specs(i,2);

    % Design bandpass
    [b,a] = butter(4, [low_cutoff high_cutoff]/(fs/2), 'bandpass');

    % Filter each trial
    filteredData = filtfilt(b, a, Cue1LFP')';

    % Build dynamic variable name and struct
    varName = sprintf('Cue1LFP_Filtered_%d', high_cutoff);
    S.(varName) = filteredData;

    % Save using -struct so variable inside .mat matches varName
    save(output_filenames{i}, '-struct', 'S');
    clear S  % clear struct for next iteration

    % Plot and save filter response
    h = figure('Visible','off');
    freqz(b, a, 2048, fs);
    title(sprintf('Filter Response %d–%d Hz', low_cutoff, high_cutoff));
    saveas(h, plot_filenames{i});
    close(h);

    % Status
    fprintf('Saved %s to %s\n', varName, output_filenames{i});
end

%%
save('Valid_Trials.mat', 'Valid_Trials');
%% 
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
%other_trials = setdiff(all_trials, union(Clean_Trials, Valid_Trials));

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
    Best_Trials, 'Best Trials';
    all_trials, 'All Trials';
};

% Time vector
time = (0:size(Cue1LFP, 2) - 1) / fs; % In seconds

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

%% 4. Plot the Frequency vs. Power (PSD) for Some Random Trials
% We'll use pwelch to get the Power Spectral Density for each trial

% Suppose your segment length is around 70, choose smaller window
winSize = 64;   % was 256
overlap = 32;   % was 128
nfft    = 128;  % was 512

tTiles = 20;
inds = round(linspace(1, numSamples, tTiles+1));

psdMat = [];
timeVec = zeros(1, tTiles);

for i = 1:tTiles
    idxStart = inds(i);
    idxEnd   = inds(i+1);
    segment  = sig(idxStart:idxEnd);

    % pwelch with smaller window
    [pxx, f] = pwelch(segment, winSize, overlap, nfft, fs);

    % Truncate frequency range if you want
    idxMax = find(f <= 100, 1, 'last');
    fTrunc = f(1:idxMax);
    pxxTrunc = pxx(1:idxMax);

    psdMat = [psdMat, pxxTrunc];
    timeVec(i) = time(idxStart);
end

figure;
imagesc(timeVec*1000, fTrunc, 10*log10(psdMat));
set(gca,'YDir','normal');
colorbar;
title('Welch Spectrogram with Smaller Window');
xlabel('Time (ms)'); ylabel('Frequency (Hz)');


%% 5. Compute and Plot the Welch Spectrogram (Time-Frequency Power)
% We'll illustrate a simple approach to creating a time-frequency plot
% for one (or multiple) trials. Below, we'll do it for a single random trial.
% You can adapt this to loop over multiple trials or do it on the average signal.

trialForSpectrogram = randTrials(1);  % pick one random trial for demonstration
sig = lfp_Trim(trialForSpectrogram, :);

% Define how many time "tiles" or segments you want.
% For instance, we split the signal into ~20 segments.
tTiles = 20;
inds = round(linspace(1, numSamples, tTiles+1));  % +1 to get boundaries

% Frequency range for pwelch analysis
maxFreq = 800;  % in Hz, or choose any range you prefer

% Initialize a matrix to store power spectra for each segment
% We will find pxx up to 'maxFreq' in each segment
timeVec    = zeros(1, tTiles);
psdMat     = [];

for i = 1:tTiles
    idxStart = inds(i);
    idxEnd   = inds(i+1);
    segment  = sig(idxStart:idxEnd);

    % pwelch for this segment
    [pxx, f] = pwelch(segment, winSize, overlap, nfft, fs);

    % Truncate frequency range
    idxMax = find(f <= maxFreq, 1, 'last');
    fTrunc = f(1:idxMax);
    pxxTrunc = pxx(1:idxMax);

    % Store for plotting
    psdMat = [psdMat, pxxTrunc];   %#ok<*AGROW>
    timeVec(i) = time(idxStart);   % approximate segment start time (seconds)
end

figure('Name','Welch Spectrogram','NumberTitle','off');
imagesc(timeVec * 1000, fTrunc, 10*log10(psdMat)); % time in ms
set(gca, 'YDir', 'normal');
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf("Welch Spectrogram (Trial %d)", trialForSpectrogram));

% Optional vertical line at 0 ms if you have an event reference
% xline(0, 'k', 'LineWidth', 2);

% Save figure
saveas(gcf, 'Welch_Spectrogram.png');
%%
%% 1. Load the trimmed LFP data
clear; close all; clc;

% Load the trimmed LFP data (1080 x 1466).
% Make sure the .mat file and variable name are correct.
load('LFP_Saccade_Trim.mat');  % Suppose the variable is 'lfp_trimmed'
% If the loaded variable is named differently, adjust accordingly, e.g.,
% lfp_trimmed = data.lfp_trimmed;

[numTrials, numSamples] = size(lfp_trimmed);
fs = 1000;  % Sampling frequency in Hz

fprintf('Loaded LFP_Saccade_Trim with size: %d trials x %d samples\n', ...
    numTrials, numSamples);


%% 2. Design a Bandpass Filter (1-400 Hz)
% For demonstration, we'll create a 4th-order Butterworth bandpass filter.
% We must normalize the passband by (fs/2) = 500 Hz for digital filter design.

lowCut  = 1;      % 1 Hz
highCut = 400;    % 400 Hz
[b, a] = butter(4, [lowCut, highCut]/(fs/2), 'bandpass');

% Optional: Inspect the filter's frequency response (uncomment if desired)
% freqz(b, a, [], fs);


%% 3. Apply the Filter to Each Trial
filtered_lfp_trimmed = zeros(size(lfp_trimmed));
for i = 1:numTrials
    % Use filtfilt for zero-phase filtering
    filtered_lfp_trimmed(i, :) = filtfilt(b, a, lfp_trimmed(i, :));
end

% Save the filtered data
save('Filtered_lfp_trimmed.mat', 'filtered_lfp_trimmed');
fprintf('Filtered data saved to "Filtered_lfp_trimmed.mat".\n');


%% 4. Plot Comparisons for Random Trials
% Decide how many random trials to look at
numRandomTrials = 3;
randomIndices   = randperm(numTrials, numRandomTrials);

% Create a time vector in seconds
timeVec = (0:numSamples-1) / fs;

% We'll create subfolders or unique names for saving if you like
% For now, let's just do them inline.

for idx = 1:numRandomTrials
    trialID = randomIndices(idx);
    
    %% 4.1 Plot Original vs. Filtered (Time-Domain)
    figure('Name', sprintf('Trial %d: Original vs Filtered', trialID), ...
           'NumberTitle','off');
    hold on;
    plot(timeVec, lfp_trimmed(trialID, :), 'b', 'DisplayName','Original');
    plot(timeVec, filtered_lfp_trimmed(trialID, :), 'r', 'DisplayName','Filtered');
    hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend('Location', 'best');
    title(sprintf('Trial %d: Time-Domain (Original vs Filtered)', trialID));
    grid on;
    
    % Save figure
    saveas(gcf, sprintf('Trial%d_Original_vs_Filtered.png', trialID));
    
    
    %% 4.2 Plot Power Spectrum (PSD) of Original vs. Filtered
    % Assume you have:
%   segData_orig: the current signal segment
%   winSize      = 256;  % or your preferred default
%   overlap      = 128;  % or your preferred default
%   nfft         = 512;
%   fs           = 1000; % sampling rate

segmentLength = length(segData_orig);

% If the segment is shorter than the chosen window, reduce the window:
if segmentLength < winSize
    localWinSize = segmentLength;                 % new window size
    localOverlap = min(overlap, localWinSize - 1);% keep overlap < window
else
    localWinSize = winSize;
    localOverlap = overlap;
end

% Now call pwelch with these local parameters:
[pxx_o, f_o] = pwelch(segData_orig, localWinSize, localOverlap, nfft, fs);

    [pxx_filt, f_filt] = pwelch(filtered_lfp_trimmed(trialID, :), ...
        winSize, overlap, nfft, fs);
    
    figure('Name', sprintf('Trial %d: PSD (Original vs Filtered)', trialID), ...
           'NumberTitle','off');
    plot(f_orig, 10*log10(pxx_orig), 'b', 'DisplayName','Original');
    hold on;
    plot(f_filt, 10*log10(pxx_filt), 'r', 'DisplayName','Filtered');
    hold off;
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    legend('Location','best');
    title(sprintf('Trial %d: PSD (Original vs Filtered)', trialID));
    grid on;
    
    % Save figure
    saveas(gcf, sprintf('Trial%d_PSD_Original_vs_Filtered.png', trialID));
    
    
    %% 4.3 Plot Welch Spectrogram for Original vs. Filtered
    % Let's define how many segments to break the signal into.
    tTiles = 10;  % if you want more or fewer time segments, adjust
    
    % Indices to segment the time axis
    inds = round(linspace(1, numSamples, tTiles+1));
    
    % We'll store PSD for both original and filtered in separate matrices
    psdMat_orig = [];
    psdMat_filt = [];
    timeVals    = zeros(1, tTiles);
    
    % Choose a maximum frequency for display (e.g., 500 or 400)
    maxFreq = 400;  % so we see the band of interest
    
    for seg = 1:tTiles
        segStart = inds(seg);
        segEnd   = inds(seg+1);
        
        segData_orig = lfp_trimmed(trialID, segStart:segEnd);
        segData_filt = filtered_lfp_trimmed(trialID, segStart:segEnd);
        
        [pxx_o, f_o] = pwelch(segData_orig, winSize, overlap, nfft, fs);
        [pxx_f, f_f] = pwelch(segData_filt, winSize, overlap, nfft, fs);
        
        % Truncate frequencies above maxFreq for plotting
        idxMax_o = find(f_o <= maxFreq, 1, 'last');
        idxMax_f = find(f_f <= maxFreq, 1, 'last');
        
        psdMat_orig = [psdMat_orig, 10*log10(pxx_o(1:idxMax_o))]; %#ok<*AGROW>
        psdMat_filt = [psdMat_filt, 10*log10(pxx_f(1:idxMax_f))];
        
        % We assume f_o and f_f have the same length up to maxFreq
        freqVec = f_o(1:idxMax_o);  % for plotting below
        
        timeVals(seg) = timeVec(segStart);  % segment start time in seconds
    end
    
    % Plot Original spectrogram
    figure('Name', sprintf('Trial %d: Welch Spectrogram (Original)', trialID), ...
           'NumberTitle','off');
    imagesc(timeVals*1000, freqVec, psdMat_orig);  % time in ms
    set(gca, 'YDir', 'normal');
    colorbar;
    xlabel('Time (ms)');
    ylabel('Frequency (Hz)');
    title(sprintf('Trial %d: Welch Spectrogram (Original)', trialID));
    
    saveas(gcf, sprintf('Trial%d_WelchSpectrogram_Original.png', trialID));
    
    % Plot Filtered spectrogram
    figure('Name', sprintf('Trial %d: Welch Spectrogram (Filtered)', trialID), ...
           'NumberTitle','off');
    imagesc(timeVals*1000, freqVec, psdMat_filt);
    set(gca, 'YDir', 'normal');
    colorbar;
    xlabel('Time (ms)');
    ylabel('Frequency (Hz)');
    title(sprintf('Trial %d: Welch Spectrogram (Filtered)', trialID));
    
    saveas(gcf, sprintf('Trial%d_WelchSpectrogram_Filtered.png', trialID));
    
end

%% Example Script: Brekka_Data.m
clear; clc; close all;

%% 1. Load your LFP data
% Here we assume the file 'LFP_Saccade_Trim.mat' has a variable 'lfp_trimmed'
load('LFP_Saccade_Trim.mat');  % or your actual file name
[numTrials, numSamples] = size(lfp_trimmed);
fs = 1000;  % sampling frequency (Hz)

disp(['Data loaded: ', num2str(numTrials), ' trials x ', num2str(numSamples), ' samples']);

%% 2. Choose a random trial for demonstration
trialID = randi(numTrials,1);  % pick one random trial
disp(['Using trial #', num2str(trialID)]);

% Extract this trial's data
segData_whole = lfp_trimmed(trialID, :);

%% 3. Define how many time segments you want
tTiles = 10;  % e.g., split the trial into 10 segments
inds = round(linspace(1, numSamples, tTiles + 1));

%% 4. Welch Parameters
% Default "desired" parameters:
winSize = 256;   
overlap = 128;   
nfft    = 512;

%% 5. Pre-allocate for storing PSD
maxFreq = 400;  % frequency limit for plotting (Hz)
timeVals = zeros(1, tTiles);

% We'll store the PSD in a matrix for a spectrogram
% But first, we need to know how many frequency bins we'll keep up to maxFreq
% We'll do an initial pwelch on a dummy signal (e.g., the entire trial) to see the freq vector
[pxx_dummy, freq_full] = pwelch(segData_whole, winSize, overlap, nfft, fs);
idxMax = find(freq_full <= maxFreq, 1, 'last');
freqVec = freq_full(1:idxMax);  
psdMat = [];  % will be freq x timeTiles in size

%% 6. Loop Over Segments
for seg = 1:tTiles
    segStart = inds(seg);
    segEnd   = inds(seg+1);
    
    % Extract this segment
    segData_orig = segData_whole(segStart:segEnd);
    segLength    = length(segData_orig);

    % Dynamically adjust the window if needed:
    if segLength < winSize
        localWinSize = segLength;                 % reduce window to segment length
        localOverlap = min(overlap, localWinSize-1);
    else
        localWinSize = winSize;
        localOverlap = overlap;
    end
    
    % Compute pwelch for this segment
    [pxx_tmp, f_tmp] = pwelch(segData_orig, localWinSize, localOverlap, nfft, fs);
    
    % Truncate to maxFreq
    idxMaxLocal = find(f_tmp <= maxFreq, 1, 'last');
    pxx_tmp_log = 10*log10(pxx_tmp(1:idxMaxLocal));  % convert to dB
    psdMat = [psdMat, pxx_tmp_log]; %#ok<*AGROW>
    
    % Record segment start time (in seconds)
    timeVals(seg) = (segStart-1)/fs;  % (segStart-1) so that the first sample is at 0
end

%% 7. Plot the Spectrogram
figure('Name','Welch Spectrogram','NumberTitle','off');
imagesc(timeVals*1000, freqVec, psdMat);  % x-axis in ms
set(gca, 'YDir', 'normal');
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf('Welch Spectrogram for Trial %d', trialID));

% Save figure if you want
saveas(gcf, sprintf('WelchSpectrogram_Trial%d.png', trialID));

%%

% Function to compute and plot Welch Spectrogram
function compute_and_plot_spectrogram(avg_signal, fs, tTiles, inds, maxFreq, winSize, overlap, nfft, trial_name)
    psdMat = [];
    timeVec = zeros(1, tTiles);

    % Loop through time segments
    for i = 1:tTiles
        idxStart = inds(i);
        idxEnd = inds(i+1);
        segment = avg_signal(idxStart:idxEnd);

        % Compute PSD for the segment
        [pxx, f] = pwelch(segment, winSize, overlap, nfft, fs);

        % Truncate frequency range
        idxMax = find(f <= maxFreq, 1, 'last');
        fTrunc = f(1:idxMax);
        pxxTrunc = pxx(1:idxMax);

        % Store power spectrum for plotting
        psdMat = [psdMat, pxxTrunc]; %#ok<AGROW>
        timeVec(i) = idxStart / fs;  % Approximate segment start time (seconds)
    end

    % Plot Welch Spectrogram
    figure('Name', ['Welch Spectrogram - ', trial_name], 'NumberTitle', 'off');
    imagesc(timeVec * 1000, fTrunc, 10*log10(psdMat)); % Time in ms
    set(gca, 'YDir', 'normal');
    colorbar;
    xlabel('Time (ms)');
    ylabel('Frequency (Hz)');
    title(['Welch Spectrogram (', trial_name, ')']);

    % Save the plot
    saveas(gcf, ['Spectrogram_', strrep(trial_name, ' ', '_'), '.png']);
end


