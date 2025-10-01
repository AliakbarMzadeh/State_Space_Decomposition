
% Load data (if not already in workspace)
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Raw_Brekka_Data(LFP)_Google_Drive copy/lfp_021422_17_2.mat');




%% == FIND SECOND SACCADE ONSET - VALID TRIALS == %%



% Parameters
thresholdDrop = 500;                   % Minimum difference between High and Low to count as saccade
numTrials     = size(Cue1Eye, 3);      % Number of trials

% Preallocate outputs
second_saccade_onset = nan(numTrials, 1);
Valid_Trials          = [];

% Loop over each trial
for trialIdx = 1:numTrials
    % Extract horizontal eye position for this trial (1×2000)
    xEye = squeeze(Cue1Eye(1, :, trialIdx));
    
    % Retrieve first saccade onset
    firstOnset = saccade_onset(trialIdx);
    
    % Skip trial if first onset is invalid
    if isnan(firstOnset) || firstOnset < 1 || firstOnset >= numel(xEye)
        continue;
    end
    
    % Define High value at last sample
    highValue = xEye(end);
    
    % Scan backwards from the last sample down to just after the first saccade
    for idx = numel(xEye)-1 : -1 : (firstOnset + 1)
        lowValue = xEye(idx);
        if (highValue - lowValue) > thresholdDrop
            % Found a sudden drop: mark this sample as second saccade onset
            second_saccade_onset(trialIdx) = idx;
            Valid_Trials(end+1, 1) = trialIdx;
            break;
        end
    end
    % If loop completes without break, second_saccade_onset(trialIdx) remains NaN
end

% Save results
save('second_saccade_onset.mat', 'second_saccade_onset', 'Valid_Trials');
save('Valid_Trials.mat',          'Valid_Trials');





%% == SHOW THE EYE POSITION CURVES == %%



% Load necessary variables if not already in workspace
load('second_saccade_onset.mat');   % contains second_saccade_onset
load('Valid_Trials.mat');           % contains Valid_Trials
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Raw_Brekka_Data(LFP)_Google_Drive copy/lfp_021422_17_2.mat');

% Number of random trials to plot
numPlots = 5;

% Randomly select trials from Valid_Trials
rng(0);  % for reproducibility
plotIndices = randsample(Valid_Trials, min(numPlots, numel(Valid_Trials)));

% Plot each selected trial
figure;
for i = 1:length(plotIndices)
    trialIdx = plotIndices(i);
    xEye     = squeeze(Cue1Eye(1, :, trialIdx));
    
    % Get saccade onset points
    firstOnset  = saccade_onset(trialIdx);
    secondOnset = second_saccade_onset(trialIdx);
    
    subplot(length(plotIndices), 1, i);
    plot(1:length(xEye), xEye, 'b'); hold on;
    xline(firstOnset, 'g--', 'LineWidth', 1.5, 'Label', 'First Saccade');
    xline(secondOnset, 'r--', 'LineWidth', 1.5, 'Label', 'Second Saccade');
    title(['Trial ', num2str(trialIdx)]);
    xlabel('Sample');
    ylabel('Eye X Position');
    grid on;
end



%% == SHOW LFP FOR VALID TRIAL AND ALL ==%%

% ------------------------------
%  SHOW LFP FOR VALID TRIAL AND ALL
% ------------------------------


% Load data
load('second_saccade_onset.mat');   % contains second_saccade_onset
load('Valid_Trials.mat');           % contains Valid_Trials
load('/Users/aliakbarmahmoodzadeh/Desktop/Pipe_Line_Process_Filter/Raw_Brekka_Data(LFP)_Google_Drive copy/lfp_021422_17_2.mat');

% --- Plot all trials and their average ---
figure;
plot(Cue1LFP', 'Color', [0.6 0.6 0.6]); hold on;  % all trials in light gray
plot(mean(Cue1LFP, 1), 'k', 'LineWidth', 2);      % average in black
title('All Trials - Cue1LFP');
xlabel('Sample');
ylabel('LFP Signal');
grid on;
legend('Individual Trials', 'Average Signal');

% --- Plot only Valid Trials and their average ---
validLFP = Cue1LFP(Valid_Trials, :);

figure;
plot(validLFP', 'Color', [0.4 0.6 1]); hold on;   % valid trials in blue
plot(mean(validLFP, 1), 'r', 'LineWidth', 2);     % average in red
title('Valid Trials - Cue1LFP');
xlabel('Sample');
ylabel('LFP Signal');
grid on;
legend('Valid Trials', 'Average Signal');



%% ==  FIND CLEAN TRIALS - DELET THE OUTEDE DATA == 

% Load data
load('/Users/aliakbarmahmoodzadeh/Desktop/PhD_UT/UT_Main_OSC/osc_decomp-main/Data/Brekka_Data/Raw_Brekka_Data(LFP)_Google_Drive/lfp_021422_15_1.mat');

% Parameters
numTrials = size(Cue1LFP, 1);
threshold = 200;
minConsecutive = 100;

% Initialize Not Clean Trials
Not_Clean_Trials = [];

% Loop through all trials
for trialIdx = 1:numTrials
    signal = Cue1LFP(trialIdx, :);
    outMask = (signal < -threshold) | (signal > threshold);  % logical array of outliers
    
    % Find consecutive outliers
    d = diff([0, outMask, 0]);
    runStarts = find(d == 1);
    runEnds   = find(d == -1) - 1;
    runLengths = runEnds - runStarts + 1;
    
    if any(runLengths >= minConsecutive)
        Not_Clean_Trials(end+1, 1) = trialIdx;
    end
end

% Compute Clean Trials
allIndices  = (1:numTrials)';
Clean_Trials = setdiff(allIndices, Not_Clean_Trials);

% Save
save('Clean_Trials.mat', 'Clean_Trials');

% ------------------------------
% Plot 1: All trials and average
% ------------------------------
figure;
plot(Cue1LFP', 'Color', [0.7 0.7 0.7]); hold on;
plot(mean(Cue1LFP, 1), 'k', 'LineWidth', 2);
title('All Trials - Cue1LFP');
xlabel('Sample');
ylabel('LFP Signal');
grid on;
legend('All Trials', 'Average');

% -----------------------------------
% Plot 2: Clean trials and their avg
% -----------------------------------
figure;
plot(Cue1LFP(Clean_Trials, :)', 'Color', [0.4 0.6 1]); hold on;
plot(mean(Cue1LFP(Clean_Trials, :), 1), 'r', 'LineWidth', 2);
title('Clean Trials - Cue1LFP');
xlabel('Sample');
ylabel('LFP Signal');
grid on;
legend('Clean Trials', 'Average');

%%

%% == FIND TRIALS IN BNOTH VALID AND CLEAN TRIALS, PUT THEM IN BEST_TRIALS == %% 

% Load Clean_Trials and Valid_Trials
load('Clean_Trials.mat');      % contains Clean_Trials
load('Valid_Trials.mat');      % contains Valid_Trials

% Find intersection
Best_Trials = intersect(Clean_Trials, Valid_Trials);

% Save result
save('Best_Trials.mat', 'Best_Trials');

% ------------------------------
% Plot 1: All trials and average
% ------------------------------
figure;
plot(Cue1LFP', 'Color', [0.7 0.7 0.7]); hold on;
plot(mean(Cue1LFP, 1), 'k', 'LineWidth', 2);
title('All Trials - Cue1LFP');
xlabel('Sample');
ylabel('LFP Signal');
grid on;
legend('All Trials', 'Average');

% ------------------------------------
% Plot 2: Best trials and their average
% ------------------------------------
figure;
plot(Cue1LFP(Best_Trials, :)', 'Color', [0.2 0.5 0.9]); hold on;
plot(mean(Cue1LFP(Best_Trials, :), 1), 'r', 'LineWidth', 2);
title('Best Trials - Cue1LFP');
xlabel('Sample');
ylabel('LFP Signal');
grid on;
legend('Best Trials', 'Average');







