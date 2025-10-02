clear; clc;

% Load the merged frequency data (cell array)
load('/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/Filtered_200_Part4_Frequency.mat')
%load('Filtered_200_Part1_Frequency.mat', 'mergedFreq_F_200_Part_1');

N = numel(mergedFreq_F_200_Part_4);  % number of trials
numSquares = 5;  % maximum number of components per trial

% Gather all available component values (from all trials) to determine the colormap limits
allVals = [];
for i = 1:N
    comp = mergedFreq_F_200_Part_4{i};
    allVals = [allVals; comp(:)];  %#ok<AGROW>
end

minVal = min(allVals);
maxVal = max(allVals);

% Choose a colormap (here, using jet with 256 colors)
cmap = jet(256);
nColors = size(cmap,1);

figure; hold on;

% Loop over each trial: each trial gets one row.
% Each row has 5 squares along the x-axis representing up to 5 components.
for i = 1:N
    comp = mergedFreq_F_200_Part_4{i};
    nComp = length(comp);
    for j = 1:numSquares
        % Determine the rectangle position:
        xPos = j - 1;      % x from 0 to 4
        yPos = i - 1;      % y from 0 to N-1
        width = 1;
        height = 1;
        
        if j <= nComp
            % Map the component value to a color using linear scaling.
            normVal = (comp(j) - minVal) / (maxVal - minVal);
            colorIdx = round(normVal*(nColors-1)) + 1;
            rectColor = cmap(colorIdx, :);
        else
            % If the trial has fewer than 5 components, color the extra square black.
            rectColor = [1 1 1];
        end
        
        rectangle('Position', [xPos, yPos, width, height], ...
                  'FaceColor', rectColor, 'EdgeColor', 'w');
    end
end

% Adjust the axes.
xlim([0, numSquares]);
ylim([0, N]);
set(gca, 'YDir', 'normal');  % so trial 1 is at the bottom
set(gca, 'XTick', 0.5:1:(numSquares-0.5), 'XTickLabel', 1:numSquares);
set(gca, 'YTick', 0.5:1:(N-0.5), 'YTickLabel', 1:N);
xlabel('Component Index');
ylabel('Trial Number');
title('Filtered 200 Part4 Frequency Components');

% Add a colorbar with the appropriate colormap and value range.
colormap(cmap);
caxis([minVal maxVal]);
colorbar;

% Save the plot as PNG and FIG files.
saveas(gcf, 'Filtered_200_Part4_Frequency.png');
%savefig('Filtered_200_Part1_Frequency.fig');

%%


clear; clc;

% Define parts and base file path (adjust the path as needed)
parts = {'Part1', 'Part2', 'Part3', 'Part4'};
basePath = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/Merged_All_Parts_F200_CleanTrials';

% Loop over each part
for idx = 1:length(parts)
    partName = parts{idx};
    % Construct the file name (e.g., 'Filtered_200_Part1_Frequency.mat')
    fileName = sprintf('Filtered_200_%s_Frequency.mat', partName);
    filePath = fullfile(basePath, fileName);
    
    % Load the merged frequency data from the .mat file.
    % The file is assumed to contain one variable (a cell array)
    dataStruct = load(filePath);
    fNames = fieldnames(dataStruct);
    mergedData = dataStruct.(fNames{1});
    
    % Determine the number of trials (each cell corresponds to one trial)
    N = numel(mergedData);
    numSquares = 5;  % Maximum number of components per trial
    
    % For colormap scaling, collect all component values that are <=205.
    % (Values >205 will be plotted in black.)
    allVals = [];
    for i = 1:N
        comp = mergedData{i};
        valid = comp(comp <= 205);
        allVals = [allVals; valid(:)];  %#ok<AGROW>
    end
    if isempty(allVals)
        minVal = 0;
        maxVal = 205;
    else
        minVal = min(allVals);
        maxVal = max(allVals);
    end
    
    % Choose a colormap (using jet with 256 colors)
    cmap = jet(256);
    nColors = size(cmap,1);
    
    % Create a new figure for the current part
    figure; hold on;
    
    % Loop over trials (each trial will be one row)
    for i = 1:N
        comp = mergedData{i};
        nComp = length(comp);
        for j = 1:numSquares
            % Determine rectangle position: x increases with component index,
            % y increases with trial number.
            xPos = j - 1;      % x from 0 to 4
            yPos = i - 1;      % y from 0 to N-1
            width = 1;
            height = 1;
            
            if j <= nComp
                % If the component exists for this square...
                if comp(j) > 205
                    % If the value is above 205, assign black.
                    rectColor = [0 0 0];
                else
                    % Otherwise, map the value linearly to the colormap.
                    normVal = (comp(j) - minVal) / (maxVal - minVal);
                    colorIdx = round(normVal*(nColors-1)) + 1;
                    rectColor = cmap(colorIdx, :);
                end
            else
                % If no component exists (fewer than 5 components), color the square black.
                rectColor = [1 1 1];
            end
            
            rectangle('Position', [xPos, yPos, width, height], ...
                      'FaceColor', rectColor, 'EdgeColor', 'w');
        end
    end
    
    % Adjust axes so that each square is 1x1; label x-axis as component index and y-axis as trial number.
    xlim([0, numSquares]);
    ylim([0, N]);
    set(gca, 'YDir', 'normal');  % so trial 1 appears at the bottom
    set(gca, 'XTick', 0.5:1:(numSquares-0.5), 'XTickLabel', 1:numSquares);
    set(gca, 'YTick', 0.5:1:(N-0.5), 'YTickLabel', 1:N);
    xlabel('Component Index');
    ylabel('Trial Number');
    title(sprintf('Filtered 200 %s Frequency Components', partName));
    
    % Set colormap and color axis limits (for values <=205)
    colormap(cmap);
    caxis([minVal maxVal]);
    colorbar;
    
    % Save the plot as PNG and FIG files.
    outFilePNG = sprintf('Filtered_200_%s_Frequency_P.png', partName);
    
    saveas(gcf, outFilePNG);
    
    
    close;  % Close the figure before moving on to the next part
end

%%
clear; clc;

% Define parts and base file path (adjust the path as needed)
parts = {'Part1', 'Part2', 'Part3', 'Part4'};
basePath = '/Users/aliakbarmahmoodzadeh/Desktop/UT/osc_decomp-main/Code/OSC_CODE/Filtered_200_Parts_Fk/';

for idx = 1:length(parts)
    partName = parts{idx};
    % Construct the file name, e.g., 'Filtered_200_Part1_Frequency.mat'
    fileName = sprintf('Filtered_200_%s_Frequency.mat', partName);
    filePath = fullfile(basePath, fileName);
    
    % Load the merged frequency data; the file must contain one variable
    dataStruct = load(filePath);
    fNames = fieldnames(dataStruct);
    mergedData = dataStruct.(fNames{1});
    
    % Determine number of trials (each cell is one trial)
    N = numel(mergedData);
    numSquares = 5;  % Maximum number of components per trial
    
    % Gather all component values (only those <=205) to set colormap limits.
    allVals = [];
    for i = 1:N
        comp = mergedData{i};
        valid = comp(comp <= 205);
        allVals = [allVals; valid(:)];  %#ok<AGROW>
    end
    if isempty(allVals)
        minVal = 0;
        maxVal = 205;
    else
        minVal = min(allVals);
        maxVal = max(allVals);
    end
    
    % Choose a colormap (using jet with 256 colors)
    cmap = jet(256);
    nColors = size(cmap, 1);
    
    % Create a new figure for the current part
    figure; hold on;
    
    % Loop over each trial (each trial gets one row)
    for i = 1:N
        comp = mergedData{i};
        nComp = length(comp);
        for j = 1:numSquares
            % Determine rectangle position:
            xPos = j - 1;      % x from 0 to 4
            yPos = i - 1;      % y from 0 to N-1
            width = 1;
            height = 1;
            
            if j <= nComp
                % If a component exists for this square...
                if comp(j) > 205
                    % If value >205, force black color.
                    rectColor = [0 0 0];
                else
                    % Map the component value to a color using linear scaling.
                    normVal = (comp(j) - minVal) / (maxVal - minVal);
                    colorIdx = round(normVal*(nColors-1)) + 1;
                    rectColor = cmap(colorIdx, :);
                end
                % Format the component value as text.
                textStr = sprintf('%.2f', comp(j));
            else
                % If no component exists, use black and no text.
                rectColor = [0 0 0];
                textStr = '';
            end
            
            % Draw the colored rectangle.
            rectangle('Position', [xPos, yPos, width, height], ...
                      'FaceColor', rectColor, 'EdgeColor', 'w');
                  
            % Choose text color: white if the background is black; otherwise, black.
            if all(rectColor == 0)
                textColor = [1 1 1];
            else
                textColor = [0 0 0];
            end
            
            % Annotate the rectangle with the value.
            text(xPos + width/2, yPos + height/2, textStr, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'Color', textColor, 'FontSize', 8);
        end
    end
    
    % Adjust the axes.
    xlim([0, numSquares]);
    ylim([0, N]);
    set(gca, 'YDir', 'normal');  % so trial 1 is at the bottom
    set(gca, 'XTick', 0.5:1:(numSquares-0.5), 'XTickLabel', 1:numSquares);
    set(gca, 'YTick', 0.5:1:(N-0.5), 'YTickLabel', 1:N);
    xlabel('Component Index');
    ylabel('Trial Number');
    title(sprintf('Filtered 200 %s Frequency Components', partName));
    
    % Set colormap and add a colorbar for values <=205.
    colormap(cmap);
    caxis([minVal maxVal]);
    colorbar;
    
    % Save the plot as PNG and FIG files.
    outFilePNG = sprintf('Filtered_200_%s_Frequency.png', partName);
    %outFileFIG = sprintf('Filtered_200_%s_Frequency.fig', partName);
    saveas(gcf, outFilePNG);
   
    
    close;  % Close the figure before processing the next part.
end


