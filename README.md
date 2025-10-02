# Oscillatory Component Decomposition Pipeline

## Overview
MATLAB pipeline for decomposing Local Field Potential (LFP) signals into oscillatory components using state-space modeling with amplitude-based filtering.

---

## Repository Structure

```
├── Pipe_Line_Process_Filter/          # Pre-processing pipeline
├── OSC_CODE/                          # Oscillatory decomposition functions
├── Filtered@BandPass@200Hz_WindowWise/ # Example filtered data
├── Main_State_Space.m                 # Main execution file
└── README.md
```

---

## Workflow

### Step 1: Pre-Processing

Process raw LFP data using **Pipe_Line_Process_Filter**:

1. Remove bad trials
2. Apply Butterworth bandpass filter (e.g., 200Hz)
3. Divide filtered LFP into 4 time windows

**Output:** `Filtered_200_Part2.mat` (Part 2, filtered @ 200Hz)

Example data: `Filtered@BandPass@200Hz_WindowWise/`

---

### Step 2: Oscillatory Decomposition

Open `Main_State_Space.m` and configure:

```matlab
% Load filtered-windowed data
load('/path/to/your/Filtered_200_Part2.mat');

% Set data variable (change for different parts)
dataCell = Filtered_200_Part2;

% Configure processing range (for RAM safety)
startIdx = 4;    % First trial to process
endIdx   = 5;    % Last trial to process

% Set amplitude threshold
AMP_THRESHOLD = 0.01;  % Components below this are filtered out
```

Run `Main_State_Space.m`

---

## Code Blocks

### Block 1: Processing & Filtering
- Runs oscillatory decomposition on each trial
- Filters components based on amplitude threshold
- Saves results to `OSC_Results/` folder

### Block 2: Summary Table
Displays summary statistics:

```
CellIndex  OrigOsc  ValidOsc  Removed  MeanFreq  MeanP2P  MeanOrder  MSE       Threshold
_________  _______  ________  _______  ________  _______  _________  ________  _________
    1        3        3         0       65.42     8.0446   0.33333    1.22e-15    NaN
    4        4        4         0       87.066    6.9322   0.25       1.88e-21    0.1
    5        4        3         1       75.349    8.6964   0.33333    1.15e-18    0.1
```

### Block 3: Valid Frequencies Report
```
========== VALID FREQUENCIES [Hz] ==========
Cell 001: [4.1 58 1.3e+02]
Cell 002: 1.3e-05
Cell 004: [4.4 51 1.1e+02 1.8e+02]
Cell 005: [5.8 66 1.5e+02]
=============================================
```

### Block 4: Component Visualization
Set trial number to plot:
```matlab
cellNum = 4;
```
Generates plots of individual components and reconstruction.

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `AMP_THRESHOLD` | Minimum peak-to-peak amplitude | 0.01 |
| `MAX_OSC` | Maximum number of oscillators | 6 |
| `fs` | Sampling frequency (Hz) | 1000 |
| `startIdx` / `endIdx` | Trial range to process | 4 / 5 |

---

## Output Files

Results saved in `OSC_Results/cell_XXX_results.mat`:

- `result.numComponents` - Valid components
- `result.numComponentsOrig` - Original components
- `result.frequencies` - Valid frequencies (Hz)
- `result.p2pRaw` - Peak-to-peak amplitudes
- `result.orderRaw` - Orders of magnitude (10^n)
- `result.mse` - Reconstruction error
- `result.ampThreshold` - Threshold used
- `result.removedCount` - Removed components

---

## Quick Start

```matlab
% 1. Load data
load('Filtered_200_Part2.mat');
dataCell = Filtered_200_Part2;

% 2. Configure
startIdx = 1;
endIdx = 10;
AMP_THRESHOLD = 0.01;

% 3. Run
run('Main_State_Space.m')
```

---

## Notes

- **RAM Management:** Adjust `startIdx` and `endIdx` to process trials in batches
- **Part Naming:** Change `Filtered_200_Part2` to `Part1/3/4` for different windows
- **Threshold:** Lower values keep more components but may include noise

---

## Dependencies

- MATLAB R2020a or later
- Signal Processing Toolbox
- `osc_decomp_uni.m` (included in `OSC_CODE/`)

---

## Contact

For questions or issues, open an issue in this repository.
