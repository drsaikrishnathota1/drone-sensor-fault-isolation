# Operating-Speed-Conditioned IMU Detection of Drone Propeller Faults

A compact and reproducible sensor-data experiment for detecting drone
propeller faults using direction-preserving onboard IMU vibration features.

## Proposed Short Communication

**Operating-Speed-Conditioned IMU Detection of Drone Propeller Faults Using
Direction-Preserving Vibration Features**

## Research objective

This study evaluates whether retaining separate X-, Y-, and Z-axis vibration
information improves drone propeller-fault detection compared with using only
the combined vibration magnitude.

## Dataset

The experiment uses the public DronePropB ground-testing dataset containing
physical vibration measurements from commercial drones with healthy and
defective propellers.

The present study uses:

- 108 recordings containing onboard three-axis IMU measurements
- 27 healthy recordings
- 81 faulty recordings
- Three operating-speed levels
- Three fault types
- Three fault-severity levels
- IMU sampling rate of 1,000 Hz

The raw dataset is not stored in this repository.

## Signal processing

A common 30-second center segment is extracted from each recording. Each signal
is divided into one-second windows with 50% overlap.

Six features are extracted from each channel:

- Root mean square
- Standard deviation
- Peak-to-peak amplitude
- Kurtosis
- Dominant frequency
- Spectral entropy

Two feature representations are compared:

1. Six magnitude-only features
2. Twenty-four X-, Y-, Z-, and magnitude-based features

## Validation

Models are trained separately for each operating-speed level.

For each speed, one C-coded experimental condition is held out for testing,
while the remaining two conditions are used for training. Complete recordings,
rather than individual windows, determine the final evaluation unit.

The exact physical interpretation of the dataset's C1, C2, and C3 identifiers
should be described conservatively unless confirmed by the dataset authors.

## Models

- Logistic Regression
- Random Forest

## Verified recording-level results

| Feature representation | Model | Accuracy | Balanced accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|---:|
| X/Y/Z/magnitude, 24 features | Random Forest | 0.8241 | 0.8086 | 0.7830 | 0.9031 |
| Magnitude only, 6 features | Random Forest | 0.7870 | 0.7840 | 0.7469 | 0.8006 |
| X/Y/Z/magnitude, 24 features | Logistic Regression | 0.6296 | 0.6790 | 0.6068 | 0.6968 |
| Magnitude only, 6 features | Logistic Regression | 0.5370 | 0.5926 | 0.5206 | 0.5921 |

## Best-model results by speed

| Speed level | Balanced accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|
| SP1 | 0.7963 | 0.8075 | 0.9506 |
| SP2 | 0.8148 | 0.8393 | 0.8807 |
| SP3 | 0.8148 | 0.7078 | 0.9095 |

## Main finding

Preserving directional IMU vibration information improved Random Forest
macro-F1 from 0.7469 to 0.7830 and AUROC from 0.8006 to 0.9031 compared with
magnitude-only features.

## Repository structure

- `src/main.py` — complete feature extraction and validation pipeline
- `config.yaml` — experiment configuration
- `requirements.txt` — Python dependencies
- `data/raw/` — local public dataset location
- `results/` — generated metrics and feature files
- `figures/` — generated publication figures

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Dataset placement

Place the DronePropB MATLAB files under:

    data/raw/dronepropb/DronePropB Ground Testing Dataset/

## Run

    python src/main.py

## Generated outputs

The experiment generates:

- Conditioned IMU feature table
- Recording-level predictions
- Fold-level and overall metrics
- Random Forest feature importance
- Recording-level confusion matrix
- Feature-importance figure

## Reproducibility note

Raw data, generated results, figures, and trained model artifacts are excluded
from Git because they can be regenerated from the public dataset and source
code.
