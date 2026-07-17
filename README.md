# Drone Sensor Fault Isolation

A compact research repository for detecting drone propeller faults from real
IMU vibration recordings using lightweight machine-learning methods.

## Proposed paper

**Speed-Robust IMU Detection of Propeller Faults in Drones Using Lightweight
Machine Learning**

## Research objective

Determine whether six lightweight vibration features extracted from onboard
three-axis IMU signals can distinguish healthy and faulty drone propellers.

## Features

- Root mean square
- Standard deviation
- Peak-to-peak amplitude
- Kurtosis
- Dominant frequency
- Spectral entropy

## Repository structure

- `data/raw/` — public drone sensor recordings
- `src/main.py` — dataset inspection and feature extraction
- `results/` — model metrics and predictions
- `figures/` — publication-quality plots
- `config.yaml` — experiment settings

## Setup

Create a Python environment and install the required packages:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Inspect the dataset

After placing CSV files inside `data/raw/`, run:

    python src/main.py --inspect

## Run feature extraction

    python src/main.py
