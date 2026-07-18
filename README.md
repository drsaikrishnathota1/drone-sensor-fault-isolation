# Multi-Rate Sensor Fusion for Drone Propeller Fault Detection

This compact RunPod-ready repository performs a rigorous secondary analysis of the public DronePropB dataset using both sensor streams available in each matched recording:

- Three-axis onboard IMU vibration at 1 kHz
- High-sensitivity piezoelectric vibration at 10 kHz

## Scientific design

The script creates one recording-level sample per physical experiment. Window-level features are aggregated before validation, preventing neighboring windows from leaking between training and testing.

Primary validation uses nine leave-one-group-out folds, where each held-out group is one dataset-defined operating-speed/C-coded subgroup. Model selection occurs only inside each outer training fold through stratified group-aware cross-validation.

Four models are evaluated:

1. IMU-only XGBoost
2. Piezo-only XGBoost
3. Early-fusion XGBoost
4. OOF-optimized weighted late fusion

The script also produces:

- Recording-level bootstrap confidence intervals
- Paired bootstrap improvement intervals
- Magnitude-only and directional-feature ablations
- Fault-severity and fault-type sensitivity analyses
- Leave-one-speed-out challenge results
- Two publication-grade multi-panel figures

## Dataset placement

Place the MATLAB files here:

```text
data/raw/dronepropb/DronePropB Ground Testing Dataset/
```

## Run on RunPod

Use an official RunPod PyTorch Pod template, open a terminal, clone or upload this repository, place the dataset at the path above, and run:

```bash
bash runpod_run.sh
```

To rerun without extracting features again, leave `force_recompute: false` in `config.yaml`.

## Main outputs

```text
outputs/
├── Figure_1_sensor_design.png/.pdf/.tiff
├── Figure_2_validation_results.png/.pdf/.tiff
├── Table_1_primary_performance.csv
├── Table_2_paired_improvements.csv
├── Table_3_severity_and_fault_type.csv
├── Table_S1_leave_one_speed_out.csv
├── outer_predictions.csv
├── fold_hyperparameters.json
└── study_summary.json
```

## Important interpretation

The code does not guarantee a particular accuracy. It protects the test folds from tuning and reports honest uncertainty. The primary claim should be based on the final confidence intervals and paired improvements, not on the largest single metric.
