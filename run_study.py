#!/usr/bin/env python3
"""RunPod-ready multi-rate sensor-fusion study for DronePropB.

The pipeline uses recording-level features from matched onboard IMU and
piezoelectric vibration signals. It performs nested group-aware model selection,
outer leave-one-speed-condition-group-out validation, bootstrap uncertainty,
paired ablation comparisons, and generates exactly two publication figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.integrate import trapezoid
from scipy.io import loadmat
from scipy.signal import detrend, welch
from scipy.stats import kurtosis, skew
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import DMatrix, XGBClassifier


LOGGER = logging.getLogger("drone_multirate_fusion")

FILENAME_PATTERN = re.compile(
    r"F(?P<fault>\d+)_SV(?P<severity>\d+)_SP(?P<speed>\d+)_C(?P<condition>\d+)"
    r"(?:_R(?P<repeat>\d+))?"
)

IMU_BANDS = (
    (0.0, 25.0),
    (25.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 350.0),
    (350.0, 500.0),
)

PIEZO_BANDS = (
    (0.0, 100.0),
    (100.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 3500.0),
    (3500.0, 5000.0),
)

AGGREGATIONS = ("median", "iqr", "p90", "mad")
MODEL_ORDER = (
    "IMU-Magnitude-XGB",
    "IMU-Directional-XGB",
    "IMU-XGB",
    "Piezo-XGB",
    "EarlyFusion-XGB",
    "OOF-Weighted-LateFusion",
)


@dataclass(frozen=True)
class StudyConfig:
    raw_directory: Path
    segment_seconds: float
    window_seconds: float
    overlap_fraction: float
    cache_file: Path
    force_recompute: bool
    inner_splits: int
    search_iterations: int
    bootstrap_iterations: int
    run_leave_one_speed_out: bool
    use_gpu: bool
    cpu_threads: int
    output_directory: Path
    figure_dpi: int
    seed: int


def load_config(path: Path) -> StudyConfig:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    return StudyConfig(
        raw_directory=Path(raw["data"]["raw_directory"]),
        segment_seconds=float(raw["data"]["segment_seconds"]),
        window_seconds=float(raw["data"]["window_seconds"]),
        overlap_fraction=float(raw["data"]["overlap_fraction"]),
        cache_file=Path(raw["data"]["cache_file"]),
        force_recompute=bool(raw["data"]["force_recompute"]),
        inner_splits=int(raw["validation"]["inner_splits"]),
        search_iterations=int(raw["validation"]["search_iterations"]),
        bootstrap_iterations=int(raw["validation"]["bootstrap_iterations"]),
        run_leave_one_speed_out=bool(raw["validation"]["run_leave_one_speed_out"]),
        use_gpu=bool(raw["compute"]["use_gpu"]),
        cpu_threads=int(raw["compute"]["cpu_threads"]),
        output_directory=Path(raw["output"]["directory"]),
        figure_dpi=int(raw["output"]["figure_dpi"]),
        seed=int(raw["project"]["seed"]),
    )


def parse_metadata(path: Path) -> dict[str, int | str]:
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unrecognized DronePropB filename: {path.name}")

    fault_type = int(match.group("fault"))
    speed = int(match.group("speed"))
    condition = int(match.group("condition"))

    return {
        "recording_id": path.stem,
        "fault_type": fault_type,
        "severity": int(match.group("severity")),
        "speed": speed,
        "condition": condition,
        "repeat": int(match.group("repeat")) if match.group("repeat") else 0,
        "label": int(fault_type > 0),
        "outer_group": f"SP{speed}_C{condition}",
    }


def center_crop(signal: np.ndarray, samples: int) -> np.ndarray:
    if signal.shape[-1] < samples:
        raise ValueError(f"Signal has {signal.shape[-1]} samples; {samples} required")
    start = (signal.shape[-1] - samples) // 2
    return signal[..., start : start + samples]


def load_recording(path: Path, segment_seconds: float) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    contents = loadmat(
        path,
        variable_names=["IMUData", "AccData"],
        simplify_cells=True,
    )
    imu = contents.get("IMUData")
    acc = contents.get("AccData")
    if not isinstance(imu, dict) or not isinstance(acc, dict):
        raise ValueError("Matched IMUData and AccData are required")

    imu_axes = np.asarray(imu["Vib_Data"], dtype=np.float64)
    if imu_axes.ndim != 2:
        raise ValueError(f"Unexpected IMU shape: {imu_axes.shape}")
    if imu_axes.shape[0] != 3 and imu_axes.shape[1] == 3:
        imu_axes = imu_axes.T
    if imu_axes.shape[0] != 3:
        raise ValueError(f"Expected three IMU axes, received {imu_axes.shape}")

    piezo = np.asarray(acc["Vib_Data"], dtype=np.float64).reshape(-1)
    imu_fs = int(imu["samplingRate"])
    piezo_fs = int(acc["samplingRate"])

    imu_axes = center_crop(imu_axes, int(round(segment_seconds * imu_fs)))
    piezo = center_crop(piezo, int(round(segment_seconds * piezo_fs)))

    imu_axes = detrend(imu_axes, axis=1, type="linear")
    piezo = detrend(piezo, type="linear")
    imu_magnitude = np.sqrt(np.sum(np.square(imu_axes), axis=0))

    return (
        {
            "imu_x": imu_axes[0],
            "imu_y": imu_axes[1],
            "imu_z": imu_axes[2],
            "imu_m": imu_magnitude,
            "piezo": piezo,
        },
        {"imu_fs": imu_fs, "piezo_fs": piezo_fs},
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + np.finfo(float).eps))


def spectral_features(signal: np.ndarray, fs: int, bands: Iterable[tuple[float, float]]) -> dict[str, float]:
    centered = signal - np.mean(signal)
    frequencies, power = welch(
        centered,
        fs=fs,
        nperseg=min(len(centered), 2048 if fs <= 1000 else 16384),
        noverlap=None,
        scaling="density",
    )
    power = np.maximum(power, np.finfo(float).eps)
    total = float(trapezoid(power, frequencies))
    normalized = power / np.sum(power)

    dominant_frequency = float(frequencies[int(np.argmax(power))])
    centroid = safe_divide(float(np.sum(frequencies * power)), float(np.sum(power)))
    bandwidth = math.sqrt(
        safe_divide(float(np.sum(np.square(frequencies - centroid) * power)), float(np.sum(power)))
    )
    entropy = float(-np.sum(normalized * np.log2(normalized + np.finfo(float).eps)))
    entropy /= max(math.log2(len(normalized)), np.finfo(float).eps)
    flatness = safe_divide(float(np.exp(np.mean(np.log(power)))), float(np.mean(power)))

    features = {
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
    }

    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        band_energy = float(trapezoid(power[mask], frequencies[mask])) if np.any(mask) else 0.0
        features[f"relative_bandpower_{int(low)}_{int(high)}"] = safe_divide(band_energy, total)

    return features


def window_features(signal: np.ndarray, fs: int, bands: Iterable[tuple[float, float]]) -> dict[str, float]:
    absolute = np.abs(signal)
    rms = float(np.sqrt(np.mean(np.square(signal))))
    mean_abs = float(np.mean(absolute))
    root_abs_mean = float(np.mean(np.sqrt(absolute + np.finfo(float).eps)))

    features = {
        "rms": rms,
        "std": float(np.std(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "skewness": float(np.nan_to_num(skew(signal, bias=False), nan=0.0)),
        "kurtosis": float(np.nan_to_num(kurtosis(signal, fisher=False, bias=False), nan=0.0)),
        "crest_factor": safe_divide(float(np.max(absolute)), rms),
        "impulse_factor": safe_divide(float(np.max(absolute)), mean_abs),
        "shape_factor": safe_divide(rms, mean_abs),
        "clearance_factor": safe_divide(float(np.max(absolute)), root_abs_mean**2),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(signal)).astype(float))),
    }
    features.update(spectral_features(signal, fs, bands))
    return features


def aggregate_windows(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    output: dict[str, float] = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        output[f"{prefix}_{column}__median"] = median
        output[f"{prefix}_{column}__iqr"] = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
        output[f"{prefix}_{column}__p90"] = float(np.nanpercentile(values, 90))
        output[f"{prefix}_{column}__mad"] = float(np.nanmedian(np.abs(values - median)))
    return output


def extract_channel(signal: np.ndarray, fs: int, window_seconds: float, overlap: float, bands: Iterable[tuple[float, float]], prefix: str) -> dict[str, float]:
    window_size = int(round(window_seconds * fs))
    step = max(1, int(round(window_size * (1.0 - overlap))))
    rows = [
        window_features(signal[start : start + window_size], fs, bands)
        for start in range(0, len(signal) - window_size + 1, step)
    ]
    if not rows:
        raise ValueError(f"No windows produced for {prefix}")
    return aggregate_windows(rows, prefix)


def process_file(path: Path, cfg: StudyConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = parse_metadata(path)
    if int(metadata["speed"]) == 0:
        raise ValueError("SP0 is excluded because no matched faulty-speed condition exists")

    signals, rates = load_recording(path, cfg.segment_seconds)
    row: dict[str, Any] = dict(metadata)

    for channel in ("imu_x", "imu_y", "imu_z", "imu_m"):
        row.update(
            extract_channel(
                signals[channel],
                rates["imu_fs"],
                cfg.window_seconds,
                cfg.overlap_fraction,
                IMU_BANDS,
                channel,
            )
        )

    row.update(
        extract_channel(
            signals["piezo"],
            rates["piezo_fs"],
            cfg.window_seconds,
            cfg.overlap_fraction,
            PIEZO_BANDS,
            "piezo",
        )
    )

    imu_f, imu_p = welch(signals["imu_m"], fs=rates["imu_fs"], nperseg=2048)
    piezo_f, piezo_p = welch(signals["piezo"], fs=rates["piezo_fs"], nperseg=16384)
    psd = {
        "label": int(metadata["label"]),
        "imu_f": imu_f,
        "imu_p": 10.0 * np.log10(np.maximum(imu_p, np.finfo(float).eps)),
        "piezo_f": piezo_f,
        "piezo_p": 10.0 * np.log10(np.maximum(piezo_p, np.finfo(float).eps)),
    }
    return row, psd


def extract_dataset(cfg: StudyConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cfg.cache_file.parent.mkdir(parents=True, exist_ok=True)
    psd_cache = cfg.cache_file.with_suffix(".psd.npz")

    if cfg.cache_file.exists() and psd_cache.exists() and not cfg.force_recompute:
        LOGGER.info("Loading cached recording-level features from %s", cfg.cache_file)
        frame = pd.read_csv(cfg.cache_file)
        data = np.load(psd_cache, allow_pickle=True)
        psd_records = list(data["records"])
        return frame, psd_records

    files = sorted(cfg.raw_directory.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No MATLAB files found under {cfg.raw_directory.resolve()}")

    rows: list[dict[str, Any]] = []
    psd_records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for index, path in enumerate(files, start=1):
        LOGGER.info("[%03d/%03d] %s", index, len(files), path.name)
        try:
            row, psd = process_file(path, cfg)
            rows.append(row)
            psd_records.append(psd)
        except ValueError as error:
            skipped.append({"file": path.name, "reason": str(error)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No matched IMU/piezo recordings were extracted")
    if frame["recording_id"].nunique() != 108:
        raise RuntimeError(f"Expected 108 matched recordings, found {frame['recording_id'].nunique()}")

    frame.to_csv(cfg.cache_file, index=False)
    np.savez_compressed(psd_cache, records=np.array(psd_records, dtype=object))
    pd.DataFrame(skipped).to_csv(cfg.cache_file.parent / "skipped_files.csv", index=False)
    return frame, psd_records


def gpu_available(requested: bool) -> bool:
    if not requested:
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, check=False, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def safe_f_score(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        scores, p_values = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    return scores, p_values


def build_pipeline(seed: int, use_cuda: bool, threads: int) -> Pipeline:
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda" if use_cuda else "cpu",
        random_state=seed,
        n_jobs=threads,
        verbosity=0,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=safe_f_score, k=20)),
            ("model", classifier),
        ]
    )


def hyperparameter_space(feature_count: int) -> dict[str, list[Any]]:
    k_candidates = sorted({min(feature_count, value) for value in (12, 20, 32, 48, 72, 96)})
    return {
        "select__k": k_candidates,
        "model__n_estimators": [250, 400, 650, 900],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.04, 0.07, 0.10],
        "model__min_child_weight": [1, 3, 5, 8],
        "model__subsample": [0.70, 0.85, 1.0],
        "model__colsample_bytree": [0.55, 0.70, 0.85, 1.0],
        "model__reg_alpha": [0.0, 0.05, 0.2, 1.0],
        "model__reg_lambda": [1.0, 3.0, 8.0, 15.0],
        "model__gamma": [0.0, 0.05, 0.2],
    }


def make_inner_cv(y: np.ndarray, groups: np.ndarray, requested_splits: int, seed: int) -> StratifiedGroupKFold:
    splits = min(requested_splits, len(np.unique(groups)))
    if splits < 2:
        raise ValueError("At least two distinct inner groups are required")
    return StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)


def tune_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: StudyConfig, use_cuda: bool, seed: int) -> tuple[Pipeline, dict[str, Any]]:
    pipeline = build_pipeline(seed, use_cuda, cfg.cpu_threads)
    cv = make_inner_cv(y, groups, cfg.inner_splits, seed)
    weights = compute_sample_weight(class_weight="balanced", y=y)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=hyperparameter_space(X.shape[1]),
        n_iter=cfg.search_iterations,
        scoring="roc_auc",
        n_jobs=1 if use_cuda else max(1, cfg.cpu_threads // 2),
        cv=cv,
        random_state=seed,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    search.fit(X, y, groups=groups, model__sample_weight=weights)
    return search.best_estimator_, {"best_score": float(search.best_score_), **search.best_params_}


def oof_probabilities(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: StudyConfig, seed: int) -> np.ndarray:
    cv = make_inner_cv(y, groups, cfg.inner_splits, seed)
    output = np.full(len(y), np.nan, dtype=float)
    for train_index, valid_index in cv.split(X, y, groups):
        model = clone(estimator)
        weights = compute_sample_weight(class_weight="balanced", y=y[train_index])
        model.fit(X.iloc[train_index], y[train_index], model__sample_weight=weights)
        output[valid_index] = model.predict_proba(X.iloc[valid_index])[:, 1]
    if np.isnan(output).any():
        raise RuntimeError("Incomplete inner out-of-fold probabilities")
    return output


def threshold_scores(y: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    predictions = probabilities[:, None] >= thresholds[None, :]
    positive = y[:, None] == 1
    negative = ~positive

    tp = np.sum(predictions & positive, axis=0).astype(float)
    fp = np.sum(predictions & negative, axis=0).astype(float)
    fn = np.sum((~predictions) & positive, axis=0).astype(float)
    tn = np.sum((~predictions) & negative, axis=0).astype(float)

    f1_positive = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn) > 0)
    f1_negative = np.divide(2.0 * tn, 2.0 * tn + fp + fn, out=np.zeros_like(tn), where=(2.0 * tn + fp + fn) > 0)
    return 0.5 * (f1_positive + f1_negative)


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.15, 0.85, 141)
    scores = threshold_scores(y, probabilities, thresholds)
    return float(thresholds[int(np.argmax(scores))])


def choose_fusion(y: np.ndarray, imu_prob: np.ndarray, piezo_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.15, 0.85, 141)
    best_alpha = 0.5
    best_threshold = 0.5
    best_score = -np.inf
    best_auc = -np.inf
    for alpha in np.linspace(0.0, 1.0, 51):
        combined = alpha * imu_prob + (1.0 - alpha) * piezo_prob
        scores = threshold_scores(y, combined, thresholds)
        index = int(np.argmax(scores))
        score = float(scores[index])
        auc = roc_auc_score(y, combined)
        if score > best_score or (math.isclose(score, best_score) and auc > best_auc):
            best_score = score
            best_auc = auc
            best_alpha = float(alpha)
            best_threshold = float(thresholds[index])
    return best_alpha, best_threshold


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    imu_directional = [column for column in frame.columns if column.startswith(("imu_x_", "imu_y_", "imu_z_"))]
    imu_magnitude = [column for column in frame.columns if column.startswith("imu_m_")]
    imu_all = imu_directional + imu_magnitude
    piezo = [column for column in frame.columns if column.startswith("piezo_")]
    return imu_directional, imu_magnitude, imu_all, piezo


def run_outer_validation(frame: pd.DataFrame, cfg: StudyConfig, outer_groups: np.ndarray, label: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    imu_directional, imu_magnitude, imu_columns, piezo_columns = feature_columns(frame)
    combined_columns = imu_columns + piezo_columns
    y = frame["label"].to_numpy(dtype=int)
    logo = LeaveOneGroupOut()
    predictions: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    use_cuda = gpu_available(cfg.use_gpu)
    LOGGER.info("XGBoost device: %s", "cuda" if use_cuda else "cpu")

    for fold, (train_index, test_index) in enumerate(logo.split(frame, y, outer_groups), start=1):
        held_out = str(np.unique(outer_groups[test_index])[0])
        LOGGER.info("%s fold %d: held out %s", label, fold, held_out)

        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)
        inner_groups = train["outer_group"].to_numpy(dtype=str)
        seed = cfg.seed + fold * 101

        magnitude_model, magnitude_params = tune_model(
            train[imu_magnitude], y_train, inner_groups, cfg, use_cuda, seed
        )
        directional_model, directional_params = tune_model(
            train[imu_directional], y_train, inner_groups, cfg, use_cuda, seed + 1
        )
        imu_model, imu_params = tune_model(
            train[imu_columns], y_train, inner_groups, cfg, use_cuda, seed + 2
        )
        piezo_model, piezo_params = tune_model(
            train[piezo_columns], y_train, inner_groups, cfg, use_cuda, seed + 3
        )
        early_model, early_params = tune_model(
            train[combined_columns], y_train, inner_groups, cfg, use_cuda, seed + 4
        )

        magnitude_oof = oof_probabilities(
            magnitude_model, train[imu_magnitude], y_train, inner_groups, cfg, seed + 5
        )
        directional_oof = oof_probabilities(
            directional_model, train[imu_directional], y_train, inner_groups, cfg, seed + 6
        )
        imu_oof = oof_probabilities(
            imu_model, train[imu_columns], y_train, inner_groups, cfg, seed + 7
        )
        piezo_oof = oof_probabilities(
            piezo_model, train[piezo_columns], y_train, inner_groups, cfg, seed + 8
        )
        early_oof = oof_probabilities(
            early_model, train[combined_columns], y_train, inner_groups, cfg, seed + 9
        )

        magnitude_threshold = choose_threshold(y_train, magnitude_oof)
        directional_threshold = choose_threshold(y_train, directional_oof)
        imu_threshold = choose_threshold(y_train, imu_oof)
        piezo_threshold = choose_threshold(y_train, piezo_oof)
        early_threshold = choose_threshold(y_train, early_oof)
        alpha, late_threshold = choose_fusion(y_train, imu_oof, piezo_oof)

        magnitude_test = magnitude_model.predict_proba(test[imu_magnitude])[:, 1]
        directional_test = directional_model.predict_proba(test[imu_directional])[:, 1]
        imu_test = imu_model.predict_proba(test[imu_columns])[:, 1]
        piezo_test = piezo_model.predict_proba(test[piezo_columns])[:, 1]
        early_test = early_model.predict_proba(test[combined_columns])[:, 1]
        late_test = alpha * imu_test + (1.0 - alpha) * piezo_test

        model_outputs = {
            "IMU-Magnitude-XGB": (magnitude_test, magnitude_threshold),
            "IMU-Directional-XGB": (directional_test, directional_threshold),
            "IMU-XGB": (imu_test, imu_threshold),
            "Piezo-XGB": (piezo_test, piezo_threshold),
            "EarlyFusion-XGB": (early_test, early_threshold),
            "OOF-Weighted-LateFusion": (late_test, late_threshold),
        }

        for model_name, (probabilities, threshold) in model_outputs.items():
            for row_index, probability in enumerate(probabilities):
                metadata = test.iloc[row_index]
                predictions.append(
                    {
                        "validation": label,
                        "fold": fold,
                        "held_out_group": held_out,
                        "recording_id": metadata["recording_id"],
                        "fault_type": int(metadata["fault_type"]),
                        "severity": int(metadata["severity"]),
                        "speed": int(metadata["speed"]),
                        "condition": int(metadata["condition"]),
                        "label": int(y_test[row_index]),
                        "model": model_name,
                        "probability": float(probability),
                        "threshold": float(threshold),
                        "prediction": int(probability >= threshold),
                    }
                )

        parameters.append(
            {
                "validation": label,
                "fold": fold,
                "held_out_group": held_out,
                "late_fusion_imu_weight": alpha,
                "thresholds": {
                    "IMU-Magnitude-XGB": magnitude_threshold,
                    "IMU-Directional-XGB": directional_threshold,
                    "IMU-XGB": imu_threshold,
                    "Piezo-XGB": piezo_threshold,
                    "EarlyFusion-XGB": early_threshold,
                    "OOF-Weighted-LateFusion": late_threshold,
                },
                "IMU-Magnitude-XGB": magnitude_params,
                "IMU-Directional-XGB": directional_params,
                "IMU-XGB": imu_params,
                "Piezo-XGB": piezo_params,
                "EarlyFusion-XGB": early_params,
            }
        )

    return pd.DataFrame(predictions), parameters


def metrics_from_arrays(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": safe_divide(float(tn), float(tn + fp)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for class_value in np.unique(y):
        indices = np.flatnonzero(y == class_value)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    combined = np.concatenate(parts)
    rng.shuffle(combined)
    return combined


def bootstrap_metrics(group: pd.DataFrame, iterations: int, seed: int) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    y = group["label"].to_numpy(dtype=int)
    prediction = group["prediction"].to_numpy(dtype=int)
    probability = group["probability"].to_numpy(dtype=float)
    point = metrics_from_arrays(y, prediction, probability)
    rng = np.random.default_rng(seed)
    distributions = {name: [] for name in point}

    for _ in range(iterations):
        indices = stratified_bootstrap_indices(y, rng)
        values = metrics_from_arrays(y[indices], prediction[indices], probability[indices])
        for name, value in values.items():
            distributions[name].append(value)

    intervals = {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in distributions.items()
    }
    return point, intervals


def summarize_primary(predictions: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    rows = []
    for order, model in enumerate(MODEL_ORDER):
        group = predictions[predictions["model"] == model].sort_values("recording_id")
        point, intervals = bootstrap_metrics(group, cfg.bootstrap_iterations, cfg.seed + order * 1000)
        row: dict[str, Any] = {"model": model, "recordings": len(group)}
        for metric, value in point.items():
            row[metric] = value
            row[f"{metric}_ci_low"] = intervals[metric][0]
            row[f"{metric}_ci_high"] = intervals[metric][1]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_improvements(predictions: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    pivot_prob = predictions.pivot(index="recording_id", columns="model", values="probability")
    pivot_pred = predictions.pivot(index="recording_id", columns="model", values="prediction")
    labels = predictions.drop_duplicates("recording_id").set_index("recording_id")["label"].loc[pivot_prob.index].to_numpy(dtype=int)
    rng = np.random.default_rng(cfg.seed + 9999)
    rows = []

    for baseline in ("IMU-XGB", "Piezo-XGB", "EarlyFusion-XGB"):
        distributions = {"auroc": [], "macro_f1": [], "balanced_accuracy": []}
        proposed_probability = pivot_prob["OOF-Weighted-LateFusion"].to_numpy(dtype=float)
        baseline_probability = pivot_prob[baseline].to_numpy(dtype=float)
        proposed_prediction = pivot_pred["OOF-Weighted-LateFusion"].to_numpy(dtype=int)
        baseline_prediction = pivot_pred[baseline].to_numpy(dtype=int)

        proposed_point = metrics_from_arrays(labels, proposed_prediction, proposed_probability)
        baseline_point = metrics_from_arrays(labels, baseline_prediction, baseline_probability)

        for _ in range(cfg.bootstrap_iterations):
            indices = stratified_bootstrap_indices(labels, rng)
            proposed = metrics_from_arrays(labels[indices], proposed_prediction[indices], proposed_probability[indices])
            comparison = metrics_from_arrays(labels[indices], baseline_prediction[indices], baseline_probability[indices])
            for metric in distributions:
                distributions[metric].append(proposed[metric] - comparison[metric])

        for metric, values in distributions.items():
            rows.append(
                {
                    "comparison": f"OOF-Weighted-LateFusion minus {baseline}",
                    "metric": metric,
                    "point_difference": proposed_point[metric] - baseline_point[metric],
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_probability_improvement_gt_0": float(np.mean(np.asarray(values) > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def severity_and_fault_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    best = predictions[predictions["model"] == "OOF-Weighted-LateFusion"].copy()
    rows = []
    for field in ("severity", "fault_type"):
        for value, group in best[best["label"] == 1].groupby(field):
            rows.append(
                {
                    "analysis": field,
                    "level": int(value),
                    "faulty_recordings": len(group),
                    "sensitivity": float(np.mean(group["prediction"] == 1)),
                    "mean_fault_probability": float(group["probability"].mean()),
                }
            )
    return pd.DataFrame(rows)


def ablation_table(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    representations = (
        ("IMU-Magnitude-XGB", "IMU magnitude only"),
        ("IMU-Directional-XGB", "IMU X/Y/Z directional only"),
        ("IMU-XGB", "IMU X/Y/Z plus magnitude"),
        ("Piezo-XGB", "10 kHz piezo only"),
        ("EarlyFusion-XGB", "Early feature fusion"),
        ("OOF-Weighted-LateFusion", "OOF-optimized weighted late fusion"),
    )

    for model, representation in representations:
        group = predictions[predictions["model"] == model]

        rows.append(
            {
                "representation": representation,
                "model": model,
                **metrics_from_arrays(
                    group["label"].to_numpy(),
                    group["prediction"].to_numpy(),
                    group["probability"].to_numpy(),
                ),
            }
        )

    return pd.DataFrame(rows)


def fit_explanatory_model(frame: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    _, _, imu_columns, piezo_columns = feature_columns(frame)
    columns = imu_columns + piezo_columns
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["outer_group"].to_numpy(dtype=str)
    use_cuda = gpu_available(cfg.use_gpu)
    model, _ = tune_model(frame[columns], y, groups, cfg, use_cuda, cfg.seed + 777)

    imputed = model.named_steps["imputer"].transform(frame[columns])
    selected = model.named_steps["select"].transform(imputed)
    support = model.named_steps["select"].get_support()
    selected_names = np.asarray(columns)[support]
    booster = model.named_steps["model"].get_booster()
    contributions = booster.predict(DMatrix(selected), pred_contribs=True)
    importance = np.mean(np.abs(contributions[:, :-1]), axis=0)
    return pd.DataFrame({"feature": selected_names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)


def save_figure_formats(figure: plt.Figure, base: Path, dpi: int) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def draw_pipeline(axis: plt.Axes) -> None:
    axis.axis("off")
    boxes = [
        (0.02, 0.57, 0.20, 0.25, "Matched physical\nrecordings"),
        (0.28, 0.57, 0.20, 0.25, "1 kHz IMU +\n10 kHz piezo"),
        (0.54, 0.57, 0.20, 0.25, "Window features →\nrecording summaries"),
        (0.80, 0.57, 0.18, 0.25, "Nested group CV +\nlate fusion"),
    ]
    for x, y, width, height, text in boxes:
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", linewidth=1.4, fill=False)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    for index in range(len(boxes) - 1):
        start = boxes[index][0] + boxes[index][2]
        end = boxes[index + 1][0]
        axis.add_patch(FancyArrowPatch((start + 0.01, 0.695), (end - 0.01, 0.695), arrowstyle="->", mutation_scale=12))
    axis.text(0.02, 0.28, "Primary test design", fontsize=10, fontweight="bold")
    axis.text(0.02, 0.16, "Nine outer folds: one SP×C dataset-defined subgroup held out per fold", fontsize=9)
    axis.text(0.02, 0.05, "All tuning, thresholds, and fusion weights selected inside each outer training fold", fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def create_figure_1(frame: pd.DataFrame, psd_records: list[dict[str, Any]], cfg: StudyConfig) -> None:
    figure = plt.figure(figsize=(13.5, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])
    draw_pipeline(axis_a)
    axis_a.set_title("(a) Multi-rate sensing and leakage-resistant evaluation", loc="left", fontweight="bold")

    for axis, sensor, title, x_limit in (
        (axis_b, "imu", "(b) Onboard IMU magnitude spectra", 500),
        (axis_c, "piezo", "(c) High-frequency piezo spectra", 5000),
    ):
        for label, label_name in ((0, "Healthy"), (1, "Faulty")):
            selected = [record for record in psd_records if int(record["label"]) == label]
            frequencies = selected[0][f"{sensor}_f"]
            powers = np.vstack([record[f"{sensor}_p"] for record in selected])
            median = np.median(powers, axis=0)
            lower = np.percentile(powers, 25, axis=0)
            upper = np.percentile(powers, 75, axis=0)
            axis.plot(frequencies, median, label=label_name)
            axis.fill_between(frequencies, lower, upper, alpha=0.18)
        axis.set_xlim(0, x_limit)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power spectral density (dB)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    figure.suptitle("Figure 1. Matched multi-rate drone sensing and experimental design", fontsize=15, fontweight="bold")
    save_figure_formats(figure, cfg.output_directory / "Figure_1_sensor_design", cfg.figure_dpi)
    plt.close(figure)


def create_figure_2(predictions: pd.DataFrame, summary: pd.DataFrame, shap: pd.DataFrame, cfg: StudyConfig) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)

    axis = axes[0, 0]
    metric_offsets = {"balanced_accuracy": -0.10, "auroc": 0.10}
    x = np.arange(len(MODEL_ORDER))
    for metric, offset in metric_offsets.items():
        ordered = summary.set_index("model").loc[list(MODEL_ORDER)]
        values = ordered[metric].to_numpy()
        low = ordered[f"{metric}_ci_low"].to_numpy()
        high = ordered[f"{metric}_ci_high"].to_numpy()
        axis.errorbar(x + offset, values, yerr=np.vstack([values - low, high - values]), marker="o", capsize=4, linestyle="none", label=metric.replace("_", " ").title())
    axis.set_xticks(x, [name.replace("-", "\n") for name in MODEL_ORDER])
    axis.set_ylim(0.35, 1.02)
    axis.set_ylabel("Recording-level score (95% bootstrap CI)")
    axis.set_title("(a) Primary performance with uncertainty", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for model in MODEL_ORDER:
        group = predictions[predictions["model"] == model]
        false_positive, true_positive, _ = roc_curve(group["label"], group["probability"])
        auc = roc_auc_score(group["label"], group["probability"])
        axis.plot(false_positive, true_positive, label=f"{model} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("(b) Outer-fold ROC curves", loc="left", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    best = predictions[predictions["model"] == "OOF-Weighted-LateFusion"]
    ConfusionMatrixDisplay.from_predictions(best["label"], best["prediction"], display_labels=["Healthy", "Faulty"], values_format="d", ax=axis, colorbar=False)
    axis.set_title("(c) Proposed fusion confusion matrix", loc="left", fontweight="bold")

    axis = axes[1, 1]
    top = shap.head(12).sort_values("mean_abs_shap")
    axis.barh(top["feature"].str.replace("__", " · ", regex=False), top["mean_abs_shap"])
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_title("(d) Explanatory feature ranking", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.25)

    figure.suptitle("Figure 2. Leakage-resistant validation and sensor-fusion evidence", fontsize=15, fontweight="bold")
    save_figure_formats(figure, cfg.output_directory / "Figure_2_validation_results", cfg.figure_dpi)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(arguments.config)
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    (cfg.output_directory / "cache").mkdir(parents=True, exist_ok=True)

    frame, psd_records = extract_dataset(cfg)
    primary_predictions, primary_parameters = run_outer_validation(
        frame,
        cfg,
        frame["outer_group"].to_numpy(dtype=str),
        "leave-one-speed-condition-group-out",
    )
    primary_predictions.to_csv(cfg.output_directory / "outer_predictions.csv", index=False)

    primary_summary = summarize_primary(primary_predictions, cfg)
    primary_summary.to_csv(cfg.output_directory / "Table_1_primary_performance.csv", index=False)

    improvements = paired_improvements(primary_predictions, cfg)
    improvements.to_csv(cfg.output_directory / "Table_2_paired_improvements.csv", index=False)

    severity = severity_and_fault_analysis(primary_predictions)
    severity.to_csv(cfg.output_directory / "Table_3_severity_and_fault_type.csv", index=False)

    ablation = ablation_table(frame, primary_predictions)
    ablation.to_csv(cfg.output_directory / "Table_4_sensor_ablation.csv", index=False)

    all_parameters = {"primary": primary_parameters}
    if cfg.run_leave_one_speed_out:
        speed_predictions, speed_parameters = run_outer_validation(
            frame,
            cfg,
            frame["speed"].astype(str).to_numpy(),
            "leave-one-speed-out",
        )
        speed_summary = summarize_primary(speed_predictions, cfg)
        speed_summary.to_csv(cfg.output_directory / "Table_S1_leave_one_speed_out.csv", index=False)
        speed_predictions.to_csv(cfg.output_directory / "speed_challenge_predictions.csv", index=False)
        all_parameters["speed_challenge"] = speed_parameters

    with (cfg.output_directory / "fold_hyperparameters.json").open("w", encoding="utf-8") as stream:
        json.dump(all_parameters, stream, indent=2)

    shap = fit_explanatory_model(frame, cfg)
    shap.to_csv(cfg.output_directory / "explanatory_shap_features.csv", index=False)

    create_figure_1(frame, psd_records, cfg)
    create_figure_2(primary_predictions, primary_summary, shap, cfg)

    summary = {
        "matched_recordings": int(frame["recording_id"].nunique()),
        "healthy_recordings": int((frame["label"] == 0).sum()),
        "faulty_recordings": int((frame["label"] == 1).sum()),
        "outer_groups": sorted(frame["outer_group"].unique().tolist()),
        "primary_validation": "leave-one-speed-condition-group-out",
        "test_tuning_leakage": "none: tuning, thresholds and fusion weights selected inside outer training folds",
        "figures": ["Figure_1_sensor_design", "Figure_2_validation_results"],
    }
    with (cfg.output_directory / "study_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("\nPRIMARY RECORDING-LEVEL RESULTS")
    print("=" * 120)
    print(primary_summary[["model", "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high", "macro_f1", "auroc", "auroc_ci_low", "auroc_ci_high", "brier"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPAIRED IMPROVEMENTS")
    print("=" * 120)
    print(improvements.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutputs saved to {cfg.output_directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
