#!/usr/bin/env python3
"""Speed-conditioned drone propeller-fault detection using DronePropB IMU data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FILENAME_PATTERN = re.compile(
    r"F(?P<fault>\d+)_"
    r"SV(?P<severity>\d+)_"
    r"SP(?P<speed>\d+)_"
    r"C(?P<configuration>\d+)"
    r"(?:_R(?P<repeat>\d+))?"
)

CHANNELS = ("x", "y", "z", "m")

FEATURE_SUFFIXES = (
    "rms",
    "std",
    "peak_to_peak",
    "kurtosis",
    "dominant_frequency",
    "spectral_entropy",
)

MAGNITUDE_FEATURES = [
    f"m_{suffix}"
    for suffix in FEATURE_SUFFIXES
]

DIRECTION_FEATURES = [
    f"{channel}_{suffix}"
    for channel in CHANNELS
    for suffix in FEATURE_SUFFIXES
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def parse_filename(path: Path) -> dict[str, int | str]:
    match = FILENAME_PATTERN.fullmatch(path.stem)

    if match is None:
        raise ValueError(f"Unrecognized filename: {path.name}")

    fault_type = int(match.group("fault"))

    return {
        "recording_id": path.stem,
        "fault_type": fault_type,
        "severity": int(match.group("severity")),
        "speed": int(match.group("speed")),
        "configuration": int(match.group("configuration")),
        "label": int(fault_type > 0),
    }


def load_imu_axes(
    path: Path,
    segment_seconds: float,
) -> tuple[np.ndarray, int]:
    contents = loadmat(
        path,
        variable_names=["IMUData"],
        simplify_cells=True,
    )

    imu = contents.get("IMUData")

    if not isinstance(imu, dict):
        raise ValueError("IMUData is unavailable")

    axes = np.asarray(
        imu["Vib_Data"],
        dtype=np.float64,
    )

    sampling_rate = int(imu["samplingRate"])

    if axes.ndim != 2:
        raise ValueError(f"Unexpected IMU shape: {axes.shape}")

    if axes.shape[0] != 3 and axes.shape[1] == 3:
        axes = axes.T

    if axes.shape[0] != 3:
        raise ValueError(
            f"Expected three IMU axes, received {axes.shape}"
        )

    required_samples = int(
        round(segment_seconds * sampling_rate)
    )

    if axes.shape[1] < required_samples:
        raise ValueError(
            f"{axes.shape[1]} samples available; "
            f"{required_samples} required"
        )

    start = (axes.shape[1] - required_samples) // 2
    axes = axes[:, start : start + required_samples]

    # Remove static bias and gravity contribution independently per axis.
    axes = axes - np.mean(
        axes,
        axis=1,
        keepdims=True,
    )

    return axes, sampling_rate


def calculate_spectral_features(
    signal: np.ndarray,
    sampling_rate: int,
) -> tuple[float, float]:
    centered = signal - np.mean(signal)

    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(
        len(centered),
        d=1.0 / sampling_rate,
    )

    # Ignore the zero-frequency component.
    power = power[1:]
    frequencies = frequencies[1:]

    total_power = float(np.sum(power))

    if total_power <= np.finfo(float).eps:
        return 0.0, 0.0

    dominant_frequency = float(
        frequencies[int(np.argmax(power))]
    )

    probabilities = power / total_power

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities + np.finfo(float).eps)
    )

    normalized_entropy = float(
        entropy / np.log2(len(probabilities))
    )

    return dominant_frequency, normalized_entropy


def extract_signal_features(
    signal: np.ndarray,
    prefix: str,
    sampling_rate: int,
) -> dict[str, float]:
    dominant_frequency, spectral_entropy = (
        calculate_spectral_features(
            signal,
            sampling_rate,
        )
    )

    return {
        f"{prefix}_rms": float(
            np.sqrt(np.mean(np.square(signal)))
        ),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_peak_to_peak": float(np.ptp(signal)),
        f"{prefix}_kurtosis": float(
            np.nan_to_num(
                kurtosis(
                    signal,
                    fisher=False,
                    bias=False,
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        ),
        f"{prefix}_dominant_frequency": dominant_frequency,
        f"{prefix}_spectral_entropy": spectral_entropy,
    }


def process_recording(
    path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = parse_filename(path)

    permitted_speeds = {
        int(speed)
        for speed in config["experiment"]["held_out_speeds"]
    }

    if int(metadata["speed"]) not in permitted_speeds:
        return []

    axes, sampling_rate = load_imu_axes(
        path,
        float(config["data"]["steady_state_seconds"]),
    )

    magnitude = np.sqrt(
        np.sum(np.square(axes), axis=0)
    )

    signals = {
        "x": axes[0],
        "y": axes[1],
        "z": axes[2],
        "m": magnitude,
    }

    window_size = int(
        round(
            float(config["data"]["window_seconds"])
            * sampling_rate
        )
    )

    overlap = float(
        config["data"]["overlap_fraction"]
    )

    step_size = max(
        1,
        int(round(window_size * (1.0 - overlap))),
    )

    rows: list[dict[str, Any]] = []

    for window_index, start in enumerate(
        range(
            0,
            axes.shape[1] - window_size + 1,
            step_size,
        )
    ):
        end = start + window_size

        row: dict[str, Any] = dict(metadata)
        row["window_index"] = window_index

        for channel, signal in signals.items():
            row.update(
                extract_signal_features(
                    signal[start:end],
                    channel,
                    sampling_rate,
                )
            )

        rows.append(row)

    return rows


def build_feature_dataset(
    config: dict[str, Any],
) -> pd.DataFrame:
    raw_directory = Path(
        config["data"]["raw_directory"]
    )

    files = sorted(raw_directory.glob("*.mat"))

    if not files:
        raise FileNotFoundError(
            f"No MAT files found under {raw_directory}"
        )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    print("Extracting direction-preserving IMU features...")

    for index, path in enumerate(files, start=1):
        print(f"[{index:03}/{len(files):03}] {path.name}")

        try:
            rows.extend(
                process_recording(path, config)
            )
        except ValueError as error:
            skipped.append(
                {
                    "file": path.name,
                    "reason": str(error),
                }
            )

    features = pd.DataFrame(rows)

    if features.empty:
        raise RuntimeError(
            "No usable feature windows were extracted"
        )

    results_directory = Path(
        config["output"]["results_directory"]
    )

    features.to_csv(
        results_directory
        / "conditioned_imu_features.csv",
        index=False,
    )

    pd.DataFrame(skipped).to_csv(
        results_directory
        / "conditioned_skipped_files.csv",
        index=False,
    )

    return features


def create_models(
    config: dict[str, Any],
) -> dict[str, Any]:
    forest = config["experiment"]["random_forest"]
    seed = int(config["project"]["random_seed"])

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=int(
                forest["number_of_trees"]
            ),
            max_depth=int(
                forest["maximum_depth"]
            ),
            min_samples_leaf=int(
                forest["minimum_samples_leaf"]
            ),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "auroc": roc_auc_score(
            labels,
            probabilities,
        ),
    }


def run_conditioned_validation(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "magnitude_6": MAGNITUDE_FEATURES,
        "xyz_magnitude_24": DIRECTION_FEATURES,
    }

    speeds = [
        int(speed)
        for speed in config["experiment"]["held_out_speeds"]
    ]

    prediction_tables: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    forest_importances: list[np.ndarray] = []

    for speed in speeds:
        speed_data = features[
            features["speed"] == speed
        ].copy()

        configurations = sorted(
            speed_data["configuration"].unique()
        )

        for held_out_configuration in configurations:
            training = speed_data[
                speed_data["configuration"]
                != held_out_configuration
            ].copy()

            testing = speed_data[
                speed_data["configuration"]
                == held_out_configuration
            ].copy()

            print(
                f"\nSP{speed}: train on other C-coded "
                f"conditions; test C{held_out_configuration}"
            )

            for feature_set_name, feature_columns in (
                feature_sets.items()
            ):
                for model_name, model in create_models(
                    config
                ).items():
                    model.fit(
                        training[feature_columns],
                        training["label"],
                    )

                    probabilities = model.predict_proba(
                        testing[feature_columns]
                    )[:, 1]

                    predictions = (
                        probabilities >= 0.5
                    ).astype(int)

                    fold_result = calculate_metrics(
                        testing["label"].to_numpy(),
                        predictions,
                        probabilities,
                    )

                    fold_result.update(
                        {
                            "speed": speed,
                            "held_out_configuration": (
                                held_out_configuration
                            ),
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "test_recordings": testing[
                                "recording_id"
                            ].nunique(),
                            "test_windows": len(testing),
                        }
                    )

                    fold_metrics.append(fold_result)

                    prediction_table = testing[
                        [
                            "recording_id",
                            "fault_type",
                            "severity",
                            "speed",
                            "configuration",
                            "label",
                            "window_index",
                        ]
                    ].copy()

                    prediction_table[
                        "held_out_configuration"
                    ] = held_out_configuration

                    prediction_table[
                        "feature_set"
                    ] = feature_set_name

                    prediction_table["model"] = model_name
                    prediction_table[
                        "probability"
                    ] = probabilities

                    prediction_tables.append(
                        prediction_table
                    )

                    if (
                        feature_set_name
                        == "xyz_magnitude_24"
                        and model_name
                        == "random_forest"
                    ):
                        forest_importances.append(
                            model.feature_importances_
                        )

    windows = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    recordings = (
        windows.groupby(
            [
                "feature_set",
                "model",
                "held_out_configuration",
                "recording_id",
                "fault_type",
                "severity",
                "speed",
                "configuration",
                "label",
            ],
            as_index=False,
        )
        .agg(
            probability=("probability", "mean"),
            number_of_windows=("window_index", "count"),
        )
    )

    recordings["prediction"] = (
        recordings["probability"] >= 0.5
    ).astype(int)

    overall_metrics: list[dict[str, Any]] = []

    for (feature_set, model), group in recordings.groupby(
        ["feature_set", "model"]
    ):
        metrics = calculate_metrics(
            group["label"].to_numpy(),
            group["prediction"].to_numpy(),
            group["probability"].to_numpy(),
        )

        metrics.update(
            {
                "feature_set": feature_set,
                "model": model,
                "recordings": len(group),
            }
        )

        overall_metrics.append(metrics)

    if forest_importances:
        importance_table = pd.DataFrame(
            {
                "feature": DIRECTION_FEATURES,
                "importance": np.mean(
                    np.vstack(forest_importances),
                    axis=0,
                ),
            }
        ).sort_values(
            "importance",
            ascending=False,
        )
    else:
        importance_table = pd.DataFrame(
            columns=["feature", "importance"]
        )

    return (
        recordings,
        pd.DataFrame(fold_metrics),
        pd.DataFrame(overall_metrics),
        importance_table,
    )


def create_figures(
    recordings: pd.DataFrame,
    importance_table: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    figure_directory = Path(
        config["output"]["figures_directory"]
    )

    best = recordings[
        (
            recordings["feature_set"]
            == "xyz_magnitude_24"
        )
        & (
            recordings["model"]
            == "random_forest"
        )
    ]

    figure, axis = plt.subplots(
        figsize=(5.2, 4.5)
    )

    ConfusionMatrixDisplay.from_predictions(
        best["label"],
        best["prediction"],
        display_labels=["Healthy", "Faulty"],
        values_format="d",
        ax=axis,
    )

    axis.set_title(
        "Speed-conditioned recording-level evaluation"
    )

    figure.tight_layout()
    figure.savefig(
        figure_directory
        / "conditioned_random_forest_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    if not importance_table.empty:
        top_features = (
            importance_table.head(12)
            .sort_values("importance")
        )

        figure, axis = plt.subplots(
            figsize=(7.2, 5.0)
        )

        axis.barh(
            top_features["feature"],
            top_features["importance"],
        )

        axis.set_xlabel(
            "Mean random-forest feature importance"
        )

        axis.set_ylabel("IMU vibration feature")

        axis.set_title(
            "Direction-preserving feature importance"
        )

        figure.tight_layout()
        figure.savefig(
            figure_directory
            / "conditioned_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


def run(config_path: Path) -> None:
    config = load_config(config_path)

    results_directory = Path(
        config["output"]["results_directory"]
    )

    figure_directory = Path(
        config["output"]["figures_directory"]
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = build_feature_dataset(config)

    recordings, fold_metrics, metrics, importance = (
        run_conditioned_validation(
            features,
            config,
        )
    )

    recordings.to_csv(
        results_directory
        / "conditioned_recording_predictions.csv",
        index=False,
    )

    fold_metrics.to_csv(
        results_directory
        / "conditioned_fold_metrics.csv",
        index=False,
    )

    metrics.to_csv(
        results_directory
        / "conditioned_recording_metrics.csv",
        index=False,
    )

    importance.to_csv(
        results_directory
        / "conditioned_feature_importance.csv",
        index=False,
    )

    create_figures(
        recordings,
        importance,
        config,
    )

    metadata = {
        "usable_recordings": int(
            features["recording_id"].nunique()
        ),
        "feature_windows": int(len(features)),
        "healthy_recordings": int(
            features.loc[
                features["label"] == 0,
                "recording_id",
            ].nunique()
        ),
        "faulty_recordings": int(
            features.loc[
                features["label"] == 1,
                "recording_id",
            ].nunique()
        ),
        "segment_seconds": float(
            config["data"]["steady_state_seconds"]
        ),
        "validation": (
            "speed-conditioned C-coded-condition holdout"
        ),
        "feature_sets": {
            "magnitude_6": MAGNITUDE_FEATURES,
            "xyz_magnitude_24": DIRECTION_FEATURES,
        },
    }

    with (
        results_directory
        / "conditioned_experiment_metadata.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print("\nSPEED-CONDITIONED CONFIGURATION-HOLDOUT RESULTS")
    print("=" * 100)

    print(
        metrics.sort_values(
            ["macro_f1", "balanced_accuracy"],
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nPER-SPEED BEST-MODEL RESULTS")
    print("=" * 100)

    best = recordings[
        (
            recordings["feature_set"]
            == "xyz_magnitude_24"
        )
        & (
            recordings["model"]
            == "random_forest"
        )
    ]

    for speed, group in best.groupby("speed"):
        speed_metrics = calculate_metrics(
            group["label"].to_numpy(),
            group["prediction"].to_numpy(),
            group["probability"].to_numpy(),
        )

        print(
            f"SP{speed}: "
            f"recordings={len(group)}, "
            f"balanced_accuracy="
            f"{speed_metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={speed_metrics['macro_f1']:.4f}, "
            f"auroc={speed_metrics['auroc']:.4f}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Speed-conditioned drone IMU "
            "propeller-fault detection"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run(arguments.config)


if __name__ == "__main__":
    main()
