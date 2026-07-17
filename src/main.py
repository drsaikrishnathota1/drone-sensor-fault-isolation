#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis


AXIS_CANDIDATES = {
    "x": ["ax", "acc_x", "accel_x", "acceleration_x", "accelerometer_x"],
    "y": ["ay", "acc_y", "accel_y", "acceleration_y", "accelerometer_y"],
    "z": ["az", "acc_z", "accel_z", "acceleration_z", "accelerometer_z"],
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def detect_axis_columns(dataframe: pd.DataFrame) -> dict[str, str]:
    normalized = {
        normalize_column_name(str(column)): str(column)
        for column in dataframe.columns
    }

    detected: dict[str, str] = {}

    for axis, candidates in AXIS_CANDIDATES.items():
        for candidate in candidates:
            if candidate in normalized:
                detected[axis] = normalized[candidate]
                break

    if len(detected) != 3:
        raise ValueError(
            "Could not detect all three accelerometer axes. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return detected


def calculate_spectral_features(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[float, float]:
    centered = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(
        len(centered),
        d=1.0 / sampling_rate_hz,
    )

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

    spectral_entropy = float(
        entropy / np.log2(len(probabilities))
    )

    return dominant_frequency, spectral_entropy


def extract_features(
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> dict[str, float]:
    dominant_frequency, spectral_entropy = (
        calculate_spectral_features(signal, sampling_rate_hz)
    )

    return {
        "rms": float(np.sqrt(np.mean(np.square(signal)))),
        "standard_deviation": float(np.std(signal)),
        "peak_to_peak": float(np.ptp(signal)),
        "kurtosis": float(
            np.nan_to_num(
                kurtosis(signal, fisher=False, bias=False),
                nan=0.0,
            )
        ),
        "dominant_frequency_hz": dominant_frequency,
        "spectral_entropy": spectral_entropy,
    }


def inspect_dataset(raw_directory: Path) -> None:
    files = sorted(raw_directory.rglob("*.csv"))

    print(f"CSV files found: {len(files)}")

    if not files:
        print(f"Add the public dataset under: {raw_directory.resolve()}")
        return

    for file_path in files[:10]:
        dataframe = pd.read_csv(file_path, nrows=5)

        print("\nFile:", file_path)
        print("Columns:", list(dataframe.columns))
        print("Preview shape:", dataframe.shape)


def process_recording(
    file_path: Path,
    sampling_rate_hz: float,
    window_seconds: float,
    overlap_fraction: float,
) -> list[dict]:
    dataframe = pd.read_csv(file_path)
    axis_columns = detect_axis_columns(dataframe)

    acceleration = dataframe[
        [
            axis_columns["x"],
            axis_columns["y"],
            axis_columns["z"],
        ]
    ].apply(pd.to_numeric, errors="coerce")

    acceleration = acceleration.dropna()

    magnitude = np.sqrt(
        np.square(acceleration.iloc[:, 0].to_numpy())
        + np.square(acceleration.iloc[:, 1].to_numpy())
        + np.square(acceleration.iloc[:, 2].to_numpy())
    )

    window_size = int(sampling_rate_hz * window_seconds)
    step_size = max(
        1,
        int(window_size * (1.0 - overlap_fraction)),
    )

    rows: list[dict] = []

    for window_index, start in enumerate(
        range(0, len(magnitude) - window_size + 1, step_size)
    ):
        window = magnitude[start : start + window_size]
        features = extract_features(window, sampling_rate_hz)

        features["recording_id"] = file_path.stem
        features["source_file"] = str(file_path)
        features["window_index"] = window_index

        rows.append(features)

    return rows


def run_feature_extraction(config: dict) -> None:
    raw_directory = Path(config["data"]["raw_directory"])
    output_directory = Path(config["output"]["directory"])

    output_directory.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_directory.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_directory.resolve()}"
        )

    all_features: list[dict] = []

    for file_path in files:
        print(f"Processing: {file_path}")

        rows = process_recording(
            file_path=file_path,
            sampling_rate_hz=float(
                config["data"]["sampling_rate_hz"]
            ),
            window_seconds=float(
                config["data"]["window_seconds"]
            ),
            overlap_fraction=float(
                config["data"]["overlap_fraction"]
            ),
        )

        all_features.extend(rows)

    feature_table = pd.DataFrame(all_features)
    output_file = output_directory / "imu_features.csv"
    feature_table.to_csv(output_file, index=False)

    metadata = {
        "number_of_recordings": len(files),
        "number_of_windows": len(feature_table),
        "feature_file": str(output_file),
    }

    with (
        output_directory / "feature_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nFeature extraction completed.")
    print(f"Recordings: {len(files)}")
    print(f"Windows: {len(feature_table)}")
    print(f"Saved to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone IMU propeller-fault experiment"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    if arguments.inspect:
        inspect_dataset(Path(config["data"]["raw_directory"]))
    else:
        run_feature_extraction(config)


if __name__ == "__main__":
    main()
