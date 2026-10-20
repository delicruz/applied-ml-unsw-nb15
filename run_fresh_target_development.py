"""Run the preregistered NF-CSE-CIC-IDS2018 target-development study.

The official CSV is streamed once to verify aggregate counts and construct
deterministic, five-significant-digit feature groups. Only the training,
calibration, and validation samples are retained. Rows assigned to the sealed
confirmation partition are counted and included in an order-bound digest, then
discarded without prediction or metric calculation.

Models are fitted only on target training, thresholds are selected only on
target calibration, and fixed candidates are compared only on target
validation. NF-ToN-IoT and NF-BoT-IoT are not read by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from run_external_netflow_validation import (
    calculate_metrics,
    per_attack_recall,
    select_threshold,
    sha256_file,
)


PROTOCOL_PATH = Path("protocols/fresh_target_development_protocol.json")
THIRD_PROTOCOL_PATH = Path("protocols/third_dataset_preregistration.json")
DATASET_GLOB = "data/external/**/NF-CSE-CIC-IDS2018.csv"
OUTPUT_DIR = Path("artifacts/fresh_target_development")
SEED = 42
CHUNK_SIZE = 250_000

ENDPOINT_FEATURES = (
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
)
CATEGORICAL_FEATURES = ("PROTOCOL", "L7_PROTO")
NUMERIC_FEATURES = (
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REDUCED_FEATURES = ("PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS")
TARGET = "Label"
ATTACK = "Attack"
REQUIRED_COLUMNS = set(ENDPOINT_FEATURES + MODEL_FEATURES + (TARGET, ATTACK))

MAX_U64 = (1 << 64) - 1
PARTITION_CUTOFFS = {
    "target_training": int(0.48 * MAX_U64),
    "target_calibration": int(0.64 * MAX_U64),
    "target_validation": int(0.80 * MAX_U64),
}
EXPECTED_ROWS = 8_392_401
SAMPLING_PROBABILITIES = {
    "target_training": min(1.0, 1_000_000 / (EXPECTED_ROWS * 0.48)),
    "target_calibration": min(1.0, 500_000 / (EXPECTED_ROWS * 0.16)),
    "target_validation": min(1.0, 500_000 / (EXPECTED_ROWS * 0.16)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def resolve_dataset_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted(Path.cwd().glob(DATASET_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one match for {DATASET_GLOB!r}, found {len(matches)}"
        )
    return matches[0].resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def round_to_significant(values: np.ndarray, digits: int = 5) -> np.ndarray:
    """Vectorised significant-digit rounding with stable zero/NaN handling."""
    array = np.asarray(values, dtype="float64")
    result = array.copy()
    finite_nonzero = np.isfinite(array) & (array != 0)
    magnitudes = np.zeros_like(array)
    magnitudes[finite_nonzero] = np.floor(
        np.log10(np.abs(array[finite_nonzero]))
    )
    steps = np.ones_like(array)
    steps[finite_nonzero] = np.power(10.0, magnitudes[finite_nonzero] - digits + 1)
    result[finite_nonzero] = (
        np.rint(array[finite_nonzero] / steps[finite_nonzero])
        * steps[finite_nonzero]
    )
    result[result == 0] = 0.0
    return result


def canonical_group_hashes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the first two unsigned 64-bit words of each group SHA-256."""
    matrix = np.empty((len(frame), len(MODEL_FEATURES)), dtype="<f8")
    for column_index, feature in enumerate(MODEL_FEATURES):
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
            dtype="float64"
        )
        if feature in NUMERIC_FEATURES:
            values = round_to_significant(values, digits=5)
        matrix[:, column_index] = values
    # Canonicalise every NaN to one quiet-NaN bit pattern before hashing.
    matrix[np.isnan(matrix)] = np.nan
    first = np.empty(len(frame), dtype="uint64")
    second = np.empty(len(frame), dtype="uint64")
    for row_index, row in enumerate(matrix):
        digest = hashlib.sha256(row.tobytes(order="C")).digest()
        first[row_index] = int.from_bytes(digest[:8], "big")
        second[row_index] = int.from_bytes(digest[8:16], "big")
    return first, second


def assign_partitions(first_hash_word: np.ndarray) -> np.ndarray:
    values = np.asarray(first_hash_word, dtype="uint64")
    partitions = np.full(len(values), "sealed_confirmation", dtype=object)
    partitions[values <= PARTITION_CUTOFFS["target_validation"]] = (
        "target_validation"
    )
    partitions[values <= PARTITION_CUTOFFS["target_calibration"]] = (
        "target_calibration"
    )
    partitions[values <= PARTITION_CUTOFFS["target_training"]] = "target_training"
    return partitions


def group_sample_mask(partition: str, second_hash_word: np.ndarray) -> np.ndarray:
    probability = SAMPLING_PROBABILITIES[partition]
    cutoff = int(probability * MAX_U64)
    return np.asarray(second_hash_word, dtype="uint64") <= cutoff


def validate_header(path: Path) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def verify_official_counts(
    total_rows: int,
    label_counts: dict[int, int],
    attack_counts: dict[str, int],
    protocol: dict[str, Any],
) -> None:
    expected = protocol["dataset"]
    observed_attacks = int(label_counts.get(1, 0))
    observed_benign = int(label_counts.get(0, 0))
    checks = {
        "rows": (total_rows, int(expected["expected_rows"])),
        "positive_rows": (observed_attacks, int(expected["expected_positive_rows"])),
        "negative_rows": (observed_benign, int(expected["expected_negative_rows"])),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected_value}
        for key, (observed, expected_value) in checks.items()
        if observed != expected_value
    }
    expected_attacks = {
        str(key): int(value) for key, value in expected["expected_attack_counts"].items()
    }
    normalised_attacks = normalise_attack_counts(attack_counts)
    if normalised_attacks != expected_attacks:
        mismatches["attack_counts"] = {
            "observed": normalised_attacks,
            "expected": expected_attacks,
        }
    if mismatches:
        raise ValueError(f"Official count verification failed: {mismatches}")


def normalise_attack_counts(raw_counts: dict[str, int]) -> dict[str, int]:
    """Map raw CIC-IDS2018 attack labels to the official parent categories."""
    parent_by_raw = {
        "Benign": "Benign",
        "Bot": "Bot",
        "Infilteration": "Infiltration",
        "FTP-BruteForce": "BruteForce",
        "SSH-Bruteforce": "BruteForce",
        "Brute Force -Web": "Web Attacks",
        "Brute Force -XSS": "Web Attacks",
        "SQL Injection": "Web Attacks",
        "DoS attacks-Hulk": "DoS",
        "DoS attacks-SlowHTTPTest": "DoS",
        "DoS attacks-GoldenEye": "DoS",
        "DoS attacks-Slowloris": "DoS",
        "DDOS attack-HOIC": "DDoS",
        "DDoS attacks-LOIC-HTTP": "DDoS",
        "DDOS attack-LOIC-UDP": "DDoS",
    }
    unknown = sorted(set(raw_counts) - set(parent_by_raw))
    if unknown:
        raise ValueError(f"Unknown raw attack labels: {unknown}")
    normalised: dict[str, int] = {}
    for raw_label, count in raw_counts.items():
        parent = parent_by_raw[raw_label]
        normalised[parent] = normalised.get(parent, 0) + int(count)
    return dict(sorted(normalised.items()))


def construct_development_partitions(
    dataset_path: Path,
    protocol: dict[str, Any],
    chunk_size: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Stream the dataset and retain only sampled development partitions."""
    validate_header(dataset_path)
    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    retained: dict[str, list[pd.DataFrame]] = {
        "target_training": [],
        "target_calibration": [],
        "target_validation": [],
    }
    partition_rows = {name: 0 for name in (*retained, "sealed_confirmation")}
    label_counts = {0: 0, 1: 0}
    attack_counts: dict[str, int] = {}
    confirmation_digest = hashlib.sha256()
    total_rows = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(dataset_path, usecols=usecols, chunksize=chunk_size), start=1
    ):
        for feature in MODEL_FEATURES:
            chunk[feature] = pd.to_numeric(chunk[feature], errors="coerce")
        chunk[TARGET] = pd.to_numeric(chunk[TARGET], errors="raise").astype("int8")
        if not set(chunk[TARGET].unique()).issubset({0, 1}):
            raise ValueError("Dataset contains a non-binary label")
        chunk[ATTACK] = chunk[ATTACK].fillna("<MISSING>").astype(str)

        total_rows += len(chunk)
        for label, count in chunk[TARGET].value_counts().items():
            label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)
        for attack, count in chunk[ATTACK].value_counts().items():
            attack_counts[str(attack)] = attack_counts.get(str(attack), 0) + int(count)

        first, second = canonical_group_hashes(chunk)
        assignments = assign_partitions(first)
        for partition in partition_rows:
            partition_mask = assignments == partition
            partition_rows[partition] += int(partition_mask.sum())
            if partition == "sealed_confirmation":
                for first_word, second_word in zip(
                    first[partition_mask], second[partition_mask]
                ):
                    confirmation_digest.update(int(first_word).to_bytes(8, "big"))
                    confirmation_digest.update(int(second_word).to_bytes(8, "big"))
                continue
            keep = partition_mask & group_sample_mask(partition, second)
            if keep.any():
                selected = chunk.loc[keep, usecols].copy()
                selected["_group_key"] = [
                    f"{int(a):016x}{int(b):016x}"
                    for a, b in zip(first[keep], second[keep])
                ]
                retained[partition].append(selected)
        print(
            f"partition_progress chunk={chunk_index} rows={total_rows}", flush=True
        )

    verify_official_counts(total_rows, label_counts, attack_counts, protocol)
    partitions = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in retained.items()
    }
    group_sets = {
        name: set(frame["_group_key"].unique()) for name, frame in partitions.items()
    }
    overlaps = {
        "training_calibration": len(
            group_sets["target_training"] & group_sets["target_calibration"]
        ),
        "training_validation": len(
            group_sets["target_training"] & group_sets["target_validation"]
        ),
        "calibration_validation": len(
            group_sets["target_calibration"] & group_sets["target_validation"]
        ),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Grouped development partitions overlap: {overlaps}")
    integrity = {
        "official_counts_verified": True,
        "total_rows": total_rows,
        "global_label_counts": {str(key): value for key, value in label_counts.items()},
        "global_raw_attack_counts": attack_counts,
        "global_official_attack_counts": normalise_attack_counts(attack_counts),
        "partition_rows_before_sampling": partition_rows,
        "development_samples": {
            name: {
                "rows": int(len(frame)),
                "groups": int(frame["_group_key"].nunique()),
                "suspicious": int(frame[TARGET].sum()),
                "prevalence": float(frame[TARGET].mean()),
                "sampling_probability": SAMPLING_PROBABILITIES[name],
            }
            for name, frame in partitions.items()
        },
        "development_group_overlap": overlaps,
        "sealed_confirmation": {
            "rows": partition_rows["sealed_confirmation"],
            "ordered_group_digest_sha256": confirmation_digest.hexdigest(),
            "predictions_computed": False,
            "metrics_computed": False,
            "labels_summarised_by_partition": False,
        },
    }
    return partitions, integrity


def log1p_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0))


def make_preprocessor(features: tuple[str, ...], scale: bool) -> ColumnTransformer:
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        (
            "log1p",
            FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one"),
        ),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_target_model(model: str, features: tuple[str, ...]) -> Pipeline:
    if model == "logistic_regression":
        estimator: Any = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            solver="saga",
            tol=1e-3,
        )
        scale = True
    elif model == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
            oob_score=True,
            bootstrap=True,
        )
        scale = False
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline(
        [("preprocess", make_preprocessor(features, scale=scale)), ("model", estimator)]
    )


def metric_row(
    candidate: str,
    scope: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    result = calculate_metrics(labels, probabilities, threshold)
    result["false_positive_rate"] = 1.0 - float(result["specificity"])
    return {"candidate": candidate, "scope": scope, **result}


def load_frozen_models() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(THIRD_PROTOCOL_PATH)
    loaded = {}
    manifest = {}
    for name, specification in protocol["frozen_inputs"]["models"].items():
        path = Path(specification["artifact"])
        observed_hash = sha256_file(path)
        if observed_hash != specification["sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
        loaded[name] = joblib.load(path)
        manifest[name] = {
            "path": str(path),
            "sha256": observed_hash,
            "source_threshold": float(specification["threshold"]),
        }
    return loaded, manifest


def evaluate_candidates(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit and compare candidates; confirmation data is intentionally absent."""
    frozen, frozen_manifest = load_frozen_models()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    attacks: list[pd.DataFrame] = []
    threshold_manifest: dict[str, Any] = {}
    fitted_manifest: dict[str, Any] = {}

    X_cal = calibration.loc[:, MODEL_FEATURES]
    y_cal = calibration[TARGET].to_numpy()
    X_val = validation.loc[:, MODEL_FEATURES]
    y_val = validation[TARGET].to_numpy()

    frozen_probabilities = {
        name: model.predict_proba(X_val)[:, 1] for name, model in frozen.items()
    }
    for name, threshold in (
        ("frozen_source_lr", frozen_manifest["logistic_regression"]["source_threshold"]),
        ("frozen_source_rf", frozen_manifest["random_forest"]["source_threshold"]),
    ):
        model_name = name.removeprefix("frozen_source_")
        lookup = "logistic_regression" if model_name == "lr" else "random_forest"
        probability = frozen_probabilities[lookup]
        metrics.append(metric_row(name, "target_validation", y_val, probability, threshold))
        attacks.append(
            per_attack_recall(
                "target_validation",
                validation[ATTACK],
                y_val,
                name,
                probability,
                threshold,
            )
        )

    frozen_rf_cal = frozen["random_forest"].predict_proba(X_cal)[:, 1]
    recalibrated_threshold, threshold_table = select_threshold(y_cal, frozen_rf_cal)
    threshold_table.to_csv(
        output_dir / "frozen_source_rf_recalibrated_threshold_grid.csv", index=False
    )
    recalibrated_name = "frozen_source_rf_recalibrated_threshold"
    metrics.append(
        metric_row(
            recalibrated_name,
            "target_validation",
            y_val,
            frozen_probabilities["random_forest"],
            recalibrated_threshold,
        )
    )
    attacks.append(
        per_attack_recall(
            "target_validation",
            validation[ATTACK],
            y_val,
            recalibrated_name,
            frozen_probabilities["random_forest"],
            recalibrated_threshold,
        )
    )
    threshold_manifest[recalibrated_name] = {
        "threshold": recalibrated_threshold,
        "selected_on": "target_calibration",
    }

    target_specs = {
        "target_lr_all_features": ("logistic_regression", MODEL_FEATURES),
        "target_rf_all_features": ("random_forest", MODEL_FEATURES),
        "target_rf_reduced_features": ("random_forest", REDUCED_FEATURES),
    }
    for candidate, (model_kind, features) in target_specs.items():
        model = make_target_model(model_kind, features)
        start = time.perf_counter()
        model.fit(training.loc[:, features], training[TARGET])
        fit_seconds = time.perf_counter() - start
        calibration_probability = model.predict_proba(calibration.loc[:, features])[
            :, 1
        ]
        threshold, table = select_threshold(y_cal, calibration_probability)
        table.to_csv(output_dir / f"{candidate}_threshold_grid.csv", index=False)
        train_probability = model.predict_proba(training.loc[:, features])[:, 1]
        validation_probability = model.predict_proba(validation.loc[:, features])[:, 1]
        metrics.append(
            metric_row(
                candidate,
                "target_training",
                training[TARGET].to_numpy(),
                train_probability,
                threshold,
            )
        )
        metrics.append(
            metric_row(
                candidate,
                "target_validation",
                y_val,
                validation_probability,
                threshold,
            )
        )
        attacks.append(
            per_attack_recall(
                "target_validation",
                validation[ATTACK],
                y_val,
                candidate,
                validation_probability,
                threshold,
            )
        )
        model_path = model_dir / f"{candidate}.joblib"
        joblib.dump(model, model_path)
        fitted_model = model.named_steps["model"]
        fitted_manifest[candidate] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "features": list(features),
            "fit_seconds": fit_seconds,
            "oob_score": (
                float(fitted_model.oob_score_)
                if hasattr(fitted_model, "oob_score_")
                else None
            ),
        }
        threshold_manifest[candidate] = {
            "threshold": threshold,
            "selected_on": "target_calibration",
        }
        print(f"candidate_complete name={candidate} seconds={fit_seconds:.2f}", flush=True)

    metrics_frame = pd.DataFrame(metrics)
    attacks_frame = pd.concat(attacks, ignore_index=True)
    validation_metrics = metrics_frame[
        metrics_frame["scope"].eq("target_validation")
    ].copy()
    baseline = validation_metrics[
        validation_metrics["candidate"].eq("frozen_source_rf")
    ].iloc[0]
    validation_metrics["mcc_improvement_over_frozen_rf"] = (
        validation_metrics["mcc"] - baseline["mcc"]
    )
    validation_metrics["balanced_accuracy_improvement_over_frozen_rf"] = (
        validation_metrics["balanced_accuracy"] - baseline["balanced_accuracy"]
    )
    validation_metrics["development_promising"] = (
        validation_metrics["mcc_improvement_over_frozen_rf"].ge(0.05)
        & validation_metrics["balanced_accuracy_improvement_over_frozen_rf"].ge(0.05)
    )

    gap_rows = []
    for candidate in target_specs:
        indexed = metrics_frame[metrics_frame["candidate"].eq(candidate)].set_index(
            "scope"
        )
        row: dict[str, Any] = {"candidate": candidate}
        for metric in (
            "average_precision",
            "f1",
            "mcc",
            "balanced_accuracy",
            "false_positive_rate",
        ):
            row[f"training_{metric}"] = float(indexed.loc["target_training", metric])
            row[f"validation_{metric}"] = float(
                indexed.loc["target_validation", metric]
            )
            row[f"training_minus_validation_{metric}"] = (
                row[f"training_{metric}"] - row[f"validation_{metric}"]
            )
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)
    manifest = {
        "frozen_models": frozen_manifest,
        "fitted_models": fitted_manifest,
        "thresholds": threshold_manifest,
    }
    return metrics_frame, validation_metrics, gaps, {
        "attacks": attacks_frame,
        "manifest": manifest,
    }


def save_plots(validation: pd.DataFrame, gaps: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered = validation.sort_values("mcc", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    axes[0].barh(ordered["candidate"], ordered["mcc"], color="#4c78a8")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Fresh target-development validation MCC")
    axes[0].set_xlabel("Matthews correlation coefficient")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(
        ordered["candidate"], ordered["balanced_accuracy"], color="#59a14f"
    )
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Fresh target-development balanced accuracy")
    axes[1].set_xlabel("Balanced accuracy")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(plot_dir / "candidate_validation_comparison.png", dpi=180)
    plt.close(fig)

    gap_columns = [
        "training_minus_validation_average_precision",
        "training_minus_validation_mcc",
        "training_minus_validation_balanced_accuracy",
    ]
    labels = ["Average Precision", "MCC", "Balanced accuracy"]
    x = np.arange(len(gaps))
    width = 0.25
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, (column, label) in enumerate(zip(gap_columns, labels)):
        axis.bar(x + (index - 1) * width, gaps[column], width, label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(gaps["candidate"], rotation=20, ha="right")
    axis.set_ylabel("Training minus validation")
    axis.set_title("Target-development overfitting gaps")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(plot_dir / "target_development_overfitting_gaps.png", dpi=180)
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if args.chunk_size < 10_000:
        raise ValueError("--chunk-size must be at least 10,000")
    dataset_path = resolve_dataset_path(args.dataset_path)
    protocol = read_json(args.protocol)
    if protocol["status"] != "REGISTERED_BEFORE_DATA_ACCESS":
        raise ValueError("Fresh target-development protocol has unexpected status")
    if tuple(protocol["features"]["model_features"]) != MODEL_FEATURES:
        raise ValueError("Protocol model features do not match the runner")
    if tuple(protocol["features"]["reduced_candidate_features"]) != REDUCED_FEATURES:
        raise ValueError("Protocol reduced features do not match the runner")

    start = time.perf_counter()
    partitions, integrity = construct_development_partitions(
        dataset_path, protocol, args.chunk_size
    )
    metrics, validation, gaps, extras = evaluate_candidates(
        partitions["target_training"],
        partitions["target_calibration"],
        partitions["target_validation"],
        args.output_dir,
    )
    attacks = extras["attacks"]
    model_manifest = extras["manifest"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
    validation.to_csv(args.output_dir / "validation_candidate_comparison.csv", index=False)
    gaps.to_csv(args.output_dir / "overfitting_gaps.csv", index=False)
    attacks.to_csv(args.output_dir / "validation_per_attack_recall.csv", index=False)
    save_plots(validation, gaps, args.output_dir)

    ranked = validation.sort_values(
        ["mcc", "balanced_accuracy", "average_precision"], ascending=False
    )
    best = ranked.iloc[0]
    result = json_safe(
        {
            "material_passport": {
                "origin_skill": "experiment-agent",
                "origin_mode": "run",
                "origin_date": "2026-10-20",
                "verification_status": "UNVERIFIED",
                "version_label": "target_dev_result_v1",
            },
            "experiment": {
                "protocol_id": protocol["protocol_id"],
                "status": "completed",
                "development_only": True,
                "confirmation_evaluated": False,
                "duration_seconds": time.perf_counter() - start,
            },
            "data_provenance": {
                "dataset": "NF-CSE-CIC-IDS2018",
                "path": str(dataset_path),
                "csv_sha256": sha256_file(dataset_path),
                "protocol_path": str(args.protocol),
                "protocol_sha256": sha256_file(args.protocol),
            },
            "integrity": integrity,
            "models": model_manifest,
            "best_development_candidate": {
                "candidate": best["candidate"],
                "mcc": best["mcc"],
                "balanced_accuracy": best["balanced_accuracy"],
                "average_precision": best["average_precision"],
                "development_promising": bool(best["development_promising"]),
            },
            "candidate_validation_records": validation.to_dict(orient="records"),
            "interpretation_boundary": (
                "Candidate ranking uses only the registered target-development "
                "partitions. The sealed confirmation partition was not scored, so "
                "no candidate is externally confirmed and no deployment claim is "
                "permitted."
            ),
            "environment": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
