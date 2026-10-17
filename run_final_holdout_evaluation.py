"""Run the frozen final evaluation on the supplied UNSW-NB15 test set.

The final model removes ``sttl``, ``dttl`` and ``ct_state_ttl`` as a conservative
response to the shortcut-feature concern. Its Random Forest structure and 0.54
threshold were fixed before this script opens the test file. The model is
refitted on all supplied training records and is not changed after test results
are calculated.

An older development workflow evaluated a deterministic 5,000-row stratified
sample of the supplied test data. This script reconstructs that sample and
reports it transparently. The primary conservative scope excludes both those
historically evaluated rows and test records whose coarsened final-feature
fingerprints overlap the supplied training data.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from run_preliminary_validation import (
    CATEGORICAL,
    DEVELOPMENT_PATH,
    SEED,
    TARGET,
    VALIDATION_PATH,
    make_models,
    split_xy,
)
from run_robustness_audit import round_significant


TEST_PATH = Path("data/raw/UNSW_NB15_testing-set.csv")
ROBUSTNESS_METRICS_PATH = Path("artifacts/robustness_audit/robustness_metrics.csv")
OUTPUT_DIR = Path("artifacts/final_holdout_evaluation")
REMOVED_FEATURES = ("sttl", "dttl", "ct_state_ttl")
NEAR_DUPLICATE_SIGNIFICANT_DIGITS = 5
HISTORICAL_TEST_SAMPLE_SIZE = 5_000
EXPECTED_THRESHOLD = 0.54
INTERVAL_METRICS = ("average_precision", "f1", "mcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_frozen_threshold() -> float:
    if not ROBUSTNESS_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Frozen robustness results not found: {ROBUSTNESS_METRICS_PATH}"
        )
    metrics = pd.read_csv(ROBUSTNESS_METRICS_PATH)
    selected = metrics[
        (metrics["evaluation_scope"] == "current_validation")
        & (metrics["variant"] == "without_all_ttl")
    ]
    if len(selected) != 1:
        raise ValueError("Could not resolve one frozen no-TTL threshold")
    threshold = float(selected.iloc[0]["threshold"])
    if not np.isclose(threshold, EXPECTED_THRESHOLD):
        raise ValueError(
            f"Frozen threshold changed from {EXPECTED_THRESHOLD:.2f} to {threshold:.2f}"
        )
    return threshold


def stratified_limit(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Reproduce the historical sampler in run_project.py exactly."""
    fractions = frame[TARGET].value_counts(normalize=True)
    parts = []
    for label, fraction in fractions.items():
        rows = max(1, int(round(limit * fraction)))
        group = frame[frame[TARGET] == label]
        parts.append(group.sample(n=min(rows, len(group)), random_state=SEED))
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def coarsened_hashes(
    frame: pd.DataFrame, columns: list[str], digits: int
) -> pd.Series:
    coarsened = frame[columns].copy()
    for column in columns:
        if column not in CATEGORICAL:
            coarsened[column] = round_significant(coarsened[column], digits)
    return pd.util.hash_pandas_object(coarsened, index=False).astype("uint64")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_point_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return (
        {
            "threshold": threshold,
            "average_precision": float(
                average_precision_score(y_true, probabilities)
            ),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(
                precision_score(y_true, predictions, zero_division=0)
            ),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, predictions)),
            "false_positive_rate": float(fp / (fp + tn)),
            "brier_score": float(brier_score_loss(y_true, probabilities)),
            "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        predictions,
    )


def weighted_interval_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "average_precision": float(
            average_precision_score(
                y_true, probabilities, sample_weight=sample_weight
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
                sample_weight=sample_weight,
            )
        ),
        "mcc": float(
            matthews_corrcoef(y_true, predictions, sample_weight=sample_weight)
        ),
    }


def cluster_bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    codes, unique_groups = pd.factorize(groups, sort=False)
    group_count = len(unique_groups)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in INTERVAL_METRICS}
    completed = 0
    while completed < iterations:
        sampled_groups = rng.integers(0, group_count, group_count)
        group_weights = np.bincount(sampled_groups, minlength=group_count)
        weights = group_weights[codes]
        if np.unique(y_true[weights > 0]).size < 2:
            continue
        values = weighted_interval_metrics(
            y_true, probabilities, threshold, weights
        )
        for metric in INTERVAL_METRICS:
            samples[metric].append(values[metric])
        completed += 1
    rows = []
    for metric, values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "metric": metric,
                "ci_95_low": float(np.quantile(array, 0.025)),
                "ci_95_high": float(np.quantile(array, 0.975)),
                "bootstrap_mean": float(array.mean()),
                "iterations": len(array),
                "resampling_unit": (
                    "five-significant-digit near-duplicate final-feature group"
                ),
            }
        )
    return rows


def per_attack_recall(
    frame: pd.DataFrame, predictions: np.ndarray, scope: str
) -> pd.DataFrame:
    evaluated = frame[["attack_cat", TARGET]].copy()
    evaluated["prediction"] = predictions
    attacks = evaluated[evaluated[TARGET] == 1]
    result = (
        attacks.groupby("attack_cat")
        .agg(support=(TARGET, "size"), detected=("prediction", "sum"))
        .reset_index()
    )
    result["recall"] = result["detected"] / result["support"]
    result.insert(0, "scope", scope)
    return result


def save_plots(
    scope_metrics: pd.DataFrame,
    per_attack: pd.DataFrame,
    primary_y: np.ndarray,
    primary_probabilities: np.ndarray,
    primary_predictions: np.ndarray,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scope_order = [
        "official_test_all",
        "historically_unexamined_remainder",
        "near_duplicate_clean_test",
        "conservative_primary",
    ]
    scope_labels = {
        "official_test_all": "Official test\nall rows",
        "historically_unexamined_remainder": "Historically unexamined\nremainder",
        "near_duplicate_clean_test": "Near-duplicate-clean\ntest",
        "conservative_primary": "Conservative\nprimary",
    }
    plot_data = (
        scope_metrics.set_index("scope")
        .reindex(scope_order)[["average_precision", "f1", "mcc"]]
        .rename(columns={"average_precision": "AP", "f1": "F1", "mcc": "MCC"})
    )
    plot_data.index = [scope_labels[index] for index in plot_data.index]
    ax = plot_data.plot.bar(
        figsize=(10.0, 5.6), color=["#4C78A8", "#F58518", "#54A24B"]
    )
    ax.set_title("Frozen Random Forest final holdout performance")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(max(0.0, float(plot_data.min().min()) - 0.05), 1.0)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "holdout_scope_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    matrix = confusion_matrix(primary_y, primary_predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=15,
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    ax.set_xticks([0, 1], ["Normal", "Suspicious"])
    ax.set_yticks([0, 1], ["Normal", "Suspicious"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Conservative primary holdout confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(plot_dir / "primary_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    primary_attack = per_attack[per_attack["scope"] == "conservative_primary"].sort_values(
        "recall"
    )
    plt.figure(figsize=(8.6, 5.6))
    plt.barh(primary_attack["attack_cat"], primary_attack["recall"], color="#4C78A8")
    plt.xlim(0, 1.02)
    plt.xlabel("Recall")
    plt.title("Attack recall on the conservative primary holdout")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "primary_attack_recall.png", dpi=200, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(primary_y, primary_probabilities)
    ap = average_precision_score(primary_y, primary_probabilities)
    plt.figure(figsize=(6.8, 5.2))
    plt.plot(recall, precision, linewidth=2.2, color="#4C78A8")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Conservative primary precision-recall curve (AP = {ap:.3f})")
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "primary_precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    threshold = load_frozen_threshold()
    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    full_training = pd.concat([development, validation], ignore_index=True)
    X_training, y_training = split_xy(full_training)
    final_columns = [
        column for column in X_training.columns if column not in REMOVED_FEATURES
    ]

    print("Fitting the frozen no-TTL Random Forest on all training rows...", flush=True)
    started = time.time()
    model = make_models(X_training[final_columns])["Random forest"]
    model.fit(X_training[final_columns], y_training)
    fit_seconds = time.time() - started

    model_path = args.output_dir / "final_random_forest_no_ttl.joblib"
    joblib.dump(model, model_path)
    model_hash = sha256_file(model_path)

    print("Opening the supplied test set for final evaluation...", flush=True)
    test = pd.read_csv(TEST_PATH)
    if list(test.columns) != list(full_training.columns):
        raise ValueError("Training and test schemas do not match")
    if not test["id"].is_unique:
        raise ValueError("Test IDs are not unique; historical-sample reconstruction is ambiguous")

    historical_sample = stratified_limit(test, HISTORICAL_TEST_SAMPLE_SIZE)
    historical_ids = set(historical_sample["id"])
    historically_unexamined = ~test["id"].isin(historical_ids)

    all_predictor_columns = list(X_training.columns)
    training_all_hashes = coarsened_hashes(
        full_training, all_predictor_columns, NEAR_DUPLICATE_SIGNIFICANT_DIGITS
    )
    test_all_hashes = coarsened_hashes(
        test, all_predictor_columns, NEAR_DUPLICATE_SIGNIFICANT_DIGITS
    )
    training_final_hashes = coarsened_hashes(
        full_training, final_columns, NEAR_DUPLICATE_SIGNIFICANT_DIGITS
    )
    test_final_hashes = coarsened_hashes(
        test, final_columns, NEAR_DUPLICATE_SIGNIFICANT_DIGITS
    )
    all_overlap = set(training_all_hashes).intersection(set(test_all_hashes))
    final_overlap = set(training_final_hashes).intersection(set(test_final_hashes))
    near_duplicate_clean = ~test_final_hashes.isin(final_overlap)
    conservative_primary = historically_unexamined & near_duplicate_clean

    masks = {
        "official_test_all": np.ones(len(test), dtype=bool),
        "historically_unexamined_remainder": historically_unexamined.to_numpy(),
        "near_duplicate_clean_test": near_duplicate_clean.to_numpy(),
        "conservative_primary": conservative_primary.to_numpy(),
    }
    if any(mask.sum() == 0 for mask in masks.values()):
        raise ValueError("At least one final evaluation scope is empty")

    X_test, y_test = split_xy(test)
    probabilities = model.predict_proba(X_test[final_columns])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    prediction_table = test[["id", "attack_cat", TARGET]].copy()
    prediction_table["probability"] = probabilities
    prediction_table["prediction"] = predictions
    prediction_table["historically_evaluated_5000"] = ~historically_unexamined
    prediction_table["training_near_duplicate_final_features"] = ~near_duplicate_clean
    prediction_table.to_csv(args.output_dir / "test_predictions.csv", index=False)

    metric_rows = []
    interval_rows = []
    attack_tables = []
    scope_predictions: dict[str, dict[str, np.ndarray]] = {}
    for position, (scope, mask) in enumerate(masks.items()):
        frame = test.loc[mask].reset_index(drop=True)
        scope_y = y_test.to_numpy(dtype=int)[mask]
        scope_probabilities = probabilities[mask]
        scope_groups = test_final_hashes.to_numpy(dtype=np.uint64)[mask]
        metrics, scope_predicted = calculate_point_metrics(
            scope_y, scope_probabilities, threshold
        )
        metric_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "suspicious_rate": float(scope_y.mean()),
                **metrics,
            }
        )
        intervals = cluster_bootstrap_intervals(
            scope_y,
            scope_probabilities,
            threshold,
            scope_groups,
            args.bootstrap_iterations,
            seed=SEED + 500 + position,
        )
        for row in intervals:
            row["scope"] = scope
        interval_rows.extend(intervals)
        attack_tables.append(per_attack_recall(frame, scope_predicted, scope))
        scope_predictions[scope] = {
            "y": scope_y,
            "probabilities": scope_probabilities,
            "predictions": scope_predicted,
        }

    metrics_table = pd.DataFrame(metric_rows)
    intervals_table = pd.DataFrame(interval_rows)
    per_attack_table = pd.concat(attack_tables, ignore_index=True)
    metrics_table.to_csv(args.output_dir / "holdout_metrics.csv", index=False)
    intervals_table.to_csv(args.output_dir / "holdout_bootstrap_intervals.csv", index=False)
    per_attack_table.to_csv(args.output_dir / "per_attack_recall.csv", index=False)
    historical_sample[["id", "attack_cat", TARGET]].to_csv(
        args.output_dir / "reconstructed_historical_test_sample.csv", index=False
    )

    primary = scope_predictions["conservative_primary"]
    save_plots(
        metrics_table,
        per_attack_table,
        primary["y"],
        primary["probabilities"],
        primary["predictions"],
        args.output_dir,
    )

    primary_metrics = metrics_table[
        metrics_table["scope"] == "conservative_primary"
    ].iloc[0]
    result = {
        "metadata": {
            "purpose": "final frozen holdout evaluation",
            "test_path": str(TEST_PATH),
            "test_opened_for_this_evaluation": True,
            "no_model_changes_after_test_opened": True,
            "historical_test_exposure": (
                "An older development artifact evaluated a deterministic "
                "5,000-row stratified sample; that sample is excluded from "
                "the conservative primary scope."
            ),
            "near_duplicate_definition": (
                "categorical predictors retained exactly; numeric final-model "
                f"predictors rounded to {NEAR_DUPLICATE_SIGNIFICANT_DIGITS} significant digits"
            ),
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": SEED,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "frozen_model": {
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 24,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "removed_features": list(REMOVED_FEATURES),
            "threshold": threshold,
            "threshold_source": (
                "internal near-duplicate-grouped development fold, frozen in robustness audit"
            ),
            "training_rows": len(full_training),
            "fit_seconds": fit_seconds,
            "artifact": str(model_path),
            "sha256": model_hash,
        },
        "holdout_integrity": {
            "official_test_rows": len(test),
            "reconstructed_historical_sample_rows": int(
                (~historically_unexamined).sum()
            ),
            "historically_unexamined_rows": int(historically_unexamined.sum()),
            "all_predictor_cross_partition_near_duplicate_groups": len(all_overlap),
            "all_predictor_test_rows_in_overlap": int(
                test_all_hashes.isin(all_overlap).sum()
            ),
            "final_feature_cross_partition_near_duplicate_groups": len(final_overlap),
            "final_feature_test_rows_in_overlap": int(
                test_final_hashes.isin(final_overlap).sum()
            ),
            "conservative_primary_rows": int(conservative_primary.sum()),
        },
        "scope_metrics": json_records(metrics_table),
        "conservative_primary_metrics": {
            key: (
                int(primary_metrics[key])
                if key in {"rows", "tn", "fp", "fn", "tp"}
                else float(primary_metrics[key])
            )
            for key in metrics_table.columns
            if key != "scope"
        },
        "bootstrap_intervals": json_records(intervals_table),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nFinal holdout metrics")
    print(
        metrics_table[
            [
                "scope",
                "rows",
                "average_precision",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
