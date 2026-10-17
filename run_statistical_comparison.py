"""Compare the final development-stage models statistically and diagnostically.

This command fits Logistic Regression and the selected regularised Random
Forest using development data only, evaluates them on the held-out validation
partition, and performs paired bootstrap comparisons. The primary bootstrap
resamples five-significant-digit near-duplicate predictor groups so closely
matching records remain dependent; a conventional row bootstrap is retained as
a sensitivity analysis. The protected test set is never opened.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold

from run_preliminary_validation import (
    DEVELOPMENT_PATH,
    SEED,
    VALIDATION_PATH,
    calculate_metrics,
    make_models,
    predictor_hashes,
    select_threshold,
    split_xy,
)
from run_robustness_audit import near_duplicate_hashes


OUTPUT_DIR = Path("artifacts/statistical_comparison")
MODEL_NAMES = ["Logistic regression", "Random forest"]
NEAR_DUPLICATE_SIGNIFICANT_DIGITS = 5
COMPARISON_METRICS = ("average_precision", "f1", "mcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def fit_models_and_predict(
    development: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    X_development, y_development = split_xy(development)
    X_validation, y_validation = split_xy(validation)
    groups = near_duplicate_hashes(
        development, NEAR_DUPLICATE_SIGNIFICANT_DIGITS
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_indices, tuning_indices = next(
        splitter.split(X_development, development["attack_cat"], groups=groups)
    )

    thresholds: dict[str, float] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        print(f"Selecting threshold for {name}...", flush=True)
        tuning_model = make_models(X_development.iloc[fit_indices])[name]
        tuning_model.fit(
            X_development.iloc[fit_indices], y_development.iloc[fit_indices]
        )
        tuning_probabilities = tuning_model.predict_proba(
            X_development.iloc[tuning_indices]
        )[:, 1]
        threshold, _ = select_threshold(
            y_development.iloc[tuning_indices], tuning_probabilities
        )
        thresholds[name] = threshold

        print(f"Refitting {name} on all development rows...", flush=True)
        model = make_models(X_development)[name]
        model.fit(X_development, y_development)
        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        probabilities[name] = validation_probabilities
        metrics[name], _ = calculate_metrics(
            y_validation, validation_probabilities, threshold
        )
    return thresholds, probabilities, metrics


def metric_values(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    indices: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    sampled_y = y_true if indices is None else y_true[indices]
    values: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        sampled_probabilities = (
            probabilities[name]
            if indices is None
            else probabilities[name][indices]
        )
        predictions = sampled_probabilities >= thresholds[name]
        values[name] = {
            "average_precision": float(
                average_precision_score(
                    sampled_y,
                    sampled_probabilities,
                    sample_weight=sample_weight,
                )
            ),
            "f1": float(
                f1_score(
                    sampled_y,
                    predictions,
                    zero_division=0,
                    sample_weight=sample_weight,
                )
            ),
            "mcc": float(
                matthews_corrcoef(
                    sampled_y, predictions, sample_weight=sample_weight
                )
            ),
        }
    return values


def bootstrap_group_codes(group_hashes: np.ndarray) -> tuple[np.ndarray, int]:
    codes, unique_groups = pd.factorize(group_hashes, sort=False)
    return codes, len(unique_groups)


def run_paired_bootstrap(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    group_hashes: np.ndarray,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    group_codes, group_count = bootstrap_group_codes(group_hashes)
    for method_index, method in enumerate(("group_cluster", "row")):
        print(f"Paired bootstrap: {method} ({iterations} iterations)...", flush=True)
        rng = np.random.default_rng(SEED + 100 + method_index)
        completed = 0
        attempts = 0
        while completed < iterations:
            attempts += 1
            if method == "group_cluster":
                sampled_groups = rng.integers(0, group_count, group_count)
                group_weights = np.bincount(
                    sampled_groups, minlength=group_count
                )
                sample_weight = group_weights[group_codes]
                if np.unique(y_true[sample_weight > 0]).size < 2:
                    continue
                values = metric_values(
                    y_true,
                    probabilities,
                    thresholds,
                    sample_weight=sample_weight,
                )
            else:
                indices = rng.integers(0, len(y_true), len(y_true))
                if np.unique(y_true[indices]).size < 2:
                    continue
                values = metric_values(
                    y_true, probabilities, thresholds, indices=indices
                )
            for metric in COMPARISON_METRICS:
                logistic = values["Logistic regression"][metric]
                forest = values["Random forest"][metric]
                rows.append(
                    {
                        "resampling": method,
                        "iteration": completed + 1,
                        "metric": metric,
                        "logistic_regression": logistic,
                        "random_forest": forest,
                        "difference_rf_minus_lr": forest - logistic,
                    }
                )
            completed += 1
        if attempts > iterations:
            print(f"Discarded {attempts - iterations} single-class resamples", flush=True)

    samples = pd.DataFrame(rows)
    point_indices = np.arange(len(y_true), dtype=np.int64)
    point_values = metric_values(y_true, probabilities, thresholds, point_indices)
    summary_rows = []
    for method in ("group_cluster", "row"):
        for metric in COMPARISON_METRICS:
            part = samples[
                (samples["resampling"] == method) & (samples["metric"] == metric)
            ]
            differences = part["difference_rf_minus_lr"].to_numpy()
            probability_positive = float(np.mean(differences > 0))
            summary_rows.append(
                {
                    "resampling": method,
                    "metric": metric,
                    "logistic_regression_point": point_values["Logistic regression"][metric],
                    "random_forest_point": point_values["Random forest"][metric],
                    "difference_rf_minus_lr_point": (
                        point_values["Random forest"][metric]
                        - point_values["Logistic regression"][metric]
                    ),
                    "difference_95_ci_low": float(np.quantile(differences, 0.025)),
                    "difference_95_ci_high": float(np.quantile(differences, 0.975)),
                    "bootstrap_probability_rf_better": probability_positive,
                    "iterations": len(part),
                }
            )
    return samples, pd.DataFrame(summary_rows)


def calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    bins: int,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.digitize(probabilities, edges[1:-1], right=True)
    rows = []
    for bin_index in range(bins):
        mask = assignments == bin_index
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(y_true[mask].mean())
            absolute_gap = abs(mean_probability - observed_rate)
        else:
            mean_probability = np.nan
            observed_rate = np.nan
            absolute_gap = np.nan
        rows.append(
            {
                "model": model_name,
                "bin": bin_index + 1,
                "lower_bound": edges[bin_index],
                "upper_bound": edges[bin_index + 1],
                "count": count,
                "mean_probability": mean_probability,
                "observed_suspicious_rate": observed_rate,
                "absolute_calibration_gap": absolute_gap,
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    bins: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    prevalence = float(y_true.mean())
    baseline_brier = brier_score_loss(y_true, np.full(len(y_true), prevalence))
    for name in MODEL_NAMES:
        model_bins = bins[(bins["model"] == name) & (bins["count"] > 0)]
        weights = model_bins["count"] / model_bins["count"].sum()
        ece = float((weights * model_bins["absolute_calibration_gap"]).sum())
        brier = float(brier_score_loss(y_true, probabilities[name]))
        rows.append(
            {
                "model": name,
                "brier_score": brier,
                "brier_skill_vs_prevalence": 1 - brier / baseline_brier,
                "log_loss": float(log_loss(y_true, probabilities[name], labels=[0, 1])),
                "expected_calibration_error": ece,
                "maximum_calibration_error": float(
                    model_bins["absolute_calibration_gap"].max()
                ),
                "calibration_bins": int(len(model_bins)),
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    calibration_bins: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colours = {"Logistic regression": "#4C78A8", "Random forest": "#F58518"}

    plt.figure(figsize=(6.4, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect calibration")
    for name in MODEL_NAMES:
        part = calibration_bins[
            (calibration_bins["model"] == name) & (calibration_bins["count"] > 0)
        ]
        plt.plot(
            part["mean_probability"],
            part["observed_suspicious_rate"],
            marker="o",
            linewidth=2,
            color=colours[name],
            label=name,
        )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed suspicious rate")
    plt.title("Validation probability calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plot_dir / "calibration_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, name in zip(axes, MODEL_NAMES):
        axis.hist(
            probabilities[name][y_true == 0],
            bins=np.linspace(0, 1, 31),
            alpha=0.65,
            label="Normal",
            color="#4C78A8",
        )
        axis.hist(
            probabilities[name][y_true == 1],
            bins=np.linspace(0, 1, 31),
            alpha=0.65,
            label="Suspicious",
            color="#F58518",
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted suspicious probability")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation records (log scale)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "probability_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()

    primary = bootstrap_summary[
        bootstrap_summary["resampling"] == "group_cluster"
    ].copy()
    positions = np.arange(len(primary))
    point = primary["difference_rf_minus_lr_point"].to_numpy()
    low = primary["difference_95_ci_low"].to_numpy()
    high = primary["difference_95_ci_high"].to_numpy()
    plt.figure(figsize=(7.4, 4.7))
    plt.axvline(0, color="#666666", linestyle="--", linewidth=1.5)
    plt.errorbar(
        point,
        positions,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        color="#54A24B",
        capsize=5,
        linewidth=2,
    )
    metric_labels = {
        "average_precision": "Average precision",
        "f1": "F1",
        "mcc": "MCC",
    }
    plt.yticks(positions, [metric_labels[value] for value in primary["metric"]])
    plt.xlabel("Random Forest minus Logistic Regression")
    plt.title("Paired group-bootstrap metric differences with 95% intervals")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "paired_bootstrap_differences.png", dpi=200, bbox_inches="tight")
    plt.close()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    overlap = set(predictor_hashes(development)).intersection(
        set(predictor_hashes(validation))
    )
    if overlap:
        raise ValueError(f"Found {len(overlap)} predictor groups across partitions")

    thresholds, probabilities, metrics = fit_models_and_predict(development, validation)
    y_validation = validation["label"].to_numpy(dtype=int)
    group_hashes = near_duplicate_hashes(
        validation, NEAR_DUPLICATE_SIGNIFICANT_DIGITS
    ).to_numpy()

    predictions = validation[["id", "attack_cat", "label"]].copy()
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        predictions[f"{slug}_probability"] = probabilities[name]
        predictions[f"{slug}_prediction"] = (
            probabilities[name] >= thresholds[name]
        ).astype(int)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    metric_rows = []
    for name in MODEL_NAMES:
        metric_rows.append(
            {
                "model": name,
                **{key: value for key, value in metrics[name].items() if key != "confusion_matrix"},
                **metrics[name]["confusion_matrix"],
            }
        )
    validation_metrics = pd.DataFrame(metric_rows)
    validation_metrics.to_csv(args.output_dir / "validation_metrics.csv", index=False)

    bootstrap_samples, bootstrap_summary = run_paired_bootstrap(
        y_validation,
        probabilities,
        thresholds,
        group_hashes,
        args.bootstrap_iterations,
    )
    bootstrap_samples.to_csv(args.output_dir / "paired_bootstrap_samples.csv", index=False)
    bootstrap_summary.to_csv(args.output_dir / "paired_bootstrap_summary.csv", index=False)

    calibration_bins = pd.concat(
        [
            calibration_table(
                y_validation, probabilities[name], name, args.calibration_bins
            )
            for name in MODEL_NAMES
        ],
        ignore_index=True,
    )
    calibration_bins.to_csv(args.output_dir / "calibration_bins.csv", index=False)
    calibration_summary = calibration_metrics(
        y_validation, probabilities, calibration_bins
    )
    calibration_summary.to_csv(args.output_dir / "calibration_metrics.csv", index=False)

    save_plots(
        y_validation,
        probabilities,
        calibration_bins,
        bootstrap_summary,
        args.output_dir,
    )

    primary = bootstrap_summary[
        bootstrap_summary["resampling"] == "group_cluster"
    ]
    result = {
        "metadata": {
            "purpose": "paired validation comparison and calibration analysis",
            "protected_test_set_opened": False,
            "seed": SEED,
            "bootstrap_iterations": args.bootstrap_iterations,
            "primary_resampling_unit": (
                "five-significant-digit near-duplicate predictor group"
            ),
            "near_duplicate_significant_digits": (
                NEAR_DUPLICATE_SIGNIFICANT_DIGITS
            ),
            "sensitivity_resampling_unit": "row",
            "calibration_bins": args.calibration_bins,
            "development_path": str(DEVELOPMENT_PATH),
            "validation_path": str(VALIDATION_PATH),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "thresholds": thresholds,
        "validation_metrics": json_records(validation_metrics),
        "paired_group_bootstrap": json_records(primary),
        "calibration_metrics": json_records(calibration_summary),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nPaired group-bootstrap comparison")
    print(primary.to_string(index=False))
    print("\nCalibration metrics")
    print(calibration_summary.to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
