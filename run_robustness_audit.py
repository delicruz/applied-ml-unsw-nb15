"""Audit TTL shortcuts and near-duplicate leakage without opening the test set.

This sensitivity analysis uses only the existing development and validation
partitions. It defines a conservative near-duplicate group by rounding every
numeric predictor to five significant digits while retaining categorical
predictors exactly. The audit then:

1. quantifies near-duplicate overlap in the current split;
2. evaluates the selected Random Forest on the current validation data and on
   a version with cross-partition near-duplicate groups removed;
3. creates a separate 80/20 robustness split that keeps near-duplicate groups
   together; and
4. repeats each evaluation after removing ``sttl`` and all TTL-related fields.

Every operating threshold is selected on an internal grouped fold from the
corresponding development partition. The supplied testing set is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

from run_preliminary_validation import (
    CATEGORICAL,
    DEVELOPMENT_PATH,
    SEED,
    TARGET,
    VALIDATION_PATH,
    calculate_metrics,
    make_models,
    select_threshold,
    split_xy,
)


OUTPUT_DIR = Path("artifacts/robustness_audit")
SIGNIFICANT_DIGITS = 5
TTL_FEATURES = ("sttl", "dttl", "ct_state_ttl")
VARIANTS = {
    "all_features": (),
    "without_sttl": ("sttl",),
    "without_all_ttl": TTL_FEATURES,
}
METRIC_NAMES = ("average_precision", "f1", "mcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--significant-digits", type=int, default=SIGNIFICANT_DIGITS)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.significant_digits <= 8:
        raise ValueError("--significant-digits must be between 2 and 8")
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")


def round_significant(values: pd.Series, digits: int) -> np.ndarray:
    """Round finite nonzero values to a fixed number of significant digits."""
    data = values.to_numpy(dtype=np.float64, copy=True)
    mask = np.isfinite(data) & (data != 0)
    magnitudes = np.floor(np.log10(np.abs(data[mask])))
    factors = np.power(10.0, digits - 1 - magnitudes)
    data[mask] = np.round(data[mask] * factors) / factors
    return data


def near_duplicate_hashes(frame: pd.DataFrame, digits: int) -> pd.Series:
    """Return reproducible hashes for coarsened predictor vectors."""
    predictors, _ = split_xy(frame)
    coarsened = predictors.copy()
    numeric_columns = [
        column for column in coarsened.columns if column not in CATEGORICAL
    ]
    for column in numeric_columns:
        coarsened[column] = round_significant(coarsened[column], digits)
    return pd.util.hash_pandas_object(coarsened, index=False).astype("uint64")


def sensitivity_table(
    development: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for digits in (5, 4, 3, 2):
        development_hashes = near_duplicate_hashes(development, digits)
        validation_hashes = near_duplicate_hashes(validation, digits)
        overlap = set(development_hashes).intersection(set(validation_hashes))
        combined = pd.concat(
            [development_hashes, validation_hashes], ignore_index=True
        )
        group_sizes = combined.value_counts()
        rows.append(
            {
                "significant_digits": digits,
                "cross_partition_groups": len(overlap),
                "development_rows_in_cross_groups": int(
                    development_hashes.isin(overlap).sum()
                ),
                "validation_rows_in_cross_groups": int(
                    validation_hashes.isin(overlap).sum()
                ),
                "validation_row_share_in_cross_groups": float(
                    validation_hashes.isin(overlap).mean()
                ),
                "combined_groups_with_multiple_rows": int((group_sizes > 1).sum()),
                "largest_group_rows": int(group_sizes.max()),
            }
        )
    return pd.DataFrame(rows)


def cross_partition_group_table(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    development_hashes: pd.Series,
    validation_hashes: pd.Series,
) -> pd.DataFrame:
    overlap = set(development_hashes).intersection(set(validation_hashes))
    rows = []
    for group_hash in sorted(overlap):
        development_mask = development_hashes.eq(group_hash)
        validation_mask = validation_hashes.eq(group_hash)
        development_part = development.loc[development_mask]
        validation_part = validation.loc[validation_mask]
        combined_labels = pd.concat(
            [development_part[TARGET], validation_part[TARGET]], ignore_index=True
        )
        combined_attacks = pd.concat(
            [
                development_part["attack_cat"],
                validation_part["attack_cat"],
            ],
            ignore_index=True,
        )
        rows.append(
            {
                "near_duplicate_group": f"{int(group_hash):016x}",
                "development_rows": int(development_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "total_rows": int(development_mask.sum() + validation_mask.sum()),
                "suspicious_rate": float(combined_labels.mean()),
                "label_values": ",".join(
                    str(value) for value in sorted(combined_labels.unique())
                ),
                "attack_categories": ",".join(
                    sorted(combined_attacks.astype(str).unique())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["total_rows", "near_duplicate_group"], ascending=[False, True]
    )


def choose_grouped_validation_fold(
    frame: pd.DataFrame, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the grouped 80/20 fold closest to the attack distribution."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 31)
    overall = frame["attack_cat"].value_counts(normalize=True)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for development_indices, validation_indices in splitter.split(
        frame, frame["attack_cat"], groups=groups
    ):
        distribution = (
            frame.iloc[validation_indices]["attack_cat"]
            .value_counts(normalize=True)
            .reindex(overall.index, fill_value=0)
        )
        size_error = abs(len(validation_indices) / len(frame) - 0.20)
        distribution_error = float((distribution - overall).abs().mean())
        candidates.append(
            (size_error + distribution_error, development_indices, validation_indices)
        )
    _, development_indices, validation_indices = min(
        candidates, key=lambda candidate: candidate[0]
    )
    return development_indices, validation_indices


def internal_tuning_split(
    development: pd.DataFrame, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    X, _ = split_xy(development)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(
        splitter.split(X, development["attack_cat"], groups=groups)
    )


def fit_variant(
    development: pd.DataFrame,
    validation_frames: dict[str, pd.DataFrame],
    development_groups: pd.Series,
    validation_groups: dict[str, pd.Series],
    variant: str,
    removed: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    X_development, y_development = split_xy(development)
    columns = [column for column in X_development.columns if column not in removed]
    fit_indices, tuning_indices = internal_tuning_split(
        development, development_groups
    )

    print(f"Selecting threshold: {variant}...", flush=True)
    tuning_model = make_models(X_development.iloc[fit_indices][columns])[
        "Random forest"
    ]
    tuning_model.fit(
        X_development.iloc[fit_indices][columns], y_development.iloc[fit_indices]
    )
    tuning_probabilities = tuning_model.predict_proba(
        X_development.iloc[tuning_indices][columns]
    )[:, 1]
    threshold, _ = select_threshold(
        y_development.iloc[tuning_indices], tuning_probabilities
    )

    print(f"Fitting final model: {variant}...", flush=True)
    started = time.time()
    model = make_models(X_development[columns])["Random forest"]
    model.fit(X_development[columns], y_development)
    fit_seconds = time.time() - started

    metric_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    for scope, validation in validation_frames.items():
        X_validation, y_validation = split_xy(validation)
        probabilities = model.predict_proba(X_validation[columns])[:, 1]
        metrics, predicted = calculate_metrics(
            y_validation, probabilities, threshold
        )
        metric_rows.append(
            {
                "evaluation_scope": scope,
                "variant": variant,
                "removed_features": ",".join(removed) or "none",
                "development_rows": len(development),
                "validation_rows": len(validation),
                "threshold": threshold,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"threshold", "confusion_matrix"}
                },
                **metrics["confusion_matrix"],
                "fit_seconds": fit_seconds,
            }
        )
        predictions[scope] = {
            "y": y_validation.to_numpy(dtype=int),
            "probabilities": probabilities,
            "predictions": predicted,
            "groups": validation_groups[scope].to_numpy(dtype=np.uint64),
            "attack_cat": validation["attack_cat"].astype(str).to_numpy(),
        }
    return metric_rows, predictions


def group_codes(groups: np.ndarray) -> tuple[np.ndarray, int]:
    codes, unique_groups = pd.factorize(groups, sort=False)
    return codes, len(unique_groups)


def grouped_bootstrap_weights(
    codes: np.ndarray, group_count: int, rng: np.random.Generator
) -> np.ndarray:
    sampled = rng.integers(0, group_count, group_count)
    group_weights = np.bincount(sampled, minlength=group_count)
    return group_weights[codes]


def metric_triplet(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
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


def paired_ablation_bootstrap(
    scenario_predictions: dict[str, dict[str, dict[str, Any]]],
    metric_table: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    """Estimate paired metric changes relative to the all-feature model."""
    rows = []
    for scope, variant_predictions in scenario_predictions.items():
        baseline = variant_predictions["all_features"]
        baseline_threshold = float(
            metric_table.query(
                "evaluation_scope == @scope and variant == 'all_features'"
            )["threshold"].iloc[0]
        )
        codes, group_count = group_codes(baseline["groups"])
        for variant in ("without_sttl", "without_all_ttl"):
            comparison = variant_predictions[variant]
            comparison_threshold = float(
                metric_table.query(
                    "evaluation_scope == @scope and variant == @variant"
                )["threshold"].iloc[0]
            )
            if not np.array_equal(baseline["y"], comparison["y"]):
                raise ValueError(f"Unpaired labels for {scope} and {variant}")
            rng = np.random.default_rng(
                SEED
                + int(
                    hashlib.sha256(f"{scope}:{variant}".encode()).hexdigest()[:8],
                    16,
                )
            )
            samples = {metric: [] for metric in METRIC_NAMES}
            completed = 0
            while completed < iterations:
                weights = grouped_bootstrap_weights(codes, group_count, rng)
                if np.unique(baseline["y"][weights > 0]).size < 2:
                    continue
                baseline_values = metric_triplet(
                    baseline["y"],
                    baseline["probabilities"],
                    baseline_threshold,
                    sample_weight=weights,
                )
                comparison_values = metric_triplet(
                    baseline["y"],
                    comparison["probabilities"],
                    comparison_threshold,
                    sample_weight=weights,
                )
                for metric in METRIC_NAMES:
                    samples[metric].append(
                        comparison_values[metric] - baseline_values[metric]
                    )
                completed += 1
            point_baseline = metric_triplet(
                baseline["y"], baseline["probabilities"], baseline_threshold
            )
            point_comparison = metric_triplet(
                comparison["y"],
                comparison["probabilities"],
                comparison_threshold,
            )
            for metric in METRIC_NAMES:
                differences = np.asarray(samples[metric])
                rows.append(
                    {
                        "evaluation_scope": scope,
                        "comparison_variant": variant,
                        "metric": metric,
                        "all_features_point": point_baseline[metric],
                        "comparison_point": point_comparison[metric],
                        "difference_comparison_minus_all": (
                            point_comparison[metric] - point_baseline[metric]
                        ),
                        "difference_95_ci_low": float(
                            np.quantile(differences, 0.025)
                        ),
                        "difference_95_ci_high": float(
                            np.quantile(differences, 0.975)
                        ),
                        "bootstrap_probability_difference_positive": float(
                            np.mean(differences > 0)
                        ),
                        "iterations": len(differences),
                        "resampling_unit": (
                            "five-significant-digit near-duplicate group"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def per_attack_recall_table(
    scenario_predictions: dict[str, dict[str, dict[str, Any]]]
) -> pd.DataFrame:
    rows = []
    for scope, variant_predictions in scenario_predictions.items():
        for variant, values in variant_predictions.items():
            attack_frame = pd.DataFrame(
                {
                    "attack_cat": values["attack_cat"],
                    "label": values["y"],
                    "prediction": values["predictions"],
                }
            )
            attacks = attack_frame[attack_frame["label"] == 1]
            summary = (
                attacks.groupby("attack_cat")
                .agg(support=("label", "size"), detected=("prediction", "sum"))
                .reset_index()
            )
            summary["recall"] = summary["detected"] / summary["support"]
            summary.insert(0, "variant", variant)
            summary.insert(0, "evaluation_scope", scope)
            rows.extend(summary.to_dict(orient="records"))
    return pd.DataFrame(rows)


def save_plots(
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    per_attack: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_data = metrics.pivot(
        index="evaluation_scope", columns="variant", values="average_precision"
    ).reindex(
        [
            "current_validation",
            "current_validation_near_duplicate_clean",
            "near_duplicate_grouped_resplit",
        ]
    ).reindex(columns=["all_features", "without_sttl", "without_all_ttl"])
    plot_data.index = [
        "Original\nvalidation",
        "Near-duplicate-clean\nvalidation",
        "Near-duplicate-grouped\nresplit",
    ]
    plot_data.columns = ["All features", "Without sttl", "Without all TTL"]
    ax = plot_data.plot.bar(
        figsize=(10.0, 5.4), color=["#4C78A8", "#F58518", "#E45756"]
    )
    ax.set_title("Average precision under TTL and near-duplicate sensitivity checks")
    ax.set_xlabel("")
    ax.set_ylabel("Average precision")
    ax.set_ylim(max(0.9, float(plot_data.min().min()) - 0.01), 1.0)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Feature set")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        plot_dir / "robustness_average_precision.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    selected = differences[
        (differences["comparison_variant"] == "without_all_ttl")
        & (differences["metric"].isin(["average_precision", "f1", "mcc"]))
    ].copy()
    scope_labels = {
        "current_validation": "Original validation",
        "current_validation_near_duplicate_clean": "Near-duplicate-clean validation",
        "near_duplicate_grouped_resplit": "Near-duplicate-grouped resplit",
    }
    metric_labels = {
        "average_precision": "AP",
        "f1": "F1",
        "mcc": "MCC",
    }
    labels = [
        f"{scope_labels[scope]} | {metric_labels[metric]}"
        for scope, metric in zip(
            selected["evaluation_scope"], selected["metric"]
        )
    ]
    positions = np.arange(len(selected))
    points = selected["difference_comparison_minus_all"].to_numpy()
    low = selected["difference_95_ci_low"].to_numpy()
    high = selected["difference_95_ci_high"].to_numpy()
    plt.figure(figsize=(9.0, 6.0))
    plt.axvline(0, color="#666666", linestyle="--", linewidth=1.3)
    plt.errorbar(
        points,
        positions,
        xerr=np.vstack([points - low, high - points]),
        fmt="o",
        capsize=4,
        color="#54A24B",
    )
    plt.yticks(positions, labels)
    plt.xlabel("Change after removing all TTL features")
    plt.title("Paired near-duplicate-group bootstrap differences")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        plot_dir / "ttl_ablation_differences.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    robust_attack = per_attack[
        (per_attack["evaluation_scope"] == "near_duplicate_grouped_resplit")
        & (per_attack["variant"].isin(["all_features", "without_all_ttl"]))
    ]
    pivot = robust_attack.pivot(
        index="attack_cat", columns="variant", values="recall"
    ).reindex(columns=["all_features", "without_all_ttl"]).sort_values(
        "all_features"
    )
    pivot.columns = ["All features", "Without all TTL"]
    ax = pivot.plot.barh(
        figsize=(10.5, 5.8), color=["#4C78A8", "#F58518"]
    )
    ax.set_title("Attack recall on the near-duplicate-grouped robustness split")
    ax.set_xlabel("Recall")
    ax.set_ylabel("")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(title="Feature set", loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(
        plot_dir / "robust_split_attack_recall.png", dpi=200, bbox_inches="tight"
    )
    plt.close()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    missing_ttl = sorted(set(TTL_FEATURES).difference(development.columns))
    if missing_ttl:
        raise ValueError(f"TTL features are missing: {missing_ttl}")

    sensitivity = sensitivity_table(development, validation)
    sensitivity.to_csv(
        args.output_dir / "near_duplicate_sensitivity.csv", index=False
    )
    development_hashes = near_duplicate_hashes(
        development, args.significant_digits
    )
    validation_hashes = near_duplicate_hashes(validation, args.significant_digits)
    overlap = set(development_hashes).intersection(set(validation_hashes))
    cross_groups = cross_partition_group_table(
        development, validation, development_hashes, validation_hashes
    )
    cross_groups.to_csv(
        args.output_dir / "cross_partition_near_duplicate_groups.csv", index=False
    )

    clean_validation_mask = ~validation_hashes.isin(overlap)
    clean_validation = validation.loc[clean_validation_mask].reset_index(drop=True)
    clean_validation_hashes = validation_hashes.loc[clean_validation_mask].reset_index(
        drop=True
    )

    combined = pd.concat([development, validation], ignore_index=True)
    combined_hashes = near_duplicate_hashes(combined, args.significant_digits)
    robust_development_indices, robust_validation_indices = (
        choose_grouped_validation_fold(combined, combined_hashes)
    )
    robust_development = combined.iloc[robust_development_indices].reset_index(drop=True)
    robust_validation = combined.iloc[robust_validation_indices].reset_index(drop=True)
    robust_development_hashes = combined_hashes.iloc[
        robust_development_indices
    ].reset_index(drop=True)
    robust_validation_hashes = combined_hashes.iloc[
        robust_validation_indices
    ].reset_index(drop=True)
    robust_overlap = set(robust_development_hashes).intersection(
        set(robust_validation_hashes)
    )
    if robust_overlap:
        raise RuntimeError("Near-duplicate groups crossed the robustness split")

    scenarios = {
        "current": {
            "development": development,
            "development_groups": development_hashes,
            "validations": {
                "current_validation": validation,
                "current_validation_near_duplicate_clean": clean_validation,
            },
            "validation_groups": {
                "current_validation": validation_hashes,
                "current_validation_near_duplicate_clean": clean_validation_hashes,
            },
        },
        "robust_resplit": {
            "development": robust_development,
            "development_groups": robust_development_hashes,
            "validations": {
                "near_duplicate_grouped_resplit": robust_validation,
            },
            "validation_groups": {
                "near_duplicate_grouped_resplit": robust_validation_hashes,
            },
        },
    }

    metric_rows: list[dict[str, Any]] = []
    all_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario_name, scenario in scenarios.items():
        print(f"\nScenario: {scenario_name}", flush=True)
        for variant, removed in VARIANTS.items():
            rows, predictions = fit_variant(
                scenario["development"],
                scenario["validations"],
                scenario["development_groups"],
                scenario["validation_groups"],
                variant,
                removed,
            )
            metric_rows.extend(rows)
            for scope, values in predictions.items():
                all_predictions.setdefault(scope, {})[variant] = values

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "robustness_metrics.csv", index=False)
    differences = paired_ablation_bootstrap(
        all_predictions, metrics, args.bootstrap_iterations
    )
    differences.to_csv(
        args.output_dir / "ttl_ablation_paired_bootstrap.csv", index=False
    )
    per_attack = per_attack_recall_table(all_predictions)
    per_attack.to_csv(args.output_dir / "per_attack_recall.csv", index=False)
    save_plots(metrics, differences, per_attack, args.output_dir)

    selected_sensitivity = sensitivity[
        sensitivity["significant_digits"] == args.significant_digits
    ].iloc[0]
    result = {
        "metadata": {
            "purpose": "TTL shortcut and near-duplicate robustness audit",
            "protected_test_set_opened": False,
            "development_path": str(DEVELOPMENT_PATH),
            "validation_path": str(VALIDATION_PATH),
            "near_duplicate_definition": (
                "categorical predictors retained exactly; numeric predictors rounded "
                f"to {args.significant_digits} significant digits"
            ),
            "seed": SEED,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_resampling_unit": (
                "five-significant-digit near-duplicate group"
            ),
            "threshold_policy": (
                "F1 selected on an internal StratifiedGroupKFold development fold"
            ),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "near_duplicate_overlap": {
            "cross_partition_groups": int(
                selected_sensitivity["cross_partition_groups"]
            ),
            "development_rows_in_cross_groups": int(
                selected_sensitivity["development_rows_in_cross_groups"]
            ),
            "validation_rows_in_cross_groups": int(
                selected_sensitivity["validation_rows_in_cross_groups"]
            ),
            "validation_row_share_in_cross_groups": float(
                selected_sensitivity["validation_row_share_in_cross_groups"]
            ),
            "clean_validation_rows": len(clean_validation),
        },
        "robustness_split": {
            "development_rows": len(robust_development),
            "validation_rows": len(robust_validation),
            "cross_partition_near_duplicate_groups": len(robust_overlap),
            "development_suspicious_rate": float(
                robust_development[TARGET].mean()
            ),
            "validation_suspicious_rate": float(robust_validation[TARGET].mean()),
        },
        "metrics": json_records(metrics),
        "paired_ttl_ablation": json_records(differences),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nRobustness metrics")
    print(
        metrics[
            [
                "evaluation_scope",
                "variant",
                "validation_rows",
                "threshold",
                "average_precision",
                "f1",
                "mcc",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
