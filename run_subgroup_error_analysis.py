"""Describe Fuzzers misses and Normal ``state=INT`` false positives.

This is a post-evaluation diagnostic of the conservative primary holdout. It
uses the frozen no-TTL Random Forest and threshold 0.54. The analysis produces
score summaries, supported categorical intersections, robust numeric feature
contrasts, and sampled SHAP contrasts for:

* missed versus detected Fuzzers attacks; and
* false-positive versus true-negative Normal records with ``state=INT``.

Nothing in this script changes the model, feature set, or threshold, and no
subgroup-specific correction rule is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp

from run_final_holdout_evaluation import REMOVED_FEATURES, TEST_PATH
from run_preliminary_validation import (
    CATEGORICAL,
    DEVELOPMENT_PATH,
    SEED,
    TARGET,
    VALIDATION_PATH,
    split_xy,
)


FINAL_DIR = Path("artifacts/final_holdout_evaluation")
OUTPUT_DIR = Path("artifacts/subgroup_error_analysis")
EXPECTED_THRESHOLD = 0.54
MIN_CATEGORY_SUPPORT = 50
SCORE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--shap-sample-per-outcome", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_conservative_holdout() -> tuple[pd.DataFrame, dict[str, Any], Path]:
    summary_path = FINAL_DIR / "summary.json"
    prediction_path = FINAL_DIR / "test_predictions.csv"
    model_path = FINAL_DIR / "final_random_forest_no_ttl.joblib"
    with summary_path.open(encoding="utf-8") as handle:
        final_summary = json.load(handle)

    threshold = float(final_summary["frozen_model"]["threshold"])
    if not np.isclose(threshold, EXPECTED_THRESHOLD):
        raise ValueError(
            f"Expected frozen threshold {EXPECTED_THRESHOLD}, found {threshold}"
        )
    if not final_summary["metadata"]["no_model_changes_after_test_opened"]:
        raise ValueError("Final evaluation does not confirm model freeze")
    actual_hash = sha256_file(model_path)
    if actual_hash != final_summary["frozen_model"]["sha256"]:
        raise ValueError("Frozen model hash does not match the final-evaluation manifest")

    test = pd.read_csv(TEST_PATH)
    predictions = pd.read_csv(prediction_path)
    if len(test) != len(predictions) or not np.array_equal(
        test["id"], predictions["id"]
    ):
        raise ValueError("Test rows do not align with saved final predictions")
    frame = test.copy()
    for column in (
        "probability",
        "prediction",
        "historically_evaluated_5000",
        "training_near_duplicate_final_features",
    ):
        frame[column] = predictions[column].to_numpy()
    conservative = frame[
        (~frame["historically_evaluated_5000"].astype(bool))
        & (~frame["training_near_duplicate_final_features"].astype(bool))
    ].copy()
    expected_rows = final_summary["holdout_integrity"]["conservative_primary_rows"]
    if len(conservative) != expected_rows:
        raise ValueError(
            f"Conservative scope has {len(conservative):,} rows; expected {expected_rows:,}"
        )
    final_summary["verified_model_sha256"] = actual_hash
    return conservative, final_summary, model_path


def outcome_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fuzzers = frame[frame["attack_cat"].eq("Fuzzers")].copy()
    state_int_normal = frame[
        frame["state"].eq("INT") & frame[TARGET].eq(0)
    ].copy()
    outcomes = {
        "fuzzers_missed": fuzzers[fuzzers["prediction"].eq(0)].copy(),
        "fuzzers_detected": fuzzers[fuzzers["prediction"].eq(1)].copy(),
        "state_int_false_positive": state_int_normal[
            state_int_normal["prediction"].eq(1)
        ].copy(),
        "state_int_true_negative": state_int_normal[
            state_int_normal["prediction"].eq(0)
        ].copy(),
    }
    if any(part.empty for part in outcomes.values()):
        empty = [name for name, part in outcomes.items() if part.empty]
        raise ValueError(f"Required outcome groups are empty: {empty}")
    return outcomes


def score_summary(outcomes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome, group in outcomes.items():
        values = group["probability"].to_numpy(dtype=float)
        quantiles = np.quantile(values, SCORE_QUANTILES)
        rows.append(
            {
                "outcome": outcome,
                "records": int(len(group)),
                "mean_probability": float(values.mean()),
                "standard_deviation": float(values.std(ddof=0)),
                **{
                    f"probability_q{int(level * 100):02d}": float(value)
                    for level, value in zip(SCORE_QUANTILES, quantiles)
                },
            }
        )
    return pd.DataFrame(rows)


def supported_categorical_profiles(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    audit_specs = (
        (
            "fuzzers",
            frame[frame["attack_cat"].eq("Fuzzers")],
            "miss_rate",
            0,
        ),
        (
            "state_int_normal",
            frame[frame["state"].eq("INT") & frame[TARGET].eq(0)],
            "false_positive_rate",
            1,
        ),
    )
    for audit, subset, error_rate_name, error_prediction in audit_specs:
        features = [feature for feature in CATEGORICAL if feature != "state"]
        if audit == "fuzzers":
            features.append("state")
        for feature in features:
            values = subset[feature].fillna("<MISSING>").astype(str)
            working = subset.assign(_category=values)
            for category, group in working.groupby("_category"):
                errors = int(group["prediction"].eq(error_prediction).sum())
                rows.append(
                    {
                        "audit": audit,
                        "feature": feature,
                        "category": str(category),
                        "records": int(len(group)),
                        "errors": errors,
                        "error_rate_name": error_rate_name,
                        "error_rate": float(errors / len(group)),
                        "mean_probability": float(group["probability"].mean()),
                        "supported": bool(len(group) >= MIN_CATEGORY_SUPPORT),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["audit", "supported", "errors", "error_rate"],
        ascending=[True, False, False, False],
    )


def robust_numeric_contrast(
    comparison: str,
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    group_a_name: str,
    group_b_name: str,
    numeric_columns: list[str],
) -> pd.DataFrame:
    combined = pd.concat([group_a, group_b], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for feature in numeric_columns:
        values_a = pd.to_numeric(group_a[feature], errors="coerce").dropna()
        values_b = pd.to_numeric(group_b[feature], errors="coerce").dropna()
        combined_values = pd.to_numeric(combined[feature], errors="coerce").dropna()
        if values_a.empty or values_b.empty:
            continue
        scale = float(
            combined_values.quantile(0.75) - combined_values.quantile(0.25)
        )
        if not np.isfinite(scale) or scale == 0:
            scale = float(combined_values.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        median_a = float(values_a.median())
        median_b = float(values_b.median())
        ks = ks_2samp(values_a, values_b, method="asymp")
        rows.append(
            {
                "comparison": comparison,
                "feature": feature,
                "group_a": group_a_name,
                "group_b": group_b_name,
                "group_a_records": int(len(group_a)),
                "group_b_records": int(len(group_b)),
                "group_a_median": median_a,
                "group_b_median": median_b,
                "signed_normalized_median_gap_a_minus_b": float(
                    (median_a - median_b) / scale
                ),
                "absolute_normalized_median_gap": float(
                    abs(median_a - median_b) / scale
                ),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "group_a_missing_rate": float(group_a[feature].isna().mean()),
                "group_b_missing_rate": float(group_b[feature].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["ks_statistic", "absolute_normalized_median_gap"], ascending=False
    )


def transformed_to_original(feature: str, original_columns: list[str]) -> str:
    if feature in original_columns:
        return feature
    for categorical in CATEGORICAL:
        if feature.startswith(f"{categorical}_"):
            return categorical
    return feature


def subgroup_shap_contrasts(
    model: Any,
    outcomes: dict[str, pd.DataFrame],
    final_columns: list[str],
    sample_per_outcome: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_parts = []
    sample_rows = []
    for outcome, group in outcomes.items():
        sample = group.sample(
            n=min(sample_per_outcome, len(group)), random_state=SEED
        ).copy()
        sample["_outcome"] = outcome
        sampled_parts.append(sample)
        sample_rows.append(
            {
                "outcome": outcome,
                "available_records": int(len(group)),
                "sampled_records": int(len(sample)),
            }
        )
    sample = pd.concat(sampled_parts, ignore_index=True)

    preprocessor = model.named_steps["preprocess"]
    forest = model.named_steps["model"]
    transformed = preprocessor.transform(sample[final_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    explanation = shap.TreeExplainer(forest)(transformed, check_additivity=False)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != transformed.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {transformed.shape}"
        )

    original_names = np.asarray(
        [transformed_to_original(name, final_columns) for name in feature_names]
    )
    aggregate = np.zeros((len(sample), len(final_columns)), dtype=float)
    for index, feature in enumerate(final_columns):
        aggregate[:, index] = values[:, original_names == feature].sum(axis=1)

    comparisons = (
        ("fuzzers_missed_vs_detected", "fuzzers_missed", "fuzzers_detected"),
        (
            "state_int_false_positive_vs_true_negative",
            "state_int_false_positive",
            "state_int_true_negative",
        ),
    )
    rows: list[dict[str, Any]] = []
    outcomes_array = sample["_outcome"].to_numpy()
    for comparison, group_a, group_b in comparisons:
        mask_a = outcomes_array == group_a
        mask_b = outcomes_array == group_b
        for index, feature in enumerate(final_columns):
            mean_a = float(aggregate[mask_a, index].mean())
            mean_b = float(aggregate[mask_b, index].mean())
            rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "group_a": group_a,
                    "group_b": group_b,
                    "group_a_mean_signed_shap": mean_a,
                    "group_b_mean_signed_shap": mean_b,
                    "signed_shap_gap_a_minus_b": mean_a - mean_b,
                    "absolute_shap_gap": abs(mean_a - mean_b),
                    "group_a_mean_absolute_shap": float(
                        np.abs(aggregate[mask_a, index]).mean()
                    ),
                    "group_b_mean_absolute_shap": float(
                        np.abs(aggregate[mask_b, index]).mean()
                    ),
                }
            )
    contrasts = pd.DataFrame(rows).sort_values(
        ["comparison", "absolute_shap_gap"], ascending=[True, False]
    )
    return contrasts, pd.DataFrame(sample_rows)


def save_plots(
    outcomes: dict[str, pd.DataFrame],
    categorical: pd.DataFrame,
    numeric: pd.DataFrame,
    shap_contrasts: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    score_specs = (
        (
            axes[0],
            "Fuzzers probability distributions",
            "fuzzers_missed",
            "fuzzers_detected",
            "Missed",
            "Detected",
        ),
        (
            axes[1],
            "Normal state=INT probability distributions",
            "state_int_false_positive",
            "state_int_true_negative",
            "False positive",
            "True negative",
        ),
    )
    for axis, title, group_a, group_b, label_a, label_b in score_specs:
        axis.hist(
            outcomes[group_a]["probability"],
            bins=30,
            density=True,
            alpha=0.65,
            label=label_a,
            color="#E45756",
        )
        axis.hist(
            outcomes[group_b]["probability"],
            bins=30,
            density=True,
            alpha=0.55,
            label=label_b,
            color="#4C78A8",
        )
        axis.axvline(
            EXPECTED_THRESHOLD,
            color="#222222",
            linestyle="--",
            linewidth=1.5,
            label="Threshold 0.54",
        )
        axis.set_title(title)
        axis.set_xlabel("Predicted attack probability")
        axis.set_ylabel("Density")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "subgroup_probability_distributions.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    numeric_specs = (
        (axes[0], "fuzzers_missed_vs_detected", "Fuzzers: missed vs detected"),
        (
            axes[1],
            "state_int_false_positive_vs_true_negative",
            "Normal state=INT: false positive vs true negative",
        ),
    )
    for axis, comparison, title in numeric_specs:
        subset = numeric[numeric["comparison"].eq(comparison)].head(12)
        subset = subset.sort_values("ks_statistic")
        axis.barh(subset["feature"], subset["ks_statistic"], color="#B279A2")
        axis.set_xlabel("KS statistic")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "numeric_outcome_contrasts.png", dpi=200)
    plt.close(figure)

    supported = categorical[categorical["supported"]].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    category_specs = (
        (axes[0], "fuzzers", "Fuzzers supported subgroups"),
        (axes[1], "state_int_normal", "Normal state=INT supported subgroups"),
    )
    for axis, audit, title in category_specs:
        subset = supported[supported["audit"].eq(audit)].nlargest(
            12, ["errors", "error_rate"]
        )
        subset = subset.sort_values("error_rate")
        labels = subset["feature"] + "=" + subset["category"]
        axis.barh(labels, subset["error_rate"], color="#F58518")
        axis.set_xlabel("Error rate")
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "categorical_subgroup_error_rates.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    shap_specs = (
        (axes[0], "fuzzers_missed_vs_detected", "Fuzzers SHAP contrast"),
        (
            axes[1],
            "state_int_false_positive_vs_true_negative",
            "Normal state=INT SHAP contrast",
        ),
    )
    for axis, comparison, title in shap_specs:
        subset = shap_contrasts[
            shap_contrasts["comparison"].eq(comparison)
        ].head(12)
        subset = subset.sort_values("signed_shap_gap_a_minus_b")
        colors = np.where(
            subset["signed_shap_gap_a_minus_b"] >= 0, "#E45756", "#4C78A8"
        )
        axis.barh(
            subset["feature"], subset["signed_shap_gap_a_minus_b"], color=colors
        )
        axis.axvline(0, color="#222222", linewidth=0.8)
        axis.set_xlabel("Mean signed SHAP gap: error minus correct outcome")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "subgroup_shap_contrasts.png", dpi=200)
    plt.close(figure)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.shap_sample_per_outcome < 50:
        raise ValueError("--shap-sample-per-outcome must be at least 50")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and verifying frozen holdout artifacts...", flush=True)
    conservative, final_summary, model_path = load_conservative_holdout()
    outcomes = outcome_frames(conservative)
    scores = score_summary(outcomes)
    categorical = supported_categorical_profiles(conservative)

    training = pd.concat(
        [pd.read_csv(DEVELOPMENT_PATH), pd.read_csv(VALIDATION_PATH)],
        ignore_index=True,
    )
    X_training, _ = split_xy(training)
    final_columns = [
        column for column in X_training.columns if column not in REMOVED_FEATURES
    ]
    numeric_columns = [
        column for column in final_columns if column not in CATEGORICAL
    ]

    print("Computing robust numeric outcome contrasts...", flush=True)
    numeric = pd.concat(
        [
            robust_numeric_contrast(
                "fuzzers_missed_vs_detected",
                outcomes["fuzzers_missed"],
                outcomes["fuzzers_detected"],
                "fuzzers_missed",
                "fuzzers_detected",
                numeric_columns,
            ),
            robust_numeric_contrast(
                "state_int_false_positive_vs_true_negative",
                outcomes["state_int_false_positive"],
                outcomes["state_int_true_negative"],
                "state_int_false_positive",
                "state_int_true_negative",
                numeric_columns,
            ),
        ],
        ignore_index=True,
    )
    numeric = numeric.sort_values(
        ["comparison", "ks_statistic", "absolute_normalized_median_gap"],
        ascending=[True, False, False],
    )

    print("Computing sampled SHAP outcome contrasts...", flush=True)
    model = joblib.load(model_path)
    shap_contrasts, shap_sample = subgroup_shap_contrasts(
        model,
        outcomes,
        final_columns,
        args.shap_sample_per_outcome,
    )

    scores.to_csv(args.output_dir / "score_summary.csv", index=False)
    categorical.to_csv(
        args.output_dir / "categorical_subgroup_profiles.csv", index=False
    )
    numeric.to_csv(args.output_dir / "numeric_outcome_contrasts.csv", index=False)
    shap_contrasts.to_csv(
        args.output_dir / "subgroup_shap_contrasts.csv", index=False
    )
    shap_sample.to_csv(args.output_dir / "shap_sample_counts.csv", index=False)
    save_plots(outcomes, categorical, numeric, shap_contrasts, args.output_dir)

    fuzzers_total = len(outcomes["fuzzers_missed"]) + len(
        outcomes["fuzzers_detected"]
    )
    state_int_total = len(outcomes["state_int_false_positive"]) + len(
        outcomes["state_int_true_negative"]
    )
    result = {
        "metadata": {
            "purpose": "descriptive Fuzzers and state=INT error audit",
            "scope": "conservative_primary_holdout",
            "descriptive_only": True,
            "model_or_threshold_changes": False,
            "subgroup_rules_created": False,
            "threshold": EXPECTED_THRESHOLD,
            "removed_features": list(REMOVED_FEATURES),
            "frozen_model_sha256_verified": final_summary[
                "verified_model_sha256"
            ],
            "seed": SEED,
        },
        "fuzzers": {
            "support": fuzzers_total,
            "detected": int(len(outcomes["fuzzers_detected"])),
            "missed": int(len(outcomes["fuzzers_missed"])),
            "recall": float(len(outcomes["fuzzers_detected"]) / fuzzers_total),
        },
        "state_int_normal": {
            "support": state_int_total,
            "false_positives": int(len(outcomes["state_int_false_positive"])),
            "true_negatives": int(len(outcomes["state_int_true_negative"])),
            "false_positive_rate": float(
                len(outcomes["state_int_false_positive"]) / state_int_total
            ),
        },
        "score_summary": json_records(scores),
        "top_supported_categorical_patterns": json_records(
            categorical[categorical["supported"]]
            .sort_values(["audit", "errors"], ascending=[True, False])
            .groupby("audit", group_keys=False)
            .head(10)
        ),
        "top_numeric_contrasts": json_records(
            numeric.groupby("comparison", group_keys=False).head(10)
        ),
        "top_shap_contrasts": json_records(
            shap_contrasts.groupby("comparison", group_keys=False).head(10)
        ),
        "shap_sample": json_records(shap_sample),
        "interpretation_guardrail": (
            "All subgroup, KS, and SHAP contrasts are descriptive associations "
            "within an already-evaluated holdout. They were not used to alter the "
            "model or threshold and do not establish causal error mechanisms or "
            "justify subgroup-specific decision rules."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("\nFuzzers and state=INT error summary", flush=True)
    print(
        pd.DataFrame(
            [
                {
                    "audit": "Fuzzers",
                    "support": fuzzers_total,
                    "errors": len(outcomes["fuzzers_missed"]),
                    "error_rate": len(outcomes["fuzzers_missed"])
                    / fuzzers_total,
                },
                {
                    "audit": "Normal state=INT",
                    "support": state_int_total,
                    "errors": len(outcomes["state_int_false_positive"]),
                    "error_rate": len(outcomes["state_int_false_positive"])
                    / state_int_total,
                },
            ]
        ).to_string(index=False),
        flush=True,
    )
    print(f"\nSaved outputs to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
