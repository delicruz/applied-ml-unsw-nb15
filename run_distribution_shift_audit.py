"""Audit holdout distribution shift and false positives without retuning.

This is a strictly descriptive post-evaluation analysis. It uses the frozen
no-TTL Random Forest, its frozen 0.54 decision threshold, and the conservative
primary holdout scope created by ``run_final_holdout_evaluation.py``. Nothing in
this script selects features, changes the model, or changes the threshold.
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
from scipy.spatial.distance import jensenshannon
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
OUTPUT_DIR = Path("artifacts/distribution_shift_audit")
EXPECTED_THRESHOLD = 0.54
MIN_GROUP_NORMAL_SUPPORT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--shap-sample-per-class", type=int, default=500)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def population_stability_index(train: pd.Series, test: pd.Series) -> tuple[float, int]:
    train_values = pd.to_numeric(train, errors="coerce").dropna().to_numpy(dtype=float)
    test_values = pd.to_numeric(test, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(train_values) or not len(test_values):
        return float("nan"), 0

    quantiles = np.quantile(train_values, np.linspace(0, 1, 11))
    interior = np.unique(quantiles[1:-1])
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    if len(edges) < 3:
        return 0.0, 1

    train_counts, _ = np.histogram(train_values, bins=edges)
    test_counts, _ = np.histogram(test_values, bins=edges)
    epsilon = 1e-6
    train_share = np.clip(train_counts / train_counts.sum(), epsilon, None)
    test_share = np.clip(test_counts / test_counts.sum(), epsilon, None)
    psi = np.sum((test_share - train_share) * np.log(test_share / train_share))
    return float(psi), int(len(edges) - 1)


def numeric_shift_table(
    training: pd.DataFrame, holdout: pd.DataFrame, numeric_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        train = pd.to_numeric(training[column], errors="coerce")
        test = pd.to_numeric(holdout[column], errors="coerce")
        train_nonmissing = train.dropna()
        test_nonmissing = test.dropna()
        train_median = float(train_nonmissing.median())
        test_median = float(test_nonmissing.median())
        train_q1 = float(train_nonmissing.quantile(0.25))
        train_q3 = float(train_nonmissing.quantile(0.75))
        scale = train_q3 - train_q1
        if not np.isfinite(scale) or scale == 0:
            scale = float(train_nonmissing.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        psi, bins = population_stability_index(train, test)
        ks = ks_2samp(train_nonmissing, test_nonmissing, method="asymp")
        rows.append(
            {
                "feature": column,
                "train_missing_rate": float(train.isna().mean()),
                "holdout_missing_rate": float(test.isna().mean()),
                "train_mean": float(train_nonmissing.mean()),
                "holdout_mean": float(test_nonmissing.mean()),
                "train_median": train_median,
                "holdout_median": test_median,
                "train_iqr": float(train_q3 - train_q1),
                "normalized_median_shift": float(abs(test_median - train_median) / scale),
                "psi": psi,
                "psi_bins": bins,
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["psi", "ks_statistic"], ascending=False, na_position="last"
    )


def categorical_distribution(series: pd.Series) -> pd.Series:
    return series.fillna("<MISSING>").astype(str).value_counts(normalize=True)


def categorical_shift_table(
    training: pd.DataFrame, holdout: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in CATEGORICAL:
        train_dist = categorical_distribution(training[column])
        test_dist = categorical_distribution(holdout[column])
        categories = train_dist.index.union(test_dist.index)
        train_prob = train_dist.reindex(categories, fill_value=0.0).to_numpy()
        test_prob = test_dist.reindex(categories, fill_value=0.0).to_numpy()
        js_divergence = float(jensenshannon(train_prob, test_prob, base=2.0) ** 2)
        unseen = ~holdout[column].fillna("<MISSING>").astype(str).isin(train_dist.index)
        differences = np.abs(test_prob - train_prob)
        max_index = int(np.argmax(differences))
        rows.append(
            {
                "feature": column,
                "train_unique": int(len(train_dist)),
                "holdout_unique": int(len(test_dist)),
                "jensen_shannon_divergence_bits": js_divergence,
                "unseen_holdout_category_rate": float(unseen.mean()),
                "largest_prevalence_shift_category": str(categories[max_index]),
                "largest_absolute_prevalence_shift": float(differences[max_index]),
                "train_prevalence_for_largest_shift": float(train_prob[max_index]),
                "holdout_prevalence_for_largest_shift": float(test_prob[max_index]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "jensen_shannon_divergence_bits", ascending=False
    )


def categorical_error_groups(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in CATEGORICAL:
        values = frame[feature].fillna("<MISSING>").astype(str)
        working = frame.assign(_category=values)
        for category, group in working.groupby("_category", dropna=False):
            normal = group[group[TARGET] == 0]
            suspicious = group[group[TARGET] == 1]
            fp = int(((normal["prediction"] == 1)).sum())
            tn = int(((normal["prediction"] == 0)).sum())
            tp = int(((suspicious["prediction"] == 1)).sum())
            fn = int(((suspicious["prediction"] == 0)).sum())
            rows.append(
                {
                    "feature": feature,
                    "category": str(category),
                    "records": int(len(group)),
                    "normal_records": int(len(normal)),
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_positive_rate": fp / len(normal) if len(normal) else np.nan,
                    "suspicious_records": int(len(suspicious)),
                    "true_positives": tp,
                    "false_negatives": fn,
                    "attack_recall": tp / len(suspicious) if len(suspicious) else np.nan,
                    "mean_probability": float(group["probability"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["false_positives", "normal_records"], ascending=False
    )


def numeric_false_positive_contrast(
    frame: pd.DataFrame, numeric_columns: list[str]
) -> pd.DataFrame:
    normal = frame[frame[TARGET] == 0]
    fp = normal[normal["prediction"] == 1]
    tn = normal[normal["prediction"] == 0]
    rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        fp_values = pd.to_numeric(fp[column], errors="coerce").dropna()
        tn_values = pd.to_numeric(tn[column], errors="coerce").dropna()
        normal_values = pd.to_numeric(normal[column], errors="coerce").dropna()
        normal_iqr = float(normal_values.quantile(0.75) - normal_values.quantile(0.25))
        scale = normal_iqr
        if not np.isfinite(scale) or scale == 0:
            scale = float(normal_values.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        fp_median = float(fp_values.median())
        tn_median = float(tn_values.median())
        ks = ks_2samp(fp_values, tn_values, method="asymp")
        rows.append(
            {
                "feature": column,
                "false_positive_median": fp_median,
                "true_negative_median": tn_median,
                "signed_normalized_median_gap": float((fp_median - tn_median) / scale),
                "absolute_normalized_median_gap": float(abs(fp_median - tn_median) / scale),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "false_positive_missing_rate": float(fp[column].isna().mean()),
                "true_negative_missing_rate": float(tn[column].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["ks_statistic", "absolute_normalized_median_gap"], ascending=False
    )


def normal_probability_bins(frame: pd.DataFrame) -> pd.DataFrame:
    normal = frame[frame[TARGET] == 0].copy()
    edges = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, EXPECTED_THRESHOLD, 0.6, 0.7, 0.8, 0.9, 1.0000001])
    labels = [f"[{left:.2f}, {right:.2f})" for left, right in zip(edges[:-1], edges[1:])]
    normal["probability_bin"] = pd.cut(
        normal["probability"], bins=edges, labels=labels, include_lowest=True, right=False
    )
    result = (
        normal.groupby("probability_bin", observed=False)
        .agg(records=("id", "size"), mean_probability=("probability", "mean"))
        .reset_index()
    )
    result["share_of_normal"] = result["records"] / len(normal)
    result["classification"] = np.where(
        result["probability_bin"].astype(str).str.startswith(("[0.54", "[0.60", "[0.70", "[0.80", "[0.90")),
        "false_positive",
        "true_negative",
    )
    return result


def transformed_to_original(feature: str, original_columns: list[str]) -> str:
    if feature in original_columns:
        return feature
    for categorical in CATEGORICAL:
        if feature.startswith(f"{categorical}_"):
            return categorical
    return feature


def false_positive_shap_drivers(
    model: Any,
    frame: pd.DataFrame,
    final_columns: list[str],
    sample_per_class: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    normal = frame[frame[TARGET] == 0]
    fp = normal[normal["prediction"] == 1]
    tn = normal[normal["prediction"] == 0]
    fp_sample = fp.sample(n=min(sample_per_class, len(fp)), random_state=SEED)
    tn_sample = tn.sample(n=min(sample_per_class, len(tn)), random_state=SEED)
    sample = pd.concat(
        [fp_sample.assign(_class="false_positive"), tn_sample.assign(_class="true_negative")],
        ignore_index=True,
    )

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

    original_names = np.array(
        [transformed_to_original(name, final_columns) for name in feature_names]
    )
    aggregate = np.zeros((len(sample), len(final_columns)), dtype=float)
    for index, feature in enumerate(final_columns):
        aggregate[:, index] = values[:, original_names == feature].sum(axis=1)

    is_fp = sample["_class"].eq("false_positive").to_numpy()
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(final_columns):
        fp_values = aggregate[is_fp, index]
        tn_values = aggregate[~is_fp, index]
        rows.append(
            {
                "feature": feature,
                "false_positive_mean_signed_shap": float(fp_values.mean()),
                "true_negative_mean_signed_shap": float(tn_values.mean()),
                "false_positive_minus_true_negative_signed_shap": float(
                    fp_values.mean() - tn_values.mean()
                ),
                "false_positive_mean_absolute_shap": float(np.abs(fp_values).mean()),
                "true_negative_mean_absolute_shap": float(np.abs(tn_values).mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "false_positive_mean_signed_shap", ascending=False
    )
    return table, {
        "false_positive_sample": int(is_fp.sum()),
        "true_negative_sample": int((~is_fp).sum()),
    }


def save_plots(
    numeric_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    groups: pd.DataFrame,
    fp_contrast: pd.DataFrame,
    shap_drivers: pd.DataFrame,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    top_shift = numeric_shift.head(15).sort_values("psi")
    plt.figure(figsize=(8.4, 6.2))
    plt.barh(top_shift["feature"], top_shift["psi"], color="#E45756")
    plt.xlabel("Population Stability Index (training vs holdout)")
    plt.title("Largest numeric feature shifts")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "numeric_feature_shift_psi.png", dpi=200, bbox_inches="tight")
    plt.close()

    cat_plot = categorical_shift.sort_values("jensen_shannon_divergence_bits")
    plt.figure(figsize=(7.4, 3.5))
    plt.barh(
        cat_plot["feature"], cat_plot["jensen_shannon_divergence_bits"], color="#72B7B2"
    )
    plt.xlabel("Jensen-Shannon divergence (bits)")
    plt.title("Categorical distribution shift")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "categorical_feature_shift.png", dpi=200, bbox_inches="tight")
    plt.close()

    eligible = groups[groups["normal_records"] >= MIN_GROUP_NORMAL_SUPPORT].copy()
    eligible = eligible.nlargest(15, "false_positive_rate").sort_values("false_positive_rate")
    eligible["group"] = eligible["feature"] + " = " + eligible["category"]
    plt.figure(figsize=(8.8, 6.3))
    plt.barh(eligible["group"], eligible["false_positive_rate"], color="#F58518")
    plt.xlabel("False-positive rate among Normal records")
    plt.title(f"Highest-risk categorical groups (at least {MIN_GROUP_NORMAL_SUPPORT} Normal records)")
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "categorical_false_positive_groups.png", dpi=200, bbox_inches="tight")
    plt.close()

    contrast = fp_contrast.head(15).sort_values("ks_statistic")
    plt.figure(figsize=(8.4, 6.2))
    plt.barh(contrast["feature"], contrast["ks_statistic"], color="#B279A2")
    plt.xlabel("KS statistic: false positives vs true negatives")
    plt.title("Numeric features separating Normal-record outcomes")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "false_positive_numeric_contrast.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap_plot = shap_drivers.head(15).sort_values("false_positive_mean_signed_shap")
    colors = np.where(shap_plot["false_positive_mean_signed_shap"] >= 0, "#E45756", "#4C78A8")
    plt.figure(figsize=(8.6, 6.2))
    plt.barh(
        shap_plot["feature"], shap_plot["false_positive_mean_signed_shap"], color=colors
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean signed SHAP value for false positives")
    plt.title("Features pushing false positives toward attack")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "false_positive_shap_drivers.png", dpi=200, bbox_inches="tight")
    plt.close()

    normal = frame[frame[TARGET] == 0]
    plt.figure(figsize=(8.2, 4.8))
    plt.hist(normal["probability"], bins=40, color="#4C78A8", alpha=0.85)
    plt.axvline(EXPECTED_THRESHOLD, color="#E45756", linestyle="--", linewidth=2, label="Frozen threshold = 0.54")
    plt.xlabel("Predicted attack probability for true Normal records")
    plt.ylabel("Records")
    plt.title("Holdout Normal-score distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "normal_probability_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.shap_sample_per_class < 50:
        raise ValueError("--shap-sample-per-class must be at least 50")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = FINAL_DIR / "summary.json"
    prediction_path = FINAL_DIR / "test_predictions.csv"
    model_path = FINAL_DIR / "final_random_forest_no_ttl.joblib"
    with summary_path.open(encoding="utf-8") as handle:
        final_summary = json.load(handle)
    threshold = float(final_summary["frozen_model"]["threshold"])
    if not np.isclose(threshold, EXPECTED_THRESHOLD):
        raise ValueError(f"Expected frozen threshold {EXPECTED_THRESHOLD}, found {threshold}")
    if not final_summary["metadata"]["no_model_changes_after_test_opened"]:
        raise ValueError("Final evaluation does not confirm model freeze")
    expected_hash = final_summary["frozen_model"]["sha256"]
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_hash:
        raise ValueError("Frozen model hash does not match the final-evaluation manifest")

    print("Loading frozen evaluation inputs...", flush=True)
    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    training = pd.concat([development, validation], ignore_index=True)
    test = pd.read_csv(TEST_PATH)
    predictions = pd.read_csv(prediction_path)
    if len(test) != len(predictions) or not np.array_equal(test["id"], predictions["id"]):
        raise ValueError("Test rows do not align with saved final predictions")
    frame = test.copy()
    for column in [
        "probability",
        "prediction",
        "historically_evaluated_5000",
        "training_near_duplicate_final_features",
    ]:
        frame[column] = predictions[column].to_numpy()
    conservative = frame[
        (~frame["historically_evaluated_5000"].astype(bool))
        & (~frame["training_near_duplicate_final_features"].astype(bool))
    ].copy()

    X_training, _ = split_xy(training)
    final_columns = [column for column in X_training.columns if column not in REMOVED_FEATURES]
    numeric_columns = [column for column in final_columns if column not in CATEGORICAL]
    if len(conservative) != final_summary["holdout_integrity"]["conservative_primary_rows"]:
        raise ValueError("Conservative primary scope does not reproduce the final evaluation")

    print("Measuring train-to-holdout feature shift...", flush=True)
    numeric_shift = numeric_shift_table(training, conservative, numeric_columns)
    categorical_shift = categorical_shift_table(training, conservative)
    print("Profiling false positives and true negatives...", flush=True)
    groups = categorical_error_groups(conservative)
    fp_contrast = numeric_false_positive_contrast(conservative, numeric_columns)
    probability_bins = normal_probability_bins(conservative)

    print("Computing SHAP drivers on a balanced Normal-record sample...", flush=True)
    model = joblib.load(model_path)
    shap_drivers, shap_counts = false_positive_shap_drivers(
        model, conservative, final_columns, args.shap_sample_per_class
    )

    numeric_shift.to_csv(args.output_dir / "numeric_feature_shift.csv", index=False)
    categorical_shift.to_csv(args.output_dir / "categorical_feature_shift.csv", index=False)
    groups.to_csv(args.output_dir / "categorical_error_groups.csv", index=False)
    fp_contrast.to_csv(args.output_dir / "false_positive_numeric_contrast.csv", index=False)
    probability_bins.to_csv(args.output_dir / "normal_probability_bins.csv", index=False)
    shap_drivers.to_csv(args.output_dir / "false_positive_shap_drivers.csv", index=False)
    save_plots(
        numeric_shift,
        categorical_shift,
        groups,
        fp_contrast,
        shap_drivers,
        conservative,
        args.output_dir,
    )

    normal = conservative[conservative[TARGET] == 0]
    fp = normal[normal["prediction"] == 1]
    tn = normal[normal["prediction"] == 0]
    supported_groups = groups[groups["normal_records"] >= MIN_GROUP_NORMAL_SUPPORT]
    result = {
        "metadata": {
            "purpose": "descriptive post-evaluation distribution-shift and false-positive audit",
            "descriptive_only": True,
            "model_or_threshold_changes": False,
            "scope": "conservative_primary",
            "threshold": threshold,
            "removed_features": list(REMOVED_FEATURES),
            "training_rows": int(len(training)),
            "holdout_rows": int(len(conservative)),
            "frozen_model_sha256_verified": actual_hash,
            "seed": SEED,
        },
        "false_positive_summary": {
            "normal_records": int(len(normal)),
            "false_positives": int(len(fp)),
            "true_negatives": int(len(tn)),
            "false_positive_rate": float(len(fp) / len(normal)),
            "false_positive_mean_probability": float(fp["probability"].mean()),
            "false_positive_median_probability": float(fp["probability"].median()),
            "high_confidence_false_positives_at_0_90": int((fp["probability"] >= 0.90).sum()),
            "high_confidence_false_positive_share": float((fp["probability"] >= 0.90).mean()),
        },
        "top_numeric_train_to_holdout_shifts_by_psi": json_records(numeric_shift.head(10)),
        "categorical_train_to_holdout_shifts": json_records(categorical_shift),
        "top_supported_categorical_false_positive_groups": json_records(
            supported_groups.nlargest(10, "false_positive_rate")
        ),
        "top_numeric_false_positive_contrasts_by_ks": json_records(fp_contrast.head(10)),
        "top_false_positive_shap_drivers": json_records(shap_drivers.head(10)),
        "shap_sample": shap_counts,
        "interpretation_guardrail": (
            "Shift statistics and SHAP values are descriptive associations. They explain "
            "where performance changed but do not establish causation and were not used "
            "to retune the locked model."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved distribution-shift audit to {args.output_dir}", flush=True)
    print(json.dumps(result["false_positive_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
