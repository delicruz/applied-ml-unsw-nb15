"""Explain the NF-UNSW-NB15 to NF-ToN-IoT generalisation failure.

This is a descriptive post-evaluation audit. It verifies the external
validation inputs, reloads the already-fitted NetFlow-v1 models and their
source-selected thresholds, and measures three distinct forms of shift:

1. feature-distribution shift;
2. feature/label association drift; and
3. frozen-model score and error-profile shift.

No model, feature set, threshold, or prediction rule is selected or changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from run_external_netflow_validation import (
    ATTACK,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    balanced_group_folds,
    exact_feature_groups,
    load_dataset,
    log1p_nonnegative,
    sha256_file,
)


VALIDATION_DIR = Path("artifacts/external_netflow_validation")
OUTPUT_DIR = Path("artifacts/external_shift_audit")
MIN_BIN_SUPPORT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_and_verify(
    validation_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    summary_path = validation_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run run_external_netflow_validation.py first: {summary_path} is missing"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    provenance = summary["data_provenance"]
    if provenance["source_row_limit"] is not None or provenance["external_row_limit"] is not None:
        raise ValueError("Shift audit requires the full external-validation run")
    if summary["experiment_scope"]["external_data_used_for_tuning"]:
        raise ValueError("External-validation manifest reports external-data tuning")

    source_path = Path(provenance["source_path"])
    external_path = Path(provenance["external_path"])
    if sha256_file(source_path) != provenance["source_sha256"]:
        raise ValueError("NF-UNSW-NB15 hash does not match the validation manifest")
    if sha256_file(external_path) != provenance["external_sha256"]:
        raise ValueError("NF-ToN-IoT hash does not match the validation manifest")
    source = load_dataset(source_path, row_limit=None)
    external = load_dataset(external_path, row_limit=None)
    if len(source) != provenance["source_rows"] or len(external) != provenance["external_rows"]:
        raise ValueError("Dataset row counts do not match the validation manifest")

    model_records: dict[str, Any] = {}
    for model_name, record in summary["models"].items():
        model_path = validation_dir / record["artifact"]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model_records[model_name] = {
            "pipeline": joblib.load(model_path),
            "threshold": float(record["threshold"]),
            "path": model_path,
            "sha256": sha256_file(model_path),
        }
    return source, external, summary, model_records


def source_quantile_edges(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numeric):
        return np.array([-np.inf, np.inf])
    quantiles = np.unique(np.quantile(numeric, np.linspace(0, 1, 11)))
    interior = quantiles[1:-1]
    return np.concatenate(([-np.inf], interior, [np.inf]))


def population_stability_index(
    source: pd.Series, external: pd.Series
) -> tuple[float, int]:
    source_values = pd.to_numeric(source, errors="coerce").dropna().to_numpy(dtype=float)
    external_values = pd.to_numeric(external, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(source_values) or not len(external_values):
        return float("nan"), 0
    edges = source_quantile_edges(source)
    if len(edges) == 2:
        return 0.0, 1
    source_counts, _ = np.histogram(source_values, bins=edges)
    external_counts, _ = np.histogram(external_values, bins=edges)
    epsilon = 1e-8
    source_share = np.clip(source_counts / source_counts.sum(), epsilon, None)
    external_share = np.clip(external_counts / external_counts.sum(), epsilon, None)
    psi = np.sum((external_share - source_share) * np.log(external_share / source_share))
    return float(psi), len(edges) - 1


def numeric_distribution_shift(
    source: pd.DataFrame, external: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        source_values = pd.to_numeric(source[feature], errors="coerce")
        external_values = pd.to_numeric(external[feature], errors="coerce")
        source_nonmissing = source_values.dropna()
        external_nonmissing = external_values.dropna()
        source_q01 = float(source_nonmissing.quantile(0.01))
        source_q99 = float(source_nonmissing.quantile(0.99))
        source_iqr = float(
            source_nonmissing.quantile(0.75) - source_nonmissing.quantile(0.25)
        )
        scale = source_iqr or float(source_nonmissing.std(ddof=0)) or 1.0
        psi, bins = population_stability_index(source_values, external_values)
        ks = ks_2samp(source_nonmissing, external_nonmissing, method="asymp")
        rows.append(
            {
                "feature": feature,
                "source_median": float(source_nonmissing.median()),
                "external_median": float(external_nonmissing.median()),
                "normalized_median_shift": float(
                    abs(external_nonmissing.median() - source_nonmissing.median()) / scale
                ),
                "source_missing_rate": float(source_values.isna().mean()),
                "external_missing_rate": float(external_values.isna().mean()),
                "external_below_source_q01_rate": float((external_nonmissing < source_q01).mean()),
                "external_above_source_q99_rate": float((external_nonmissing > source_q99).mean()),
                "psi": psi,
                "psi_bins": bins,
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows).sort_values(["psi", "ks_statistic"], ascending=False)


def categorical_distribution(series: pd.Series) -> pd.Series:
    return series.fillna("<MISSING>").astype(str).value_counts(normalize=True)


def categorical_distribution_shift(
    source: pd.DataFrame, external: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for feature in CATEGORICAL_FEATURES:
        source_values = source[feature].fillna("<MISSING>").astype(str)
        external_values = external[feature].fillna("<MISSING>").astype(str)
        source_dist = categorical_distribution(source_values)
        external_dist = categorical_distribution(external_values)
        categories = source_dist.index.union(external_dist.index)
        source_probability = source_dist.reindex(categories, fill_value=0.0).to_numpy()
        external_probability = external_dist.reindex(categories, fill_value=0.0).to_numpy()
        differences = np.abs(external_probability - source_probability)
        largest = int(np.argmax(differences))
        rows.append(
            {
                "feature": feature,
                "source_unique": int(source_values.nunique()),
                "external_unique": int(external_values.nunique()),
                "jensen_shannon_divergence_bits": float(
                    jensenshannon(source_probability, external_probability, base=2.0) ** 2
                ),
                "external_unseen_category_rate": float(
                    (~external_values.isin(source_dist.index)).mean()
                ),
                "largest_shift_category": str(categories[largest]),
                "source_share_largest_shift": float(source_probability[largest]),
                "external_share_largest_shift": float(external_probability[largest]),
                "largest_absolute_share_shift": float(differences[largest]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "jensen_shannon_divergence_bits", ascending=False
    )


def binned_feature(
    source: pd.Series, external: pd.Series, categorical: bool
) -> tuple[pd.Series, pd.Series]:
    if categorical:
        return (
            source.fillna("<MISSING>").astype(str),
            external.fillna("<MISSING>").astype(str),
        )
    edges = source_quantile_edges(source)
    labels = [f"bin_{index:02d}" for index in range(len(edges) - 1)]
    return (
        pd.cut(
            pd.to_numeric(source, errors="coerce"),
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype("string").fillna("<MISSING>"),
        pd.cut(
            pd.to_numeric(external, errors="coerce"),
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype("string").fillna("<MISSING>"),
    )


def feature_label_association_drift(
    source: pd.DataFrame, external: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def relative_log_odds(rate: pd.Series, rows: pd.Series, prevalence: float) -> pd.Series:
        smoothed_rate = (rate * rows + 0.5) / (rows + 1.0)
        smoothed_prevalence = (
            prevalence * float(rows.sum()) + 0.5
        ) / (float(rows.sum()) + 1.0)
        return np.log(smoothed_rate / (1 - smoothed_rate)) - np.log(
            smoothed_prevalence / (1 - smoothed_prevalence)
        )

    detail_tables = []
    summary_rows = []
    source_prevalence = float(source[TARGET].mean())
    external_prevalence = float(external[TARGET].mean())
    for feature in MODEL_FEATURES:
        source_bin, external_bin = binned_feature(
            source[feature], external[feature], feature in CATEGORICAL_FEATURES
        )
        source_stats = (
            pd.DataFrame({"bin": source_bin, "label": source[TARGET].to_numpy()})
            .groupby("bin", dropna=False)["label"]
            .agg(source_rows="size", source_attack_rate="mean")
        )
        external_stats = (
            pd.DataFrame({"bin": external_bin, "label": external[TARGET].to_numpy()})
            .groupby("bin", dropna=False)["label"]
            .agg(external_rows="size", external_attack_rate="mean")
        )
        detail = source_stats.join(external_stats, how="outer").reset_index()
        detail.insert(0, "feature", feature)
        detail["source_rows"] = detail["source_rows"].fillna(0).astype(int)
        detail["external_rows"] = detail["external_rows"].fillna(0).astype(int)
        detail["supported_both"] = (
            (detail["source_rows"] >= MIN_BIN_SUPPORT)
            & (detail["external_rows"] >= MIN_BIN_SUPPORT)
        )
        detail["signed_attack_rate_change"] = (
            detail["external_attack_rate"] - detail["source_attack_rate"]
        )
        detail["absolute_attack_rate_change"] = detail[
            "signed_attack_rate_change"
        ].abs()
        supported = detail[detail["supported_both"]].copy()
        supported["source_relative_log_odds"] = relative_log_odds(
            supported["source_attack_rate"],
            supported["source_rows"],
            source_prevalence,
        )
        supported["external_relative_log_odds"] = relative_log_odds(
            supported["external_attack_rate"],
            supported["external_rows"],
            external_prevalence,
        )
        supported["relative_log_odds_drift"] = (
            supported["external_relative_log_odds"]
            - supported["source_relative_log_odds"]
        )
        supported["absolute_relative_log_odds_drift"] = supported[
            "relative_log_odds_drift"
        ].abs()
        for column in (
            "source_relative_log_odds",
            "external_relative_log_odds",
            "relative_log_odds_drift",
            "absolute_relative_log_odds_drift",
        ):
            detail[column] = np.nan
            detail.loc[supported.index, column] = supported[column]
        supported_external_rows = int(supported["external_rows"].sum())
        total_external_rows = int(detail["external_rows"].sum())
        weights = (
            supported["external_rows"] / supported_external_rows
            if supported_external_rows
            else pd.Series(dtype=float)
        )
        summary_rows.append(
            {
                "feature": feature,
                "supported_bins": int(len(supported)),
                "external_coverage_supported_bins": float(
                    supported_external_rows / total_external_rows
                ),
                "external_weighted_absolute_attack_rate_change": float(
                    (weights * supported["absolute_attack_rate_change"]).sum()
                )
                if supported_external_rows
                else np.nan,
                "external_weighted_signed_attack_rate_change": float(
                    (weights * supported["signed_attack_rate_change"]).sum()
                )
                if supported_external_rows
                else np.nan,
                "largest_supported_absolute_attack_rate_change": float(
                    supported["absolute_attack_rate_change"].max()
                )
                if len(supported)
                else np.nan,
                "external_weighted_absolute_relative_log_odds_drift": float(
                    (weights * supported["absolute_relative_log_odds_drift"]).sum()
                )
                if supported_external_rows
                else np.nan,
                "external_weighted_signed_relative_log_odds_drift": float(
                    (weights * supported["relative_log_odds_drift"]).sum()
                )
                if supported_external_rows
                else np.nan,
                "largest_supported_absolute_relative_log_odds_drift": float(
                    supported["absolute_relative_log_odds_drift"].max()
                )
                if len(supported)
                else np.nan,
            }
        )
        detail_tables.append(detail)
    summary = pd.DataFrame(summary_rows).sort_values(
        "external_weighted_absolute_relative_log_odds_drift", ascending=False
    )
    return summary, pd.concat(detail_tables, ignore_index=True)


def model_score_shift(
    source_validation: pd.DataFrame,
    external: pd.DataFrame,
    models: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "source_validation": {},
        "external": {},
    }
    for model_name, record in models.items():
        for dataset, frame in (
            ("source_validation", source_validation),
            ("external", external),
        ):
            values = record["pipeline"].predict_proba(frame.loc[:, MODEL_FEATURES])[:, 1]
            probabilities[dataset][model_name] = values
            threshold = record["threshold"]
            labels = frame[TARGET].to_numpy()
            quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "rows": int(len(frame)),
                    "prevalence": float(labels.mean()),
                    "threshold": threshold,
                    "predicted_positive_rate": float((values >= threshold).mean()),
                    "mean_probability": float(values.mean()),
                    "normal_mean_probability": float(values[labels == 0].mean()),
                    "attack_mean_probability": float(values[labels == 1].mean()),
                    **{
                        f"probability_q{int(level * 100):02d}": float(value)
                        for level, value in zip(
                            [0.05, 0.25, 0.50, 0.75, 0.95], quantiles
                        )
                    },
                }
            )
    return pd.DataFrame(rows), probabilities


def external_error_profiles(
    source: pd.DataFrame,
    external: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    models: dict[str, Any],
) -> pd.DataFrame:
    tables = []
    for feature in MODEL_FEATURES:
        _, bins = binned_feature(
            source[feature], external[feature], feature in CATEGORICAL_FEATURES
        )
        for model_name, values in probabilities.items():
            predictions = values >= models[model_name]["threshold"]
            working = pd.DataFrame(
                {
                    "bin": bins,
                    "label": external[TARGET].to_numpy(),
                    "prediction": predictions,
                }
            )
            rows = []
            for category, group in working.groupby("bin", dropna=False):
                normal = group["label"].eq(0)
                attack = group["label"].eq(1)
                rows.append(
                    {
                        "feature": feature,
                        "bin": str(category),
                        "model": model_name,
                        "rows": int(len(group)),
                        "normal_rows": int(normal.sum()),
                        "attack_rows": int(attack.sum()),
                        "false_positive_rate": float(group.loc[normal, "prediction"].mean())
                        if normal.any()
                        else np.nan,
                        "attack_recall": float(group.loc[attack, "prediction"].mean())
                        if attack.any()
                        else np.nan,
                    }
                )
            tables.append(pd.DataFrame(rows))
    return pd.concat(tables, ignore_index=True)


def save_plots(
    numeric_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    association_shift: pd.DataFrame,
    probabilities: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    numeric_plot = numeric_shift.sort_values("psi")
    axes[0].barh(numeric_plot["feature"], numeric_plot["psi"], color="#e45756")
    axes[0].set_xlabel("Population Stability Index")
    axes[0].set_title("Numeric distribution shift")
    axes[0].grid(axis="x", alpha=0.25)
    categorical_plot = categorical_shift.sort_values("jensen_shannon_divergence_bits")
    axes[1].barh(
        categorical_plot["feature"],
        categorical_plot["jensen_shannon_divergence_bits"],
        color="#4c78a8",
    )
    axes[1].set_xlabel("Jensen-Shannon divergence (bits)")
    axes[1].set_title("Categorical distribution shift")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_distribution_shift.png", dpi=180)
    plt.close(fig)

    association_plot = association_shift.sort_values(
        "external_weighted_absolute_relative_log_odds_drift"
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(
        association_plot["feature"],
        association_plot["external_weighted_absolute_relative_log_odds_drift"],
        color="#f58518",
    )
    axis.set_xlabel("External-weighted absolute relative log-odds drift")
    axis.set_title("Prevalence-adjusted feature/label association drift")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_label_association_drift.png", dpi=180)
    plt.close(fig)

    bins = np.linspace(0, 1, 51)
    fig, axes = plt.subplots(1, len(probabilities["external"]), figsize=(13, 4.8))
    axes = np.atleast_1d(axes)
    for axis, model_name in zip(axes, probabilities["external"]):
        for dataset, color in (("source_validation", "#4c78a8"), ("external", "#e45756")):
            values = probabilities[dataset][model_name]
            counts, _ = np.histogram(values, bins=bins)
            axis.plot(
                bins[:-1],
                counts / counts.sum(),
                drawstyle="steps-post",
                label=dataset.replace("_", " "),
                color=color,
            )
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_xlabel("Predicted attack probability")
        axis.set_ylabel("Share of records")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Frozen-model score distribution shift")
    fig.tight_layout()
    fig.savefig(plot_dir / "model_score_distribution_shift.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, external, validation_summary, models = load_and_verify(args.validation_dir)

    source_groups = exact_feature_groups(source)
    source_folds, _ = balanced_group_folds(source_groups, source[TARGET])
    source_validation = source.loc[source_folds.eq(4)].copy()
    expected_rows = next(
        row["rows"]
        for row in validation_summary["split_integrity"]["splits"]
        if row["split"] == "source_validation"
    )
    if len(source_validation) != expected_rows:
        raise ValueError("Reconstructed source-validation row count does not match")

    numeric_shift = numeric_distribution_shift(source, external)
    categorical_shift = categorical_distribution_shift(source, external)
    association_shift, association_detail = feature_label_association_drift(
        source, external
    )
    score_shift, probabilities = model_score_shift(
        source_validation, external, models
    )
    error_profiles = external_error_profiles(
        source, external, probabilities["external"], models
    )

    numeric_shift.to_csv(args.output_dir / "numeric_distribution_shift.csv", index=False)
    categorical_shift.to_csv(
        args.output_dir / "categorical_distribution_shift.csv", index=False
    )
    association_shift.to_csv(
        args.output_dir / "feature_label_association_drift.csv", index=False
    )
    association_detail.to_csv(
        args.output_dir / "feature_label_association_bins.csv", index=False
    )
    score_shift.to_csv(args.output_dir / "model_score_shift.csv", index=False)
    error_profiles.to_csv(
        args.output_dir / "external_error_profiles_by_feature_bin.csv", index=False
    )
    save_plots(
        numeric_shift,
        categorical_shift,
        association_shift,
        probabilities,
        args.output_dir,
    )

    summary = {
        "scope": {
            "analysis": "Post-evaluation descriptive cross-dataset shift audit",
            "model_changes": False,
            "feature_changes": False,
            "threshold_changes": False,
            "external_tuning": False,
        },
        "verified_inputs": {
            "external_validation_summary": str(args.validation_dir / "summary.json"),
            "source_rows": int(len(source)),
            "external_rows": int(len(external)),
            "source_validation_rows": int(len(source_validation)),
            "model_sha256": {
                model_name: record["sha256"] for model_name, record in models.items()
            },
        },
        "largest_numeric_distribution_shifts": numeric_shift.head(6).to_dict(
            orient="records"
        ),
        "categorical_distribution_shifts": categorical_shift.to_dict(
            orient="records"
        ),
        "largest_feature_label_association_drifts": association_shift.head(8).to_dict(
            orient="records"
        ),
        "model_score_shift": score_shift.to_dict(orient="records"),
        "interpretation_boundary": (
            "Distribution and association drift are descriptive evidence and do not "
            "identify a causal mechanism or authorise external-data retuning."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
