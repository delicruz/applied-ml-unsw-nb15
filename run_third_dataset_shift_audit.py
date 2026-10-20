"""Explain the NF-UNSW-NB15 to NF-BoT-IoT transfer failure.

This is a descriptive post-evaluation audit. It verifies the preregistered
third-dataset result, reloads the already-fitted NetFlow-v1 models and their
source-selected thresholds, and measures feature-distribution shift,
feature/label association drift, and frozen-model score shift. It also places
the score result beside the earlier NF-ToN-IoT audit when that artifact exists.

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

from run_external_netflow_validation import (
    MODEL_FEATURES,
    TARGET,
    balanced_group_folds,
    exact_feature_groups,
    load_dataset,
    log1p_nonnegative,
    sha256_file,
)
from run_external_shift_audit import (
    categorical_distribution_shift,
    external_error_profiles,
    feature_label_association_drift,
    model_score_shift,
    numeric_distribution_shift,
    save_plots,
)


SOURCE_VALIDATION_DIR = Path("artifacts/external_netflow_validation")
THIRD_VALIDATION_DIR = Path("artifacts/third_dataset_validation")
PREVIOUS_SHIFT_DIR = Path("artifacts/external_shift_audit")
PROTOCOL_PATH = Path("protocols/third_dataset_preregistration.json")
OUTPUT_DIR = Path("artifacts/third_dataset_shift_audit")
DATASET_ORDER = (
    "NF-UNSW-NB15 source validation",
    "NF-ToN-IoT",
    "NF-BoT-IoT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-validation-dir", type=Path, default=SOURCE_VALIDATION_DIR
    )
    parser.add_argument(
        "--third-validation-dir", type=Path, default=THIRD_VALIDATION_DIR
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--previous-shift-dir", type=Path, default=PREVIOUS_SHIFT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_and_verify(
    source_validation_dir: Path,
    third_validation_dir: Path,
    protocol_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    source_summary = read_json(source_validation_dir / "summary.json")
    third_summary = read_json(third_validation_dir / "summary.json")
    protocol = read_json(protocol_path)

    provenance = source_summary["data_provenance"]
    if provenance["source_row_limit"] is not None:
        raise ValueError("Third shift audit requires the full source run")
    if source_summary["experiment_scope"]["external_data_used_for_tuning"]:
        raise ValueError("Source-validation manifest reports external-data tuning")
    if third_summary["model_or_threshold_changes"] or third_summary["external_tuning"]:
        raise ValueError("Third-validation manifest reports a prohibited change")
    if third_summary["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Third-validation protocol ID does not match")
    if third_summary["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Third-validation protocol hash does not match")

    source_path = Path(provenance["source_path"])
    third_path = Path(third_summary["dataset"]["path"])
    if sha256_file(source_path) != provenance["source_sha256"]:
        raise ValueError("NF-UNSW-NB15 hash does not match the source manifest")
    if sha256_file(third_path) != third_summary["dataset"]["sha256"]:
        raise ValueError("NF-BoT-IoT hash does not match the third-validation manifest")

    source = load_dataset(source_path, row_limit=None)
    third = load_dataset(third_path, row_limit=None)
    if len(source) != provenance["source_rows"]:
        raise ValueError("NF-UNSW-NB15 row count does not match the source manifest")
    if len(third) != third_summary["dataset"]["rows"]:
        raise ValueError("NF-BoT-IoT row count does not match the third manifest")

    source_groups = exact_feature_groups(source)
    source_folds, _ = balanced_group_folds(source_groups, source[TARGET])
    source_validation = source.loc[source_folds.eq(4)].copy()
    expected_rows = next(
        row["rows"]
        for row in source_summary["split_integrity"]["splits"]
        if row["split"] == "source_validation"
    )
    if len(source_validation) != expected_rows:
        raise ValueError("Reconstructed source-validation row count does not match")

    models: dict[str, Any] = {}
    for model_name, record in source_summary["models"].items():
        model_path = source_validation_dir / record["artifact"]
        frozen = third_summary["frozen_models"][model_name]
        model_hash = sha256_file(model_path)
        if model_hash != frozen["sha256"]:
            raise ValueError(f"Frozen {model_name} hash does not match")
        if not np.isclose(float(record["threshold"]), float(frozen["threshold"])):
            raise ValueError(f"Frozen {model_name} threshold does not match")
        models[model_name] = {
            "pipeline": joblib.load(model_path),
            "threshold": float(frozen["threshold"]),
            "path": model_path,
            "sha256": model_hash,
        }
    return source, source_validation, third, models, source_summary, third_summary


def labelled_score_rows(score_shift: pd.DataFrame, external_name: str) -> pd.DataFrame:
    labelled = score_shift.copy()
    labelled["dataset"] = labelled["dataset"].replace(
        {
            "source_validation": "NF-UNSW-NB15 source validation",
            "external": external_name,
        }
    )
    labelled["attack_minus_normal_mean_probability"] = (
        labelled["attack_mean_probability"] - labelled["normal_mean_probability"]
    )
    labelled["score_ordering"] = np.where(
        labelled["attack_minus_normal_mean_probability"] > 0,
        "attack_above_benign",
        "attack_at_or_below_benign",
    )
    return labelled


def cross_external_score_comparison(
    current_score_shift: pd.DataFrame, previous_shift_dir: Path
) -> pd.DataFrame:
    current = labelled_score_rows(current_score_shift, "NF-BoT-IoT")
    previous_path = previous_shift_dir / "model_score_shift.csv"
    if not previous_path.exists():
        return current
    previous = labelled_score_rows(pd.read_csv(previous_path), "NF-ToN-IoT")
    previous = previous[previous["dataset"].eq("NF-ToN-IoT")]
    combined = pd.concat([current, previous], ignore_index=True)
    combined["dataset"] = pd.Categorical(
        combined["dataset"], categories=DATASET_ORDER, ordered=True
    )
    combined = combined.sort_values(["model", "dataset"])
    combined["dataset"] = combined["dataset"].astype("string")
    return combined.reset_index(drop=True)


def save_cross_external_score_plot(comparison: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = list(comparison["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, model_name in zip(axes, models):
        subset = comparison[comparison["model"].eq(model_name)].copy()
        x = np.arange(len(subset))
        width = 0.36
        axis.bar(
            x - width / 2,
            subset["normal_mean_probability"],
            width,
            label="Benign",
            color="#4c78a8",
        )
        axis.bar(
            x + width / 2,
            subset["attack_mean_probability"],
            width,
            label="Attack",
            color="#e45756",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(subset["dataset"], rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_ylabel("Mean predicted attack probability")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Frozen score ordering across source and external datasets")
    fig.tight_layout()
    fig.savefig(plot_dir / "cross_external_score_ordering.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        source,
        source_validation,
        third,
        models,
        source_summary,
        third_summary,
    ) = load_and_verify(
        args.source_validation_dir,
        args.third_validation_dir,
        args.protocol,
    )

    numeric_shift = numeric_distribution_shift(source, third)
    categorical_shift = categorical_distribution_shift(source, third)
    association_shift, association_detail = feature_label_association_drift(
        source, third
    )
    score_shift, probabilities = model_score_shift(source_validation, third, models)
    error_profiles = external_error_profiles(
        source, third, probabilities["external"], models
    )
    comparison = cross_external_score_comparison(
        score_shift, args.previous_shift_dir
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
    comparison.to_csv(
        args.output_dir / "cross_external_score_comparison.csv", index=False
    )
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
    save_cross_external_score_plot(comparison, args.output_dir)

    summary = {
        "scope": {
            "analysis": "Post-evaluation descriptive NF-BoT-IoT shift audit",
            "model_changes": False,
            "feature_changes": False,
            "threshold_changes": False,
            "external_tuning": False,
        },
        "verified_inputs": {
            "protocol_id": third_summary["protocol_id"],
            "protocol_sha256": third_summary["protocol_sha256"],
            "source_sha256": source_summary["data_provenance"]["source_sha256"],
            "third_dataset_sha256": third_summary["dataset"]["sha256"],
            "source_rows": int(len(source)),
            "source_validation_rows": int(len(source_validation)),
            "third_dataset_rows": int(len(third)),
            "model_sha256": {
                model_name: record["sha256"] for model_name, record in models.items()
            },
            "thresholds": {
                model_name: record["threshold"] for model_name, record in models.items()
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
        "model_score_shift": labelled_score_rows(
            score_shift, "NF-BoT-IoT"
        ).to_dict(orient="records"),
        "cross_external_score_comparison": comparison.to_dict(orient="records"),
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
