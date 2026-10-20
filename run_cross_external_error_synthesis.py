"""Synthesize aligned feature-bin errors across two external datasets.

The inputs are the saved post-evaluation NF-ToN-IoT and NF-BoT-IoT error
profiles. Because both audits used NF-UNSW-NB15-derived numeric bins, matching
feature/bin rows can be compared directly. The reporting rules are descriptive:

* consistently low recall: at least 100 attacks in each external dataset and
  attack recall <= 0.20 in both;
* consistently high false-positive rate: at least 100 benign rows in each
  external dataset and false-positive rate >= 0.20 in both.

These post-hoc flags generate hypotheses only. They do not select features,
change a model or threshold, or authorise tuning on either external dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TON_AUDIT_DIR = Path("artifacts/external_shift_audit")
BOT_AUDIT_DIR = Path("artifacts/third_dataset_shift_audit")
OUTPUT_DIR = Path("artifacts/cross_external_error_synthesis")
MIN_CLASS_SUPPORT = 100
LOW_RECALL_THRESHOLD = 0.20
HIGH_FALSE_POSITIVE_RATE_THRESHOLD = 0.20
KEY_COLUMNS = ["feature", "bin", "model"]
REQUIRED_COLUMNS = {
    *KEY_COLUMNS,
    "rows",
    "normal_rows",
    "attack_rows",
    "false_positive_rate",
    "attack_recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ton-audit-dir", type=Path, default=TON_AUDIT_DIR)
    parser.add_argument("--bot-audit-dir", type=Path, default=BOT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_profile(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        dtype={"feature": "string", "bin": "string", "model": "string"},
    )
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing error-profile columns: {sorted(missing)}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Error profile contains duplicate feature/bin/model rows")
    return frame


def verify_audit_summaries(
    ton_summary: dict[str, Any], bot_summary: dict[str, Any]
) -> dict[str, Any]:
    for name, summary in (("NF-ToN-IoT", ton_summary), ("NF-BoT-IoT", bot_summary)):
        scope = summary["scope"]
        prohibited = (
            scope["model_changes"],
            scope["feature_changes"],
            scope["threshold_changes"],
            scope["external_tuning"],
        )
        if any(prohibited):
            raise ValueError(f"{name} audit reports a prohibited change")
    ton_inputs = ton_summary["verified_inputs"]
    bot_inputs = bot_summary["verified_inputs"]
    if ton_inputs["source_rows"] != bot_inputs["source_rows"]:
        raise ValueError("External audits do not use the same source row count")
    if ton_inputs["source_validation_rows"] != bot_inputs["source_validation_rows"]:
        raise ValueError("External audits do not use the same source-validation fold")
    if ton_inputs["model_sha256"] != bot_inputs["model_sha256"]:
        raise ValueError("External audits do not use identical frozen models")
    return {
        "source_rows": ton_inputs["source_rows"],
        "source_validation_rows": ton_inputs["source_validation_rows"],
        "model_sha256": ton_inputs["model_sha256"],
        "nf_ton_external_rows": ton_inputs["external_rows"],
        "nf_bot_external_rows": bot_inputs["third_dataset_rows"],
        "nf_bot_protocol_id": bot_inputs["protocol_id"],
        "nf_bot_protocol_sha256": bot_inputs["protocol_sha256"],
    }


def align_error_profiles(ton: pd.DataFrame, bot: pd.DataFrame) -> pd.DataFrame:
    aligned = ton.merge(
        bot,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_ton", "_bot"),
        validate="one_to_one",
    )
    aligned["attack_supported_both"] = (
        aligned["attack_rows_ton"].ge(MIN_CLASS_SUPPORT)
        & aligned["attack_rows_bot"].ge(MIN_CLASS_SUPPORT)
        & aligned["attack_recall_ton"].notna()
        & aligned["attack_recall_bot"].notna()
    )
    aligned["normal_supported_both"] = (
        aligned["normal_rows_ton"].ge(MIN_CLASS_SUPPORT)
        & aligned["normal_rows_bot"].ge(MIN_CLASS_SUPPORT)
        & aligned["false_positive_rate_ton"].notna()
        & aligned["false_positive_rate_bot"].notna()
    )
    aligned["consistent_low_recall"] = (
        aligned["attack_supported_both"]
        & aligned["attack_recall_ton"].le(LOW_RECALL_THRESHOLD)
        & aligned["attack_recall_bot"].le(LOW_RECALL_THRESHOLD)
    )
    aligned["consistent_high_false_positive_rate"] = (
        aligned["normal_supported_both"]
        & aligned["false_positive_rate_ton"].ge(HIGH_FALSE_POSITIVE_RATE_THRESHOLD)
        & aligned["false_positive_rate_bot"].ge(HIGH_FALSE_POSITIVE_RATE_THRESHOLD)
    )
    aligned["worst_external_attack_recall"] = aligned[
        ["attack_recall_ton", "attack_recall_bot"]
    ].max(axis=1)
    aligned["minimum_external_false_positive_rate"] = aligned[
        ["false_positive_rate_ton", "false_positive_rate_bot"]
    ].min(axis=1)
    aligned["attack_recall_absolute_gap"] = (
        aligned["attack_recall_ton"] - aligned["attack_recall_bot"]
    ).abs()
    aligned["false_positive_rate_absolute_gap"] = (
        aligned["false_positive_rate_ton"] - aligned["false_positive_rate_bot"]
    ).abs()
    return aligned.sort_values(KEY_COLUMNS).reset_index(drop=True)


def feature_consistency_summary(
    aligned: pd.DataFrame, ton: pd.DataFrame, bot: pd.DataFrame
) -> pd.DataFrame:
    denominator_rows = []
    for dataset, frame in (("ton", ton), ("bot", bot)):
        grouped = frame.groupby(["model", "feature"], as_index=False).agg(
            **{
                f"total_attack_rows_{dataset}": ("attack_rows", "sum"),
                f"total_normal_rows_{dataset}": ("normal_rows", "sum"),
            }
        )
        denominator_rows.append(grouped)
    denominators = denominator_rows[0].merge(
        denominator_rows[1], on=["model", "feature"], validate="one_to_one"
    )

    rows = []
    for (model, feature), group in aligned.groupby(["model", "feature"]):
        low = group[group["consistent_low_recall"]]
        high = group[group["consistent_high_false_positive_rate"]]
        rows.append(
            {
                "model": model,
                "feature": feature,
                "aligned_bins": int(len(group)),
                "attack_supported_bins": int(group["attack_supported_both"].sum()),
                "consistent_low_recall_bins": int(len(low)),
                "ton_attack_rows_in_consistent_low_recall_bins": int(
                    low["attack_rows_ton"].sum()
                ),
                "bot_attack_rows_in_consistent_low_recall_bins": int(
                    low["attack_rows_bot"].sum()
                ),
                "normal_supported_bins": int(group["normal_supported_both"].sum()),
                "consistent_high_false_positive_bins": int(len(high)),
                "ton_normal_rows_in_consistent_high_false_positive_bins": int(
                    high["normal_rows_ton"].sum()
                ),
                "bot_normal_rows_in_consistent_high_false_positive_bins": int(
                    high["normal_rows_bot"].sum()
                ),
            }
        )
    summary = pd.DataFrame(rows).merge(
        denominators, on=["model", "feature"], validate="one_to_one"
    )
    for dataset in ("ton", "bot"):
        summary[f"{dataset}_attack_share_in_consistent_low_recall_bins"] = (
            summary[f"{dataset}_attack_rows_in_consistent_low_recall_bins"]
            / summary[f"total_attack_rows_{dataset}"]
        )
        summary[f"{dataset}_normal_share_in_consistent_high_false_positive_bins"] = (
            summary[f"{dataset}_normal_rows_in_consistent_high_false_positive_bins"]
            / summary[f"total_normal_rows_{dataset}"]
        )
    return summary.sort_values(["model", "feature"]).reset_index(drop=True)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records, representing missing metrics as ``null``."""
    safe = frame.astype(object).where(pd.notna(frame), None)
    return safe.to_dict(orient="records")


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(
        len(models), 2, figsize=(14, 4.8 * len(models)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row_index, model in enumerate(models):
        subset = summary[summary["model"].eq(model)].copy()
        x = np.arange(len(subset))
        width = 0.36
        attack_axis = axes[row_index, 0]
        attack_axis.bar(
            x - width / 2,
            subset["ton_attack_share_in_consistent_low_recall_bins"],
            width,
            label="NF-ToN-IoT",
            color="#4c78a8",
        )
        attack_axis.bar(
            x + width / 2,
            subset["bot_attack_share_in_consistent_low_recall_bins"],
            width,
            label="NF-BoT-IoT",
            color="#e45756",
        )
        model_label = model.replace("_", " ").title()
        attack_axis.set_title(f"{model_label}: recurring low-recall attack bins")
        attack_axis.set_ylabel("Share of all attacks")
        attack_axis.set_ylim(0, 1)
        attack_axis.set_xticks(x)
        attack_axis.set_xticklabels(subset["feature"], rotation=35, ha="right")
        attack_axis.grid(axis="y", alpha=0.25)
        attack_axis.legend()

        normal_axis = axes[row_index, 1]
        normal_axis.bar(
            x - width / 2,
            subset["ton_normal_share_in_consistent_high_false_positive_bins"],
            width,
            label="NF-ToN-IoT",
            color="#4c78a8",
        )
        normal_axis.bar(
            x + width / 2,
            subset["bot_normal_share_in_consistent_high_false_positive_bins"],
            width,
            label="NF-BoT-IoT",
            color="#e45756",
        )
        normal_axis.set_title(f"{model_label}: recurring high-FPR benign bins")
        normal_axis.set_ylabel("Share of all benign rows")
        normal_axis.set_ylim(0, 1)
        normal_axis.set_xticks(x)
        normal_axis.set_xticklabels(subset["feature"], rotation=35, ha="right")
        normal_axis.grid(axis="y", alpha=0.25)
        normal_axis.legend()
    fig.suptitle("Cross-external consistency of feature-bin failures")
    fig.savefig(plot_dir / "consistent_feature_bin_failures.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ton_summary = read_json(args.ton_audit_dir / "summary.json")
    bot_summary = read_json(args.bot_audit_dir / "summary.json")
    verified_inputs = verify_audit_summaries(ton_summary, bot_summary)
    ton = load_profile(
        args.ton_audit_dir / "external_error_profiles_by_feature_bin.csv"
    )
    bot = load_profile(
        args.bot_audit_dir / "external_error_profiles_by_feature_bin.csv"
    )
    if set(ton["model"]) != set(bot["model"]):
        raise ValueError("External error profiles do not contain the same models")

    aligned = align_error_profiles(ton, bot)
    feature_summary = feature_consistency_summary(aligned, ton, bot)
    low_recall = aligned[aligned["consistent_low_recall"]].sort_values(
        ["model", "worst_external_attack_recall", "feature", "bin"]
    )
    high_false_positive = aligned[
        aligned["consistent_high_false_positive_rate"]
    ].sort_values(
        ["model", "minimum_external_false_positive_rate", "feature", "bin"],
        ascending=[True, False, True, True],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.output_dir / "aligned_feature_bin_errors.csv", index=False)
    low_recall.to_csv(args.output_dir / "consistent_low_recall_bins.csv", index=False)
    high_false_positive.to_csv(
        args.output_dir / "consistent_high_false_positive_bins.csv", index=False
    )
    feature_summary.to_csv(
        args.output_dir / "feature_consistency_summary.csv", index=False
    )
    save_plots(feature_summary, args.output_dir)

    result = {
        "scope": {
            "analysis": "Post-hoc cross-external feature-bin error synthesis",
            "model_changes": False,
            "feature_changes": False,
            "threshold_changes": False,
            "external_tuning": False,
            "rules_are_model_selection_criteria": False,
        },
        "reporting_rules": {
            "minimum_class_support_each_dataset": MIN_CLASS_SUPPORT,
            "consistent_low_recall_maximum_each_dataset": LOW_RECALL_THRESHOLD,
            "consistent_high_false_positive_rate_minimum_each_dataset": (
                HIGH_FALSE_POSITIVE_RATE_THRESHOLD
            ),
        },
        "verified_inputs": verified_inputs,
        "aligned_rows": int(len(aligned)),
        "consistent_low_recall_rows": int(len(low_recall)),
        "consistent_high_false_positive_rows": int(len(high_false_positive)),
        "feature_consistency_summary": json_records(feature_summary),
        "strongest_consistent_low_recall_bins": json_records(low_recall.head(20)),
        "strongest_consistent_high_false_positive_bins": json_records(
            high_false_positive.head(20)
        ),
        "interpretation_boundary": (
            "The aligned-bin findings are post-hoc descriptions of two already-opened "
            "external datasets. They may motivate hypotheses for a separately designated "
            "development dataset, but cannot validate a correction or authorise tuning on "
            "NF-ToN-IoT or NF-BoT-IoT."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
