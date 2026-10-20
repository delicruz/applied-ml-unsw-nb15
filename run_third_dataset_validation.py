"""Execute the preregistered untouched NF-BoT-IoT validation exactly once.

The protocol must be committed before the dataset is downloaded or opened.
This runner verifies the protocol, official dataset counts, and frozen model
hashes; scores the existing models without refitting; and applies the sealed
decision rule. It contains no training or threshold-selection path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from run_external_netflow_validation import (
    ATTACK,
    MODEL_FEATURES,
    TARGET,
    calculate_metrics,
    confidence_intervals,
    exact_feature_groups,
    grouped_bootstrap,
    load_dataset,
    log1p_nonnegative,
    per_attack_recall,
    sha256_file,
)


DEFAULT_PROTOCOL = Path("protocols/third_dataset_preregistration.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/third_dataset_validation")
EXPECTED_PROTOCOL_ID = "nf-bot-iot-v1-third-holdout-2026-10-20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Unexpected third-dataset protocol ID")
    if protocol.get("status") != "PREREGISTERED_BEFORE_DATA_ACCESS":
        raise ValueError("Protocol is not marked as preregistered before data access")
    analysis = protocol["analysis"]
    if analysis["external_threshold_tuning"] or analysis["model_refitting"]:
        raise ValueError("Protocol unexpectedly permits tuning or refitting")
    if analysis["bootstrap_iterations"] < 100:
        raise ValueError("Protocol bootstrap count is below the integrity minimum")
    if tuple(protocol["frozen_inputs"]["model_features"]) != MODEL_FEATURES:
        raise ValueError("Protocol model features do not match the frozen runner")
    return protocol


def verify_official_counts(frame: pd.DataFrame, protocol: dict[str, Any]) -> None:
    dataset = protocol["dataset"]
    checks = {
        "rows": (len(frame), dataset["expected_rows"]),
        "positive_rows": (int(frame[TARGET].sum()), dataset["expected_positive_rows"]),
        "negative_rows": (
            int(frame[TARGET].eq(0).sum()),
            dataset["expected_negative_rows"],
        ),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in checks.items()
        if observed != expected
    }
    observed_attacks = frame[ATTACK].value_counts().to_dict()
    expected_attacks = dataset["expected_attack_counts"]
    if observed_attacks != expected_attacks:
        mismatches["attack_counts"] = {
            "observed": observed_attacks,
            "expected": expected_attacks,
        }
    if mismatches:
        raise ValueError(f"Official NF-BoT-IoT count verification failed: {mismatches}")


def load_frozen_models(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for model_name, record in protocol["frozen_inputs"]["models"].items():
        path = Path(record["artifact"])
        observed_hash = sha256_file(path)
        if observed_hash != record["sha256"]:
            raise ValueError(f"Frozen {model_name} hash does not match protocol")
        models[model_name] = {
            "pipeline": joblib.load(path),
            "threshold": float(record["threshold"]),
            "sha256": observed_hash,
            "role": record["role"],
        }
    return models


def apply_decision_rule(
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    model_name = protocol["analysis"]["primary_model"]
    point = metrics[metrics["model"].eq(model_name)].set_index("metric")["value"]
    mcc_interval = intervals[
        intervals["model"].eq(model_name) & intervals["metric"].eq("mcc")
    ]
    if len(mcc_interval) != 1:
        raise ValueError("Expected exactly one primary-model MCC interval")
    mcc_lower = float(mcc_interval.iloc[0]["ci_lower"])
    requirements = {
        "mcc_ci_lower_gt_zero": bool(mcc_lower > 0),
        "balanced_accuracy_gt_half": bool(point["balanced_accuracy"] > 0.50),
        "average_precision_lift_gt_one": bool(
            point["average_precision_lift"] > 1.00
        ),
    }
    return {
        "primary_model": model_name,
        "primary_metric": protocol["analysis"]["primary_metric"],
        "mcc_ci_lower": mcc_lower,
        "requirements": requirements,
        "verdict": (
            "useful transfer supported"
            if all(requirements.values())
            else "useful transfer not supported"
        ),
        "deployment_claim_permitted": False,
    }


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    frame = load_dataset(dataset_path, row_limit=None)
    verify_official_counts(frame, protocol)
    models = load_frozen_models(protocol)

    labels = frame[TARGET].to_numpy()
    groups = exact_feature_groups(frame)
    probabilities: dict[str, np.ndarray] = {}
    metric_rows = []
    attack_tables = []
    thresholds = {}
    for model_name, record in models.items():
        values = record["pipeline"].predict_proba(frame.loc[:, MODEL_FEATURES])[:, 1]
        probabilities[model_name] = values
        thresholds[model_name] = record["threshold"]
        result = calculate_metrics(labels, values, record["threshold"])
        for metric, value in result.items():
            metric_rows.append({"model": model_name, "metric": metric, "value": value})
        attack_tables.append(
            per_attack_recall(
                "NF-BoT-IoT",
                frame[ATTACK],
                labels,
                model_name,
                values,
                record["threshold"],
            )
        )

    metrics = pd.DataFrame(metric_rows)
    bootstrap_samples = grouped_bootstrap(
        "NF-BoT-IoT",
        labels,
        groups,
        probabilities,
        thresholds,
        int(protocol["analysis"]["bootstrap_iterations"]),
    )
    intervals = confidence_intervals(bootstrap_samples)
    decision = apply_decision_rule(metrics, intervals, protocol)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics_long.csv", index=False)
    pd.concat(attack_tables, ignore_index=True).to_csv(
        args.output_dir / "per_attack_recall.csv", index=False
    )
    bootstrap_samples.to_csv(
        args.output_dir / "group_bootstrap_samples.csv", index=False
    )
    intervals.to_csv(args.output_dir / "group_bootstrap_intervals.csv", index=False)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "rows": int(len(frame)),
            "positive_rows": int(labels.sum()),
            "negative_rows": int((labels == 0).sum()),
            "exact_feature_groups": int(groups.nunique()),
        },
        "frozen_models": {
            name: {
                "sha256": record["sha256"],
                "threshold": record["threshold"],
                "role": record["role"],
            }
            for name, record in models.items()
        },
        "decision": decision,
        "model_or_threshold_changes": False,
        "external_tuning": False,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
