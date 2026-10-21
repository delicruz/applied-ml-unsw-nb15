"""Run the preregistered NF-CSE-CIC-IDS2018 sealed confirmation once.

This runner verifies the dataset, prior-development record, partition digest,
and frozen model hashes before scoring. It contains no model fitting, threshold
selection, feature selection, candidate selection, or confirmation sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_external_netflow_validation import (
    calculate_metrics,
    confidence_intervals,
    grouped_bootstrap,
    per_attack_recall,
    sha256_file,
)
from run_fresh_target_development import (
    ATTACK,
    MODEL_FEATURES,
    TARGET,
    assign_partitions,
    canonical_group_hashes,
    log1p_nonnegative,
    normalise_attack_counts,
    validate_header,
    verify_official_counts,
)


DEFAULT_PROTOCOL = Path("protocols/fresh_confirmation_protocol.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/fresh_confirmation_evaluation")
DATASET_GLOB = "data/external/**/NF-CSE-CIC-IDS2018.csv"
EXPECTED_PROTOCOL_ID = "nf-cse-cic-ids2018-v1-sealed-confirmation-2026-10-21"
CHUNK_SIZE = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def resolve_dataset_path(explicit: Path | None) -> Path:
    if explicit is not None:
        resolved = explicit.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    matches = sorted(Path().glob(DATASET_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one NF-CSE-CIC-IDS2018 CSV, found {len(matches)}"
        )
    return matches[0].resolve()


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Unexpected fresh-confirmation protocol ID")
    if protocol.get("status") != "PREREGISTERED_BEFORE_CONFIRMATION_ACCESS":
        raise ValueError("Confirmation protocol is not sealed")
    analysis = protocol["analysis"]
    if any(
        analysis[key]
        for key in ("model_refitting", "threshold_selection", "candidate_reselection")
    ):
        raise ValueError("Protocol unexpectedly permits adaptation")
    if int(analysis["bootstrap_iterations"]) < 100:
        raise ValueError("Bootstrap count is below the integrity minimum")
    if tuple(protocol["frozen_inputs"]["model_features"]) != MODEL_FEATURES:
        raise ValueError("Protocol features do not match the frozen runner")
    return protocol


def verify_locked_provenance(protocol: dict[str, Any]) -> None:
    lock = protocol["provenance_lock"]
    for path_key, hash_key in (
        ("development_protocol", "development_protocol_sha256"),
        ("development_summary", "development_summary_sha256"),
        ("development_runner", "development_runner_sha256"),
    ):
        path = Path(lock[path_key])
        if sha256_file(path) != lock[hash_key]:
            raise ValueError(f"Locked provenance hash mismatch: {path}")


def expose_legacy_pickle_globals() -> None:
    """Expose the development runner's transform for its __main__ pickle path."""
    setattr(sys.modules["__main__"], "log1p_nonnegative", log1p_nonnegative)


def load_frozen_models(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expose_legacy_pickle_globals()
    records = {
        "target_rf_all_features": protocol["frozen_inputs"]["selected_candidate"],
        "frozen_source_rf": protocol["frozen_inputs"]["comparison_baseline"],
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        if record["id"] != name:
            raise ValueError(f"Frozen model ID mismatch for {name}")
        path = Path(record["artifact"])
        if path.stat().st_size != int(record["artifact_bytes"]):
            raise ValueError(f"Frozen model byte size mismatch for {name}")
        observed_hash = sha256_file(path)
        if observed_hash != record["sha256"]:
            raise ValueError(f"Frozen model hash mismatch for {name}")
        loaded[name] = {
            "pipeline": joblib.load(path),
            "threshold": float(record["threshold"]),
            "sha256": observed_hash,
            "role": record["role"],
        }
    return loaded


def claim_one_time_execution(output_dir: Path, protocol: dict[str, Any]) -> None:
    """Create a durable marker and refuse a silent repeat or overwrite."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Confirmation output directory is not empty: {output_dir}. "
            "Do not rerun without documenting the prior attempt and obtaining "
            "an explicit decision."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "protocol_id": protocol["protocol_id"],
        "status": "confirmation_execution_started",
        "started_at": datetime.now(ZoneInfo("Australia/Adelaide")).isoformat(),
        "silent_retry_permitted": False,
    }
    with (output_dir / "execution_started.json").open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, allow_nan=False)


def load_confirmation_rows(
    dataset_path: Path,
    protocol: dict[str, Any],
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_header(dataset_path)
    if sha256_file(dataset_path) != protocol["dataset"]["expected_csv_sha256"]:
        raise ValueError("Dataset hash does not match the confirmation protocol")

    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    retained: list[pd.DataFrame] = []
    total_rows = 0
    confirmation_rows = 0
    label_counts = {0: 0, 1: 0}
    attack_counts: dict[str, int] = {}
    confirmation_digest = hashlib.sha256()

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
        keep = assign_partitions(first) == "sealed_confirmation"
        confirmation_rows += int(keep.sum())
        if keep.any():
            for first_word, second_word in zip(first[keep], second[keep]):
                confirmation_digest.update(int(first_word).to_bytes(8, "big"))
                confirmation_digest.update(int(second_word).to_bytes(8, "big"))
            selected = chunk.loc[keep, usecols].copy()
            selected["_group_key"] = [
                f"{int(a):016x}{int(b):016x}" for a, b in zip(first[keep], second[keep])
            ]
            retained.append(selected)
        print(
            f"confirmation_partition_progress chunk={chunk_index} rows={total_rows}",
            flush=True,
        )

    verify_official_counts(total_rows, label_counts, attack_counts, protocol)
    expected_rows = int(protocol["dataset"]["expected_confirmation_rows"])
    observed_digest = confirmation_digest.hexdigest()
    expected_digest = protocol["dataset"][
        "expected_confirmation_ordered_group_digest_sha256"
    ]
    if confirmation_rows != expected_rows:
        raise ValueError(
            f"Confirmation row mismatch: observed {confirmation_rows}, expected {expected_rows}"
        )
    if observed_digest != expected_digest:
        raise ValueError("Confirmation ordered group digest mismatch")

    frame = pd.concat(retained, ignore_index=True)
    integrity = {
        "official_counts_verified": True,
        "global_rows": total_rows,
        "global_label_counts": {str(key): value for key, value in label_counts.items()},
        "global_official_attack_counts": normalise_attack_counts(attack_counts),
        "confirmation_rows": confirmation_rows,
        "confirmation_groups": int(frame["_group_key"].nunique()),
        "confirmation_ordered_group_digest_sha256": observed_digest,
    }
    return frame, integrity


def paired_difference_samples(
    samples: pd.DataFrame,
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    pivot = samples.pivot(
        index=["dataset", "iteration", "metric"], columns="model", values="value"
    ).reset_index()
    required = {candidate, baseline}
    if not required.issubset(pivot.columns):
        raise ValueError("Bootstrap samples do not contain both frozen models")
    pivot["candidate_minus_baseline"] = pivot[candidate] - pivot[baseline]
    return pivot[
        ["dataset", "iteration", "metric", "candidate_minus_baseline"]
    ]


def paired_difference_intervals(samples: pd.DataFrame) -> pd.DataFrame:
    return (
        samples.groupby(["dataset", "metric"])["candidate_minus_baseline"]
        .agg(
            estimate="median",
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )


def apply_decision_rules(
    metrics: pd.DataFrame,
    paired_intervals: pd.DataFrame,
    attacks: pd.DataFrame,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    candidate = protocol["frozen_inputs"]["selected_candidate"]["id"]
    point = metrics[metrics["model"].eq(candidate)].set_index("metric")["value"]
    paired = paired_intervals.set_index("metric")
    aggregate_rule = protocol["decision_rules"]["aggregate_improvement_supported"][
        "requirements"
    ]
    aggregate_requirements = {
        "candidate_mcc_at_least_0_60": bool(
            point["mcc"] >= aggregate_rule["selected_candidate_mcc_at_least"]
        ),
        "candidate_balanced_accuracy_at_least_0_80": bool(
            point["balanced_accuracy"]
            >= aggregate_rule["selected_candidate_balanced_accuracy_at_least"]
        ),
        "candidate_average_precision_lift_above_1": bool(
            point["average_precision_lift"]
            > aggregate_rule["selected_candidate_average_precision_lift_above"]
        ),
        "paired_mcc_ci_lower_above_0": bool(
            paired.loc["mcc", "ci_lower"]
            > aggregate_rule["paired_mcc_difference_ci_lower_above"]
        ),
        "paired_balanced_accuracy_ci_lower_above_0": bool(
            paired.loc["balanced_accuracy", "ci_lower"]
            > aggregate_rule["paired_balanced_accuracy_difference_ci_lower_above"]
        ),
    }
    aggregate_supported = all(aggregate_requirements.values())

    coverage_rule = protocol["decision_rules"]["broad_attack_coverage_supported"][
        "requirements"
    ]
    minimum_support = int(
        protocol["analysis"]["attack_reporting"]["minimum_support_for_coverage_rule"]
    )
    supported_attacks = attacks[
        attacks["model"].eq(candidate)
        & attacks["positive_rows"].ge(minimum_support)
    ]
    minimum_supported_recall = (
        float(supported_attacks["recall"].min()) if len(supported_attacks) else None
    )
    coverage_requirements = {
        "candidate_false_positive_rate_at_most_0_10": bool(
            point["false_positive_rate"]
            <= coverage_rule["selected_candidate_false_positive_rate_at_most"]
        ),
        "every_supported_attack_recall_at_least_0_50": bool(
            minimum_supported_recall is not None
            and minimum_supported_recall
            >= coverage_rule[
                "selected_candidate_recall_for_every_attack_with_at_least_100_rows_at_least"
            ]
        ),
    }
    coverage_supported = all(coverage_requirements.values())

    labels = protocol["decision_rules"]["verdict_labels"]
    if not aggregate_supported:
        verdict = labels["aggregate_fail"]
    elif coverage_supported:
        verdict = labels["aggregate_and_coverage_pass"]
    else:
        verdict = labels["aggregate_pass_coverage_fail"]
    return {
        "aggregate_improvement_supported": aggregate_supported,
        "aggregate_requirements": aggregate_requirements,
        "broad_attack_coverage_supported": coverage_supported,
        "coverage_requirements": coverage_requirements,
        "supported_attack_count": int(len(supported_attacks)),
        "minimum_supported_attack_recall": minimum_supported_recall,
        "verdict": verdict,
        "deployment_claim_permitted": False,
    }


def save_plots(
    metrics: pd.DataFrame,
    attacks: pd.DataFrame,
    protocol: dict[str, Any],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = ["frozen_source_rf", "target_rf_all_features"]
    measures = ["average_precision", "mcc", "balanced_accuracy"]
    table = metrics[
        metrics["model"].isin(models) & metrics["metric"].isin(measures)
    ].pivot(index="model", columns="metric", values="value").loc[models]
    axis = table[measures].plot.bar(figsize=(10, 5), rot=0)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Metric value")
    axis.set_title("Sealed-confirmation frozen-model comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(plot_dir / "confirmation_metric_comparison.png", dpi=180)
    plt.close(axis.figure)

    candidate = protocol["frozen_inputs"]["selected_candidate"]["id"]
    minimum_support = int(
        protocol["analysis"]["attack_reporting"]["minimum_support_for_coverage_rule"]
    )
    supported = attacks[
        attacks["model"].eq(candidate)
        & attacks["positive_rows"].ge(minimum_support)
    ].sort_values("recall")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.barh(supported["attack"], supported["recall"], color="#4c78a8")
    axis.axvline(0.50, color="#c44e52", linestyle="--", label="coverage rule")
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_title("Selected-model recall for supported attack labels")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "selected_attack_recall.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    verify_locked_provenance(protocol)
    dataset_path = resolve_dataset_path(args.dataset_path)
    models = load_frozen_models(protocol)
    if sha256_file(dataset_path) != protocol["dataset"]["expected_csv_sha256"]:
        raise ValueError("Dataset hash does not match the confirmation protocol")
    claim_one_time_execution(args.output_dir, protocol)
    frame, integrity = load_confirmation_rows(dataset_path, protocol, args.chunk_size)

    labels = frame[TARGET].to_numpy()
    groups = frame["_group_key"]
    probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    metric_rows: list[dict[str, Any]] = []
    attack_tables: list[pd.DataFrame] = []
    for model_name, record in models.items():
        values = record["pipeline"].predict_proba(frame.loc[:, MODEL_FEATURES])[:, 1]
        probabilities[model_name] = values
        thresholds[model_name] = record["threshold"]
        result = calculate_metrics(labels, values, record["threshold"])
        result["false_positive_rate"] = 1.0 - float(result["specificity"])
        for metric, value in result.items():
            metric_rows.append({"model": model_name, "metric": metric, "value": value})
        attack_tables.append(
            per_attack_recall(
                "sealed_confirmation",
                frame[ATTACK],
                labels,
                model_name,
                values,
                record["threshold"],
            )
        )

    metrics = pd.DataFrame(metric_rows)
    attacks = pd.concat(attack_tables, ignore_index=True)
    bootstrap_samples = grouped_bootstrap(
        "sealed_confirmation",
        labels,
        groups,
        probabilities,
        thresholds,
        int(protocol["analysis"]["bootstrap_iterations"]),
    )
    intervals = confidence_intervals(bootstrap_samples)
    paired_samples = paired_difference_samples(
        bootstrap_samples, "target_rf_all_features", "frozen_source_rf"
    )
    paired_intervals = paired_difference_intervals(paired_samples)
    decision = apply_decision_rules(metrics, paired_intervals, attacks, protocol)

    metrics.to_csv(args.output_dir / "metrics_long.csv", index=False)
    attacks.to_csv(args.output_dir / "per_attack_recall.csv", index=False)
    bootstrap_samples.to_csv(
        args.output_dir / "group_bootstrap_samples.csv", index=False
    )
    intervals.to_csv(args.output_dir / "group_bootstrap_intervals.csv", index=False)
    paired_samples.to_csv(
        args.output_dir / "paired_bootstrap_differences.csv", index=False
    )
    paired_intervals.to_csv(
        args.output_dir / "paired_bootstrap_intervals.csv", index=False
    )
    save_plots(metrics, attacks, protocol, args.output_dir)

    summary = {
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(ZoneInfo("Australia/Adelaide")).date().isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "sealed_confirmation_result_v1",
        },
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "integrity": integrity,
        "frozen_models": {
            name: {
                "sha256": record["sha256"],
                "threshold": record["threshold"],
                "role": record["role"],
            }
            for name, record in models.items()
        },
        "decision": decision,
        "confirmation_opened": True,
        "one_time_evaluation_completed": True,
        "model_or_threshold_changes": False,
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
