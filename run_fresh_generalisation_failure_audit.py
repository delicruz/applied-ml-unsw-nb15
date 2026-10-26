"""Audit the failed NF-CSE-CIC-IDS2018 sealed confirmation post hoc.

This runner reconstructs the registered target-validation sample and full
sealed-confirmation partition, verifies the frozen target Random Forest's saved
metrics, and describes attack-mix, feature, feature/label, score, and repeated-
group differences. It has no fitting, threshold selection, candidate selection,
or deployment-decision path. Findings are diagnostic hypotheses only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_external_netflow_validation import calculate_metrics, sha256_file
from run_external_shift_audit import (
    binned_feature,
    categorical_distribution_shift,
    feature_label_association_drift,
    numeric_distribution_shift,
)
from run_fresh_confirmation_evaluation import load_frozen_models, load_protocol
from run_fresh_target_development import (
    ATTACK,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    assign_partitions,
    canonical_group_hashes,
    group_sample_mask,
    validate_header,
)


AUDIT_PROTOCOL = Path("protocols/fresh_generalisation_failure_audit.json")
OUTPUT_DIR = Path("artifacts/fresh_generalisation_failure_audit")
DATASET_GLOB = "data/external/**/NF-CSE-CIC-IDS2018.csv"
CHUNK_SIZE = 250_000
MODEL_ID = "target_rf_all_features"
EXPECTED_PROTOCOL_ID = "nf-cse-cic-ids2018-v1-posthoc-generalisation-audit-2026-10-26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--protocol", type=Path, default=AUDIT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def resolve_dataset_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted(Path().glob(DATASET_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one NF-CSE-CIC-IDS2018 CSV, found {len(matches)}"
        )
    return matches[0].resolve()


def load_audit_protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Unexpected generalisation-audit protocol ID")
    if protocol.get("status") != "REGISTERED_POSTHOC_SCOPE_BEFORE_AUDIT_RUN":
        raise ValueError("Generalisation-audit scope is not frozen")
    prohibited = protocol["scope_boundaries"]
    if any(
        prohibited[key]
        for key in (
            "model_fitting",
            "feature_selection",
            "threshold_selection",
            "candidate_selection",
            "deployment_claim",
        )
    ):
        raise ValueError("Audit protocol unexpectedly permits adaptation")
    runner = protocol["runner"]
    if sha256_file(Path(runner["path"])) != runner["sha256"]:
        raise ValueError("Audit runner hash does not match the frozen protocol")
    for record in protocol["locked_inputs"].values():
        source = Path(record["path"])
        if sha256_file(source) != record["sha256"]:
            raise ValueError(f"Locked input hash mismatch: {source}")
    return protocol


def load_registered_partitions(
    dataset_path: Path,
    protocol: dict[str, Any],
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_header(dataset_path)
    expected = protocol["expected_integrity"]
    observed_dataset_hash = sha256_file(dataset_path)
    if observed_dataset_hash != expected["dataset_sha256"]:
        raise ValueError("Dataset hash does not match the audit protocol")

    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    validation_frames: list[pd.DataFrame] = []
    confirmation_frames: list[pd.DataFrame] = []
    confirmation_digest = hashlib.sha256()
    total_rows = 0

    for chunk_index, chunk in enumerate(
        pd.read_csv(dataset_path, usecols=usecols, chunksize=chunk_size), start=1
    ):
        for feature in MODEL_FEATURES:
            chunk[feature] = pd.to_numeric(chunk[feature], errors="coerce")
        chunk[TARGET] = pd.to_numeric(chunk[TARGET], errors="raise").astype("int8")
        chunk[ATTACK] = chunk[ATTACK].fillna("<MISSING>").astype(str)
        total_rows += len(chunk)

        first, second = canonical_group_hashes(chunk)
        assignments = assign_partitions(first)
        validation_keep = assignments == "target_validation"
        validation_keep &= group_sample_mask("target_validation", second)
        confirmation_keep = assignments == "sealed_confirmation"

        if validation_keep.any():
            selected = chunk.loc[validation_keep, usecols].copy()
            selected["_group_key"] = [
                f"{int(a):016x}{int(b):016x}"
                for a, b in zip(first[validation_keep], second[validation_keep])
            ]
            validation_frames.append(selected)
        if confirmation_keep.any():
            for first_word, second_word in zip(
                first[confirmation_keep], second[confirmation_keep]
            ):
                confirmation_digest.update(int(first_word).to_bytes(8, "big"))
                confirmation_digest.update(int(second_word).to_bytes(8, "big"))
            selected = chunk.loc[confirmation_keep, usecols].copy()
            selected["_group_key"] = [
                f"{int(a):016x}{int(b):016x}"
                for a, b in zip(first[confirmation_keep], second[confirmation_keep])
            ]
            confirmation_frames.append(selected)
        print(f"audit_partition_progress chunk={chunk_index} rows={total_rows}", flush=True)

    validation = pd.concat(validation_frames, ignore_index=True)
    confirmation = pd.concat(confirmation_frames, ignore_index=True)
    observed_digest = confirmation_digest.hexdigest()
    checks = {
        "global_rows": total_rows,
        "target_validation_rows": int(len(validation)),
        "target_validation_groups": int(validation["_group_key"].nunique()),
        "sealed_confirmation_rows": int(len(confirmation)),
        "sealed_confirmation_groups": int(confirmation["_group_key"].nunique()),
        "sealed_confirmation_ordered_group_digest_sha256": observed_digest,
    }
    for key, value in checks.items():
        if value != expected[key]:
            raise ValueError(
                f"Audit partition integrity mismatch for {key}: {value!r}"
            )
    checks["dataset_sha256"] = observed_dataset_hash
    return validation, confirmation, checks


def verify_saved_metrics(
    validation: pd.DataFrame,
    confirmation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    threshold: float,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, frame in (
        ("target_validation", validation),
        ("sealed_confirmation", confirmation),
    ):
        metrics = calculate_metrics(
            frame[TARGET].to_numpy(), probabilities[scope], threshold
        )
        metrics["false_positive_rate"] = 1.0 - float(metrics["specificity"])
        expected = protocol["expected_metrics"][scope]
        for metric, expected_value in expected.items():
            observed = float(metrics[metric])
            if not np.isclose(observed, expected_value, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"Saved {scope} {metric} was not reproduced: {observed}"
                )
        rows.extend(
            {
                "scope": scope,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        )
    return pd.DataFrame(rows)


def attack_comparison(
    validation: pd.DataFrame,
    confirmation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    def summarise(scope: str, frame: pd.DataFrame) -> pd.DataFrame:
        prediction = probabilities[scope] >= threshold
        working = pd.DataFrame(
            {
                "attack": frame[ATTACK].to_numpy(),
                "label": frame[TARGET].to_numpy(),
                "prediction": prediction,
            }
        )
        total_positive = int(working["label"].sum())
        rows = []
        for attack, group in working.groupby("attack", dropna=False):
            positives = group["label"].eq(1)
            positive_rows = int(positives.sum())
            rows.append(
                {
                    "attack": str(attack),
                    f"{scope}_rows": int(len(group)),
                    f"{scope}_positive_rows": positive_rows,
                    f"{scope}_attack_share": (
                        positive_rows / total_positive if total_positive else np.nan
                    ),
                    f"{scope}_recall": (
                        float(group.loc[positives, "prediction"].mean())
                        if positive_rows
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows)

    result = summarise("target_validation", validation).merge(
        summarise("sealed_confirmation", confirmation), on="attack", how="outer"
    )
    for column in result.columns:
        if column.endswith("_rows"):
            result[column] = result[column].fillna(0).astype(int)
        elif column.endswith("_attack_share"):
            result[column] = result[column].fillna(0.0)
    result["attack_share_change"] = (
        result["sealed_confirmation_attack_share"]
        - result["target_validation_attack_share"]
    )
    result["recall_change"] = (
        result["sealed_confirmation_recall"] - result["target_validation_recall"]
    )
    result["confirmation_coverage_eligible"] = result[
        "sealed_confirmation_positive_rows"
    ].ge(100)
    return result.sort_values(
        "attack_share_change", key=lambda values: values.abs(), ascending=False
    )


GROUP_SIZE_BINS = [-np.inf, 1, 9, 99, 999, np.inf]
GROUP_SIZE_LABELS = ["1", "2-9", "10-99", "100-999", "1000+"]


def group_size_analysis(
    scope: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = pd.DataFrame(
        {
            "group": frame["_group_key"].to_numpy(),
            "label": frame[TARGET].to_numpy(),
            "prediction": probabilities >= threshold,
        }
    )
    working["true_positive"] = working["label"].eq(1) & working["prediction"]
    working["false_positive"] = working["label"].eq(0) & working["prediction"]
    grouped = working.groupby("group", sort=False).agg(
        rows=("label", "size"),
        positive_rows=("label", "sum"),
        true_positive=("true_positive", "sum"),
        false_positive=("false_positive", "sum"),
    )
    grouped["negative_rows"] = grouped["rows"] - grouped["positive_rows"]
    grouped["size_bucket"] = pd.cut(
        grouped["rows"],
        bins=GROUP_SIZE_BINS,
        labels=GROUP_SIZE_LABELS,
        include_lowest=True,
    )
    profile = (
        grouped.groupby("size_bucket", observed=False)
        .agg(
            groups=("rows", "size"),
            rows=("rows", "sum"),
            positive_rows=("positive_rows", "sum"),
            negative_rows=("negative_rows", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
        )
        .reset_index()
    )
    profile.insert(0, "scope", scope)
    profile["prevalence"] = profile["positive_rows"] / profile["rows"]
    profile["recall"] = profile["true_positive"] / profile["positive_rows"].replace(0, np.nan)
    profile["false_positive_rate"] = profile["false_positive"] / profile[
        "negative_rows"
    ].replace(0, np.nan)

    largest_count = max(1, int(np.ceil(0.01 * len(grouped))))
    largest = grouped.nlargest(largest_count, "rows")
    mixed = grouped["positive_rows"].gt(0) & grouped["positive_rows"].lt(grouped["rows"])
    total_positive = int(grouped["positive_rows"].sum())
    summary = {
        "scope": scope,
        "rows": int(grouped["rows"].sum()),
        "groups": int(len(grouped)),
        "median_rows_per_group": float(grouped["rows"].median()),
        "maximum_rows_per_group": int(grouped["rows"].max()),
        "top_one_percent_group_row_share": float(largest["rows"].sum() / grouped["rows"].sum()),
        "top_one_percent_group_positive_share": float(
            largest["positive_rows"].sum() / total_positive
        )
        if total_positive
        else None,
        "mixed_label_groups": int(mixed.sum()),
        "mixed_label_group_row_share": float(grouped.loc[mixed, "rows"].sum() / grouped["rows"].sum()),
    }
    return profile, summary


def score_shift_table(
    validation: pd.DataFrame,
    confirmation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for scope, frame in (
        ("target_validation", validation),
        ("sealed_confirmation", confirmation),
    ):
        labels = frame[TARGET].to_numpy()
        values = probabilities[scope]
        normal_mean = float(values[labels == 0].mean())
        attack_mean = float(values[labels == 1].mean())
        quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append(
            {
                "scope": scope,
                "rows": int(len(frame)),
                "prevalence": float(labels.mean()),
                "threshold": threshold,
                "predicted_positive_rate": float((values >= threshold).mean()),
                "mean_probability": float(values.mean()),
                "normal_mean_probability": normal_mean,
                "attack_mean_probability": attack_mean,
                "attack_minus_normal_mean_probability": attack_mean - normal_mean,
                **{
                    f"probability_q{int(level * 100):02d}": float(value)
                    for level, value in zip(
                        [0.05, 0.25, 0.50, 0.75, 0.95], quantiles
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def confirmation_error_profiles(
    validation: pd.DataFrame,
    confirmation: pd.DataFrame,
    confirmation_probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    tables = []
    prediction = confirmation_probability >= threshold
    for feature in MODEL_FEATURES:
        _, bins = binned_feature(
            validation[feature],
            confirmation[feature],
            feature in CATEGORICAL_FEATURES,
        )
        working = pd.DataFrame(
            {
                "bin": bins,
                "label": confirmation[TARGET].to_numpy(),
                "prediction": prediction,
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
                    "rows": int(len(group)),
                    "normal_rows": int(normal.sum()),
                    "attack_rows": int(attack.sum()),
                    "false_positive_rate": (
                        float(group.loc[normal, "prediction"].mean())
                        if normal.any()
                        else np.nan
                    ),
                    "attack_recall": (
                        float(group.loc[attack, "prediction"].mean())
                        if attack.any()
                        else np.nan
                    ),
                }
            )
        tables.append(pd.DataFrame(rows))
    return pd.concat(tables, ignore_index=True)


def save_plots(
    numeric_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    attacks: pd.DataFrame,
    score_shift: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    group_profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    numeric_plot = numeric_shift.sort_values("psi")
    axes[0].barh(numeric_plot["feature"], numeric_plot["psi"], color="#e45756")
    axes[0].set_xlabel("Population Stability Index")
    axes[0].set_title("Validation-to-confirmation numeric shift")
    axes[0].grid(axis="x", alpha=0.25)
    categorical_plot = categorical_shift.sort_values("jensen_shannon_divergence_bits")
    axes[1].barh(
        categorical_plot["feature"],
        categorical_plot["jensen_shannon_divergence_bits"],
        color="#4c78a8",
    )
    axes[1].set_xlabel("Jensen-Shannon divergence (bits)")
    axes[1].set_title("Categorical shift")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_distribution_shift.png", dpi=180)
    plt.close(fig)

    supported = attacks[attacks["confirmation_coverage_eligible"]].copy()
    supported = supported.sort_values("sealed_confirmation_attack_share")
    positions = np.arange(len(supported))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(
        positions - 0.18,
        supported["target_validation_attack_share"],
        height=0.35,
        label="target validation",
    )
    axes[0].barh(
        positions + 0.18,
        supported["sealed_confirmation_attack_share"],
        height=0.35,
        label="sealed confirmation",
    )
    axes[0].set_yticks(positions, supported["attack"])
    axes[0].set_xlabel("Share of positive rows")
    axes[0].set_title("Attack composition")
    axes[0].legend()
    axes[1].barh(
        positions - 0.18,
        supported["target_validation_recall"].fillna(0),
        height=0.35,
        label="target validation",
    )
    axes[1].barh(
        positions + 0.18,
        supported["sealed_confirmation_recall"].fillna(0),
        height=0.35,
        label="sealed confirmation",
    )
    axes[1].axvline(0.50, color="#c44e52", linestyle="--")
    axes[1].set_yticks(positions, supported["attack"])
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("Recall")
    axes[1].set_title("Frozen-model attack recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "attack_mix_and_recall.png", dpi=180)
    plt.close(fig)

    bins = np.linspace(0, 1, 51)
    fig, axis = plt.subplots(figsize=(9, 5))
    for scope, color in (
        ("target_validation", "#4c78a8"),
        ("sealed_confirmation", "#e45756"),
    ):
        counts, _ = np.histogram(probabilities[scope], bins=bins)
        axis.plot(
            bins[:-1],
            counts / counts.sum(),
            drawstyle="steps-post",
            label=scope.replace("_", " "),
            color=color,
        )
    axis.axvline(float(score_shift.iloc[0]["threshold"]), color="black", linestyle="--")
    axis.set_yscale("log")
    axis.set_xlabel("Predicted attack probability")
    axis.set_ylabel("Share of records")
    axis.set_title("Frozen target-RF score shift")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "model_score_shift.png", dpi=180)
    plt.close(fig)

    pivot = group_profiles.pivot(
        index="size_bucket", columns="scope", values="recall"
    ).reindex(GROUP_SIZE_LABELS)
    axis = pivot.plot.bar(figsize=(10, 5), rot=0)
    axis.axhline(0.50, color="#c44e52", linestyle="--")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Recall")
    axis.set_title("Recall by repeated-group size")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(plot_dir / "group_size_recall.png", dpi=180)
    plt.close(axis.figure)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    protocol = load_audit_protocol(args.protocol)
    dataset_path = resolve_dataset_path(args.dataset_path)
    validation, confirmation, integrity = load_registered_partitions(
        dataset_path, protocol, args.chunk_size
    )

    confirmation_protocol = load_protocol(
        Path(protocol["confirmation_protocol"]["path"])
    )
    models = load_frozen_models(confirmation_protocol)
    model = models[MODEL_ID]
    threshold = float(model["threshold"])
    probabilities = {
        "target_validation": model["pipeline"].predict_proba(
            validation.loc[:, MODEL_FEATURES]
        )[:, 1],
        "sealed_confirmation": model["pipeline"].predict_proba(
            confirmation.loc[:, MODEL_FEATURES]
        )[:, 1],
    }
    reproduced_metrics = verify_saved_metrics(
        validation, confirmation, probabilities, threshold, protocol
    )

    numeric_shift = numeric_distribution_shift(validation, confirmation)
    categorical_shift = categorical_distribution_shift(validation, confirmation)
    association_shift, association_bins = feature_label_association_drift(
        validation, confirmation
    )
    attacks = attack_comparison(
        validation, confirmation, probabilities, threshold
    )
    score_shift = score_shift_table(
        validation, confirmation, probabilities, threshold
    )
    error_profiles = confirmation_error_profiles(
        validation,
        confirmation,
        probabilities["sealed_confirmation"],
        threshold,
    )
    group_profiles_list = []
    group_summaries = []
    for scope, frame in (
        ("target_validation", validation),
        ("sealed_confirmation", confirmation),
    ):
        profile, summary = group_size_analysis(
            scope, frame, probabilities[scope], threshold
        )
        group_profiles_list.append(profile)
        group_summaries.append(summary)
    group_profiles = pd.concat(group_profiles_list, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    reproduced_metrics.to_csv(args.output_dir / "reproduced_metrics.csv", index=False)
    numeric_shift.to_csv(args.output_dir / "numeric_distribution_shift.csv", index=False)
    categorical_shift.to_csv(
        args.output_dir / "categorical_distribution_shift.csv", index=False
    )
    association_shift.to_csv(
        args.output_dir / "feature_label_association_drift.csv", index=False
    )
    association_bins.to_csv(
        args.output_dir / "feature_label_association_bins.csv", index=False
    )
    attacks.to_csv(args.output_dir / "attack_mix_and_recall.csv", index=False)
    score_shift.to_csv(args.output_dir / "model_score_shift.csv", index=False)
    error_profiles.to_csv(
        args.output_dir / "confirmation_error_profiles_by_feature_bin.csv",
        index=False,
    )
    group_profiles.to_csv(args.output_dir / "group_size_profiles.csv", index=False)
    save_plots(
        numeric_shift,
        categorical_shift,
        attacks,
        score_shift,
        probabilities,
        group_profiles,
        args.output_dir,
    )

    eligible = attacks[attacks["confirmation_coverage_eligible"]]
    summary = json_safe(
        {
            "material_passport": {
                "origin_skill": "experiment-agent",
                "origin_mode": "run and validate",
                "origin_date": "2026-10-26",
                "verification_status": "VERIFIED",
                "version_label": "fresh_generalisation_failure_audit_v1",
            },
            "scope": {
                "analysis": "Post-hoc descriptive development-to-confirmation failure audit",
                "model_changes": False,
                "feature_changes": False,
                "threshold_changes": False,
                "candidate_selection": False,
                "deployment_claim": False,
            },
            "protocol": {
                "path": str(args.protocol),
                "sha256": sha256_file(args.protocol),
            },
            "integrity": integrity,
            "reproduced_saved_metrics": True,
            "top_attack_composition_changes": attacks.head(8).to_dict(
                orient="records"
            ),
            "supported_confirmation_attacks_below_recall_rule": eligible[
                eligible["sealed_confirmation_recall"].lt(0.50)
            ].to_dict(orient="records"),
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
            "group_concentration": group_summaries,
            "interpretation_boundary": (
                "This audit describes differences observed after confirmation was opened. "
                "It cannot identify a unique causal mechanism, validate a correction, "
                "authorise tuning on confirmation, or establish deployment readiness."
            ),
        }
    )
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
