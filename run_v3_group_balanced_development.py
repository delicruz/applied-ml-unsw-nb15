"""Run the preregistered Version 3 group-balanced development experiment.

The four datasets are already-opened development domains. This runner does not
provide independent confirmation and refuses to overwrite completed output.
Use --preflight-only to verify protocol, implementation, and datasets without
scanning rows, constructing samples, fitting models, or scoring domains.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import run_v2_generalisation_development as core


PROTOCOL_PATH = Path("protocols/v3_group_balanced_development_protocol.json")
MANIFEST_PATH = Path("protocols/v3_group_balanced_implementation_manifest.json")
OUTPUT_DIR = Path("artifacts/v3_group_balanced_development")
SEED = 42
CHUNK_SIZE = 250_000

DATASET_GLOBS = core.DATASET_GLOBS
ENDPOINT_FEATURES = core.ENDPOINT_FEATURES
SHORTCUT_FEATURES = core.SHORTCUT_FEATURES
CATEGORICAL_FEATURES = core.CATEGORICAL_FEATURES
BASE_NUMERIC_FEATURES = core.NUMERIC_FEATURES
BASE_FEATURES = core.MODEL_FEATURES
DERIVED_FEATURES = (
    "TOTAL_BYTES",
    "TOTAL_PKTS",
    "IN_AVG_PKT_BYTES",
    "OUT_AVG_PKT_BYTES",
    "BYTE_ASYMMETRY",
    "PKT_ASYMMETRY",
    "BYTES_PER_MILLISECOND",
    "PKTS_PER_MILLISECOND",
)
TARGET = core.TARGET
ATTACK = core.ATTACK
DOMAIN = core.DOMAIN
GROUP = core.GROUP
FAMILY = core.FAMILY
PARTITIONS = ("training", "calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--preflight-only", action="store_true")
    for domain in DATASET_GLOBS:
        option = domain.lower().replace("-", "_") + "_path"
        parser.add_argument(f"--{option.replace('_', '-')}", dest=option, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    return core.resolve_dataset_paths(args)


def expected_dataset_manifest(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    item = protocol["locked_inputs"]["v2_summary"]
    summary = read_json(Path(item["path"]))
    return {
        row["domain"]: {
            "sha256": row["sha256"],
            "rows": int(row["expected_rows"]),
        }
        for row in summary["preflight"]["datasets"]
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["status"] != "REGISTERED_BEFORE_V3_IMPLEMENTATION_OR_MODEL_FITTING":
        raise ValueError("Version 3 protocol has the wrong registration status")
    if set(protocol["development_domains"]) != set(DATASET_GLOBS):
        raise ValueError("Registered domains do not match runner domains")
    features = protocol["features"]
    if tuple(features["base_features"]) != BASE_FEATURES:
        raise ValueError("Registered base features do not match the runner")
    if tuple(features["derived_features"]) != DERIVED_FEATURES:
        raise ValueError("Registered derived features do not match the runner")
    if set(features["excluded_endpoint_features"]) != set(ENDPOINT_FEATURES):
        raise ValueError("Endpoint exclusion does not match the runner")
    if set(features["excluded_shortcut_features"]) != set(SHORTCUT_FEATURES):
        raise ValueError("Shortcut exclusion does not match the runner")
    sampling = protocol["group_and_sampling_protocol"]
    if int(sampling["maximum_retained_rows_per_group_per_partition"]) != 64:
        raise ValueError("Version 3 group cap must remain 64")
    registered_candidates = {
        item["id"] for item in protocol["candidate_experiments"]
    }
    expected_candidates = {
        "group_capped_lr_base_features",
        "group_capped_lr_derived_features",
        "group_capped_rf_derived_features",
    }
    if registered_candidates != expected_candidates:
        raise ValueError("Registered candidate set does not match the runner")
    if protocol["independent_confirmation_boundary"]["included_in_this_protocol"]:
        raise ValueError("This runner cannot include independent confirmation")
    for item in protocol["locked_inputs"].values():
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Locked input hash mismatch: {path}")


def verify_implementation(manifest_path: Path, runner_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest["status"] != "IMPLEMENTATION_FROZEN_BEFORE_EXECUTION":
        raise ValueError("Implementation manifest is not frozen")
    checks = {
        Path(manifest["protocol"]["path"]): manifest["protocol"]["sha256"],
        runner_path: manifest["runner"]["sha256"],
        Path(manifest["tests"]["path"]): manifest["tests"]["sha256"],
        Path(manifest["dependency"]["path"]): manifest["dependency"]["sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Implementation hash mismatch: {path}")
    return manifest


def preflight(
    protocol: dict[str, Any], paths: dict[str, Path], chunk_size: int
) -> dict[str, Any]:
    if chunk_size < 10_000:
        raise ValueError("--chunk-size must be at least 10,000")
    verify_protocol(protocol)
    expected = expected_dataset_manifest(protocol)
    if set(expected) != set(DATASET_GLOBS):
        raise ValueError("Locked Version 2 dataset manifest is incomplete")
    rows = []
    for domain, path in paths.items():
        core.validate_header(path)
        observed_hash = sha256_file(path)
        if observed_hash != expected[domain]["sha256"]:
            raise ValueError(f"Dataset hash mismatch for {domain}")
        rows.append(
            {
                "domain": domain,
                "path": str(path),
                "sha256": observed_hash,
                "expected_rows": expected[domain]["rows"],
            }
        )
    return {"protocol_verified": True, "datasets": rows}


def capped_group_contributions(
    strata: pd.DataFrame, partition: str, group_cap: int
) -> pd.DataFrame:
    subset = strata.loc[strata["_partition"].eq(partition)].copy()
    if subset.empty:
        raise ValueError(f"No groups available for {partition}")
    groups = (
        subset.groupby([GROUP, "_second"], sort=False, observed=True)
        .agg(original_rows=("stratum_rows", "sum"))
        .reset_index()
    )
    groups["capped_rows"] = groups["original_rows"].clip(upper=group_cap)
    return groups


def select_groups_capped(
    strata: pd.DataFrame, partition: str, row_target: int, group_cap: int
) -> set[str]:
    subset = strata.loc[strata["_partition"].eq(partition)].copy()
    contributions = capped_group_contributions(strata, partition, group_cap)
    capped_rows = contributions.set_index(GROUP)["capped_rows"].to_dict()
    selected: set[str] = set()
    retained_rows = 0

    coverage = subset.sort_values([FAMILY, "_second", GROUP]).groupby(
        FAMILY, sort=True, observed=True
    ).head(1)
    for key in coverage[GROUP]:
        if key not in selected:
            selected.add(key)
            retained_rows += int(capped_rows[key])

    queues = {
        family: list(group.sort_values(["_second", GROUP])[GROUP].drop_duplicates())
        for family, group in subset.groupby(FAMILY, sort=True, observed=True)
    }
    positions = {family: 0 for family in queues}
    families = sorted(queues)
    while retained_rows < row_target:
        added = False
        for family in families:
            queue = queues[family]
            while positions[family] < len(queue) and queue[positions[family]] in selected:
                positions[family] += 1
            if positions[family] >= len(queue):
                continue
            key = queue[positions[family]]
            positions[family] += 1
            selected.add(key)
            retained_rows += int(capped_rows[key])
            added = True
            if retained_rows >= row_target:
                break
        if not added:
            break
    return selected


def allocate_stratum_quotas(
    strata: pd.DataFrame, selected: set[str], partition: str, group_cap: int
) -> pd.DataFrame:
    subset = strata.loc[
        strata["_partition"].eq(partition) & strata[GROUP].isin(selected)
    ].copy()
    records: list[dict[str, Any]] = []
    for key, group in subset.groupby(GROUP, sort=True, observed=True):
        group = group.sort_values([FAMILY, TARGET, "_second"]).reset_index(drop=True)
        available = group["stratum_rows"].astype(int).to_numpy()
        quota = np.zeros(len(group), dtype=int)
        capacity = min(group_cap, int(available.sum()))
        for index in range(min(capacity, len(group))):
            quota[index] = 1
        remaining = capacity - int(quota.sum())
        while remaining > 0:
            progressed = False
            for index in range(len(group)):
                if quota[index] < available[index]:
                    quota[index] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
        for index, row in group.iterrows():
            if quota[index] > 0:
                records.append(
                    {
                        GROUP: key,
                        TARGET: int(row[TARGET]),
                        FAMILY: str(row[FAMILY]),
                        "quota": int(quota[index]),
                    }
                )
    result = pd.DataFrame(records)
    if result.empty:
        raise RuntimeError(f"No quotas allocated for {partition}")
    if result.groupby(GROUP)["quota"].sum().max() > group_cap:
        raise RuntimeError("A group quota exceeds the registered cap")
    return result


def deterministic_row_hash(frame: pd.DataFrame) -> np.ndarray:
    columns = list(BASE_FEATURES) + [TARGET, ATTACK, "_source_row"]
    return pd.util.hash_pandas_object(
        frame.loc[:, columns], index=False, categorize=True
    ).to_numpy(dtype="uint64")


def materialise_capped_groups(
    path: Path,
    domain: str,
    quotas: dict[str, pd.DataFrame],
    chunk_size: int,
) -> dict[str, pd.DataFrame]:
    usecols = list(BASE_FEATURES) + [TARGET, ATTACK]
    selected = {
        partition: set(table[GROUP]) for partition, table in quotas.items()
    }
    reservoirs: dict[str, pd.DataFrame | None] = {name: None for name in PARTITIONS}
    source_row = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk = core.normalise_chunk(chunk, domain)
        hashes = core.canonical_group_hashes(chunk, domain)
        chunk[GROUP] = hashes[GROUP].to_numpy()
        chunk["_partition"] = core.group_partition(hashes["_first"])
        chunk["_source_row"] = np.arange(source_row, source_row + len(chunk))
        source_row += len(chunk)
        for partition in PARTITIONS:
            mask = chunk["_partition"].eq(partition) & chunk[GROUP].isin(
                selected[partition]
            )
            if not mask.any():
                continue
            candidate = chunk.loc[mask].copy()
            candidate = candidate.merge(
                quotas[partition],
                on=[GROUP, TARGET, FAMILY],
                how="inner",
                validate="many_to_one",
            )
            candidate["_row_hash"] = deterministic_row_hash(candidate)
            previous = reservoirs[partition]
            combined = (
                candidate
                if previous is None
                else pd.concat([previous, candidate], ignore_index=True)
            )
            keys = [GROUP, TARGET, FAMILY]
            combined = combined.sort_values(
                keys + ["_row_hash", "_source_row"], kind="mergesort"
            )
            combined["_rank"] = combined.groupby(keys, sort=False).cumcount()
            reservoirs[partition] = combined.loc[
                combined["_rank"] < combined["quota"]
            ].drop(columns="_rank")

    frames: dict[str, pd.DataFrame] = {}
    for partition in PARTITIONS:
        frame = reservoirs[partition]
        if frame is None:
            raise RuntimeError(f"No rows materialised for {domain} {partition}")
        expected = int(quotas[partition]["quota"].sum())
        if len(frame) != expected:
            raise RuntimeError(
                f"Incomplete capped materialisation for {domain} {partition}: "
                f"{len(frame)} != {expected}"
            )
        if frame.groupby(GROUP).size().max() > 64:
            raise RuntimeError("A materialised group exceeds the registered cap")
        frames[partition] = frame.drop(
            columns=["_partition", "_source_row", "_row_hash", "quota"]
        ).reset_index(drop=True)
    if set(frames["training"][GROUP]) & set(frames["calibration"][GROUP]):
        raise RuntimeError(f"{domain} source partitions overlap")
    return frames


def prepare_domain_samples(
    path: Path,
    domain: str,
    expected_rows: int,
    protocol: dict[str, Any],
    chunk_size: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    strata, integrity = core.scan_domain_groups(
        path, domain, expected_rows, chunk_size
    )
    sampling = protocol["group_and_sampling_protocol"]
    group_cap = int(sampling["maximum_retained_rows_per_group_per_partition"])
    targets = {
        "training": int(sampling["target_training_rows_per_source_domain"]),
        "calibration": int(sampling["target_calibration_rows_per_source_domain"]),
    }
    selected = {
        partition: select_groups_capped(
            strata, partition, targets[partition], group_cap
        )
        for partition in PARTITIONS
    }
    quotas = {
        partition: allocate_stratum_quotas(
            strata, selected[partition], partition, group_cap
        )
        for partition in PARTITIONS
    }
    frames = materialise_capped_groups(path, domain, quotas, chunk_size)
    diagnostics = []
    integrity["samples"] = {}
    for partition in PARTITIONS:
        frame = frames[partition]
        original = strata.loc[
            strata["_partition"].eq(partition)
            & strata[GROUP].isin(selected[partition])
        ]
        original_sizes = original.groupby(GROUP)["stratum_rows"].sum()
        retained_sizes = frame.groupby(GROUP).size()
        details = {
            "rows": int(len(frame)),
            "groups": int(frame[GROUP].nunique()),
            "positive_rows": int(frame[TARGET].sum()),
            "families": int(frame[FAMILY].nunique()),
            "largest_original_group_rows": int(original_sizes.max()),
            "largest_retained_group_rows": int(retained_sizes.max()),
            "largest_retained_group_share": float(retained_sizes.max() / len(frame)),
        }
        integrity["samples"][partition] = details
        diagnostics.append({"domain": domain, "partition": partition, **details})
    return frames, integrity, pd.DataFrame(diagnostics)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    total_bytes = result["IN_BYTES"] + result["OUT_BYTES"]
    total_pkts = result["IN_PKTS"] + result["OUT_PKTS"]
    result["TOTAL_BYTES"] = total_bytes
    result["TOTAL_PKTS"] = total_pkts
    result["IN_AVG_PKT_BYTES"] = result["IN_BYTES"] / np.maximum(
        result["IN_PKTS"], 1.0
    )
    result["OUT_AVG_PKT_BYTES"] = result["OUT_BYTES"] / np.maximum(
        result["OUT_PKTS"], 1.0
    )
    result["BYTE_ASYMMETRY"] = (
        result["IN_BYTES"] - result["OUT_BYTES"]
    ) / (total_bytes + 1.0)
    result["PKT_ASYMMETRY"] = (
        result["IN_PKTS"] - result["OUT_PKTS"]
    ) / (total_pkts + 1.0)
    duration = np.maximum(result["FLOW_DURATION_MILLISECONDS"], 1.0)
    result["BYTES_PER_MILLISECOND"] = total_bytes / duration
    result["PKTS_PER_MILLISECOND"] = total_pkts / duration
    return result


def signed_log1p(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    return np.sign(array) * np.log1p(np.abs(array))


def candidate_output_features(specification: dict[str, Any]) -> tuple[str, ...]:
    declaration = specification["features"]
    if declaration == "base_features":
        return BASE_FEATURES
    if declaration == "base_features plus derived_features":
        return BASE_FEATURES + DERIVED_FEATURES
    raise ValueError(f"Unknown feature declaration: {declaration}")


def make_preprocessor(features: tuple[str, ...], scale: bool) -> ColumnTransformer:
    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    numeric = [name for name in features if name not in CATEGORICAL_FEATURES]
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        (
            "signed_log1p",
            FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
        ),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                        ),
                    ]
                ),
                categorical,
            ),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
    )


def make_candidate(specification: dict[str, Any]) -> Pipeline:
    settings = specification["settings"]
    output_features = candidate_output_features(specification)
    if specification["model"] == "Logistic Regression":
        estimator: Any = LogisticRegression(
            C=float(settings["C"]),
            class_weight=settings["class_weight"],
            max_iter=int(settings["max_iter"]),
            random_state=int(settings["random_state"]),
            solver="saga",
            tol=1e-3,
        )
        scale = True
    elif specification["model"] == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=int(settings["n_estimators"]),
            max_depth=int(settings["max_depth"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            max_features=settings["max_features"],
            class_weight=settings["class_weight"],
            n_jobs=int(settings["n_jobs"]),
            random_state=int(settings["random_state"]),
            bootstrap=True,
            oob_score=True,
        )
        scale = False
    else:
        raise ValueError(f"Unknown model: {specification['model']}")
    return Pipeline(
        [
            (
                "derive",
                FunctionTransformer(add_derived_features, validate=False),
            ),
            ("preprocess", make_preprocessor(output_features, scale)),
            ("model", estimator),
        ]
    )


def fit_candidate(specification: dict[str, Any], training: pd.DataFrame) -> Pipeline:
    model = make_candidate(specification)
    model.fit(training.loc[:, BASE_FEATURES], training[TARGET])
    return model


def select_constrained_threshold(
    calibration: pd.DataFrame,
    probabilities: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[float, pd.DataFrame, bool]:
    rule = protocol["threshold_rule"]
    grid = rule["grid"]
    thresholds = np.round(
        np.arange(grid["minimum"], grid["maximum"] + grid["step"] / 2, grid["step"]),
        10,
    )
    constraints = rule["constraints"]
    minimum_rows = int(constraints["minimum_rows_for_supported_stratum"])
    working = calibration[[DOMAIN, TARGET, ATTACK]].copy()
    working["probability"] = probabilities
    supported = (
        working.loc[working[TARGET].eq(1)]
        .groupby([DOMAIN, ATTACK], observed=True)
        .size()
    )
    supported = supported[supported >= minimum_rows].index.tolist()
    rows = []
    for threshold in thresholds:
        domain_mcc = []
        domain_fpr = []
        for _, group in working.groupby(DOMAIN, sort=True, observed=True):
            labels = group[TARGET].to_numpy(dtype="int8")
            scores = group["probability"].to_numpy(dtype="float64")
            tn, fp, fn, tp = core.confusion_components(labels, scores, float(threshold))
            domain_mcc.append(core.mcc_from_counts(tn, fp, fn, tp))
            domain_fpr.append(fp / (fp + tn) if fp + tn else 0.0)
        recalls = []
        for domain, attack in supported:
            group = working.loc[
                working[DOMAIN].eq(domain)
                & working[ATTACK].eq(attack)
                & working[TARGET].eq(1)
            ]
            recalls.append(float((group["probability"] >= threshold).mean()))
        minimum_recall = min(recalls) if recalls else 1.0
        maximum_fpr = max(domain_fpr)
        feasible = (
            maximum_fpr
            <= float(constraints["maximum_false_positive_rate_in_each_source_domain"])
            and minimum_recall
            >= float(constraints["minimum_recall_for_each_supported_attack_domain_stratum"])
        )
        rows.append(
            {
                "threshold": float(threshold),
                "minimum_domain_mcc": float(min(domain_mcc)),
                "mean_domain_mcc": float(np.mean(domain_mcc)),
                "minimum_supported_attack_recall": float(minimum_recall),
                "maximum_domain_false_positive_rate": float(maximum_fpr),
                "supported_attack_strata": int(len(supported)),
                "feasible": bool(feasible),
            }
        )
    table = pd.DataFrame(rows)
    pool = table.loc[table["feasible"]]
    passed = not pool.empty
    if pool.empty:
        pool = table
    chosen = pool.sort_values(
        [
            "minimum_domain_mcc",
            "mean_domain_mcc",
            "minimum_supported_attack_recall",
            "maximum_domain_false_positive_rate",
            "threshold",
        ],
        ascending=[False, False, False, True, False],
    ).iloc[0]
    return float(chosen["threshold"]), table, passed


def group_balanced_metrics(groups: pd.DataFrame, candidate: str) -> dict[str, Any]:
    rows = groups["rows"].to_numpy(dtype="float64")
    scale = np.divide(1.0, rows, out=np.zeros_like(rows), where=rows > 0)
    positives = float((groups["positives"].to_numpy(dtype="float64") * scale).sum())
    negatives = float((groups["negatives"].to_numpy(dtype="float64") * scale).sum())
    tp = float(
        (groups[f"{candidate}__true_positive"].to_numpy(dtype="float64") * scale).sum()
    )
    fp = float(
        (groups[f"{candidate}__false_positive"].to_numpy(dtype="float64") * scale).sum()
    )
    fn = positives - tp
    tn = negatives - fp
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    return {
        "groups": int(len(groups)),
        "mcc": core.mcc_from_counts(tn, fp, fn, tp),
        "balanced_accuracy": float((recall + specificity) / 2),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_positive_rate": float(1.0 - specificity),
        "true_negative_weight": tn,
        "false_positive_weight": fp,
        "false_negative_weight": fn,
        "true_positive_weight": tp,
    }


def aggregate_candidates(
    fold_metrics: pd.DataFrame,
    group_balanced: pd.DataFrame,
    attacks: pd.DataFrame,
    thresholds: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    minimum_rows = int(
        protocol["threshold_rule"]["constraints"][
            "minimum_rows_for_supported_stratum"
        ]
    )
    supported = attacks.loc[attacks["positive_rows"] >= minimum_rows]
    rows = []
    for candidate, group in fold_metrics.groupby("candidate", sort=True):
        candidate_attacks = supported.loc[supported["candidate"].eq(candidate)]
        candidate_thresholds = thresholds.loc[thresholds["candidate"].eq(candidate)]
        candidate_group = group_balanced.loc[group_balanced["candidate"].eq(candidate)]
        rows.append(
            {
                "candidate": candidate,
                "macro_mcc": float(group["mcc"].mean()),
                "worst_domain_mcc": float(group["mcc"].min()),
                "worst_domain_balanced_accuracy": float(group["balanced_accuracy"].min()),
                "macro_supported_attack_recall": float(candidate_attacks["recall"].mean()),
                "minimum_supported_attack_recall": float(candidate_attacks["recall"].min()),
                "maximum_domain_false_positive_rate": float(group["false_positive_rate"].max()),
                "worst_domain_group_balanced_mcc": float(candidate_group["mcc"].min()),
                "macro_group_balanced_mcc": float(candidate_group["mcc"].mean()),
                "threshold_constraints_all_folds": bool(
                    candidate_thresholds["constraints_passed"].all()
                ),
            }
        )
    result = pd.DataFrame(rows)
    promotion = protocol["promotion_rule"]
    baseline_id = promotion["reference"]
    baseline = result.set_index("candidate").loc[baseline_id]
    result["macro_mcc_improvement"] = result["macro_mcc"] - baseline["macro_mcc"]
    result["worst_balanced_accuracy_improvement"] = (
        result["worst_domain_balanced_accuracy"]
        - baseline["worst_domain_balanced_accuracy"]
    )
    result["maximum_fpr_increase"] = (
        result["maximum_domain_false_positive_rate"]
        - baseline["maximum_domain_false_positive_rate"]
    )
    result["eligibility_passed"] = (
        result["threshold_constraints_all_folds"]
        & result["worst_domain_balanced_accuracy"].gt(0.50)
        & result["maximum_domain_false_positive_rate"].le(0.25)
        & result["minimum_supported_attack_recall"].ge(0.20)
        & result["worst_domain_group_balanced_mcc"].gt(0.0)
    )
    result["improvement_passed"] = (
        result["macro_mcc_improvement"].ge(0.03)
        & result["worst_balanced_accuracy_improvement"].ge(0.02)
        & result["maximum_fpr_increase"].le(0.02)
    )
    result["promotion_eligible"] = (
        result["candidate"].isin(promotion["candidate_scope"])
        & result["eligibility_passed"]
        & result["improvement_passed"]
    )
    return result


def bootstrap_metric(
    groups: pd.DataFrame,
    candidate: str,
    multiplicity: np.ndarray,
    equal_group_weight: bool,
) -> dict[str, float]:
    if equal_group_weight:
        row_scale = np.divide(
            multiplicity,
            groups["rows"].to_numpy(dtype="float64"),
            out=np.zeros_like(multiplicity, dtype="float64"),
            where=groups["rows"].to_numpy(dtype="float64") > 0,
        )
    else:
        row_scale = multiplicity.astype("float64")
    positives = float((groups["positives"].to_numpy(dtype="float64") * row_scale).sum())
    negatives = float((groups["negatives"].to_numpy(dtype="float64") * row_scale).sum())
    tp = float(
        (groups[f"{candidate}__true_positive"].to_numpy(dtype="float64") * row_scale).sum()
    )
    fp = float(
        (groups[f"{candidate}__false_positive"].to_numpy(dtype="float64") * row_scale).sum()
    )
    fn = positives - tp
    tn = negatives - fp
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    return {
        "mcc": core.mcc_from_counts(tn, fp, fn, tp),
        "balanced_accuracy": float((recall + specificity) / 2),
        "recall": float(recall),
        "false_positive_rate": float(1.0 - specificity),
    }


def paired_bootstrap_intervals(
    group_tables: dict[tuple[str, str], pd.DataFrame],
    aggregate: pd.DataFrame,
    protocol: dict[str, Any],
    iterations: int = 200,
) -> pd.DataFrame:
    baseline = protocol["promotion_rule"]["reference"]
    candidates = aggregate.loc[aggregate["promotion_eligible"], "candidate"].tolist()
    comparisons = [(baseline, None)] + [(candidate, baseline) for candidate in candidates]
    samples = []
    for domain in sorted({key[0] for key in group_tables}):
        base = group_tables[(domain, baseline)].copy()
        tables = {baseline: base}
        for candidate in candidates:
            candidate_counts = group_tables[(domain, candidate)][
                [GROUP, f"{candidate}__true_positive", f"{candidate}__false_positive"]
            ]
            tables[candidate] = base.merge(
                candidate_counts, on=GROUP, validate="one_to_one"
            )
        rng = np.random.default_rng(SEED + sum(domain.encode("utf-8")))
        for iteration in range(iterations):
            multiplicity = np.bincount(
                rng.integers(0, len(base), size=len(base)), minlength=len(base)
            ).astype("float64")
            for weighting, equal_group in (
                ("row_weighted", False),
                ("equal_group_weight", True),
            ):
                baseline_metrics = bootstrap_metric(
                    base, baseline, multiplicity, equal_group
                )
                for candidate, reference in comparisons:
                    if reference is None:
                        values = baseline_metrics
                        comparison = baseline
                    else:
                        candidate_metrics = bootstrap_metric(
                            tables[candidate], candidate, multiplicity, equal_group
                        )
                        values = {
                            metric: candidate_metrics[metric] - baseline_metrics[metric]
                            for metric in baseline_metrics
                        }
                        comparison = f"{candidate}_minus_{baseline}"
                    for metric, value in values.items():
                        samples.append(
                            {
                                "heldout_domain": domain,
                                "comparison": comparison,
                                "weighting": weighting,
                                "iteration": iteration,
                                "metric": metric,
                                "value": value,
                            }
                        )
    frame = pd.DataFrame(samples)
    return (
        frame.groupby(["heldout_domain", "comparison", "weighting", "metric"])[
            "value"
        ]
        .agg(
            bootstrap_median="median",
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )


def create_plots(
    fold_metrics: pd.DataFrame,
    group_balanced: pd.DataFrame,
    attacks: pd.DataFrame,
    output_dir: Path,
) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    chart = fold_metrics.pivot(index="heldout_domain", columns="candidate", values="mcc")
    axis = chart.plot(kind="bar", figsize=(12, 6))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Row-weighted MCC")
    axis.set_title("Version 3 leave-one-dataset-out development performance")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "domain_generalisation.png", dpi=180)
    plt.close()

    supported = attacks.loc[attacks["positive_rows"] >= 100].copy()
    supported["label"] = supported["heldout_domain"] + " / " + supported[ATTACK]
    chart = supported.pivot(index="label", columns="candidate", values="recall")
    axis = chart.plot(kind="bar", figsize=(14, max(6, len(chart) * 0.24)))
    axis.axhline(0.20, color="red", linestyle="--", linewidth=1)
    axis.set_ylabel("Recall")
    axis.set_title("Supported held-out attack-family recall")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "attack_family_recall.png", dpi=180)
    plt.close()

    row = fold_metrics[["heldout_domain", "candidate", "mcc"]].copy()
    row["weighting"] = "row_weighted"
    equal = group_balanced[["heldout_domain", "candidate", "mcc"]].copy()
    equal["weighting"] = "equal_group_weight"
    comparison = pd.concat([row, equal], ignore_index=True)
    comparison["label"] = comparison["heldout_domain"] + " / " + comparison["candidate"]
    chart = comparison.pivot(index="label", columns="weighting", values="mcc")
    axis = chart.plot(kind="bar", figsize=(13, 7))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("MCC")
    axis.set_title("Row-weighted versus equal-group-weight MCC")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "group_influence.png", dpi=180)
    plt.close()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def run_experiment(
    protocol: dict[str, Any],
    paths: dict[str, Path],
    preflight_result: dict[str, Any],
    output_dir: Path,
    chunk_size: int,
) -> None:
    if (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"Completed output already exists at {output_dir}; no automatic rerun is allowed"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    expected = expected_dataset_manifest(protocol)
    samples: dict[str, dict[str, pd.DataFrame]] = {}
    integrity: dict[str, Any] = {}
    sampling_tables = []
    for domain, path in paths.items():
        samples[domain], integrity[domain], diagnostics = prepare_domain_samples(
            path, domain, int(expected[domain]["rows"]), protocol, chunk_size
        )
        sampling_tables.append(diagnostics)

    fold_rows = []
    group_balanced_rows = []
    attack_tables = []
    threshold_rows = []
    overfitting_rows = []
    group_metric_tables = []
    group_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for heldout_domain in DATASET_GLOBS:
        source_domains = [name for name in DATASET_GLOBS if name != heldout_domain]
        training = pd.concat(
            [samples[name]["training"] for name in source_domains], ignore_index=True
        )
        calibration = pd.concat(
            [samples[name]["calibration"] for name in source_domains], ignore_index=True
        )
        if heldout_domain in set(training[DOMAIN]) | set(calibration[DOMAIN]):
            raise RuntimeError("Held-out domain leaked into source development data")
        for specification in protocol["candidate_experiments"]:
            candidate = specification["id"]
            print(f"fit_start heldout={heldout_domain} candidate={candidate}", flush=True)
            model = fit_candidate(specification, training)
            calibration_scores = model.predict_proba(calibration.loc[:, BASE_FEATURES])[:, 1]
            threshold, table, passed = select_constrained_threshold(
                calibration, calibration_scores, protocol
            )
            table.to_csv(
                output_dir / f"threshold_grid__{heldout_domain}__{candidate}.csv",
                index=False,
            )
            threshold_rows.append(
                {
                    "heldout_domain": heldout_domain,
                    "candidate": candidate,
                    "threshold": threshold,
                    "constraints_passed": passed,
                }
            )
            training_metrics = core.evaluate_frame(
                training, model, BASE_FEATURES, threshold
            )
            calibration_metrics = core.evaluate_frame(
                calibration, model, BASE_FEATURES, threshold
            )
            heldout, attacks, group_metrics, heldout_groups = (
                core.stream_heldout_evaluation(
                    paths[heldout_domain],
                    heldout_domain,
                    model,
                    BASE_FEATURES,
                    threshold,
                    candidate,
                    chunk_size,
                )
            )
            heldout.update(
                {
                    "heldout_domain": heldout_domain,
                    "candidate": candidate,
                    "source_domains": "|".join(source_domains),
                    "threshold_constraints_passed": passed,
                    "predicted_positive_rate": (
                        heldout["true_positive"] + heldout["false_positive"]
                    )
                    / heldout["rows"],
                    "oob_score": (
                        float(model.named_steps["model"].oob_score_)
                        if hasattr(model.named_steps["model"], "oob_score_")
                        else None
                    ),
                }
            )
            fold_rows.append(heldout)
            equal_group = group_balanced_metrics(heldout_groups, candidate)
            group_balanced_rows.append(
                {
                    "heldout_domain": heldout_domain,
                    "candidate": candidate,
                    **equal_group,
                }
            )
            attack_tables.append(attacks)
            group_metric_tables.append(group_metrics)
            group_tables[(heldout_domain, candidate)] = heldout_groups
            for metric in ("mcc", "balanced_accuracy", "recall", "false_positive_rate"):
                overfitting_rows.extend(
                    [
                        {
                            "heldout_domain": heldout_domain,
                            "candidate": candidate,
                            "metric": metric,
                            "gap": "training_minus_calibration",
                            "value": training_metrics[metric] - calibration_metrics[metric],
                        },
                        {
                            "heldout_domain": heldout_domain,
                            "candidate": candidate,
                            "metric": metric,
                            "gap": "calibration_minus_heldout",
                            "value": calibration_metrics[metric] - heldout[metric],
                        },
                    ]
                )
            del model
            gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    group_balanced = pd.DataFrame(group_balanced_rows)
    attacks = pd.concat(attack_tables, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    overfitting = pd.DataFrame(overfitting_rows)
    group_metrics = pd.concat(group_metric_tables, ignore_index=True)
    sampling = pd.concat(sampling_tables, ignore_index=True)
    aggregate = aggregate_candidates(
        fold_metrics, group_balanced, attacks, thresholds, protocol
    )
    eligible = aggregate.loc[aggregate["promotion_eligible"]].sort_values(
        [
            "macro_mcc",
            "worst_domain_balanced_accuracy",
            "minimum_supported_attack_recall",
            "worst_domain_group_balanced_mcc",
            "maximum_domain_false_positive_rate",
            "candidate",
        ],
        ascending=[False, False, False, False, True, True],
    )
    selected = None if eligible.empty else str(eligible.iloc[0]["candidate"])
    verdict = (
        "no Version 3 candidate selected"
        if selected is None
        else "development-promising research prototype selected"
    )
    bootstrap = paired_bootstrap_intervals(group_tables, aggregate, protocol)

    sampling.to_csv(output_dir / "sampling_diagnostics.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    group_balanced.to_csv(
        output_dir / "group_balanced_fold_metrics.csv", index=False
    )
    aggregate.to_csv(output_dir / "aggregate_candidate_metrics.csv", index=False)
    attacks.to_csv(output_dir / "per_attack_recall.csv", index=False)
    thresholds.to_csv(output_dir / "thresholds.csv", index=False)
    overfitting.to_csv(output_dir / "overfitting_gaps.csv", index=False)
    group_metrics.to_csv(output_dir / "group_size_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_group_bootstrap_intervals.csv", index=False)
    create_plots(fold_metrics, group_balanced, attacks, output_dir)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "v3_group_balanced_development_v1",
        },
        "protocol_id": protocol["protocol_id"],
        "development_only": True,
        "independent_confirmation": False,
        "preflight": preflight_result,
        "sample_integrity": integrity,
        "selected_candidate": selected,
        "promotion_verdict": verdict,
        "aggregate_candidates": aggregate.to_dict(orient="records"),
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    protocol = read_json(args.protocol)
    verify_implementation(args.manifest, Path(__file__).resolve())
    paths = resolve_dataset_paths(args)
    result = preflight(protocol, paths, args.chunk_size)
    if args.preflight_only:
        print(json.dumps(result, indent=2))
        return
    run_experiment(protocol, paths, result, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
