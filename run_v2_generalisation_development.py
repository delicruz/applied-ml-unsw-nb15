"""Run the registered Version 2 multi-domain development experiment.

The four previously opened NetFlow-v1 datasets are development domains only.
Each leave-one-dataset-out fold fits preprocessing and models on bounded,
group-separated samples from three source domains, selects its threshold on
separate source calibration samples, and streams the complete fourth domain for
development evaluation. No dataset handled here is an untouched final test.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROTOCOL_PATH = Path("protocols/v2_generalisation_development_protocol.json")
MANIFEST_PATH = Path("protocols/v2_generalisation_implementation_manifest.json")
OUTPUT_DIR = Path("artifacts/v2_generalisation_development")
SEED = 42
CHUNK_SIZE = 250_000
MAX_U64 = (1 << 64) - 1
TRAINING_CUTOFF = int(0.75 * MAX_U64)

DATASET_GLOBS = {
    "NF-UNSW-NB15": "data/external/**/NF-UNSW-NB15.csv",
    "NF-ToN-IoT": "data/external/**/NF-ToN-IoT.csv",
    "NF-BoT-IoT": "data/external/**/NF-BoT-IoT.csv",
    "NF-CSE-CIC-IDS2018": "data/external/**/NF-CSE-CIC-IDS2018.csv",
}
ENDPOINT_FEATURES = (
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
)
SHORTCUT_FEATURES = ("sttl", "dttl", "source_ttl", "destination_ttl")
CATEGORICAL_FEATURES = ("PROTOCOL", "L7_PROTO")
NUMERIC_FEATURES = (
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REDUCED_FEATURES = ("PROTOCOL", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS")
TARGET = "Label"
ATTACK = "Attack"
DOMAIN = "_domain"
GROUP = "_group_key"
FAMILY = "_family"
REQUIRED_COLUMNS = set(ENDPOINT_FEATURES + MODEL_FEATURES + (TARGET, ATTACK))
GROUP_BUCKETS = ("1", "2-9", "10-99", "100-999", "1000+")


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
    paths: dict[str, Path] = {}
    for domain, pattern in DATASET_GLOBS.items():
        attribute = domain.lower().replace("-", "_") + "_path"
        explicit = getattr(args, attribute)
        if explicit is not None:
            path = explicit.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            matches = sorted(Path.cwd().glob(pattern))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one {domain} file for {pattern!r}; found {len(matches)}"
                )
            path = matches[0].resolve()
        paths[domain] = path
    return paths


def expected_dataset_manifest(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    locked = protocol["locked_inputs"]
    external = read_json(Path(locked["external_netflow_validation_summary"]["path"]))
    bot = read_json(Path(locked["third_dataset_validation_summary"]["path"]))
    cse = read_json(Path(locked["fresh_generalisation_audit_summary"]["path"]))
    return {
        "NF-UNSW-NB15": {
            "sha256": external["data_provenance"]["source_sha256"],
            "rows": int(external["data_provenance"]["source_rows"]),
        },
        "NF-ToN-IoT": {
            "sha256": external["data_provenance"]["external_sha256"],
            "rows": int(external["data_provenance"]["external_rows"]),
        },
        "NF-BoT-IoT": {
            "sha256": bot["dataset"]["sha256"],
            "rows": int(bot["dataset"]["rows"]),
        },
        "NF-CSE-CIC-IDS2018": {
            "sha256": cse["integrity"]["dataset_sha256"],
            "rows": int(cse["integrity"]["global_rows"]),
        },
    }


def verify_protocol(protocol: dict[str, Any]) -> None:
    if protocol["status"] != "REGISTERED_BEFORE_V2_IMPLEMENTATION_OR_MODEL_FITTING":
        raise ValueError("Version 2 protocol has the wrong registration status")
    registered_domains = {item["name"] for item in protocol["development_domains"]}
    if registered_domains != set(DATASET_GLOBS):
        raise ValueError("Registered domains do not match runner domains")
    features = protocol["features"]
    if tuple(features["all_common_endpoint_free_features"]) != MODEL_FEATURES:
        raise ValueError("Registered all-feature set does not match the runner")
    if tuple(features["reduced_ablation_features"]) != REDUCED_FEATURES:
        raise ValueError("Registered reduced feature set does not match the runner")
    if set(features["excluded_endpoint_features"]) != set(ENDPOINT_FEATURES):
        raise ValueError("Endpoint exclusion does not match the runner")
    if set(features["excluded_shortcut_features"]) != set(SHORTCUT_FEATURES):
        raise ValueError("Shortcut exclusion does not match the runner")
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
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Implementation hash mismatch: {path}")
    return manifest


def validate_header(path: Path) -> None:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_COLUMNS - header
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")


def preflight(
    protocol: dict[str, Any], paths: dict[str, Path], chunk_size: int
) -> dict[str, Any]:
    if chunk_size < 10_000:
        raise ValueError("--chunk-size must be at least 10,000")
    verify_protocol(protocol)
    expected = expected_dataset_manifest(protocol)
    rows = []
    for domain, path in paths.items():
        validate_header(path)
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


def round_to_significant(values: np.ndarray, digits: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    result = array.copy()
    finite_nonzero = np.isfinite(array) & (array != 0)
    magnitudes = np.zeros_like(array)
    magnitudes[finite_nonzero] = np.floor(np.log10(np.abs(array[finite_nonzero])))
    steps = np.ones_like(array)
    steps[finite_nonzero] = np.power(10.0, magnitudes[finite_nonzero] - digits + 1)
    result[finite_nonzero] = (
        np.rint(array[finite_nonzero] / steps[finite_nonzero])
        * steps[finite_nonzero]
    )
    result[result == 0] = 0.0
    return result


def canonical_group_hashes(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Return dataset-namespaced SHA-256 keys and first two hash words."""
    matrix = np.empty((len(frame), len(MODEL_FEATURES)), dtype="<f8")
    for column_index, feature in enumerate(MODEL_FEATURES):
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype="float64")
        if feature in NUMERIC_FEATURES:
            values = round_to_significant(values)
        matrix[:, column_index] = values
    matrix[np.isnan(matrix)] = np.nan
    prefix = domain.encode("utf-8") + b"\0"
    keys: list[str] = []
    first = np.empty(len(frame), dtype="uint64")
    second = np.empty(len(frame), dtype="uint64")
    for index, row in enumerate(matrix):
        digest = hashlib.sha256(prefix + row.tobytes(order="C")).digest()
        keys.append(digest.hex())
        first[index] = int.from_bytes(digest[:8], "big")
        second[index] = int.from_bytes(digest[8:16], "big")
    return pd.DataFrame({GROUP: keys, "_first": first, "_second": second})


def normalise_chunk(chunk: pd.DataFrame, domain: str) -> pd.DataFrame:
    for feature in MODEL_FEATURES:
        chunk[feature] = pd.to_numeric(chunk[feature], errors="coerce")
    chunk[TARGET] = pd.to_numeric(chunk[TARGET], errors="raise").astype("int8")
    if not set(chunk[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{domain} contains a non-binary target")
    chunk[ATTACK] = chunk[ATTACK].fillna("<MISSING>").astype(str)
    chunk[FAMILY] = np.where(
        chunk[TARGET].eq(0), "Normal", chunk[ATTACK].astype(str)
    )
    chunk[DOMAIN] = domain
    return chunk


def group_partition(first_hash_word: pd.Series | np.ndarray) -> np.ndarray:
    return np.where(
        np.asarray(first_hash_word, dtype="uint64") <= TRAINING_CUTOFF,
        "training",
        "calibration",
    )


def scan_domain_groups(
    path: Path, domain: str, expected_rows: int, chunk_size: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    pieces = []
    total_rows = 0
    label_counts = {0: 0, 1: 0}
    attack_counts: dict[str, int] = {}
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunk_size), start=1
    ):
        chunk = normalise_chunk(chunk, domain)
        hashes = canonical_group_hashes(chunk, domain)
        chunk[GROUP] = hashes[GROUP].to_numpy()
        chunk["_second"] = hashes["_second"].to_numpy()
        chunk["_partition"] = group_partition(hashes["_first"])
        total_rows += len(chunk)
        for label, count in chunk[TARGET].value_counts().items():
            label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)
        for attack, count in chunk[ATTACK].value_counts().items():
            attack_counts[str(attack)] = attack_counts.get(str(attack), 0) + int(count)
        pieces.append(
            chunk.groupby(
                [GROUP, "_second", "_partition", TARGET, FAMILY],
                sort=False,
                observed=True,
            )
            .size()
            .rename("stratum_rows")
            .reset_index()
        )
        print(f"scan_progress domain={domain} chunk={chunk_index} rows={total_rows}", flush=True)
    if total_rows != expected_rows:
        raise ValueError(
            f"Row-count mismatch for {domain}: {total_rows} != {expected_rows}"
        )
    strata = pd.concat(pieces, ignore_index=True)
    strata = (
        strata.groupby(
            [GROUP, "_second", "_partition", TARGET, FAMILY],
            sort=False,
            observed=True,
        )["stratum_rows"]
        .sum()
        .reset_index()
    )
    group_rows = strata.groupby([GROUP, "_second", "_partition"], sort=False)[
        "stratum_rows"
    ].sum()
    if group_rows.reset_index().groupby(GROUP)["_partition"].nunique().max() != 1:
        raise RuntimeError(f"A {domain} group crossed source partitions")
    integrity = {
        "rows": total_rows,
        "label_counts": {str(key): value for key, value in label_counts.items()},
        "attack_counts": dict(sorted(attack_counts.items())),
        "groups": int(strata[GROUP].nunique()),
    }
    return strata, integrity


def select_groups(strata: pd.DataFrame, partition: str, cap: int) -> set[str]:
    subset = strata.loc[strata["_partition"].eq(partition)].copy()
    if subset.empty:
        raise ValueError(f"No groups available for {partition}")
    group_rows = subset.groupby(GROUP, sort=False)["stratum_rows"].sum().to_dict()
    selected: set[str] = set()
    selected_rows = 0

    # First preserve every observed class/raw-attack stratum with its lowest-hash group.
    coverage = subset.sort_values([FAMILY, "_second", GROUP]).groupby(
        FAMILY, sort=True, observed=True
    ).head(1)
    for key in coverage[GROUP]:
        if key not in selected:
            selected.add(key)
            selected_rows += int(group_rows[key])

    # Then cycle deterministically through strata so a large family cannot consume the cap.
    queues = {
        family: list(
            group.sort_values(["_second", GROUP])[GROUP].drop_duplicates()
        )
        for family, group in subset.groupby(FAMILY, sort=True, observed=True)
    }
    positions = {family: 0 for family in queues}
    families = sorted(queues)
    while selected_rows < cap:
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
            selected_rows += int(group_rows[key])
            added = True
            if selected_rows >= cap:
                break
        if not added:
            break
    return selected


def materialise_selected_groups(
    path: Path,
    domain: str,
    selected: dict[str, set[str]],
    chunk_size: int,
) -> dict[str, pd.DataFrame]:
    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    retained: dict[str, list[pd.DataFrame]] = {"training": [], "calibration": []}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk = normalise_chunk(chunk, domain)
        hashes = canonical_group_hashes(chunk, domain)
        chunk[GROUP] = hashes[GROUP].to_numpy()
        partitions = group_partition(hashes["_first"])
        for partition in retained:
            mask = (partitions == partition) & chunk[GROUP].isin(selected[partition])
            if mask.any():
                retained[partition].append(chunk.loc[mask].copy())
    frames = {
        partition: pd.concat(parts, ignore_index=True)
        for partition, parts in retained.items()
    }
    for partition, frame in frames.items():
        if set(frame[GROUP]) != selected[partition]:
            raise RuntimeError(f"Failed to materialise every {domain} {partition} group")
    if set(frames["training"][GROUP]) & set(frames["calibration"][GROUP]):
        raise RuntimeError(f"{domain} source partitions overlap")
    return frames


def prepare_domain_samples(
    path: Path,
    domain: str,
    expected_rows: int,
    protocol: dict[str, Any],
    chunk_size: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    strata, integrity = scan_domain_groups(path, domain, expected_rows, chunk_size)
    sampling = protocol["group_and_sampling_protocol"]
    selected = {
        "training": select_groups(
            strata, "training", int(sampling["training_cap_per_source_domain"])
        ),
        "calibration": select_groups(
            strata, "calibration", int(sampling["calibration_cap_per_source_domain"])
        ),
    }
    frames = materialise_selected_groups(path, domain, selected, chunk_size)
    integrity["samples"] = {
        name: {
            "rows": int(len(frame)),
            "groups": int(frame[GROUP].nunique()),
            "positive_rows": int(frame[TARGET].sum()),
            "families": int(frame[FAMILY].nunique()),
        }
        for name, frame in frames.items()
    }
    return frames, integrity


def log1p_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0))


def make_preprocessor(features: tuple[str, ...], scale: bool) -> ColumnTransformer:
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        (
            "log1p",
            FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one"),
        ),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append(
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
            )
        )
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def candidate_features(specification: dict[str, Any]) -> tuple[str, ...]:
    if specification["features"] == "all eight common endpoint-free features":
        return MODEL_FEATURES
    if specification["features"] == "reduced five-feature set":
        return REDUCED_FEATURES
    raise ValueError(f"Unknown feature declaration: {specification['features']}")


def make_candidate(specification: dict[str, Any]) -> Pipeline:
    settings = specification["settings"]
    features = candidate_features(specification)
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
        [("preprocess", make_preprocessor(features, scale)), ("model", estimator)]
    )


def domain_family_weights(frame: pd.DataFrame, cap_multiple: float = 10.0) -> np.ndarray:
    counts = frame.groupby([DOMAIN, FAMILY], observed=True).size().rename("rows")
    domain_family_counts = counts.groupby(level=0).size().to_dict()
    raw = np.empty(len(frame), dtype="float64")
    for index, (domain, family) in enumerate(zip(frame[DOMAIN], frame[FAMILY])):
        raw[index] = 1.0 / (
            len(domain_family_counts)
            * domain_family_counts[domain]
            * int(counts.loc[(domain, family)])
        )
    cap = cap_multiple * float(np.median(raw[raw > 0]))
    raw = np.minimum(raw, cap)
    return raw / raw.mean()


def fit_candidate(
    specification: dict[str, Any], training: pd.DataFrame
) -> tuple[Pipeline, np.ndarray | None]:
    model = make_candidate(specification)
    features = candidate_features(specification)
    weights = domain_family_weights(training) if specification["sample_weight"] else None
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    model.fit(training.loc[:, features], training[TARGET], **fit_kwargs)
    return model, weights


def confusion_components(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    predictions = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    tn = int(np.sum(~predictions & ~positive))
    return tn, fp, fn, tp


def row_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_components(labels, probabilities, threshold)
    prevalence = float(labels.mean())
    specificity = tn / (tn + fp) if tn + fp else 0.0
    normal = labels == 0
    attack = labels == 1
    normal_mean = float(probabilities[normal].mean()) if normal.any() else None
    attack_mean = float(probabilities[attack].mean()) if attack.any() else None
    average_precision = float(average_precision_score(labels, probabilities))
    return {
        "rows": int(len(labels)),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "average_precision": average_precision,
        "average_precision_lift": average_precision / max(prevalence, 1e-12),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(1.0 - specificity),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "normal_mean_probability": normal_mean,
        "attack_mean_probability": attack_mean,
        "attack_minus_normal_mean_probability": (
            attack_mean - normal_mean
            if normal_mean is not None and attack_mean is not None
            else None
        ),
    }


def select_constrained_threshold(
    calibration: pd.DataFrame,
    probabilities: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[float, pd.DataFrame, bool]:
    threshold_rule = protocol["threshold_rule"]
    grid = threshold_rule["grid"]
    thresholds = np.round(
        np.arange(grid["minimum"], grid["maximum"] + grid["step"] / 2, grid["step"]),
        10,
    )
    minimum_rows = int(protocol["group_and_sampling_protocol"]["minimum_supported_attack_rows"])
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
            tn, fp, fn, tp = confusion_components(labels, scores, float(threshold))
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            domain_mcc.append((tp * tn - fp * fn) / denominator if denominator else 0.0)
            domain_fpr.append(fp / (fp + tn) if fp + tn else 0.0)
        recalls = []
        for domain, attack in supported:
            group = working.loc[
                working[DOMAIN].eq(domain)
                & working[ATTACK].eq(attack)
                & working[TARGET].eq(1)
            ]
            recalls.append(float((group["probability"] >= threshold).mean()))
        min_recall = min(recalls) if recalls else 1.0
        max_fpr = max(domain_fpr)
        feasible = (
            max_fpr
            <= float(threshold_rule["constraints"]["maximum_false_positive_rate_in_each_source_domain"])
            and min_recall
            >= float(threshold_rule["constraints"]["minimum_recall_for_each_supported_attack_label"])
        )
        rows.append(
            {
                "threshold": float(threshold),
                "minimum_domain_mcc": float(min(domain_mcc)),
                "mean_domain_mcc": float(np.mean(domain_mcc)),
                "minimum_supported_attack_recall": float(min_recall),
                "maximum_domain_false_positive_rate": float(max_fpr),
                "supported_attack_strata": int(len(supported)),
                "feasible": bool(feasible),
            }
        )
    table = pd.DataFrame(rows)
    pool = table.loc[table["feasible"]]
    constraints_passed = not pool.empty
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
    return float(chosen["threshold"]), table, constraints_passed


def metrics_from_group_confusions(groups: pd.DataFrame) -> dict[str, Any]:
    positives = float(groups["positives"].sum())
    negatives = float(groups["negatives"].sum())
    tp = float(groups["true_positive"].sum())
    fp = float(groups["false_positive"].sum())
    fn = positives - tp
    tn = negatives - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else None
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "rows": int(positives + negatives),
        "positive_rows": int(positives),
        "negative_rows": int(negatives),
        "mcc": float((tp * tn - fp * fn) / denominator if denominator else 0.0),
        "balanced_accuracy": (
            float((recall + specificity) / 2) if specificity is not None else None
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity) if specificity is not None else None,
        "false_positive_rate": (
            float(1.0 - specificity) if specificity is not None else None
        ),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_frame(
    frame: pd.DataFrame,
    model: Pipeline,
    features: tuple[str, ...],
    threshold: float,
) -> dict[str, Any]:
    scores = model.predict_proba(frame.loc[:, features])[:, 1]
    return row_metrics(frame[TARGET].to_numpy(dtype="int8"), scores, threshold)


def stream_heldout_evaluation(
    path: Path,
    domain: str,
    model: Pipeline,
    features: tuple[str, ...],
    threshold: float,
    candidate: str,
    chunk_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    group_pieces = []
    attack_pieces = []
    label_pieces = []
    score_pieces = []
    total_rows = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunk_size), start=1
    ):
        chunk = normalise_chunk(chunk, domain)
        hashes = canonical_group_hashes(chunk, domain)
        scores = model.predict_proba(chunk.loc[:, features])[:, 1]
        predictions = scores >= threshold
        temporary = pd.DataFrame(
            {
                GROUP: hashes[GROUP].to_numpy(),
                "label": chunk[TARGET].to_numpy(),
                "score": scores,
                ATTACK: chunk[ATTACK].to_numpy(),
                "predicted_positive": predictions.astype("int8"),
            }
        )
        temporary["true_positive"] = (
            temporary["predicted_positive"].eq(1) & temporary["label"].eq(1)
        ).astype("int8")
        temporary["false_positive"] = (
            temporary["predicted_positive"].eq(1) & temporary["label"].eq(0)
        ).astype("int8")
        group_pieces.append(
            temporary.groupby(GROUP, sort=False).agg(
                rows=("label", "size"),
                positives=("label", "sum"),
                true_positive=("true_positive", "sum"),
                false_positive=("false_positive", "sum"),
            )
        )
        positives = temporary.loc[temporary["label"].eq(1)].copy()
        attack_pieces.append(
            positives.groupby(ATTACK, sort=False).agg(
                positive_rows=("label", "size"),
                true_positive=("predicted_positive", "sum"),
            )
        )
        label_pieces.append(chunk[TARGET].to_numpy(dtype="int8"))
        score_pieces.append(scores)
        total_rows += len(chunk)
        print(
            f"evaluation_progress domain={domain} candidate={candidate} "
            f"chunk={chunk_index} rows={total_rows}",
            flush=True,
        )
    groups = pd.concat(group_pieces).reset_index()
    groups = (
        groups.groupby(GROUP, sort=False)
        .agg(
            rows=("rows", "sum"),
            positives=("positives", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
        )
        .reset_index()
    )
    groups["negatives"] = groups["rows"] - groups["positives"]
    metrics = row_metrics(
        np.concatenate(label_pieces), np.concatenate(score_pieces), threshold
    )
    attacks = pd.concat(attack_pieces).groupby(level=0).sum().reset_index()
    attacks["recall"] = attacks["true_positive"] / attacks["positive_rows"]
    attacks.insert(0, "candidate", candidate)
    attacks.insert(0, "heldout_domain", domain)

    groups["size_bucket"] = pd.cut(
        groups["rows"],
        bins=[0, 1, 9, 99, 999, np.inf],
        labels=GROUP_BUCKETS,
        include_lowest=True,
    ).astype(str)
    groups["mixed_label"] = groups["positives"].gt(0) & groups["negatives"].gt(0)
    subgroup_rows = []
    for dimension in ("size_bucket", "mixed_label"):
        for value, subset in groups.groupby(dimension, sort=True, observed=True):
            row = metrics_from_group_confusions(subset)
            subgroup_rows.append(
                {
                    "heldout_domain": domain,
                    "candidate": candidate,
                    "dimension": dimension,
                    "value": str(value),
                    **row,
                }
            )
    group_metrics = pd.DataFrame(subgroup_rows)
    groups = groups.rename(
        columns={
            "true_positive": f"{candidate}__true_positive",
            "false_positive": f"{candidate}__false_positive",
        }
    )
    return metrics, attacks, group_metrics, groups


def aggregate_candidates(
    fold_metrics: pd.DataFrame,
    attacks: pd.DataFrame,
    thresholds: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    minimum_rows = int(protocol["group_and_sampling_protocol"]["minimum_supported_attack_rows"])
    supported = attacks.loc[attacks["positive_rows"] >= minimum_rows]
    rows = []
    for candidate, group in fold_metrics.groupby("candidate", sort=True):
        candidate_attacks = supported.loc[supported["candidate"].eq(candidate)]
        candidate_thresholds = thresholds.loc[thresholds["candidate"].eq(candidate)]
        rows.append(
            {
                "candidate": candidate,
                "macro_mcc": float(group["mcc"].mean()),
                "worst_domain_mcc": float(group["mcc"].min()),
                "worst_domain_balanced_accuracy": float(group["balanced_accuracy"].min()),
                "macro_supported_attack_recall": float(candidate_attacks["recall"].mean()),
                "minimum_supported_attack_recall": float(candidate_attacks["recall"].min()),
                "maximum_domain_false_positive_rate": float(group["false_positive_rate"].max()),
                "threshold_constraints_all_folds": bool(candidate_thresholds["constraints_passed"].all()),
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
    )
    result["improvement_passed"] = (
        result["macro_mcc_improvement"].ge(0.05)
        & result["worst_balanced_accuracy_improvement"].ge(0.02)
        & result["maximum_fpr_increase"].le(0.02)
    )
    result["promotion_eligible"] = (
        result["candidate"].ne(baseline_id)
        & result["eligibility_passed"]
        & result["improvement_passed"]
    )
    return result


def bootstrap_metric(
    groups: pd.DataFrame, candidate: str, multiplicity: np.ndarray
) -> dict[str, float]:
    positive = groups["positives"].to_numpy(dtype="float64") * multiplicity
    negative = groups["negatives"].to_numpy(dtype="float64") * multiplicity
    tp = (
        groups[f"{candidate}__true_positive"].to_numpy(dtype="float64")
        * multiplicity
    ).sum()
    fp = (
        groups[f"{candidate}__false_positive"].to_numpy(dtype="float64")
        * multiplicity
    ).sum()
    positives = positive.sum()
    negatives = negative.sum()
    fn = positives - tp
    tn = negatives - fp
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "mcc": float((tp * tn - fp * fn) / denominator if denominator else 0.0),
        "balanced_accuracy": float((recall + specificity) / 2),
        "recall": float(recall),
        "false_positive_rate": float(1.0 - specificity),
    }


def paired_bootstrap_intervals(
    group_tables: dict[tuple[str, str], pd.DataFrame],
    aggregate: pd.DataFrame,
    iterations: int = 200,
) -> pd.DataFrame:
    baseline = "pooled_rf_unweighted_all_features"
    candidates = aggregate.loc[aggregate["promotion_eligible"], "candidate"].tolist()
    comparisons = [(baseline, None)] + [(candidate, baseline) for candidate in candidates]
    samples = []
    for domain in sorted({key[0] for key in group_tables}):
        base = group_tables[(domain, baseline)].copy()
        tables = {baseline: base}
        for candidate in candidates:
            candidate_counts = group_tables[(domain, candidate)][
                [
                    GROUP,
                    f"{candidate}__true_positive",
                    f"{candidate}__false_positive",
                ]
            ]
            tables[candidate] = base.merge(
                candidate_counts, on=GROUP, validate="one_to_one"
            )
        rng = np.random.default_rng(SEED + sum(domain.encode("utf-8")))
        for iteration in range(iterations):
            multiplicity = np.bincount(
                rng.integers(0, len(base), size=len(base)), minlength=len(base)
            ).astype("float64")
            baseline_metrics = bootstrap_metric(
                base,
                baseline,
                multiplicity,
            )
            for candidate, reference in comparisons:
                if reference is None:
                    values = baseline_metrics
                    comparison = baseline
                else:
                    candidate_metrics = bootstrap_metric(
                        tables[candidate],
                        candidate,
                        multiplicity,
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
                            "iteration": iteration,
                            "metric": metric,
                            "value": value,
                        }
                    )
    frame = pd.DataFrame(samples)
    return (
        frame.groupby(["heldout_domain", "comparison", "metric"])["value"]
        .agg(
            bootstrap_median="median",
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )


def create_plots(
    fold_metrics: pd.DataFrame,
    attacks: pd.DataFrame,
    overfitting: pd.DataFrame,
    output_dir: Path,
) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    pivot = fold_metrics.pivot(index="heldout_domain", columns="candidate", values="mcc")
    axis = pivot.plot(kind="bar", figsize=(13, 6))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("MCC")
    axis.set_title("Version 2 leave-one-dataset-out development performance")
    axis.legend(fontsize=7)
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
    axis.legend(fontsize=7)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "attack_family_recall.png", dpi=180)
    plt.close()

    overfitting = overfitting.copy()
    overfitting["label"] = overfitting["candidate"] + " / " + overfitting["metric"]
    chart = overfitting.pivot_table(
        index="label", columns="gap", values="value", aggfunc="mean"
    )
    axis = chart.plot(kind="bar", figsize=(12, 6))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Metric gap")
    axis.set_title("Average training, calibration and held-out gaps")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "overfitting_gaps.png", dpi=180)
    plt.close()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
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
    for domain, path in paths.items():
        samples[domain], integrity[domain] = prepare_domain_samples(
            path, domain, int(expected[domain]["rows"]), protocol, chunk_size
        )

    fold_metric_rows = []
    attack_tables = []
    threshold_rows = []
    overfitting_rows = []
    group_metric_tables = []
    group_tables: dict[tuple[str, str], pd.DataFrame] = {}
    candidates = protocol["candidate_experiments"]
    for heldout_domain in DATASET_GLOBS:
        source_domains = [domain for domain in DATASET_GLOBS if domain != heldout_domain]
        training = pd.concat(
            [samples[domain]["training"] for domain in source_domains],
            ignore_index=True,
        )
        calibration = pd.concat(
            [samples[domain]["calibration"] for domain in source_domains],
            ignore_index=True,
        )
        if heldout_domain in set(training[DOMAIN]) | set(calibration[DOMAIN]):
            raise RuntimeError("Held-out domain leaked into source development data")
        for specification in candidates:
            candidate = specification["id"]
            print(f"fit_start heldout={heldout_domain} candidate={candidate}", flush=True)
            model, weights = fit_candidate(specification, training)
            features = candidate_features(specification)
            calibration_scores = model.predict_proba(calibration.loc[:, features])[:, 1]
            threshold, threshold_table, constraints_passed = select_constrained_threshold(
                calibration, calibration_scores, protocol
            )
            threshold_table.to_csv(
                output_dir / f"threshold_grid__{heldout_domain}__{candidate}.csv",
                index=False,
            )
            threshold_rows.append(
                {
                    "heldout_domain": heldout_domain,
                    "candidate": candidate,
                    "threshold": threshold,
                    "constraints_passed": constraints_passed,
                }
            )
            training_metrics = evaluate_frame(training, model, features, threshold)
            calibration_working = calibration.copy()
            calibration_working[GROUP] = calibration[GROUP]
            calibration_metrics = evaluate_frame(
                calibration_working, model, features, threshold
            )
            heldout_metrics, attacks, group_metrics, heldout_groups = stream_heldout_evaluation(
                paths[heldout_domain],
                heldout_domain,
                model,
                features,
                threshold,
                candidate,
                chunk_size,
            )
            heldout_metrics.update(
                {
                    "heldout_domain": heldout_domain,
                    "candidate": candidate,
                    "source_domains": "|".join(source_domains),
                    "threshold_constraints_passed": constraints_passed,
                    "oob_score": (
                        float(model.named_steps["model"].oob_score_)
                        if hasattr(model.named_steps["model"], "oob_score_")
                        else None
                    ),
                    "weighted_fit": weights is not None,
                }
            )
            fold_metric_rows.append(heldout_metrics)
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
                            "value": calibration_metrics[metric] - heldout_metrics[metric],
                        },
                    ]
                )
            del model
            gc.collect()

    fold_metrics = pd.DataFrame(fold_metric_rows)
    attacks = pd.concat(attack_tables, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    overfitting = pd.DataFrame(overfitting_rows)
    group_metrics = pd.concat(group_metric_tables, ignore_index=True)
    aggregate = aggregate_candidates(fold_metrics, attacks, thresholds, protocol)
    eligible = aggregate.loc[aggregate["promotion_eligible"]].sort_values(
        [
            "macro_mcc",
            "worst_domain_balanced_accuracy",
            "minimum_supported_attack_recall",
            "maximum_domain_false_positive_rate",
            "candidate",
        ],
        ascending=[False, False, False, True, True],
    )
    selected = None if eligible.empty else str(eligible.iloc[0]["candidate"])
    verdict = "no Version 2 candidate selected" if selected is None else "development-promising research prototype selected"
    bootstrap = paired_bootstrap_intervals(group_tables, aggregate)

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_candidate_metrics.csv", index=False)
    attacks.to_csv(output_dir / "per_attack_recall.csv", index=False)
    thresholds.to_csv(output_dir / "thresholds.csv", index=False)
    overfitting.to_csv(output_dir / "overfitting_gaps.csv", index=False)
    group_metrics.to_csv(output_dir / "group_size_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_group_bootstrap_intervals.csv", index=False)
    create_plots(fold_metrics, attacks, overfitting, output_dir)
    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run",
            "origin_date": time.strftime("%Y-%m-%d"),
            "verification_status": "ANALYZED",
            "version_label": "v2_generalisation_development_v1",
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
