"""Run a leakage-resistant cross-dataset NetFlow validation experiment.

This experiment uses the common NetFlow-v1 representation published by the
University of Queensland. Models are developed only on NF-UNSW-NB15 and are
then evaluated, without refitting or threshold changes, on NF-ToN-IoT.

The four endpoint identifiers (source/destination IP address and port) are
excluded before splitting or modelling. Exact duplicates of the remaining
eight traffic features are assigned to the same source-data fold. Three folds
train the models, one fold selects an MCC-maximising threshold, and the final
source fold is left untouched for an in-domain check. NF-ToN-IoT is used only
after both models and thresholds have been frozen.

This is a matched-representation external stress test of the modelling
approach. It is not a direct test of the previously frozen 44-feature model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


OUTPUT_DIR = Path("artifacts/external_netflow_validation")
SOURCE_GLOB = "data/external/**/NF-UNSW-NB15.csv"
EXTERNAL_GLOB = "data/external/**/NF-ToN-IoT.csv"
SEED = 42

ENDPOINT_FEATURES = (
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
)
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
TARGET = "Label"
ATTACK = "Attack"
REQUIRED_COLUMNS = set(ENDPOINT_FEATURES + MODEL_FEATURES + (TARGET, ATTACK))
MODEL_NAMES = ("logistic_regression", "random_forest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--external-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rf-trees", type=int, default=120)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument(
        "--source-row-limit",
        type=int,
        help="Development-only smoke-test limit; omit for the full experiment.",
    )
    parser.add_argument(
        "--external-row-limit",
        type=int,
        help="External-data smoke-test limit; omit for the full experiment.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rf_trees < 20:
        raise ValueError("--rf-trees must be at least 20")
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be 0 or at least 100")
    for name in ("source_row_limit", "external_row_limit"):
        value = getattr(args, name)
        if value is not None and value < 5_000:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 5,000")


def resolve_single_path(explicit: Path | None, pattern: str) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = sorted(Path.cwd().glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one match for {pattern!r}, found {len(matches)}"
        )
    return matches[0].resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path, row_limit: int | None) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_COLUMNS - header
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    usecols = list(MODEL_FEATURES) + [TARGET, ATTACK]
    frame = pd.read_csv(path, usecols=usecols)
    if row_limit is not None and len(frame) > row_limit:
        # The smoke-test path is deterministic and label-stratified. Full results
        # must always be produced with both limits omitted.
        parts = []
        for label, group in frame.groupby(TARGET, sort=True):
            size = max(1, round(row_limit * len(group) / len(frame)))
            parts.append(group.sample(n=min(size, len(group)), random_state=SEED + int(label)))
        frame = pd.concat(parts, ignore_index=True).sample(
            frac=1.0, random_state=SEED
        ).reset_index(drop=True)
    for feature in MODEL_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name} has non-binary target values")
    frame[ATTACK] = frame[ATTACK].fillna("<MISSING>").astype(str)
    return frame


def exact_feature_groups(frame: pd.DataFrame) -> pd.Series:
    """Hash the endpoint-free model representation for duplicate grouping."""
    return pd.util.hash_pandas_object(
        frame.loc[:, MODEL_FEATURES], index=False
    ).astype("uint64")


def balanced_group_folds(
    groups: pd.Series, labels: pd.Series, n_splits: int = 5
) -> tuple[pd.Series, pd.DataFrame]:
    """Greedily balance row and class counts while keeping groups intact."""
    grouped = (
        pd.DataFrame({"group": groups.to_numpy(), "label": labels.to_numpy()})
        .groupby("group", sort=False)["label"]
        .agg(rows="size", positives="sum")
        .reset_index()
    )
    grouped["negatives"] = grouped["rows"] - grouped["positives"]
    targets = np.array(
        [grouped["rows"].sum(), grouped["positives"].sum(), grouped["negatives"].sum()],
        dtype=float,
    ) / n_splits
    rng = np.random.default_rng(SEED)
    grouped["tie_break"] = rng.random(len(grouped))
    grouped["priority"] = np.maximum.reduce(
        [
            grouped["rows"].to_numpy() / max(targets[0], 1.0),
            grouped["positives"].to_numpy() / max(targets[1], 1.0),
            grouped["negatives"].to_numpy() / max(targets[2], 1.0),
        ]
    )
    grouped = grouped.sort_values(
        ["priority", "tie_break"], ascending=[False, True]
    ).reset_index(drop=True)

    loads = np.zeros((n_splits, 3), dtype=float)
    assignments = np.empty(len(grouped), dtype=np.int8)
    safe_targets = np.maximum(targets, 1.0)
    for position, row in enumerate(
        grouped[["rows", "positives", "negatives"]].itertuples(index=False, name=None)
    ):
        contribution = np.asarray(row, dtype=float)
        projected = (loads + contribution) / safe_targets
        scores = np.square(projected).sum(axis=1)
        fold = int(np.argmin(scores))
        assignments[position] = fold
        loads[fold] += contribution
    grouped["fold"] = assignments
    mapping = pd.Series(grouped["fold"].to_numpy(), index=grouped["group"])
    row_folds = groups.map(mapping).astype("int8")
    if row_folds.isna().any():
        raise RuntimeError("Some source rows did not receive a fold")
    fold_summary = (
        pd.DataFrame({"fold": row_folds, "label": labels.to_numpy()})
        .groupby("fold")["label"]
        .agg(rows="size", suspicious="sum", suspicious_rate="mean")
        .reset_index()
    )
    fold_summary["normal"] = fold_summary["rows"] - fold_summary["suspicious"]
    return row_folds, fold_summary


def log1p_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(values, 0))


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        (
            "log1p",
            FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one"),
        ),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
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
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )


def make_models(rf_trees: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=True)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=300,
                        random_state=SEED,
                        solver="saga",
                        tol=1e-3,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=rf_trees,
                        max_depth=20,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        max_samples=0.70,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 197):
        predictions = (probabilities >= threshold).astype("int8")
        rows.append(
            {
                "threshold": float(threshold),
                "mcc": float(matthews_corrcoef(labels, predictions)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "precision": float(
                    precision_score(labels, predictions, zero_division=0)
                ),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
            }
        )
    table = pd.DataFrame(rows)
    selected = table.sort_values(
        ["mcc", "f1", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    return float(selected["threshold"]), table


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "rows": int(len(labels)),
        "prevalence": float(np.mean(labels)),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "average_precision_lift": float(
            average_precision_score(labels, probabilities) / max(np.mean(labels), 1e-12)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def per_attack_recall(
    dataset: str,
    attacks: pd.Series,
    labels: np.ndarray,
    model: str,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predictions = probabilities >= threshold
    rows = []
    working = pd.DataFrame(
        {"attack": attacks.to_numpy(), "label": labels, "prediction": predictions}
    )
    for attack, group in working.groupby("attack", sort=True):
        positives = group["label"].eq(1)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "attack": attack,
                "rows": int(len(group)),
                "positive_rows": int(positives.sum()),
                "recall": (
                    float(group.loc[positives, "prediction"].mean())
                    if positives.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_from_group_stats(
    positives: np.ndarray,
    negatives: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    total_positive = positives.sum()
    total_negative = negatives.sum()
    predicted_positive = scores >= threshold
    tp = positives[predicted_positive].sum()
    fp = negatives[predicted_positive].sum()
    fn = total_positive - tp
    tn = total_negative - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_positive if total_positive else 0.0
    specificity = tn / total_negative if total_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0

    unique_scores, inverse = np.unique(scores, return_inverse=True)
    descending_bin = len(unique_scores) - 1 - inverse
    bin_positive = np.bincount(
        descending_bin, weights=positives, minlength=len(unique_scores)
    )
    bin_negative = np.bincount(
        descending_bin, weights=negatives, minlength=len(unique_scores)
    )
    cumulative_positive = np.cumsum(bin_positive)
    cumulative_total = np.cumsum(bin_positive + bin_negative)
    average_precision = (
        np.sum(
            np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros_like(cumulative_positive),
                where=cumulative_total > 0,
            )
            * bin_positive
        )
        / total_positive
        if total_positive
        else np.nan
    )
    brier = (
        np.sum(positives * np.square(1 - scores) + negatives * np.square(scores))
        / (total_positive + total_negative)
    )
    return {
        "average_precision": float(average_precision),
        "f1": float(f1),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "brier_score": float(brier),
    }


def grouped_bootstrap(
    dataset: str,
    labels: np.ndarray,
    groups: pd.Series,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    iterations: int,
) -> pd.DataFrame:
    if iterations == 0:
        return pd.DataFrame()
    working = pd.DataFrame({"group": groups.to_numpy(), "label": labels})
    for model, values in probabilities.items():
        working[model] = values
    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("label", "size"),
        "positives": ("label", "sum"),
    }
    aggregations.update({model: (model, "first") for model in probabilities})
    stats = working.groupby("group", sort=False).agg(**aggregations).reset_index()
    stats["negatives"] = stats["rows"] - stats["positives"]
    group_count = len(stats)
    rng = np.random.default_rng(SEED + (0 if dataset == "source_validation" else 10_000))
    rows = []
    for iteration in range(iterations):
        multiplicity = np.bincount(
            rng.integers(0, group_count, size=group_count), minlength=group_count
        ).astype(float)
        positive = stats["positives"].to_numpy(dtype=float) * multiplicity
        negative = stats["negatives"].to_numpy(dtype=float) * multiplicity
        for model in probabilities:
            metrics = metric_from_group_stats(
                positive,
                negative,
                stats[model].to_numpy(dtype=float),
                thresholds[model],
            )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "iteration": iteration,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def confidence_intervals(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    return (
        samples.groupby(["dataset", "model", "metric"])["value"]
        .agg(
            estimate="median",
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )


def dataset_comparison(source: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in (("NF-UNSW-NB15", source), ("NF-ToN-IoT", external)):
        for feature in MODEL_FEATURES:
            values = frame[feature]
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "rows": int(len(frame)),
                    "prevalence": float(frame[TARGET].mean()),
                    "missing_rate": float(values.isna().mean()),
                    "unique_values": int(values.nunique(dropna=True)),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    output_dir: Path,
    evaluations: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    metrics: pd.DataFrame,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, (dataset, models) in zip(axes, evaluations.items()):
        for model, (labels, probabilities) in models.items():
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            ap = average_precision_score(labels, probabilities)
            axis.plot(recall, precision, label=f"{model.replace('_', ' ')} (AP={ap:.3f})")
        axis.axhline(np.mean(labels), color="grey", linestyle="--", label="prevalence")
        axis.set_title(dataset.replace("_", " ").title())
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    fig.suptitle("Locked-model precision-recall curves")
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_recall_curves.png", dpi=180)
    plt.close(fig)

    chart = metrics[metrics["metric"].isin(["average_precision", "mcc", "f1"])].copy()
    labels = [f"{row.dataset}\n{row.model.replace('_', ' ')}" for row in chart.itertuples()]
    fig, axis = plt.subplots(figsize=(12, 5))
    x = np.arange(len(chart))
    axis.bar(x, chart["value"], color=["#3b82f6" if d == "source_validation" else "#f97316" for d in chart["dataset"]])
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1.02)
    axis.set_title("In-domain versus external performance")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "generalisation_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_single_path(args.source_path, SOURCE_GLOB)
    external_path = resolve_single_path(args.external_path, EXTERNAL_GLOB)

    source = load_dataset(source_path, args.source_row_limit)
    external = load_dataset(external_path, args.external_row_limit)
    source_groups = exact_feature_groups(source)
    external_groups = exact_feature_groups(external)
    source_folds, fold_summary = balanced_group_folds(source_groups, source[TARGET])

    masks = {
        "training": source_folds.isin([0, 1, 2]).to_numpy(),
        "threshold_calibration": source_folds.eq(3).to_numpy(),
        "source_validation": source_folds.eq(4).to_numpy(),
    }
    split_summary_rows = []
    for split, mask in masks.items():
        split_summary_rows.append(
            {
                "split": split,
                "rows": int(mask.sum()),
                "suspicious": int(source.loc[mask, TARGET].sum()),
                "suspicious_rate": float(source.loc[mask, TARGET].mean()),
                "exact_feature_groups": int(source_groups.loc[mask].nunique()),
            }
        )
    split_summary = pd.DataFrame(split_summary_rows)
    overlap_checks = {
        "training_calibration": int(
            len(set(source_groups[masks["training"]]) & set(source_groups[masks["threshold_calibration"]]))
        ),
        "training_validation": int(
            len(set(source_groups[masks["training"]]) & set(source_groups[masks["source_validation"]]))
        ),
        "calibration_validation": int(
            len(set(source_groups[masks["threshold_calibration"]]) & set(source_groups[masks["source_validation"]]))
        ),
    }
    if any(overlap_checks.values()):
        raise RuntimeError(f"Exact-feature groups crossed source splits: {overlap_checks}")

    X = source.loc[:, MODEL_FEATURES]
    y = source[TARGET].to_numpy()
    X_external = external.loc[:, MODEL_FEATURES]
    y_external = external[TARGET].to_numpy()
    models = make_models(args.rf_trees)
    thresholds: dict[str, float] = {}
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "source_validation": {},
        "external": {},
    }
    metric_rows = []
    attack_tables = []
    fit_times = {}

    for model_name, pipeline in models.items():
        fit_start = time.time()
        pipeline.fit(X.loc[masks["training"]], y[masks["training"]])
        fit_times[model_name] = time.time() - fit_start
        calibration_probability = pipeline.predict_proba(
            X.loc[masks["threshold_calibration"]]
        )[:, 1]
        threshold, threshold_table = select_threshold(
            y[masks["threshold_calibration"]], calibration_probability
        )
        thresholds[model_name] = threshold
        threshold_table.insert(0, "model", model_name)
        threshold_table.to_csv(
            output_dir / f"{model_name}_threshold_calibration.csv", index=False
        )

        source_probability = pipeline.predict_proba(
            X.loc[masks["source_validation"]]
        )[:, 1]
        external_probability = pipeline.predict_proba(X_external)[:, 1]
        probabilities["source_validation"][model_name] = source_probability
        probabilities["external"][model_name] = external_probability
        for dataset, labels, values in (
            ("source_validation", y[masks["source_validation"]], source_probability),
            ("external", y_external, external_probability),
        ):
            result = calculate_metrics(labels, values, threshold)
            for metric, value in result.items():
                metric_rows.append(
                    {"dataset": dataset, "model": model_name, "metric": metric, "value": value}
                )
        attack_tables.append(
            per_attack_recall(
                "source_validation",
                source.loc[masks["source_validation"], ATTACK],
                y[masks["source_validation"]],
                model_name,
                source_probability,
                threshold,
            )
        )
        attack_tables.append(
            per_attack_recall(
                "external",
                external[ATTACK],
                y_external,
                model_name,
                external_probability,
                threshold,
            )
        )
        joblib.dump(pipeline, output_dir / f"{model_name}_netflow_v1.joblib")

    metrics = pd.DataFrame(metric_rows)
    per_attack = pd.concat(attack_tables, ignore_index=True)
    bootstrap_tables = []
    for dataset, labels, group_values in (
        (
            "source_validation",
            y[masks["source_validation"]],
            source_groups.loc[masks["source_validation"]].reset_index(drop=True),
        ),
        ("external", y_external, external_groups.reset_index(drop=True)),
    ):
        bootstrap_tables.append(
            grouped_bootstrap(
                dataset,
                labels,
                group_values,
                probabilities[dataset],
                thresholds,
                args.bootstrap_iterations,
            )
        )
    bootstrap_samples = pd.concat(bootstrap_tables, ignore_index=True)
    intervals = confidence_intervals(bootstrap_samples)
    comparison = dataset_comparison(source, external)

    metrics.to_csv(output_dir / "metrics_long.csv", index=False)
    split_summary.to_csv(output_dir / "source_split_summary.csv", index=False)
    fold_summary.to_csv(output_dir / "source_fold_summary.csv", index=False)
    per_attack.to_csv(output_dir / "per_attack_recall.csv", index=False)
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    bootstrap_samples.to_csv(output_dir / "group_bootstrap_samples.csv", index=False)
    intervals.to_csv(output_dir / "group_bootstrap_intervals.csv", index=False)
    make_plots(
        output_dir,
        {
            "source_validation": {
                model: (y[masks["source_validation"]], values)
                for model, values in probabilities["source_validation"].items()
            },
            "external": {
                model: (y_external, values)
                for model, values in probabilities["external"].items()
            },
        },
        metrics,
    )

    summary_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for dataset in ("source_validation", "external"):
        summary_metrics[dataset] = {}
        for model in MODEL_NAMES:
            subset = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == model)]
            summary_metrics[dataset][model] = {
                row.metric: row.value for row in subset.itertuples()
            }
    summary = {
        "experiment_scope": {
            "claim": "Matched-representation external stress test of the modelling approach",
            "not_claimed": "Direct external validation of the frozen 44-feature UNSW-NB15 model",
            "source_dataset": "NF-UNSW-NB15 NetFlow v1",
            "external_dataset": "NF-ToN-IoT NetFlow v1",
            "endpoint_features_excluded": list(ENDPOINT_FEATURES),
            "model_features": list(MODEL_FEATURES),
            "threshold_selection": "MCC maximised on source fold 3 only",
            "external_data_used_for_tuning": False,
        },
        "data_provenance": {
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "external_path": str(external_path),
            "external_sha256": sha256_file(external_path),
            "source_rows": int(len(source)),
            "external_rows": int(len(external)),
            "source_row_limit": args.source_row_limit,
            "external_row_limit": args.external_row_limit,
        },
        "split_integrity": {
            "group_definition": "Exact duplicate across the eight endpoint-free model features",
            "overlap_checks": overlap_checks,
            "source_unique_groups": int(source_groups.nunique()),
            "external_unique_groups": int(external_groups.nunique()),
            "folds": fold_summary.to_dict(orient="records"),
            "splits": split_summary.to_dict(orient="records"),
        },
        "models": {
            model: {
                "threshold": thresholds[model],
                "fit_seconds": fit_times[model],
                "artifact": f"{model}_netflow_v1.joblib",
            }
            for model in MODEL_NAMES
        },
        "metrics": summary_metrics,
        "uncertainty": {
            "method": "Exact-feature-group bootstrap",
            "iterations": args.bootstrap_iterations,
            "interval_file": "group_bootstrap_intervals.csv",
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
