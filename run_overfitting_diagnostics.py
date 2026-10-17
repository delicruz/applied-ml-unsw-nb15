"""Run reproducible overfitting diagnostics without opening the protected test set.

The analysis uses only the development-training and validation partitions. Exact
predictor duplicates remain grouped during cross-validation and learning-curve
construction. Outputs are written to ``artifacts/overfitting_diagnostics``.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from run_preliminary_validation import (
    ANALYSIS_ONLY,
    CATEGORICAL,
    DEVELOPMENT_PATH,
    SEED,
    TARGET,
    VALIDATION_PATH,
    calculate_metrics,
    predictor_hashes,
    select_threshold,
    split_xy,
)


OUTPUT_DIR = Path("artifacts/overfitting_diagnostics")
ABLATION_VARIANTS = {
    "all_features": [],
    "without_proto": ["proto"],
    "without_sttl": ["sttl"],
    "without_ct_state_ttl": ["ct_state_ttl"],
    "without_all_three": ["proto", "sttl", "ct_state_ttl"],
}
METRICS = [
    "average_precision",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "false_positive_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-trees", type=int, default=250)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for tables, metadata and plots.",
    )
    return parser.parse_args()


def make_random_forest(
    frame: pd.DataFrame,
    trees: int,
    *,
    excluded: tuple[str, ...] = (),
    oob: bool = False,
) -> Pipeline:
    columns = [column for column in frame.columns if column not in excluded]
    categorical = [column for column in CATEGORICAL if column in columns]
    numeric = [column for column in columns if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=trees,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    bootstrap=True,
                    oob_score=oob,
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )


def flatten_metrics(
    split: str, metrics: dict[str, Any], *, seconds: float | None = None
) -> dict[str, Any]:
    row = {"split": split}
    row.update({key: value for key, value in metrics.items() if key != "confusion_matrix"})
    row.update(metrics["confusion_matrix"])
    if seconds is not None:
        row["seconds"] = seconds
    return row


def validate_partitions(
    development: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, Any]:
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    expected = {0, 1}
    if set(development[TARGET].unique()) != expected:
        raise ValueError("Development target is not binary")
    if set(validation[TARGET].unique()) != expected:
        raise ValueError("Validation target is not binary")
    overlap = set(predictor_hashes(development)).intersection(
        set(predictor_hashes(validation))
    )
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} exact predictor groups across development and validation"
        )
    X_development, _ = split_xy(development)
    excluded_present = sorted(set(ANALYSIS_ONLY).intersection(X_development.columns))
    if excluded_present:
        raise ValueError(f"Analysis-only fields entered model inputs: {excluded_present}")
    return {
        "same_schema": True,
        "development_rows": len(development),
        "validation_rows": len(validation),
        "predictor_columns": X_development.shape[1],
        "exact_predictor_overlap": 0,
        "development_suspicious_rate": float(development[TARGET].mean()),
        "validation_suspicious_rate": float(validation[TARGET].mean()),
        "analysis_only_fields_in_model": excluded_present,
    }


def audit_univariate_features(
    development: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    X_development, y_development = split_xy(development)
    X_validation, y_validation = split_xy(validation)
    rows: list[dict[str, Any]] = []
    for feature in X_development.columns:
        if feature in CATEGORICAL:
            means = development.groupby(feature, dropna=False)[TARGET].mean()
            probabilities = (
                validation[feature]
                .map(means)
                .fillna(float(y_development.mean()))
                .to_numpy()
            )
            grouped = development.groupby(feature, dropna=False)[TARGET].agg(
                rows="size", classes="nunique"
            )
            repeated = grouped[grouped["rows"] >= 20]
            auc = roc_auc_score(y_validation, probabilities)
        else:
            median = float(X_development[feature].median())
            train_values = X_development[[feature]].fillna(median)
            validation_values = X_validation[[feature]].fillna(median)
            stump = DecisionTreeClassifier(
                max_depth=1, min_samples_leaf=100, random_state=SEED
            )
            stump.fit(train_values, y_development)
            stump_probabilities = stump.predict_proba(validation_values)[:, 1]
            raw_auc = roc_auc_score(
                y_validation, validation_values[feature].to_numpy()
            )
            auc = max(
                roc_auc_score(y_validation, stump_probabilities),
                raw_auc,
                1 - raw_auc,
            )
            grouped = development.groupby(feature, dropna=False)[TARGET].agg(
                rows="size", classes="nunique"
            )
            repeated = grouped[grouped["rows"] >= 20]
        deterministic = repeated[repeated["classes"] == 1]
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical" if feature in CATEGORICAL else "numeric",
                "unique_values": int(development[feature].nunique(dropna=False)),
                "best_univariate_validation_auc": float(auc),
                "deterministic_repeated_values": int(len(deterministic)),
                "deterministic_rows": int(deterministic["rows"].sum()),
                "deterministic_row_share": float(
                    deterministic["rows"].sum() / len(development)
                ),
                "near_perfect_auc_flag": bool(auc >= 0.98),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "best_univariate_validation_auc", ascending=False
    )


def first_grouped_split(
    X: pd.DataFrame, development: pd.DataFrame, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(splitter.split(X, development["attack_cat"], groups=groups))


def tune_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    fit_indices: np.ndarray,
    tuning_indices: np.ndarray,
    trees: int,
    excluded: tuple[str, ...] = (),
) -> float:
    columns = [column for column in X.columns if column not in excluded]
    model = make_random_forest(X.iloc[fit_indices][columns], trees, excluded=excluded)
    model.fit(X.iloc[fit_indices][columns], y.iloc[fit_indices])
    probabilities = model.predict_proba(X.iloc[tuning_indices][columns])[:, 1]
    threshold, _ = select_threshold(y.iloc[tuning_indices], probabilities)
    return threshold


def run_generalisation_and_oob(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float,
    trees: int,
) -> tuple[pd.DataFrame, Pipeline]:
    print("Fitting full Random Forest with OOB predictions...", flush=True)
    started = time.time()
    model = make_random_forest(X_development, trees, oob=True)
    model.fit(X_development, y_development)
    fit_seconds = time.time() - started
    rows = []
    for name, X, y in (
        ("training", X_development, y_development),
        ("validation", X_validation, y_validation),
    ):
        metrics, _ = calculate_metrics(y, model.predict_proba(X)[:, 1], threshold)
        rows.append(flatten_metrics(name, metrics, seconds=fit_seconds if name == "training" else None))
    oob_probabilities = model.named_steps["model"].oob_decision_function_[:, 1]
    if not np.isfinite(oob_probabilities).all():
        raise ValueError("Some development rows did not receive OOB predictions")
    oob_metrics, _ = calculate_metrics(y_development, oob_probabilities, threshold)
    rows.append(flatten_metrics("out_of_bag", oob_metrics))
    return pd.DataFrame(rows), model


def run_grouped_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    development: pd.DataFrame,
    groups: pd.Series,
    threshold: float,
    trees: int,
    folds: int,
) -> pd.DataFrame:
    rows = []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X, development["attack_cat"], groups=groups), start=1
    ):
        started = time.time()
        print(f"Grouped CV fold {fold}/{folds}...", flush=True)
        model = make_random_forest(X.iloc[train_indices], trees)
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        probabilities = model.predict_proba(X.iloc[validation_indices])[:, 1]
        metrics, _ = calculate_metrics(y.iloc[validation_indices], probabilities, threshold)
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "validation_suspicious_rate": float(y.iloc[validation_indices].mean()),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                **metrics["confusion_matrix"],
                "seconds": time.time() - started,
            }
        )
    return pd.DataFrame(rows)


def run_ablations(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    fit_indices: np.ndarray,
    tuning_indices: np.ndarray,
    trees: int,
    baseline_threshold: float,
    baseline_model: Pipeline,
) -> pd.DataFrame:
    rows = []
    for name, excluded_list in ABLATION_VARIANTS.items():
        excluded = tuple(excluded_list)
        columns = [column for column in X_development.columns if column not in excluded]
        started = time.time()
        print(f"Ablation: {name}...", flush=True)
        if name == "all_features":
            threshold = baseline_threshold
            model = baseline_model
        else:
            threshold = tune_threshold(
                X_development,
                y_development,
                fit_indices,
                tuning_indices,
                trees,
                excluded,
            )
            model = make_random_forest(
                X_development[columns], trees, excluded=excluded
            )
            model.fit(X_development[columns], y_development)
        probabilities = model.predict_proba(X_validation[columns])[:, 1]
        metrics, _ = calculate_metrics(y_validation, probabilities, threshold)
        rows.append(
            {
                "variant": name,
                "removed": ",".join(excluded) or "none",
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                **metrics["confusion_matrix"],
                "seconds": time.time() - started,
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.iloc[0]
    for metric in METRICS:
        result[f"delta_{metric}"] = result[metric] - baseline[metric]
    return result


def run_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    development: pd.DataFrame,
    groups: pd.Series,
    fit_indices: np.ndarray,
    tuning_indices: np.ndarray,
    threshold: float,
    trees: int,
) -> pd.DataFrame:
    X_pool = X.iloc[fit_indices].reset_index(drop=True)
    y_pool = y.iloc[fit_indices].reset_index(drop=True)
    attack_pool = development["attack_cat"].iloc[fit_indices].reset_index(drop=True)
    group_pool = groups.iloc[fit_indices].reset_index(drop=True)
    X_holdout = X.iloc[tuning_indices]
    y_holdout = y.iloc[tuning_indices]
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    blocks = [
        indices
        for _, indices in splitter.split(X_pool, attack_pool, groups=group_pool)
    ]
    rows = []
    for block_count in range(1, 5):
        subset_indices = np.concatenate(blocks[:block_count])
        fraction = block_count / 4
        print(f"Learning curve: {fraction:.0%} of training pool...", flush=True)
        started = time.time()
        model = make_random_forest(X_pool.iloc[subset_indices], trees)
        model.fit(X_pool.iloc[subset_indices], y_pool.iloc[subset_indices])
        for split, X_evaluation, y_evaluation in (
            ("training", X_pool.iloc[subset_indices], y_pool.iloc[subset_indices]),
            ("grouped_holdout", X_holdout, y_holdout),
        ):
            probabilities = model.predict_proba(X_evaluation)[:, 1]
            metrics, _ = calculate_metrics(y_evaluation, probabilities, threshold)
            rows.append(
                {
                    "training_fraction": fraction,
                    "training_rows": len(subset_indices),
                    "evaluation_split": split,
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                    "seconds": time.time() - started,
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    generalisation: pd.DataFrame,
    cross_validation: pd.DataFrame,
    ablations: pd.DataFrame,
    learning_curve: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colours = ["#4C78A8", "#F58518", "#54A24B"]

    selected = ["accuracy", "f1", "mcc"]
    ax = generalisation.set_index("split")[selected].plot.bar(
        figsize=(8.0, 4.8), color=colours
    )
    ax.set_title("Random Forest generalisation diagnostics")
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 1.01)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "generalisation_gap.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(cross_validation["fold"], cross_validation["f1"], marker="o")
    axes[0].set_title("Grouped CV F1")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("F1")
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        cross_validation["fold"],
        cross_validation["false_positive_rate"],
        marker="o",
        color="#E45756",
    )
    axes[1].set_title("Grouped CV false-positive rate")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("False-positive rate")
    axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "cross_validation_stability.png", dpi=200, bbox_inches="tight")
    plt.close()

    ablation_plot = ablations.set_index("variant")[["f1", "mcc"]]
    ax = ablation_plot.plot.bar(figsize=(9.0, 4.8), color=["#F58518", "#54A24B"])
    ax.set_title("Shortcut-feature ablation")
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.0)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_ablation.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for split, colour in (("training", "#4C78A8"), ("grouped_holdout", "#F58518")):
        part = learning_curve[learning_curve["evaluation_split"] == split]
        label = split.replace("_", " ").title()
        axes[0].plot(part["training_rows"], part["f1"], marker="o", label=label, color=colour)
        axes[1].plot(
            part["training_rows"],
            part["average_precision"],
            marker="o",
            label=label,
            color=colour,
        )
    axes[0].set_title("Learning curve: F1")
    axes[1].set_title("Learning curve: average precision")
    for axis in axes:
        axis.set_xlabel("Training rows")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(plot_dir / "learning_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def serialise_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.rf_trees < 1:
        raise ValueError("--rf-trees must be positive")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    integrity = validate_partitions(development, validation)
    X_development, y_development = split_xy(development)
    X_validation, y_validation = split_xy(validation)
    groups = predictor_hashes(development)
    fit_indices, tuning_indices = first_grouped_split(
        X_development, development, groups
    )

    print("Auditing individual predictors...", flush=True)
    feature_audit = audit_univariate_features(development, validation)
    feature_audit.to_csv(args.output_dir / "feature_leakage_audit.csv", index=False)

    print("Selecting the diagnostic operating threshold...", flush=True)
    threshold = tune_threshold(
        X_development,
        y_development,
        fit_indices,
        tuning_indices,
        args.rf_trees,
    )
    print(f"Selected threshold: {threshold:.2f}", flush=True)

    generalisation, baseline_model = run_generalisation_and_oob(
        X_development,
        y_development,
        X_validation,
        y_validation,
        threshold,
        args.rf_trees,
    )
    generalisation.to_csv(args.output_dir / "generalisation_metrics.csv", index=False)

    cross_validation = run_grouped_cross_validation(
        X_development,
        y_development,
        development,
        groups,
        threshold,
        args.rf_trees,
        args.cv_folds,
    )
    cross_validation.to_csv(args.output_dir / "cross_validation_folds.csv", index=False)
    cv_summary = cross_validation[METRICS].agg(["mean", "std", "min", "max"]).T
    cv_summary["range"] = cv_summary["max"] - cv_summary["min"]
    cv_summary.to_csv(args.output_dir / "cross_validation_summary.csv")

    ablations = run_ablations(
        X_development,
        y_development,
        X_validation,
        y_validation,
        fit_indices,
        tuning_indices,
        args.rf_trees,
        threshold,
        baseline_model,
    )
    ablations.to_csv(args.output_dir / "feature_ablation.csv", index=False)

    learning_curve = run_learning_curve(
        X_development,
        y_development,
        development,
        groups,
        fit_indices,
        tuning_indices,
        threshold,
        args.rf_trees,
    )
    learning_curve.to_csv(args.output_dir / "learning_curve.csv", index=False)

    save_plots(
        generalisation,
        cross_validation,
        ablations,
        learning_curve,
        args.output_dir,
    )

    indexed = generalisation.set_index("split")
    gaps = {
        metric: float(indexed.loc["training", metric] - indexed.loc["validation", metric])
        for metric in METRICS
    }
    metadata = {
        "purpose": "overfitting diagnostics only",
        "protected_test_set_opened": False,
        "seed": SEED,
        "random_forest_trees": args.rf_trees,
        "grouped_cv_folds": args.cv_folds,
        "selected_threshold": threshold,
        "development_path": str(DEVELOPMENT_PATH),
        "validation_path": str(VALIDATION_PATH),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    summary = {
        "metadata": metadata,
        "split_integrity": integrity,
        "training_minus_validation_gaps": gaps,
        "generalisation_metrics": serialise_records(generalisation),
        "grouped_cross_validation": json.loads(cv_summary.to_json(orient="index")),
        "feature_ablation": serialise_records(ablations),
        "learning_curve": serialise_records(learning_curve),
        "feature_audit_flags": serialise_records(
            feature_audit[
                (feature_audit["near_perfect_auc_flag"])
                | (feature_audit["deterministic_row_share"] >= 0.10)
            ]
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nOverfitting diagnostics complete.")
    print(generalisation[["split", *METRICS]].to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
