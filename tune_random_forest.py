"""Tune and regularise the Random Forest using development data only.

Stage one screens a deliberately bounded set of structural configurations with
group-aware cross-validation. Stage two confirms the selected structure with
200 and 500 trees on five grouped folds. Average Precision is the selection
metric; configurations within a small tolerance are resolved in favour of the
shallower, more regularised model. The supplied test set is never opened.
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
from sklearn.model_selection import StratifiedGroupKFold

from run_overfitting_diagnostics import METRICS, make_random_forest
from run_preliminary_validation import (
    DEVELOPMENT_PATH,
    SEED,
    VALIDATION_PATH,
    calculate_metrics,
    predictor_hashes,
    select_threshold,
    split_xy,
)


OUTPUT_DIR = Path("artifacts/random_forest_tuning")
SCREENING_CANDIDATES = [
    {"candidate": "unrestricted_leaf_1_sqrt", "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"candidate": "current_leaf_2_sqrt", "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"candidate": "unrestricted_leaf_5_sqrt", "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"candidate": "unrestricted_leaf_10_sqrt", "max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"candidate": "depth_24_leaf_2_sqrt", "max_depth": 24, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"candidate": "depth_24_leaf_5_sqrt", "max_depth": 24, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"candidate": "depth_16_leaf_2_sqrt", "max_depth": 16, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"candidate": "depth_16_leaf_5_sqrt", "max_depth": 16, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"candidate": "depth_12_leaf_5_sqrt", "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"candidate": "depth_12_leaf_10_sqrt", "max_depth": 12, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"candidate": "unrestricted_leaf_2_log2", "max_depth": None, "min_samples_leaf": 2, "max_features": "log2"},
    {"candidate": "depth_20_leaf_5_log2", "max_depth": 20, "min_samples_leaf": 5, "max_features": "log2"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-trees", type=int, default=100)
    parser.add_argument("--screening-folds", type=int, default=3)
    parser.add_argument("--confirmation-folds", type=int, default=5)
    parser.add_argument("--tree-counts", type=int, nargs="+", default=[200, 500])
    parser.add_argument(
        "--screening-ap-tolerance",
        type=float,
        default=0.0002,
        help="Prefer a simpler structure when mean AP is within this distance of the best.",
    )
    parser.add_argument(
        "--tree-ap-tolerance",
        type=float,
        default=0.0001,
        help="Prefer fewer trees when mean AP is within this distance of the best.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.screening_trees < 1:
        raise ValueError("--screening-trees must be positive")
    if args.screening_folds < 2 or args.confirmation_folds < 2:
        raise ValueError("Cross-validation folds must be at least 2")
    if not args.tree_counts or any(value < 1 for value in args.tree_counts):
        raise ValueError("--tree-counts must contain positive integers")
    if args.screening_ap_tolerance < 0 or args.tree_ap_tolerance < 0:
        raise ValueError("AP tolerances cannot be negative")


def make_configured_model(
    frame: pd.DataFrame, trees: int, config: dict[str, Any]
):
    model = make_random_forest(frame, trees)
    model.set_params(
        model__max_depth=config["max_depth"],
        model__min_samples_leaf=config["min_samples_leaf"],
        model__max_features=config["max_features"],
    )
    return model


def grouped_splits(
    X: pd.DataFrame,
    development: pd.DataFrame,
    groups: pd.Series,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(X, development["attack_cat"], groups=groups))


def evaluate_configurations(
    X: pd.DataFrame,
    y: pd.Series,
    configurations: list[dict[str, Any]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    trees_for: dict[str, int],
    stage: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(configurations)
    for position, config in enumerate(configurations, start=1):
        label = config["candidate"]
        trees = trees_for[label]
        print(f"{stage}: {position}/{total} {label} ({trees} trees)", flush=True)
        for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
            started = time.time()
            model = make_configured_model(X.iloc[train_indices], trees, config)
            model.fit(X.iloc[train_indices], y.iloc[train_indices])
            probabilities = model.predict_proba(X.iloc[validation_indices])[:, 1]
            metrics, _ = calculate_metrics(y.iloc[validation_indices], probabilities, 0.5)
            rows.append(
                {
                    "stage": stage,
                    "candidate": label,
                    "trees": trees,
                    "max_depth": config["max_depth"],
                    "min_samples_leaf": config["min_samples_leaf"],
                    "max_features": config["max_features"],
                    "fold": fold,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                    **metrics["confusion_matrix"],
                    "seconds": time.time() - started,
                }
            )
    return pd.DataFrame(rows)


def summarise_results(
    fold_results: pd.DataFrame, configurations: list[dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for config in configurations:
        part = fold_results[fold_results["candidate"] == config["candidate"]]
        row: dict[str, Any] = {
            "candidate": config["candidate"],
            "trees": int(part["trees"].iloc[0]),
            "max_depth": config["max_depth"],
            "min_samples_leaf": config["min_samples_leaf"],
            "max_features": config["max_features"],
            "folds": len(part),
            "total_seconds": float(part["seconds"].sum()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("average_precision_mean", ascending=False)


def structure_complexity_key(row: pd.Series) -> tuple[float, int, int, float]:
    depth = 10_000 if pd.isna(row["max_depth"]) else int(row["max_depth"])
    feature_rank = 0 if row["max_features"] == "log2" else 1
    return (
        depth,
        -int(row["min_samples_leaf"]),
        feature_rank,
        -float(row["average_precision_mean"]),
    )


def choose_structure(summary: pd.DataFrame, tolerance: float) -> tuple[pd.Series, pd.DataFrame]:
    best_ap = float(summary["average_precision_mean"].max())
    eligible = summary[
        summary["average_precision_mean"] >= best_ap - tolerance
    ].copy()
    eligible["selection_reason"] = (
        f"mean AP within {tolerance:.6f} of best; resolve by shallower depth, "
        "larger leaf, then fewer candidate features"
    )
    selected_index = min(eligible.index, key=lambda index: structure_complexity_key(eligible.loc[index]))
    return eligible.loc[selected_index], eligible


def choose_tree_count(summary: pd.DataFrame, tolerance: float) -> pd.Series:
    best_ap = float(summary["average_precision_mean"].max())
    eligible = summary[
        summary["average_precision_mean"] >= best_ap - tolerance
    ].copy()
    return eligible.sort_values(["trees", "average_precision_mean"], ascending=[True, False]).iloc[0]


def tune_selected_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    split: tuple[np.ndarray, np.ndarray],
    trees: int,
    config: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    train_indices, tuning_indices = split
    model = make_configured_model(X.iloc[train_indices], trees, config)
    model.fit(X.iloc[train_indices], y.iloc[train_indices])
    probabilities = model.predict_proba(X.iloc[tuning_indices])[:, 1]
    return select_threshold(y.iloc[tuning_indices], probabilities)


def final_validation_metrics(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    trees: int,
    config: dict[str, Any],
    threshold: float,
) -> tuple[pd.DataFrame, Any]:
    print("Refitting selected model on all development rows...", flush=True)
    started = time.time()
    model = make_configured_model(X_development, trees, config)
    model.fit(X_development, y_development)
    rows = []
    for split, X, y in (
        ("training", X_development, y_development),
        ("validation", X_validation, y_validation),
    ):
        probabilities = model.predict_proba(X)[:, 1]
        metrics, _ = calculate_metrics(y, probabilities, threshold)
        rows.append(
            {
                "model": "selected_regularised_random_forest",
                "split": split,
                "trees": trees,
                "max_depth": config["max_depth"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": config["max_features"],
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                **metrics["confusion_matrix"],
                "seconds": time.time() - started,
            }
        )
    return pd.DataFrame(rows), model


def add_existing_baseline(selected: pd.DataFrame) -> pd.DataFrame:
    path = Path("artifacts/overfitting_diagnostics/generalisation_metrics.csv")
    if not path.exists():
        return selected
    baseline = pd.read_csv(path)
    baseline = baseline[baseline["split"].isin(["training", "validation"])].copy()
    baseline.insert(0, "model", "current_random_forest")
    baseline.insert(2, "trees", 250)
    baseline.insert(3, "max_depth", None)
    baseline.insert(4, "min_samples_leaf", 2)
    baseline.insert(5, "max_features", "sqrt")
    all_columns = list(dict.fromkeys([*selected.columns, *baseline.columns]))
    return pd.concat(
        [baseline.reindex(columns=all_columns), selected.reindex(columns=all_columns)],
        ignore_index=True,
    )


def save_plots(
    screening: pd.DataFrame,
    confirmation: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    selected_candidate: str,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ordered = screening.sort_values("average_precision_mean")
    colours = ["#F58518" if value == selected_candidate else "#4C78A8" for value in ordered["candidate"]]
    plt.figure(figsize=(9.0, 6.0))
    labels = ordered["candidate"].str.replace("_", " ", regex=False)
    plt.barh(
        labels,
        ordered["average_precision_mean"],
        xerr=ordered["average_precision_std"],
        color=colours,
        alpha=0.9,
    )
    plt.xlabel("Grouped CV mean Average Precision")
    plt.title("Random Forest regularisation screening")
    lower = float(
        (ordered["average_precision_mean"] - ordered["average_precision_std"]).min()
        - 0.0001
    )
    upper = float(
        (ordered["average_precision_mean"] + ordered["average_precision_std"]).max()
        + 0.0001
    )
    plt.xlim(max(0.0, lower), min(1.0, upper))
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "screening_average_precision.png", dpi=200, bbox_inches="tight")
    plt.close()

    ordered = confirmation.sort_values("trees")
    plt.figure(figsize=(7.4, 4.7))
    plt.errorbar(
        ordered["trees"],
        ordered["average_precision_mean"],
        yerr=ordered["average_precision_std"],
        marker="o",
        linewidth=2,
        capsize=5,
        color="#4C78A8",
    )
    plt.xlabel("Trees")
    plt.ylabel("Grouped CV mean Average Precision")
    plt.title("Tree-count confirmation")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "tree_count_confirmation.png", dpi=200, bbox_inches="tight")
    plt.close()

    validation = validation_comparison[validation_comparison["split"] == "validation"]
    plot_data = validation.set_index("model")[["f1", "mcc"]]
    plot_data.index = [
        "Current 250-tree forest"
        if value == "current_random_forest"
        else "Selected regularised forest"
        for value in plot_data.index
    ]
    ax = plot_data.plot.bar(figsize=(8.0, 4.8), color=["#F58518", "#54A24B"])
    ax.set_title("Current and selected Random Forest on validation")
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.0)
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_dir / "validation_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    overlap = set(predictor_hashes(development)).intersection(
        set(predictor_hashes(validation))
    )
    if overlap:
        raise ValueError(f"Found {len(overlap)} duplicate predictor groups across partitions")

    X_development, y_development = split_xy(development)
    X_validation, y_validation = split_xy(validation)
    groups = predictor_hashes(development)

    screening_splits = grouped_splits(
        X_development, development, groups, args.screening_folds, SEED + 10
    )
    screening_trees = {
        config["candidate"]: args.screening_trees for config in SCREENING_CANDIDATES
    }
    screening_folds = evaluate_configurations(
        X_development,
        y_development,
        SCREENING_CANDIDATES,
        screening_splits,
        screening_trees,
        "screening",
    )
    screening_summary = summarise_results(screening_folds, SCREENING_CANDIDATES)
    selected_structure_row, eligible_structures = choose_structure(
        screening_summary, args.screening_ap_tolerance
    )
    selected_structure = {
        "candidate": str(selected_structure_row["candidate"]),
        "max_depth": None
        if pd.isna(selected_structure_row["max_depth"])
        else int(selected_structure_row["max_depth"]),
        "min_samples_leaf": int(selected_structure_row["min_samples_leaf"]),
        "max_features": str(selected_structure_row["max_features"]),
    }
    print(f"Selected structure after screening: {selected_structure['candidate']}", flush=True)

    confirmation_configs = []
    trees_for: dict[str, int] = {}
    for trees in sorted(set(args.tree_counts)):
        config = {**selected_structure, "candidate": f"selected_{trees}_trees"}
        confirmation_configs.append(config)
        trees_for[config["candidate"]] = trees
    confirmation_splits = grouped_splits(
        X_development, development, groups, args.confirmation_folds, SEED
    )
    confirmation_folds = evaluate_configurations(
        X_development,
        y_development,
        confirmation_configs,
        confirmation_splits,
        trees_for,
        "confirmation",
    )
    confirmation_summary = summarise_results(
        confirmation_folds, confirmation_configs
    )
    selected_tree_row = choose_tree_count(
        confirmation_summary, args.tree_ap_tolerance
    )
    selected_trees = int(selected_tree_row["trees"])
    print(f"Selected tree count: {selected_trees}", flush=True)

    threshold, threshold_table = tune_selected_threshold(
        X_development,
        y_development,
        confirmation_splits[0],
        selected_trees,
        selected_structure,
    )
    print(f"Selected threshold: {threshold:.2f}", flush=True)
    selected_validation, _ = final_validation_metrics(
        X_development,
        y_development,
        X_validation,
        y_validation,
        selected_trees,
        selected_structure,
        threshold,
    )
    validation_comparison = add_existing_baseline(selected_validation)

    screening_folds.to_csv(args.output_dir / "screening_folds.csv", index=False)
    screening_summary.to_csv(args.output_dir / "screening_summary.csv", index=False)
    eligible_structures.to_csv(
        args.output_dir / "screening_tied_candidates.csv", index=False
    )
    confirmation_folds.to_csv(args.output_dir / "confirmation_folds.csv", index=False)
    confirmation_summary.to_csv(
        args.output_dir / "confirmation_summary.csv", index=False
    )
    threshold_table.to_csv(args.output_dir / "selected_threshold.csv", index=False)
    validation_comparison.to_csv(
        args.output_dir / "validation_comparison.csv", index=False
    )
    save_plots(
        screening_summary,
        confirmation_summary,
        validation_comparison,
        selected_structure["candidate"],
        args.output_dir,
    )

    validation_selected = selected_validation[
        selected_validation["split"] == "validation"
    ].iloc[0]
    result = {
        "metadata": {
            "purpose": "development-only Random Forest tuning",
            "protected_test_set_opened": False,
            "seed": SEED,
            "screening_trees": args.screening_trees,
            "screening_folds": args.screening_folds,
            "confirmation_folds": args.confirmation_folds,
            "screening_ap_tolerance": args.screening_ap_tolerance,
            "tree_ap_tolerance": args.tree_ap_tolerance,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "selected_configuration": {
            "n_estimators": selected_trees,
            "max_depth": selected_structure["max_depth"],
            "min_samples_leaf": selected_structure["min_samples_leaf"],
            "max_features": selected_structure["max_features"],
            "threshold": threshold,
        },
        "selection_policy": {
            "primary_metric": "grouped cross-validation mean Average Precision",
            "structural_tie_break": "shallower depth, then larger minimum leaf, then log2 features",
            "tree_count_tie_break": "fewer trees",
        },
        "screening_tied_candidates": json_records(eligible_structures),
        "confirmation_summary": json_records(confirmation_summary),
        "selected_validation_metrics": {
            metric: float(validation_selected[metric]) for metric in METRICS
        },
    }
    with (args.output_dir / "selected_configuration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    print("\nRandom Forest tuning complete.")
    print(json.dumps(result["selected_configuration"], indent=2))
    print(
        selected_validation[
            ["split", "average_precision", "accuracy", "precision", "recall", "f1", "mcc", "false_positive_rate"]
        ].to_string(index=False)
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
