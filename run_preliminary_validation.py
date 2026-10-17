"""Generate leakage-aware preliminary validation results for UNSW-NB15.

The protected UNSW-NB15 testing set is deliberately never opened by this script.
Thresholds are selected on an internal fold from the development partition, after
which each model is refitted on all development rows and evaluated once on the
held-out validation partition.
"""

from __future__ import annotations

import json
import platform
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
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET = "label"
ANALYSIS_ONLY = ["id", "attack_cat"]
CATEGORICAL = ["proto", "service", "state"]
DEVELOPMENT_PATH = Path("data/processed/UNSW_NB15_development-training.csv")
VALIDATION_PATH = Path("data/processed/UNSW_NB15_validation.csv")
OUTPUT_DIR = Path("artifacts/preliminary_validation")


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = [c for c in frame.columns if c not in ANALYSIS_ONLY + [TARGET]]
    return frame[columns], frame[TARGET].astype(int)


def predictor_hashes(frame: pd.DataFrame) -> pd.Series:
    predictors, _ = split_xy(frame)
    return pd.util.hash_pandas_object(predictors, index=False)


def make_preprocessor(frame: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = [c for c in frame.columns if c not in CATEGORICAL]
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_models(frame: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline(
            [
                ("preprocess", make_preprocessor(frame, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1_500,
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("preprocess", make_preprocessor(frame, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=24,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    # Prefer recall when rounded F1 scores tie: missed attacks are the costlier error.
    best = table.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), table


def calculate_metrics(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "average_precision": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predictions),
        "false_positive_rate": fp / (fp + tn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics, predictions


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int = 500,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    ap_values: list[float] = []
    f1_values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        sampled_probabilities = probabilities[indices]
        if np.unique(sampled_y).size < 2:
            continue
        ap_values.append(average_precision_score(sampled_y, sampled_probabilities))
        f1_values.append(f1_score(sampled_y, sampled_probabilities >= threshold))
    return {
        "average_precision_95_ci": np.quantile(ap_values, [0.025, 0.975]).tolist(),
        "f1_95_ci": np.quantile(f1_values, [0.025, 0.975]).tolist(),
        "bootstrap_iterations": len(ap_values),
    }


def save_plots(
    y_validation: pd.Series,
    probabilities: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    colours = {"Logistic regression": "#4C78A8", "Random forest": "#F58518"}
    plt.figure(figsize=(7.4, 4.7))
    for name, values in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_validation, values)
        ap = results[name]["validation"]["average_precision"]
        plt.plot(recall, precision, linewidth=2.2, color=colours[name], label=f"{name} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Preliminary validation precision–recall curves")
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.2)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=200, bbox_inches="tight")
    plt.close()

    comparison = pd.DataFrame(
        {
            name: {
                "Average precision": values["validation"]["average_precision"],
                "F1": values["validation"]["f1"],
                "MCC": values["validation"]["mcc"],
            }
            for name, values in results.items()
        }
    ).T
    ax = comparison.plot.bar(figsize=(7.4, 4.7), color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Preliminary validation comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.75, 1.01)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    overlap = set(predictor_hashes(development)).intersection(set(predictor_hashes(validation)))
    if overlap:
        raise ValueError(f"Found {len(overlap)} exact predictor groups across development and validation")

    X_development, y_development = split_xy(development)
    X_validation, y_validation = split_xy(validation)

    groups = predictor_hashes(development)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_indices, tuning_indices = next(
        splitter.split(X_development, development["attack_cat"], groups=groups)
    )
    X_fit = X_development.iloc[fit_indices]
    y_fit = y_development.iloc[fit_indices]
    X_tuning = X_development.iloc[tuning_indices]
    y_tuning = y_development.iloc[tuning_indices]

    results: dict[str, dict[str, Any]] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    threshold_tables = []

    for name in make_models(X_fit):
        print(f"Selecting threshold for {name}...", flush=True)
        tuning_model = make_models(X_fit)[name]
        tuning_model.fit(X_fit, y_fit)
        tuning_probabilities = tuning_model.predict_proba(X_tuning)[:, 1]
        threshold, threshold_table = select_threshold(y_tuning, tuning_probabilities)
        threshold_table.insert(0, "model", name)
        threshold_tables.append(threshold_table)

        print(f"Refitting {name} on all development rows...", flush=True)
        final_model = make_models(X_development)[name]
        final_model.fit(X_development, y_development)
        probabilities = final_model.predict_proba(X_validation)[:, 1]
        validation_probabilities[name] = probabilities
        validation_metrics, predictions = calculate_metrics(y_validation, probabilities, threshold)
        validation_metrics.update(
            bootstrap_intervals(y_validation.to_numpy(), probabilities, threshold)
        )
        results[name] = {
            "threshold_selection": {
                "source": "internal development tuning fold",
                "fit_rows": int(len(fit_indices)),
                "tuning_rows": int(len(tuning_indices)),
                "selected_threshold": threshold,
            },
            "validation": validation_metrics,
        }

        per_attack = (
            validation.assign(prediction=predictions)
            .query("label == 1")
            .groupby("attack_cat")
            .agg(support=("label", "size"), detected=("prediction", "sum"))
            .assign(recall=lambda d: d["detected"] / d["support"], model=name)
            .reset_index()
        )
        per_attack.to_csv(
            OUTPUT_DIR / f"per_attack_recall_{name.lower().replace(' ', '_')}.csv", index=False
        )

    baseline_prediction = np.ones(len(y_validation), dtype=int)
    baseline_metrics, _ = calculate_metrics(
        y_validation, baseline_prediction.astype(float), threshold=0.5
    )

    metadata = {
        "purpose": "preliminary validation only",
        "protected_test_set_opened": False,
        "seed": SEED,
        "development_path": str(DEVELOPMENT_PATH),
        "validation_path": str(VALIDATION_PATH),
        "development_rows": len(development),
        "validation_rows": len(validation),
        "predictor_columns": X_development.shape[1],
        "categorical_columns": CATEGORICAL,
        "model_parameters": {
            "Logistic regression": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1_500,
            },
            "Random forest": {
                "n_estimators": 200,
                "max_depth": 24,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
            },
        },
        "exact_predictor_overlap": len(overlap),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "always_suspicious_baseline": baseline_metrics, "models": results},
            handle,
            indent=2,
        )

    rows = []
    for name, values in results.items():
        metric_values = values["validation"]
        rows.append(
            {
                "model": name,
                **{k: v for k, v in metric_values.items() if isinstance(v, (int, float))},
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "threshold_selection.csv", index=False
    )
    save_plots(y_validation, validation_probabilities, results, OUTPUT_DIR)
    print(json.dumps({name: values["validation"] for name, values in results.items()}, indent=2))


if __name__ == "__main__":
    main()
