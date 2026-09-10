"""End-to-end explainable ML workflow for the UNSW-NB15 binary task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET = "label"
ANALYSIS_ONLY = ["id", "attack_cat"]
CATEGORICAL = ["proto", "service", "state"]
DEFAULT_THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", choices=["all", "audit", "train"], default="all")
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--shap-samples", type=int, default=300)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--train-limit", type=int, help="Stratified sample size for development runs")
    parser.add_argument("--test-limit", type=int, help="Stratified sample size for development runs")
    return parser.parse_args()


def locate_files(data_dir: Path) -> tuple[Path, Path]:
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s): " + ", ".join(missing))
    return train_path, test_path


def stratified_limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    fractions = frame[TARGET].value_counts(normalize=True)
    parts = []
    for label, fraction in fractions.items():
        n = max(1, int(round(limit * fraction)))
        group = frame[frame[TARGET] == label]
        parts.append(group.sample(n=min(n, len(group)), random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_data(data_dir: Path, train_limit: int | None = None, test_limit: int | None = None):
    train_path, test_path = locate_files(data_dir)
    train = stratified_limit(pd.read_csv(train_path), train_limit)
    test = stratified_limit(pd.read_csv(test_path), test_limit)
    expected = {0, 1}
    if set(train[TARGET].unique()) != expected or set(test[TARGET].unique()) != expected:
        raise ValueError("Expected binary label values {0, 1} in both dataset splits")
    if not (train[TARGET].eq(train["attack_cat"].ne("Normal").astype(int))).all():
        raise ValueError("Training labels disagree with attack_cat; leakage audit cannot be trusted")
    return train, test


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def run_audit(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    audit_dir = output_dir / "audit"
    plot_dir = output_dir / "plots"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, frame in (("train", train), ("test", test)):
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "columns": frame.shape[1],
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "normal": int((frame[TARGET] == 0).sum()),
                "suspicious": int((frame[TARGET] == 1).sum()),
                "suspicious_pct": round(100 * frame[TARGET].mean(), 2),
            }
        )
    pd.DataFrame(rows).to_csv(audit_dir / "dataset_summary.csv", index=False)

    feature_types = pd.DataFrame(
        {
            "feature": train.columns,
            "dtype": train.dtypes.astype(str).values,
            "role": [
                "target" if c == TARGET else "analysis_only" if c in ANALYSIS_ONLY else "predictor"
                for c in train.columns
            ],
            "missing_train": train.isna().sum().values,
            "unique_train": train.nunique(dropna=False).values,
        }
    )
    feature_types.to_csv(audit_dir / "feature_types.csv", index=False)
    train["attack_cat"].value_counts().rename_axis("attack_cat").reset_index(name="count").to_csv(
        audit_dir / "attack_category_counts.csv", index=False
    )

    class_counts = pd.concat(
        [
            train.assign(split="train"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    ).groupby(["split", TARGET]).size().reset_index(name="count")
    class_counts["class"] = class_counts[TARGET].map({0: "Normal", 1: "Suspicious"})
    sns.barplot(data=class_counts, x="split", y="count", hue="class")
    plt.title("Class distribution by provided split")
    save_plot(plot_dir / "class_distribution.png")

    attack_counts = train["attack_cat"].value_counts().reset_index()
    sns.barplot(data=attack_counts, y="attack_cat", x="count", color="#d55e00")
    plt.title("Training attack-category distribution")
    save_plot(plot_dir / "attack_categories.png")

    selected = [c for c in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"] if c in train]
    sample = train.sample(n=min(20_000, len(train)), random_state=SEED).copy()
    sample["class"] = sample[TARGET].map({0: "Normal", 1: "Suspicious"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for feature, axis in zip(selected, axes.flat):
        sns.boxplot(data=sample, x="class", y=feature, showfliers=False, ax=axis)
        axis.set_title(feature)
    fig.suptitle("Selected numerical features by class (outliers hidden)")
    save_plot(plot_dir / "selected_feature_boxplots.png")

    numeric = [c for c in train.select_dtypes(include=np.number).columns if c not in ["id"]]
    corr_to_target = train[numeric].corr(numeric_only=True)[TARGET].abs().sort_values(ascending=False)
    top_corr = corr_to_target.drop(TARGET).head(14).index.tolist() + [TARGET]
    sns.heatmap(train[top_corr].corr(), cmap="vlag", center=0, square=False)
    plt.title("Correlation among features most associated with the target")
    save_plot(plot_dir / "correlation_heatmap.png")


def split_xy(frame: pd.DataFrame):
    feature_columns = [c for c in frame.columns if c not in ANALYSIS_ONLY + [TARGET]]
    return frame[feature_columns], frame[TARGET].astype(int)


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


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in DEFAULT_THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        )
    return pd.DataFrame(rows)


def basic_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc_average_precision": average_precision_score(y_true, probabilities),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
    }


def plot_confusion(y_true, probabilities, path: Path, title: str, threshold: float = 0.5) -> None:
    matrix = confusion_matrix(y_true, probabilities >= threshold, labels=[0, 1])
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", xticklabels=["Normal", "Suspicious"], yticklabels=["Normal", "Suspicious"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    save_plot(path)


def save_feature_importance(pipeline: Pipeline, output_dir: Path) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    importance = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    sns.barplot(data=importance.head(20), y="feature", x="importance", color="#0072b2")
    plt.title("Random Forest feature importance (top 20)")
    save_plot(output_dir / "plots" / "feature_importance.png")
    return importance


def run_shap(pipeline: Pipeline, X_test: pd.DataFrame, cases: pd.DataFrame, output_dir: Path, sample_size: int) -> None:
    import shap

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=SEED)
    transformed = preprocess.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explanation = shap.TreeExplainer(model)(transformed, check_additivity=False)
    if explanation.values.ndim == 3:
        explanation = shap.Explanation(
            values=explanation.values[:, :, 1],
            base_values=explanation.base_values[:, 1],
            data=explanation.data,
            feature_names=feature_names,
        )
    shap.summary_plot(explanation.values, transformed, feature_names=feature_names, show=False, max_display=20)
    save_plot(output_dir / "plots" / "shap_summary.png")

    shap_dir = output_dir / "shap_cases"
    shap_dir.mkdir(exist_ok=True)
    for case_name, row in cases.iterrows():
        row_frame = X_test.loc[[row["row_index"]]]
        row_transformed = preprocess.transform(row_frame)
        if hasattr(row_transformed, "toarray"):
            row_transformed = row_transformed.toarray()
        row_exp = shap.TreeExplainer(model)(row_transformed, check_additivity=False)
        if row_exp.values.ndim == 3:
            row_exp = shap.Explanation(
                values=row_exp.values[0, :, 1],
                base_values=row_exp.base_values[0, 1],
                data=row_exp.data[0],
                feature_names=feature_names,
            )
        else:
            row_exp = row_exp[0]
        shap.plots.waterfall(row_exp, max_display=15, show=False)
        save_plot(shap_dir / f"{case_name}_waterfall.png")
        pd.DataFrame(
            {"feature": feature_names, "value": row_transformed[0], "shap_value": row_exp.values}
        ).assign(abs_shap=lambda d: d.shap_value.abs()).sort_values("abs_shap", ascending=False).to_csv(
            shap_dir / f"{case_name}_contributions.csv", index=False
        )


def run_training(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path, rf_trees: int, shap_samples: int, skip_shap: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    baseline = Pipeline(
        [
            ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    forest = Pipeline(
        [
            ("preprocess", make_preprocessor(X_train, scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=rf_trees, random_state=SEED, n_jobs=-1)),
        ]
    )

    results: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, model in (("logistic_regression", baseline), ("random_forest", forest)):
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        probabilities[name] = model.predict_proba(X_test)[:, 1]
        results[name] = basic_metrics(y_test, probabilities[name])
        plot_confusion(y_test, probabilities[name], output_dir / "plots" / f"{name}_confusion_matrix.png", name.replace("_", " ").title())

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    pd.DataFrame(
        [{"model": name, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}} for name, metrics in results.items()]
    ).to_csv(output_dir / "model_comparison.csv", index=False)

    for name, probs in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        plt.plot(recall, precision, label=f"{name.replace('_', ' ')} (AP={average_precision_score(y_test, probs):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend()
    save_plot(output_dir / "plots" / "precision_recall_curves.png")

    rf_probs = probabilities["random_forest"]
    thresholds = threshold_metrics(y_test, rf_probs)
    thresholds.to_csv(output_dir / "threshold_analysis.csv", index=False)
    plt.plot(thresholds.threshold, thresholds.precision, marker="o", label="Precision")
    plt.plot(thresholds.threshold, thresholds.recall, marker="o", label="Recall")
    plt.plot(thresholds.threshold, thresholds.f1, marker="o", label="F1")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.03)
    plt.legend()
    plt.title("Random Forest threshold trade-offs")
    save_plot(output_dir / "plots" / "threshold_tradeoffs.png")

    importance = save_feature_importance(forest, output_dir)
    predictions = (rf_probs >= 0.5).astype(int)
    errors = test.copy()
    errors.insert(0, "row_index", test.index)
    errors.insert(1, "predicted_label", predictions)
    errors.insert(2, "suspicious_probability", rf_probs)
    errors["error_type"] = np.select(
        [(errors[TARGET] == 0) & (predictions == 1), (errors[TARGET] == 1) & (predictions == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors[errors.error_type == "false_positive"].sort_values("suspicious_probability", ascending=False).to_csv(
        output_dir / "false_positives.csv", index=False
    )
    errors[errors.error_type == "false_negative"].sort_values("suspicious_probability").to_csv(
        output_dir / "false_negatives.csv", index=False
    )
    errors[errors.error_type != "correct"].groupby(["error_type", "attack_cat"]).size().reset_index(name="count").to_csv(
        output_dir / "errors_by_attack_category.csv", index=False
    )

    case_masks = {
        "true_positive": (y_test.to_numpy() == 1) & (predictions == 1),
        "false_positive": (y_test.to_numpy() == 0) & (predictions == 1),
        "false_negative": (y_test.to_numpy() == 1) & (predictions == 0),
    }
    cases = {}
    for name, mask in case_masks.items():
        positions = np.flatnonzero(mask)
        if len(positions):
            pos = positions[np.argmax(np.abs(rf_probs[positions] - 0.5))] if name == "true_positive" else positions[np.argmin(np.abs(rf_probs[positions] - 0.5))]
            cases[name] = {"row_index": X_test.index[pos], "actual": int(y_test.iloc[pos]), "predicted": int(predictions[pos]), "probability": float(rf_probs[pos]), "attack_cat": test.iloc[pos]["attack_cat"]}
    case_frame = pd.DataFrame.from_dict(cases, orient="index")
    case_frame.to_csv(output_dir / "explanation_cases.csv", index_label="case")

    joblib.dump(forest, output_dir / "random_forest_pipeline.joblib", compress=3)
    if not skip_shap and len(case_frame):
        run_shap(forest, X_test, case_frame, output_dir, shap_samples)

    print("Top Random Forest features:")
    print(importance.head(10).to_string(index=False))


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    train, test = load_data(args.data_dir, args.train_limit, args.test_limit)
    print(f"Loaded train={train.shape}, test={test.shape}")
    if args.mode in {"all", "audit"}:
        run_audit(train, test, args.output_dir)
    if args.mode in {"all", "train"}:
        run_training(train, test, args.output_dir, args.rf_trees, args.shap_samples, args.skip_shap)


if __name__ == "__main__":
    main()
