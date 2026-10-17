"""Explain and audit the frozen validation-stage intrusion detector.

The command uses only the development-training and validation partitions. It
fits the selected regularised Random Forest, compares its mistakes with Logistic
Regression, computes original-feature permutation importance, produces global
and local SHAP explanations, analyses attack-specific and high-confidence
errors, and serialises the frozen pipeline. The protected test set is never
opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from run_preliminary_validation import (
    CATEGORICAL,
    DEVELOPMENT_PATH,
    SEED,
    VALIDATION_PATH,
    calculate_metrics,
    make_models,
    predictor_hashes,
    select_threshold,
    split_xy,
)


OUTPUT_DIR = Path("artifacts/explainability_analysis")
MODEL_NAMES = ["Logistic regression", "Random forest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--shap-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_models(
    development: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    X_development, y_development = split_xy(development)
    X_validation, _ = split_xy(validation)
    groups = predictor_hashes(development)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_indices, tuning_indices = next(
        splitter.split(X_development, development["attack_cat"], groups=groups)
    )
    models: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    probabilities: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        print(f"Selecting threshold for {name}...", flush=True)
        tuning_model = make_models(X_development.iloc[fit_indices])[name]
        tuning_model.fit(
            X_development.iloc[fit_indices], y_development.iloc[fit_indices]
        )
        tuning_probabilities = tuning_model.predict_proba(
            X_development.iloc[tuning_indices]
        )[:, 1]
        threshold, _ = select_threshold(
            y_development.iloc[tuning_indices], tuning_probabilities
        )
        thresholds[name] = threshold

        print(f"Refitting {name} on all development rows...", flush=True)
        model = make_models(X_development)[name]
        model.fit(X_development, y_development)
        models[name] = model
        probabilities[name] = model.predict_proba(X_validation)[:, 1]
        predictions[name] = (probabilities[name] >= threshold).astype(int)
    return models, thresholds, probabilities, predictions


def build_error_frame(
    validation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    frame = validation.copy()
    frame.insert(0, "validation_position", np.arange(len(validation)))
    frame.insert(1, "random_forest_probability", probabilities["Random forest"])
    frame.insert(2, "random_forest_prediction", predictions["Random forest"])
    frame.insert(3, "logistic_regression_probability", probabilities["Logistic regression"])
    frame.insert(4, "logistic_regression_prediction", predictions["Logistic regression"])
    y = frame["label"].to_numpy()
    rf_prediction = predictions["Random forest"]
    lr_prediction = predictions["Logistic regression"]
    frame["random_forest_error_type"] = np.select(
        [(y == 0) & (rf_prediction == 1), (y == 1) & (rf_prediction == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    frame["random_forest_confidence"] = np.where(
        rf_prediction == 1,
        probabilities["Random forest"],
        1 - probabilities["Random forest"],
    )
    frame["distance_from_threshold"] = np.abs(
        probabilities["Random forest"] - threshold
    )
    rf_correct = rf_prediction == y
    lr_correct = lr_prediction == y
    frame["model_comparison"] = np.select(
        [
            rf_correct & lr_correct,
            rf_correct & ~lr_correct,
            ~rf_correct & lr_correct,
        ],
        ["both_correct", "only_random_forest_correct", "only_logistic_regression_correct"],
        default="both_wrong",
    )
    return frame


def save_error_tables(error_frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    errors = error_frame[error_frame["random_forest_error_type"] != "correct"].copy()
    false_positives = errors[errors["random_forest_error_type"] == "false_positive"].sort_values(
        "random_forest_probability", ascending=False
    )
    false_negatives = errors[errors["random_forest_error_type"] == "false_negative"].sort_values(
        "random_forest_probability"
    )
    errors.to_csv(output_dir / "all_random_forest_errors.csv", index=False)
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)

    confident = pd.concat(
        [
            false_positives[false_positives["random_forest_probability"] >= 0.90],
            false_negatives[false_negatives["random_forest_probability"] <= 0.10],
        ],
        ignore_index=True,
    )
    confident.to_csv(output_dir / "high_confidence_errors.csv", index=False)

    by_attack = (
        error_frame.groupby("attack_cat", observed=True)
        .agg(
            support=("label", "size"),
            actual_label=("label", "first"),
            predicted_suspicious=("random_forest_prediction", "sum"),
            correct=("random_forest_error_type", lambda values: int((values == "correct").sum())),
            false_positives=("random_forest_error_type", lambda values: int((values == "false_positive").sum())),
            false_negatives=("random_forest_error_type", lambda values: int((values == "false_negative").sum())),
        )
        .reset_index()
    )
    by_attack["error_rate"] = 1 - by_attack["correct"] / by_attack["support"]
    by_attack["attack_recall"] = np.where(
        by_attack["actual_label"] == 1,
        by_attack["predicted_suspicious"] / by_attack["support"],
        np.nan,
    )
    by_attack["normal_specificity"] = np.where(
        by_attack["actual_label"] == 0,
        by_attack["correct"] / by_attack["support"],
        np.nan,
    )
    by_attack.sort_values(["actual_label", "attack_recall"], na_position="last").to_csv(
        output_dir / "performance_by_attack_category.csv", index=False
    )

    comparison = (
        error_frame.groupby(["attack_cat", "model_comparison"], observed=True)
        .size()
        .reset_index(name="count")
    )
    comparison.to_csv(output_dir / "model_error_comparison_by_attack.csv", index=False)
    error_frame[error_frame["random_forest_prediction"] != error_frame["logistic_regression_prediction"]].to_csv(
        output_dir / "model_disagreements.csv", index=False
    )

    confidence_bins = pd.cut(
        error_frame["random_forest_confidence"],
        bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001],
        labels=["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"],
        include_lowest=True,
    )
    confidence_table = (
        error_frame.assign(confidence_bin=confidence_bins)
        .groupby("confidence_bin", observed=False)
        .agg(
            records=("label", "size"),
            errors=("random_forest_error_type", lambda values: int((values != "correct").sum())),
            mean_confidence=("random_forest_confidence", "mean"),
        )
        .reset_index()
    )
    confidence_table["error_rate"] = (
        confidence_table["errors"] / confidence_table["records"]
    )
    confidence_table.to_csv(output_dir / "errors_by_confidence.csv", index=False)

    return {
        "total_errors": len(errors),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "high_confidence_errors": len(confident),
        "model_disagreements": int(
            (error_frame["random_forest_prediction"] != error_frame["logistic_regression_prediction"]).sum()
        ),
    }


def run_permutation_importance(
    forest: Any,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    repeats: int,
    output_dir: Path,
) -> pd.DataFrame:
    print(f"Permutation importance ({repeats} repeats)...", flush=True)
    result = permutation_importance(
        forest,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=SEED,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_validation.columns,
            "ap_decrease_mean": result.importances_mean,
            "ap_decrease_std": result.importances_std,
        }
    ).sort_values("ap_decrease_mean", ascending=False)
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)

    top = importance.head(20).sort_values("ap_decrease_mean")
    plt.figure(figsize=(8.2, 6.2))
    plt.barh(
        top["feature"],
        top["ap_decrease_mean"],
        xerr=top["ap_decrease_std"],
        color="#4C78A8",
        alpha=0.9,
    )
    plt.xlabel("Decrease in validation Average Precision")
    plt.title("Permutation importance of original predictors")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "permutation_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    return importance


def transformed_to_original(feature: str, original_columns: list[str]) -> str:
    if feature in original_columns:
        return feature
    for categorical in CATEGORICAL:
        if feature.startswith(f"{categorical}_"):
            return categorical
    return feature


def normalise_shap_explanation(
    raw_explanation: Any, transformed: np.ndarray, feature_names: np.ndarray
) -> shap.Explanation:
    values = raw_explanation.values
    base_values = raw_explanation.base_values
    if values.ndim == 3:
        values = values[:, :, 1]
        if np.ndim(base_values) == 2:
            base_values = base_values[:, 1]
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed,
        feature_names=feature_names,
    )


def choose_explanation_cases(
    error_frame: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    probability = error_frame["random_forest_probability"].to_numpy()
    prediction = error_frame["random_forest_prediction"].to_numpy()
    actual = error_frame["label"].to_numpy()
    specifications = {
        "high_confidence_false_positive": ((actual == 0) & (prediction == 1), "max_probability"),
        "borderline_false_positive": ((actual == 0) & (prediction == 1), "closest_threshold"),
        "high_confidence_false_negative": ((actual == 1) & (prediction == 0), "min_probability"),
        "borderline_false_negative": ((actual == 1) & (prediction == 0), "closest_threshold"),
        "representative_true_positive": ((actual == 1) & (prediction == 1), "max_probability"),
        "representative_true_negative": ((actual == 0) & (prediction == 0), "min_probability"),
    }
    rows = []
    for case_name, (mask, method) in specifications.items():
        positions = np.flatnonzero(mask)
        if not len(positions):
            continue
        if method == "max_probability":
            position = positions[np.argmax(probability[positions])]
        elif method == "min_probability":
            position = positions[np.argmin(probability[positions])]
        else:
            position = positions[np.argmin(np.abs(probability[positions] - threshold))]
        source = error_frame.iloc[position]
        rows.append(
            {
                "case": case_name,
                "validation_position": int(position),
                "id": source["id"],
                "attack_cat": source["attack_cat"],
                "actual_label": int(source["label"]),
                "predicted_label": int(source["random_forest_prediction"]),
                "suspicious_probability": float(source["random_forest_probability"]),
                "threshold": threshold,
            }
        )
    return pd.DataFrame(rows)


def run_shap_analysis(
    forest: Any,
    X_validation: pd.DataFrame,
    error_frame: pd.DataFrame,
    threshold: float,
    sample_size: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Global SHAP analysis ({sample_size} sampled rows)...", flush=True)
    preprocess = forest.named_steps["preprocess"]
    model = forest.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    sample = X_validation.sample(n=min(sample_size, len(X_validation)), random_state=SEED)
    transformed = preprocess.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    explainer = shap.TreeExplainer(model)
    explanation = normalise_shap_explanation(
        explainer(transformed, check_additivity=False), transformed, feature_names
    )

    transformed_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.abs(explanation.values).mean(axis=0),
            "mean_signed_shap": explanation.values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    transformed_importance.to_csv(
        output_dir / "shap_importance_transformed.csv", index=False
    )
    transformed_importance["original_feature"] = transformed_importance["feature"].map(
        lambda value: transformed_to_original(value, list(X_validation.columns))
    )
    original_importance = (
        transformed_importance.groupby("original_feature", as_index=False)
        .agg(
            mean_absolute_shap=("mean_absolute_shap", "sum"),
            mean_signed_shap=("mean_signed_shap", "sum"),
            transformed_columns=("feature", "size"),
        )
        .sort_values("mean_absolute_shap", ascending=False)
    )
    original_importance.to_csv(output_dir / "shap_importance_original.csv", index=False)

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("Random Forest SHAP summary")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    cases = choose_explanation_cases(error_frame, threshold)
    cases.to_csv(output_dir / "explanation_cases.csv", index=False)
    case_dir = output_dir / "shap_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for _, case in cases.iterrows():
        case_name = str(case["case"])
        position = int(case["validation_position"])
        row_frame = X_validation.iloc[[position]]
        row_transformed = preprocess.transform(row_frame)
        if hasattr(row_transformed, "toarray"):
            row_transformed = row_transformed.toarray()
        row_transformed = np.asarray(row_transformed)
        row_explanation = normalise_shap_explanation(
            explainer(row_transformed, check_additivity=False),
            row_transformed,
            feature_names,
        )[0]
        shap.plots.waterfall(row_explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(case_dir / f"{case_name}_waterfall.png", dpi=200, bbox_inches="tight")
        plt.close()
        contributions = pd.DataFrame(
            {
                "feature": feature_names,
                "transformed_value": row_transformed[0],
                "shap_value": row_explanation.values,
            }
        )
        contributions["absolute_shap"] = contributions["shap_value"].abs()
        contributions.sort_values("absolute_shap", ascending=False).to_csv(
            case_dir / f"{case_name}_contributions.csv", index=False
        )
    return transformed_importance, original_importance, cases


def save_error_plots(error_frame: pd.DataFrame, output_dir: Path) -> None:
    y = error_frame["label"].to_numpy()
    prediction = error_frame["random_forest_prediction"].to_numpy()
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Records")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=12)
    plt.xticks([0, 1], ["Normal", "Suspicious"])
    plt.yticks([0, 1], ["Normal", "Suspicious"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Selected Random Forest validation confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    attack = (
        error_frame[error_frame["label"] == 1]
        .groupby("attack_cat", observed=True)
        .agg(
            support=("label", "size"),
            recall=("random_forest_prediction", "mean"),
        )
        .reset_index()
        .sort_values("recall")
    )
    plt.figure(figsize=(8.0, 5.4))
    bars = plt.barh(attack["attack_cat"], attack["recall"], color="#F58518")
    for bar, support in zip(bars, attack["support"]):
        plt.text(
            min(bar.get_width() + 0.008, 1.005),
            bar.get_y() + bar.get_height() / 2,
            f"n={support:,}",
            va="center",
            fontsize=8,
        )
    plt.xlabel("Recall")
    plt.title("Validation recall by attack category")
    plt.xlim(0, 1.08)
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "recall_by_attack_category.png", dpi=200, bbox_inches="tight")
    plt.close()

    comparison = error_frame["model_comparison"].value_counts()
    order = [
        "both_correct",
        "only_random_forest_correct",
        "only_logistic_regression_correct",
        "both_wrong",
    ]
    comparison = comparison.reindex(order, fill_value=0)
    labels = [value.replace("_", " ").title() for value in comparison.index]
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(labels, comparison.values, color=["#54A24B", "#F58518", "#4C78A8", "#E45756"])
    plt.ylabel("Validation records")
    plt.title("Where the two models agree and disagree")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "model_error_overlap.png", dpi=200, bbox_inches="tight")
    plt.close()


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be positive")
    if args.shap_samples < 1:
        raise ValueError("--shap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")
    overlap = set(predictor_hashes(development)).intersection(
        set(predictor_hashes(validation))
    )
    if overlap:
        raise ValueError(f"Found {len(overlap)} predictor groups across partitions")

    X_validation, y_validation = split_xy(validation)
    models, thresholds, probabilities, predictions = fit_models(development, validation)
    error_frame = build_error_frame(
        validation, probabilities, predictions, thresholds["Random forest"]
    )
    error_summary = save_error_tables(error_frame, args.output_dir)
    save_error_plots(error_frame, args.output_dir)

    permutation = run_permutation_importance(
        models["Random forest"],
        X_validation,
        y_validation,
        args.permutation_repeats,
        args.output_dir,
    )
    _, shap_original, cases = run_shap_analysis(
        models["Random forest"],
        X_validation,
        error_frame,
        thresholds["Random forest"],
        args.shap_samples,
        args.output_dir,
    )

    model_path = args.output_dir / "selected_random_forest_pipeline.joblib"
    joblib.dump(models["Random forest"], model_path, compress=3)

    validation_metrics = {}
    for name in MODEL_NAMES:
        metric_values, _ = calculate_metrics(
            y_validation, probabilities[name], thresholds[name]
        )
        validation_metrics[name] = metric_values

    manifest = {
        "metadata": {
            "purpose": "frozen validation-stage model explainability and error analysis",
            "protected_test_set_opened": False,
            "seed": SEED,
            "development_path": str(DEVELOPMENT_PATH),
            "development_sha256": sha256_file(DEVELOPMENT_PATH),
            "validation_path": str(VALIDATION_PATH),
            "validation_sha256": sha256_file(VALIDATION_PATH),
            "development_rows": len(development),
            "validation_rows": len(validation),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "shap_version": shap.__version__,
        },
        "frozen_random_forest": {
            "n_estimators": 200,
            "max_depth": 24,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "threshold": thresholds["Random forest"],
            "pipeline_path": str(model_path),
        },
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "error_summary": error_summary,
        "top_permutation_features": json_records(permutation.head(15)),
        "top_shap_features": json_records(shap_original.head(15)),
        "explanation_cases": json_records(cases),
    }
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nExplainability and error analysis complete.")
    print(f"Errors: {json.dumps(error_summary)}")
    print("\nTop permutation features")
    print(permutation.head(10).to_string(index=False))
    print("\nTop original-feature SHAP importance")
    print(shap_original.head(10).to_string(index=False))
    print(f"\nOutputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
