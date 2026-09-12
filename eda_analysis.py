"""Exploratory data analysis for the UNSW-NB15 project milestone.

Run from the project folder:
    .\.venv\Scripts\python.exe eda_analysis.py

"""

# %% Imports and configuration
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TARGET = "label"
CATEGORICAL_FEATURES = ["proto", "service", "state"]
ANALYSIS_ONLY = ["id", "attack_cat"]
CLASS_NAMES = {0: "Normal", 1: "Suspicious"}
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    parser.add_argument(
        "--plot-sample",
        type=int,
        default=30_000,
        help="Maximum training rows used in distribution plots",
    )
    return parser.parse_args()


# %% Load and validate the supplied files
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"

    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    required = {TARGET, "attack_cat", "id", *CATEGORICAL_FEATURES}
    for split_name, frame in (("training", train), ("testing", test)):
        missing_columns = required.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{split_name} data lacks columns: {sorted(missing_columns)}")
        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError(f"{split_name} label must contain only 0 and 1")

        # attack_cat is useful for analysis, but it must never become a predictor.
        expected_label = frame["attack_cat"].ne("Normal").astype(int)
        if not frame[TARGET].equals(expected_label):
            raise ValueError(f"{split_name} label and attack_cat disagree")

    return train, test


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


# %% Dataset structure and data-quality audit
def create_data_quality_outputs(
    train: pd.DataFrame, test: pd.DataFrame, output_dir: Path
) -> None:
    summary_rows = []
    predictor_columns = [
        column
        for column in train.columns
        if column not in ANALYSIS_ONLY + [TARGET]
    ]

    for split_name, frame in (("Training", train), ("Testing", test)):
        numeric = frame.select_dtypes(include=np.number)
        summary_rows.append(
            {
                "split": split_name,
                "rows": len(frame),
                "columns": frame.shape[1],
                "numeric_predictors": len(
                    [c for c in predictor_columns if c in numeric.columns]
                ),
                "categorical_predictors": len(CATEGORICAL_FEATURES),
                "missing_cells": int(frame.isna().sum().sum()),
                "non_finite_numeric_cells": int(
                    (~np.isfinite(numeric.to_numpy())).sum()
                ),
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "repeated_predictor_vectors": int(
                    frame.duplicated(subset=predictor_columns).sum()
                ),
                "normal_records": int(frame[TARGET].eq(0).sum()),
                "suspicious_records": int(frame[TARGET].eq(1).sum()),
                "suspicious_percent": round(100 * frame[TARGET].mean(), 2),
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "dataset_quality_summary.csv", index=False
    )

    feature_summary = pd.DataFrame(
        {
            "feature": train.columns,
            "dtype": train.dtypes.astype(str).to_numpy(),
            "role": [
                "target"
                if column == TARGET
                else "analysis_only"
                if column in ANALYSIS_ONLY
                else "predictor"
                for column in train.columns
            ],
            "training_unique_values": train.nunique(dropna=False).to_numpy(),
            "training_missing": train.isna().sum().to_numpy(),
            "testing_missing": test.isna().sum().to_numpy(),
        }
    )
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)

    category_shift = []
    for feature in CATEGORICAL_FEATURES:
        unseen = sorted(set(test[feature].astype(str)) - set(train[feature].astype(str)))
        category_shift.append(
            {
                "feature": feature,
                "training_categories": train[feature].nunique(),
                "testing_categories": test[feature].nunique(),
                "testing_categories_absent_from_training": ", ".join(unseen)
                if unseen
                else "None",
            }
        )
    pd.DataFrame(category_shift).to_csv(
        output_dir / "categorical_shift_summary.csv", index=False
    )


# %% Target and attack-category distributions
def plot_class_distribution(
    train: pd.DataFrame, test: pd.DataFrame, output_dir: Path
) -> None:
    combined = pd.concat(
        [train.assign(split="Training"), test.assign(split="Testing")],
        ignore_index=True,
    )
    counts = (
        combined.groupby(["split", TARGET], observed=True)
        .size()
        .reset_index(name="records")
    )
    counts["class"] = counts[TARGET].map(CLASS_NAMES)
    counts.to_csv(output_dir / "class_counts.csv", index=False)

    plt.figure(figsize=(8, 5))
    axis = sns.barplot(
        data=counts,
        x="split",
        y="records",
        hue="class",
        palette={"Normal": "#8179F2", "Suspicious": "#F77B5D"},
    )
    axis.set(title="Normal and suspicious records by supplied split", xlabel="", ylabel="Records")
    axis.ticklabel_format(style="plain", axis="y")
    for container in axis.containers:
        axis.bar_label(container, fmt="{:,.0f}", padding=3)
    save_figure(output_dir / "class_distribution.png")


def plot_attack_categories(train: pd.DataFrame, output_dir: Path) -> None:
    attacks = (
        train.loc[train[TARGET].eq(1), "attack_cat"]
        .value_counts()
        .rename_axis("attack_category")
        .reset_index(name="records")
    )
    attacks.to_csv(output_dir / "training_attack_counts.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    axis = sns.barplot(
        data=attacks,
        y="attack_category",
        x="records",
        hue="attack_category",
        palette="crest",
        legend=False,
    )
    axis.set(title="Training records by attack category", xlabel="Records", ylabel="")
    axis.ticklabel_format(style="plain", axis="x")
    for container in axis.containers:
        axis.bar_label(container, fmt="{:,.0f}", padding=3, fontsize=8)
    save_figure(output_dir / "attack_category_distribution.png")


# %% Categorical predictors
def plot_categorical_features(train: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for feature, axis in zip(CATEGORICAL_FEATURES, axes):
        top_counts = train[feature].value_counts().head(10).sort_values()
        top_counts.plot.barh(ax=axis, color="#2EC4C7")
        axis.set_title(f"Most common {feature} values")
        axis.set_xlabel("Training records")
        axis.set_ylabel("")
        axis.ticklabel_format(style="plain", axis="x")
    save_figure(output_dir / "categorical_feature_frequencies.png")

    rows = []
    for feature in CATEGORICAL_FEATURES:
        counts = train[feature].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "training_records": count,
                    "training_percent": round(100 * count / len(train), 3),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "categorical_value_counts.csv", index=False)


# %% Numerical distributions and outliers
def plot_numerical_distributions(
    train: pd.DataFrame, output_dir: Path, sample_size: int
) -> None:
    selected = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"]
    sample = train.sample(
        n=min(sample_size, len(train)), random_state=RANDOM_SEED
    ).copy()
    sample["class"] = sample[TARGET].map(CLASS_NAMES)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for feature, axis in zip(selected, axes.flat):
        plot_data = sample[[feature, "class"]].copy()
        plot_data[f"log1p_{feature}"] = np.log1p(plot_data[feature].clip(lower=0))
        sns.histplot(
            data=plot_data,
            x=f"log1p_{feature}",
            hue="class",
            stat="density",
            common_norm=False,
            element="step",
            bins=45,
            fill=False,
            ax=axis,
            palette={"Normal": "#8179F2", "Suspicious": "#F77B5D"},
        )
        axis.set_title(feature)
        axis.set_xlabel(f"log(1 + {feature})")
        axis.set_ylabel("Density")
    fig.suptitle("Selected numerical distributions by class", fontsize=16)
    save_figure(output_dir / "numerical_distributions_by_class.png")

    numeric_predictors = [
        column
        for column in train.select_dtypes(include=np.number).columns
        if column not in ["id", TARGET]
    ]
    descriptive = train[numeric_predictors].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    ).T
    descriptive.to_csv(output_dir / "numerical_descriptive_statistics.csv")

    outlier_rows = []
    for feature in numeric_predictors:
        q1 = train[feature].quantile(0.25)
        q3 = train[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((train[feature] < lower) | (train[feature] > upper)).sum())
        outlier_rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "iqr_outlier_count": count,
                "iqr_outlier_percent": round(100 * count / len(train), 3),
            }
        )
    pd.DataFrame(outlier_rows).sort_values(
        "iqr_outlier_percent", ascending=False
    ).to_csv(output_dir / "iqr_outlier_summary.csv", index=False)


# %% Correlation analysis
def plot_correlations(train: pd.DataFrame, output_dir: Path) -> None:
    numeric = train.select_dtypes(include=np.number).drop(columns="id")
    correlation_matrix = numeric.corr()
    target_correlations = (
        correlation_matrix[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    target_table = target_correlations.rename("correlation_with_label").to_frame()
    target_table["absolute_correlation"] = target_table[
        "correlation_with_label"
    ].abs()
    target_table.to_csv(output_dir / "target_correlations.csv")

    top_features = target_table.head(12).index
    plot_table = target_table.loc[top_features].sort_values("absolute_correlation")
    colours = np.where(plot_table["correlation_with_label"] >= 0, "#F77B5D", "#8179F2")
    plt.figure(figsize=(9, 6))
    plt.barh(plot_table.index, plot_table["absolute_correlation"], color=colours)
    plt.xlabel("Absolute Pearson correlation with binary label")
    plt.ylabel("")
    plt.title("Numerical features most associated with the target")
    for index, value in enumerate(plot_table["absolute_correlation"]):
        plt.text(value + 0.008, index, f"{value:.3f}", va="center")
    plt.xlim(0, max(plot_table["absolute_correlation"]) + 0.1)
    save_figure(output_dir / "target_correlation_ranking.png")

    heatmap_columns = list(top_features[:10]) + [TARGET]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        train[heatmap_columns].corr(),
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
    )
    plt.title("Correlation among target-associated numerical features")
    save_figure(output_dir / "correlation_heatmap.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    train, test = load_data(args.data_dir)
    print(f"Training shape: {train.shape}")
    print(f"Testing shape:  {test.shape}")

    create_data_quality_outputs(train, test, args.output_dir)
    plot_class_distribution(train, test, args.output_dir)
    plot_attack_categories(train, args.output_dir)
    plot_categorical_features(train, args.output_dir)
    plot_numerical_distributions(train, args.output_dir, args.plot_sample)
    plot_correlations(train, args.output_dir)

    print(f"EDA outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
