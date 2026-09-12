"""Create leakage-aware development and validation splits for UNSW-NB15.

The supplied testing file is never modified or included in the split. Exact
predictor vectors are assigned to one partition only, reducing the chance that
repeated traffic patterns leak between development and validation data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


TARGET = "label"
STRATIFY_COLUMN = "attack_cat"
EXCLUDED_FROM_PREDICTORS = ["id", STRATIFY_COLUMN, TARGET]
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--summary-dir", type=Path, default=Path("artifacts/eda")
    )
    return parser.parse_args()


def choose_validation_fold(
    frame: pd.DataFrame, groups: pd.Series
) -> tuple[pd.Index, pd.Index]:
    """Return the 80/20 grouped fold closest to the full attack distribution."""
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_SEED
    )
    overall_distribution = frame[STRATIFY_COLUMN].value_counts(normalize=True)
    candidates = []

    for development_index, validation_index in splitter.split(
        frame, y=frame[STRATIFY_COLUMN], groups=groups
    ):
        validation_distribution = (
            frame.iloc[validation_index][STRATIFY_COLUMN]
            .value_counts(normalize=True)
            .reindex(overall_distribution.index, fill_value=0)
        )
        size_error = abs(len(validation_index) / len(frame) - 0.20)
        distribution_error = (
            validation_distribution - overall_distribution
        ).abs().mean()
        candidates.append(
            (size_error + distribution_error, development_index, validation_index)
        )

    _, development_index, validation_index = min(
        candidates, key=lambda candidate: candidate[0]
    )
    return pd.Index(development_index), pd.Index(validation_index)


def write_summaries(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    supplied_test: pd.DataFrame,
    summary_dir: Path,
) -> None:
    partitions = {
        "Development training": development,
        "Validation": validation,
        "Supplied testing": supplied_test,
    }
    total_records = sum(len(partition) for partition in partitions.values())

    summary_rows = []
    attack_rows = []
    for name, partition in partitions.items():
        summary_rows.append(
            {
                "partition": name,
                "records": len(partition),
                "percent_of_all_records": round(100 * len(partition) / total_records, 3),
                "normal_records": int(partition[TARGET].eq(0).sum()),
                "suspicious_records": int(partition[TARGET].eq(1).sum()),
                "suspicious_percent": round(100 * partition[TARGET].mean(), 3),
            }
        )
        for attack_category, count in partition[STRATIFY_COLUMN].value_counts().items():
            attack_rows.append(
                {
                    "partition": name,
                    "attack_category": attack_category,
                    "records": count,
                    "partition_percent": round(100 * count / len(partition), 3),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        summary_dir / "split_summary.csv", index=False
    )
    pd.DataFrame(attack_rows).to_csv(
        summary_dir / "split_attack_category_counts.csv", index=False
    )


def main() -> None:
    args = parse_args()
    training_path = args.data_dir / "UNSW_NB15_training-set.csv"
    testing_path = args.data_dir / "UNSW_NB15_testing-set.csv"
    for path in (training_path, testing_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    supplied_training = pd.read_csv(training_path)
    supplied_test = pd.read_csv(testing_path)
    predictor_columns = [
        column
        for column in supplied_training.columns
        if column not in EXCLUDED_FROM_PREDICTORS
    ]

    predictor_groups = pd.util.hash_pandas_object(
        supplied_training[predictor_columns], index=False
    )
    development_index, validation_index = choose_validation_fold(
        supplied_training, predictor_groups
    )

    development = supplied_training.iloc[development_index].sample(
        frac=1, random_state=RANDOM_SEED
    )
    validation = supplied_training.iloc[validation_index].sample(
        frac=1, random_state=RANDOM_SEED
    )

    development_groups = set(predictor_groups.iloc[development_index])
    validation_groups = set(predictor_groups.iloc[validation_index])
    if development_groups.intersection(validation_groups):
        raise RuntimeError("Predictor groups overlap between the generated splits")
    if len(development) + len(validation) != len(supplied_training):
        raise RuntimeError("Generated splits do not cover the supplied training data")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(
        args.output_dir / "UNSW_NB15_development-training.csv", index=False
    )
    validation.to_csv(args.output_dir / "UNSW_NB15_validation.csv", index=False)
    write_summaries(development, validation, supplied_test, args.summary_dir)

    print(f"Development training shape: {development.shape}")
    print(f"Validation shape:           {validation.shape}")
    print(f"Supplied testing shape:     {supplied_test.shape} (unchanged)")
    print(f"Processed splits written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
