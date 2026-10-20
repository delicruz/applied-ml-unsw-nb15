"""Quantify alert burden and development-only threshold cost trade-offs.

This stage preserves the final evaluation design:

1. candidate operating policies are selected only on an internal
   near-duplicate-grouped fold of the development data;
2. those policies are evaluated on the existing validation partition; and
3. the supplied test set is used only to report the already-frozen threshold
   of 0.54 on the conservative primary holdout scope.

No threshold is selected or changed using supplied-test performance.
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
from sklearn.metrics import confusion_matrix

from run_preliminary_validation import (
    DEVELOPMENT_PATH,
    SEED,
    VALIDATION_PATH,
    make_models,
    split_xy,
)
from run_robustness_audit import (
    SIGNIFICANT_DIGITS,
    TTL_FEATURES,
    internal_tuning_split,
    near_duplicate_hashes,
)


OUTPUT_DIR = Path("artifacts/operational_burden")
HOLDOUT_PREDICTIONS = Path(
    "artifacts/final_holdout_evaluation/test_predictions.csv"
)
FROZEN_THRESHOLD = 0.54
FALSE_NEGATIVE_COST_RATIOS = (1, 2, 5, 10, 20)
ALERT_BUDGETS_PER_10K_NORMAL = (100, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold grid step between 0.01 and 0.99.",
    )
    return parser.parse_args()


def threshold_grid(step: float) -> np.ndarray:
    if not 0.005 <= step <= 0.10:
        raise ValueError("--threshold-step must be between 0.005 and 0.10")
    grid = np.arange(0.01, 0.991, step)
    return np.unique(np.append(np.round(grid, 4), FROZEN_THRESHOLD))


def burden_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted.astype(int), labels=[0, 1]
    ).ravel()
    normal_count = tn + fp
    suspicious_count = fn + tp
    row_count = len(y_true)
    false_positive_rate = fp / normal_count if normal_count else float("nan")
    false_negative_rate = fn / suspicious_count if suspicious_count else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / suspicious_count if suspicious_count else 0.0
    return {
        "threshold": float(threshold),
        "rows": int(row_count),
        "normal_rows": int(normal_count),
        "suspicious_rows": int(suspicious_count),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_alerts_per_10000_normal": float(false_positive_rate * 10_000),
        "missed_attacks_per_10000_suspicious": float(
            false_negative_rate * 10_000
        ),
        "total_alerts_per_10000_flows": float(predicted.mean() * 10_000),
        "true_attacks_per_100_alerts": float(precision * 100),
    }


def build_threshold_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = [burden_metrics(y_true, probabilities, value) for value in thresholds]
    table = pd.DataFrame(rows)
    for ratio in FALSE_NEGATIVE_COST_RATIOS:
        table[f"cost_fp1_fn{ratio}_per_10000_flows"] = (
            table["fp"] + ratio * table["fn"]
        ) / table["rows"] * 10_000
    return table


def select_policies(table: pd.DataFrame) -> pd.DataFrame:
    policies: list[dict[str, Any]] = []
    for ratio in FALSE_NEGATIVE_COST_RATIOS:
        cost_column = f"cost_fp1_fn{ratio}_per_10000_flows"
        best = table.sort_values(
            [cost_column, "recall", "threshold"],
            ascending=[True, False, True],
        ).iloc[0]
        policies.append(
            {
                "policy": f"minimum_cost_fp1_fn{ratio}",
                "selection_basis": "internal_grouped_development_fold",
                "false_positive_cost": 1,
                "false_negative_cost": ratio,
                "alert_budget_per_10000_normal": np.nan,
                "feasible": True,
                **best.to_dict(),
            }
        )

    for budget in ALERT_BUDGETS_PER_10K_NORMAL:
        feasible = table[table["false_alerts_per_10000_normal"] <= budget]
        if feasible.empty:
            policies.append(
                {
                    "policy": f"alert_budget_{budget}_per_10000_normal",
                    "selection_basis": "internal_grouped_development_fold",
                    "false_positive_cost": np.nan,
                    "false_negative_cost": np.nan,
                    "alert_budget_per_10000_normal": budget,
                    "feasible": False,
                }
            )
            continue
        best = feasible.sort_values(
            ["recall", "false_alerts_per_10000_normal", "threshold"],
            ascending=[False, True, True],
        ).iloc[0]
        policies.append(
            {
                "policy": f"alert_budget_{budget}_per_10000_normal",
                "selection_basis": "internal_grouped_development_fold",
                "false_positive_cost": np.nan,
                "false_negative_cost": np.nan,
                "alert_budget_per_10000_normal": budget,
                "feasible": True,
                **best.to_dict(),
            }
        )
    return pd.DataFrame(policies)


def evaluate_policies(
    policies: pd.DataFrame,
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in policies.to_dict(orient="records"):
        if not policy.get("feasible", False):
            continue
        metrics = burden_metrics(
            y_validation,
            validation_probabilities,
            float(policy["threshold"]),
        )
        rows.append(
            {
                "policy": policy["policy"],
                "selection_basis": policy["selection_basis"],
                "evaluation_scope": "current_validation",
                "false_positive_cost": policy["false_positive_cost"],
                "false_negative_cost": policy["false_negative_cost"],
                "alert_budget_per_10000_normal": policy[
                    "alert_budget_per_10000_normal"
                ],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def conservative_holdout() -> tuple[np.ndarray, np.ndarray]:
    if not HOLDOUT_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing {HOLDOUT_PREDICTIONS}; run final holdout evaluation first"
        )
    predictions = pd.read_csv(HOLDOUT_PREDICTIONS)
    required = {
        "label",
        "probability",
        "historically_evaluated_5000",
        "training_near_duplicate_final_features",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Holdout prediction columns missing: {sorted(missing)}")
    mask = (
        ~predictions["historically_evaluated_5000"].astype(bool)
        & ~predictions["training_near_duplicate_final_features"].astype(bool)
    )
    primary = predictions.loc[mask]
    if len(primary) != 69_222:
        raise ValueError(
            f"Conservative holdout mask returned {len(primary):,} rows, expected 69,222"
        )
    return (
        primary["label"].to_numpy(dtype=int),
        primary["probability"].to_numpy(dtype=float),
    )


def save_plots(
    threshold_table: pd.DataFrame,
    frozen_burden: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        threshold_table["false_alerts_per_10000_normal"],
        threshold_table["missed_attacks_per_10000_suspicious"],
        color="#0072B2",
        linewidth=2,
    )
    frozen = threshold_table.loc[
        np.isclose(threshold_table["threshold"], FROZEN_THRESHOLD)
    ].iloc[0]
    axis.scatter(
        [frozen["false_alerts_per_10000_normal"]],
        [frozen["missed_attacks_per_10000_suspicious"]],
        color="#D55E00",
        s=70,
        zorder=3,
        label=f"Frozen threshold {FROZEN_THRESHOLD:.2f}",
    )
    axis.set_xlabel("False alerts per 10,000 Normal flows")
    axis.set_ylabel("Missed attacks per 10,000 Suspicious flows")
    axis.set_title("Internal development trade-off: alert burden vs missed attacks")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "alert_miss_tradeoff.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for ratio in FALSE_NEGATIVE_COST_RATIOS:
        column = f"cost_fp1_fn{ratio}_per_10000_flows"
        axis.plot(
            threshold_table["threshold"],
            threshold_table[column],
            linewidth=1.7,
            label=f"FN cost = {ratio}× FP",
        )
    axis.axvline(FROZEN_THRESHOLD, color="#333333", linestyle="--", linewidth=1.5)
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Relative cost per 10,000 flows")
    axis.set_title("Development-only cost sensitivity")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "cost_sensitivity.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.7))
    display = frozen_burden.copy()
    labels = {
        "internal_grouped_development_fold": "Internal tuning fold",
        "current_validation": "Validation",
        "conservative_primary_holdout": "Conservative holdout",
    }
    axis.bar(
        display["scope"].map(labels),
        display["false_alerts_per_10000_normal"],
        color=["#009E73", "#56B4E9", "#D55E00"],
    )
    axis.set_ylabel("False alerts per 10,000 Normal flows")
    axis.set_title(f"Frozen-threshold alert burden (threshold = {FROZEN_THRESHOLD:.2f})")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "frozen_threshold_burden.png", dpi=200)
    plt.close(figure)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    if list(development.columns) != list(validation.columns):
        raise ValueError("Development and validation schemas do not match")

    X_development, y_development = split_xy(development)
    X_validation, y_validation = split_xy(validation)
    columns = [
        column for column in X_development.columns if column not in TTL_FEATURES
    ]
    development_groups = near_duplicate_hashes(
        development, SIGNIFICANT_DIGITS
    )
    fit_indices, tuning_indices = internal_tuning_split(
        development, development_groups
    )

    print("Fitting internal-fold no-TTL model...", flush=True)
    internal_started = time.time()
    internal_model = make_models(X_development.iloc[fit_indices][columns])[
        "Random forest"
    ]
    internal_model.fit(
        X_development.iloc[fit_indices][columns],
        y_development.iloc[fit_indices],
    )
    internal_seconds = time.time() - internal_started
    internal_probabilities = internal_model.predict_proba(
        X_development.iloc[tuning_indices][columns]
    )[:, 1]
    internal_y = y_development.iloc[tuning_indices].to_numpy(dtype=int)

    thresholds = threshold_grid(args.threshold_step)
    internal_table = build_threshold_table(
        internal_y, internal_probabilities, thresholds
    )
    policies = select_policies(internal_table)

    print("Fitting full-development no-TTL model for validation...", flush=True)
    validation_started = time.time()
    validation_model = make_models(X_development[columns])["Random forest"]
    validation_model.fit(X_development[columns], y_development)
    validation_seconds = time.time() - validation_started
    validation_probabilities = validation_model.predict_proba(
        X_validation[columns]
    )[:, 1]
    validation_y = y_validation.to_numpy(dtype=int)
    policy_evaluation = evaluate_policies(
        policies, validation_y, validation_probabilities
    )

    holdout_y, holdout_probabilities = conservative_holdout()
    frozen_rows = []
    for scope, y_true, probabilities in (
        (
            "internal_grouped_development_fold",
            internal_y,
            internal_probabilities,
        ),
        ("current_validation", validation_y, validation_probabilities),
        (
            "conservative_primary_holdout",
            holdout_y,
            holdout_probabilities,
        ),
    ):
        frozen_rows.append(
            {"scope": scope, **burden_metrics(y_true, probabilities, FROZEN_THRESHOLD)}
        )
    frozen_burden = pd.DataFrame(frozen_rows)

    internal_table.to_csv(
        args.output_dir / "threshold_tradeoffs_internal.csv", index=False
    )
    policies.to_csv(args.output_dir / "selected_cost_policies.csv", index=False)
    policy_evaluation.to_csv(
        args.output_dir / "validation_policy_evaluation.csv", index=False
    )
    frozen_burden.to_csv(
        args.output_dir / "frozen_threshold_burden.csv", index=False
    )
    save_plots(internal_table, frozen_burden, args.output_dir)

    result = {
        "metadata": {
            "purpose": "operational alert burden and cost sensitivity",
            "selection_data": "internal near-duplicate-grouped development fold",
            "validation_role": "policy evaluation only",
            "holdout_role": "report frozen threshold only",
            "holdout_used_for_threshold_selection": False,
            "frozen_threshold_changed": False,
            "frozen_threshold": FROZEN_THRESHOLD,
            "removed_features": list(TTL_FEATURES),
            "near_duplicate_significant_digits": SIGNIFICANT_DIGITS,
            "false_positive_cost_unit": 1,
            "false_negative_cost_ratios": list(FALSE_NEGATIVE_COST_RATIOS),
            "alert_budgets_per_10000_normal": list(
                ALERT_BUDGETS_PER_10K_NORMAL
            ),
            "seed": SEED,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "internal_model_fit_seconds": internal_seconds,
            "validation_model_fit_seconds": validation_seconds,
        },
        "frozen_threshold_burden": json_records(frozen_burden),
        "selected_cost_policies": json_records(policies),
        "validation_policy_evaluation": json_records(policy_evaluation),
        "interpretation_boundary": (
            "Costs are relative sensitivity scenarios, not measured operational "
            "utilities. Holdout results did not select or change any threshold."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\nFrozen threshold burden", flush=True)
    print(
        frozen_burden[
            [
                "scope",
                "false_alerts_per_10000_normal",
                "missed_attacks_per_10000_suspicious",
                "precision",
                "recall",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nSaved outputs to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
