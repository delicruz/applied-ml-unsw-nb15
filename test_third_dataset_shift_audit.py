"""Focused integrity tests for the third-dataset shift audit."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import pandas as pd

import run_third_dataset_shift_audit
from run_third_dataset_shift_audit import (
    cross_external_score_comparison,
    labelled_score_rows,
)


class ThirdDatasetShiftAuditTests(unittest.TestCase):
    def test_runner_contains_no_model_fit_path(self) -> None:
        source = inspect.getsource(run_third_dataset_shift_audit)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("select_threshold", source)

    def test_score_ordering_is_labelled_from_mean_margin(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "source_validation",
                    "model": "random_forest",
                    "attack_mean_probability": 0.8,
                    "normal_mean_probability": 0.2,
                },
                {
                    "dataset": "external",
                    "model": "random_forest",
                    "attack_mean_probability": 0.1,
                    "normal_mean_probability": 0.4,
                },
            ]
        )
        labelled = labelled_score_rows(frame, "NF-BoT-IoT")
        self.assertEqual(
            list(labelled["score_ordering"]),
            ["attack_above_benign", "attack_at_or_below_benign"],
        )
        self.assertAlmostEqual(
            labelled.iloc[1]["attack_minus_normal_mean_probability"], -0.3
        )

    def test_missing_previous_audit_keeps_current_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset": "external",
                    "model": "logistic_regression",
                    "attack_mean_probability": 0.7,
                    "normal_mean_probability": 0.6,
                }
            ]
        )
        result = cross_external_score_comparison(
            frame, Path("definitely-missing-third-shift-test-directory")
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["dataset"], "NF-BoT-IoT")

    def test_cross_dataset_order_starts_with_source(self) -> None:
        current = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "model": "random_forest",
                    "attack_mean_probability": attack,
                    "normal_mean_probability": normal,
                }
                for dataset, attack, normal in (
                    ("source_validation", 0.9, 0.1),
                    ("external", 0.2, 0.4),
                )
            ]
        )
        result = cross_external_score_comparison(
            current, Path("definitely-missing-third-shift-test-directory")
        )
        self.assertEqual(
            list(result["dataset"]),
            ["NF-UNSW-NB15 source validation", "NF-BoT-IoT"],
        )


if __name__ == "__main__":
    unittest.main()
