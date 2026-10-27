"""Integrity tests for the completed Version 2 development outputs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "v2_generalisation_development"


class V2GeneralisationResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        cls.folds = pd.read_csv(OUTPUT / "fold_metrics.csv")
        cls.aggregate = pd.read_csv(OUTPUT / "aggregate_candidate_metrics.csv")
        cls.attacks = pd.read_csv(OUTPUT / "per_attack_recall.csv")
        cls.thresholds = pd.read_csv(OUTPUT / "thresholds.csv")

    def test_complete_registered_fold_candidate_grid(self) -> None:
        self.assertEqual(len(self.folds), 16)
        self.assertEqual(self.folds["heldout_domain"].nunique(), 4)
        self.assertEqual(self.folds["candidate"].nunique(), 4)
        self.assertEqual(len(self.aggregate), 4)

    def test_confusion_metrics_are_internally_consistent(self) -> None:
        confusion_total = self.folds[
            ["true_negative", "false_positive", "false_negative", "true_positive"]
        ].sum(axis=1)
        self.assertTrue((confusion_total == self.folds["rows"]).all())
        np.testing.assert_allclose(
            self.folds["balanced_accuracy"],
            (self.folds["recall"] + self.folds["specificity"]) / 2,
        )
        np.testing.assert_allclose(
            self.folds["false_positive_rate"], 1 - self.folds["specificity"]
        )

    def test_threshold_records_match_fold_metrics(self) -> None:
        merged = self.folds.merge(
            self.thresholds,
            on=["heldout_domain", "candidate"],
            suffixes=("_metric", "_record"),
            validate="one_to_one",
        )
        np.testing.assert_allclose(merged["threshold_metric"], merged["threshold_record"])
        self.assertTrue(
            (
                merged["threshold_constraints_passed"]
                == merged["constraints_passed"]
            ).all()
        )

    def test_attack_totals_match_binary_positive_totals(self) -> None:
        expected = self.folds.set_index(["heldout_domain", "candidate"])[
            "true_positive"
        ] + self.folds.set_index(["heldout_domain", "candidate"])["false_negative"]
        observed = self.attacks.groupby(["heldout_domain", "candidate"])[
            "positive_rows"
        ].sum()
        pd.testing.assert_series_equal(
            expected.sort_index(), observed.sort_index(), check_names=False
        )

    def test_fixed_verdict_selects_no_candidate(self) -> None:
        self.assertFalse(self.aggregate["promotion_eligible"].any())
        self.assertIsNone(self.summary["selected_candidate"])
        self.assertEqual(
            self.summary["promotion_verdict"], "no Version 2 candidate selected"
        )
        self.assertTrue(self.summary["development_only"])
        self.assertFalse(self.summary["independent_confirmation"])


if __name__ == "__main__":
    unittest.main()
