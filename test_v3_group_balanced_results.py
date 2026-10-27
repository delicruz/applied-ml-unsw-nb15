"""Integrity tests for the completed Version 3 development result."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "v3_group_balanced_development"


class V3GroupBalancedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        cls.fold = pd.read_csv(OUTPUT / "fold_metrics.csv")
        cls.group = pd.read_csv(OUTPUT / "group_balanced_fold_metrics.csv")
        cls.aggregate = pd.read_csv(OUTPUT / "aggregate_candidate_metrics.csv")
        cls.attacks = pd.read_csv(OUTPUT / "per_attack_recall.csv")
        cls.thresholds = pd.read_csv(OUTPUT / "thresholds.csv")
        cls.sampling = pd.read_csv(OUTPUT / "sampling_diagnostics.csv")

    def test_fixed_verdict_selects_no_candidate(self) -> None:
        self.assertIsNone(self.summary["selected_candidate"])
        self.assertEqual(
            self.summary["promotion_verdict"], "no Version 3 candidate selected"
        )
        self.assertTrue(self.summary["development_only"])
        self.assertFalse(self.summary["independent_confirmation"])

    def test_execution_record_locks_completed_outputs(self) -> None:
        record = json.loads(
            (
                ROOT
                / "protocols"
                / "v3_group_balanced_execution_attempt_001.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "COMPLETED")
        self.assertEqual(record["exit_code"], 0)
        self.assertFalse(record["automatic_retry"])
        for item in record["locked_outputs"].values():
            path = ROOT / item["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"], item["path"])

    def test_complete_registered_fold_candidate_grid(self) -> None:
        self.assertEqual(len(self.fold), 12)
        self.assertEqual(len(self.group), 12)
        self.assertEqual(len(self.thresholds), 12)
        self.assertEqual(self.fold["heldout_domain"].nunique(), 4)
        self.assertEqual(self.fold["candidate"].nunique(), 3)
        counts = self.fold.groupby("candidate")["heldout_domain"].nunique()
        self.assertTrue(counts.eq(4).all())

    def test_group_cap_was_enforced_and_concentration_reduced(self) -> None:
        self.assertTrue(self.sampling["largest_retained_group_rows"].le(64).all())
        self.assertGreaterEqual(int(self.sampling["groups"].min()), 6258)
        self.assertLess(float(self.sampling["largest_retained_group_share"].max()), 0.002)

    def test_derived_logistic_improved_reference_but_failed_eligibility(self) -> None:
        row = self.aggregate.set_index("candidate").loc[
            "group_capped_lr_derived_features"
        ]
        self.assertGreaterEqual(float(row["macro_mcc_improvement"]), 0.03)
        self.assertGreaterEqual(
            float(row["worst_balanced_accuracy_improvement"]), 0.02
        )
        self.assertTrue(bool(row["improvement_passed"]))
        self.assertFalse(bool(row["eligibility_passed"]))
        self.assertFalse(bool(row["promotion_eligible"]))
        self.assertLess(float(row["worst_domain_balanced_accuracy"]), 0.50)
        self.assertGreater(float(row["maximum_domain_false_positive_rate"]), 0.25)
        self.assertLess(float(row["worst_domain_group_balanced_mcc"]), 0.0)

    def test_no_candidate_passed_threshold_constraints_in_all_folds(self) -> None:
        by_candidate = self.thresholds.groupby("candidate")["constraints_passed"].all()
        self.assertFalse(bool(by_candidate.any()))
        self.assertEqual(int(self.thresholds["constraints_passed"].sum()), 1)

    def test_every_candidate_has_a_supported_attack_with_zero_recall(self) -> None:
        supported = self.attacks.loc[self.attacks["positive_rows"] >= 100]
        minima = supported.groupby("candidate")["recall"].min()
        self.assertTrue(np.isclose(minima.to_numpy(), 0.0).all())

    def test_metrics_are_finite_and_group_counts_match_by_domain(self) -> None:
        for column in ("mcc", "balanced_accuracy", "recall", "false_positive_rate"):
            self.assertTrue(np.isfinite(self.fold[column]).all())
            self.assertTrue(np.isfinite(self.group[column]).all())
        group_counts = self.group.groupby("heldout_domain")["groups"].nunique()
        self.assertTrue(group_counts.eq(1).all())


if __name__ == "__main__":
    unittest.main()
