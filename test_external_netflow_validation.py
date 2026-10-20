"""Focused integrity tests for the external NetFlow validation runner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_external_netflow_validation import (
    ENDPOINT_FEATURES,
    MODEL_FEATURES,
    balanced_group_folds,
    metric_from_group_stats,
)


class ExternalValidationIntegrityTests(unittest.TestCase):
    def test_endpoint_identifiers_are_not_model_features(self) -> None:
        self.assertTrue(set(ENDPOINT_FEATURES).isdisjoint(MODEL_FEATURES))

    def test_exact_groups_never_cross_folds(self) -> None:
        groups = pd.Series(
            np.repeat(np.arange(40, dtype="uint64"), np.tile([2, 3, 4, 5], 10))
        )
        labels = pd.Series((groups.to_numpy() % 7 == 0).astype("int8"))
        row_folds, summary = balanced_group_folds(groups, labels)
        per_group_fold_count = (
            pd.DataFrame({"group": groups, "fold": row_folds})
            .groupby("group")["fold"]
            .nunique()
        )
        self.assertEqual(int(per_group_fold_count.max()), 1)
        self.assertEqual(set(summary["fold"]), {0, 1, 2, 3, 4})
        self.assertEqual(int(summary["rows"].sum()), len(groups))

    def test_grouped_average_precision_matches_row_level_metric(self) -> None:
        positives = np.array([2.0, 0.0, 1.0, 3.0])
        negatives = np.array([1.0, 4.0, 2.0, 0.0])
        scores = np.array([0.9, 0.8, 0.8, 0.2])
        grouped = metric_from_group_stats(
            positives, negatives, scores, threshold=0.5
        )["average_precision"]

        labels = []
        row_scores = []
        for positive, negative, score in zip(positives, negatives, scores):
            labels.extend([1] * int(positive) + [0] * int(negative))
            row_scores.extend([score] * int(positive + negative))
        expected = average_precision_score(labels, row_scores)
        self.assertAlmostEqual(grouped, expected, places=12)


if __name__ == "__main__":
    unittest.main()
