"""Integrity tests for the cross-external error-bin synthesis."""

from __future__ import annotations

import inspect
import json
import unittest

import pandas as pd

import run_cross_external_error_synthesis
from run_cross_external_error_synthesis import (
    align_error_profiles,
    feature_consistency_summary,
    json_records,
)


def profile(
    dataset_recall: tuple[float, float],
    dataset_fpr: tuple[float, float],
    support: tuple[int, int] = (200, 200),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for recall, fpr, rows in zip(dataset_recall, dataset_fpr, support):
        frames.append(
            pd.DataFrame(
                [
                    {
                        "feature": "IN_PKTS",
                        "bin": "bin_00",
                        "model": "random_forest",
                        "rows": rows * 2,
                        "normal_rows": rows,
                        "attack_rows": rows,
                        "false_positive_rate": fpr,
                        "attack_recall": recall,
                    }
                ]
            )
        )
    return frames[0], frames[1]


class CrossExternalErrorSynthesisTests(unittest.TestCase):
    def test_runner_contains_no_model_fit_or_threshold_selection(self) -> None:
        source = inspect.getsource(run_cross_external_error_synthesis)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("select_threshold", source)

    def test_consistent_flags_require_both_datasets(self) -> None:
        ton, bot = profile((0.10, 0.15), (0.30, 0.40))
        aligned = align_error_profiles(ton, bot)
        self.assertTrue(bool(aligned.iloc[0]["consistent_low_recall"]))
        self.assertTrue(
            bool(aligned.iloc[0]["consistent_high_false_positive_rate"])
        )

        ton, bot = profile((0.10, 0.25), (0.30, 0.10))
        aligned = align_error_profiles(ton, bot)
        self.assertFalse(bool(aligned.iloc[0]["consistent_low_recall"]))
        self.assertFalse(
            bool(aligned.iloc[0]["consistent_high_false_positive_rate"])
        )

    def test_support_gate_prevents_small_bin_flags(self) -> None:
        ton, bot = profile((0.01, 0.01), (0.99, 0.99), support=(99, 200))
        aligned = align_error_profiles(ton, bot)
        self.assertFalse(bool(aligned.iloc[0]["attack_supported_both"]))
        self.assertFalse(bool(aligned.iloc[0]["normal_supported_both"]))
        self.assertFalse(bool(aligned.iloc[0]["consistent_low_recall"]))
        self.assertFalse(
            bool(aligned.iloc[0]["consistent_high_false_positive_rate"])
        )

    def test_feature_summary_shares_are_bounded(self) -> None:
        ton, bot = profile((0.10, 0.15), (0.30, 0.40))
        aligned = align_error_profiles(ton, bot)
        summary = feature_consistency_summary(aligned, ton, bot)
        share_columns = [column for column in summary if "_share_" in column]
        for column in share_columns:
            self.assertGreaterEqual(float(summary.iloc[0][column]), 0.0)
            self.assertLessEqual(float(summary.iloc[0][column]), 1.0)

    def test_json_records_converts_missing_metrics_to_null(self) -> None:
        records = json_records(pd.DataFrame([{"metric": float("nan")}]))
        self.assertIsNone(records[0]["metric"])
        json.dumps(records, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
