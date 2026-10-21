"""Integrity tests for the sealed fresh-confirmation runner."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run_fresh_confirmation_evaluation
from run_external_netflow_validation import sha256_file
from run_fresh_confirmation_evaluation import (
    apply_decision_rules,
    claim_one_time_execution,
    load_protocol,
    paired_difference_intervals,
    paired_difference_samples,
    verify_locked_provenance,
)


PROTOCOL_PATH = Path("protocols/fresh_confirmation_protocol.json")


class FreshConfirmationTests(unittest.TestCase):
    def test_protocol_is_sealed_before_confirmation_access(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(
            protocol["status"], "PREREGISTERED_BEFORE_CONFIRMATION_ACCESS"
        )
        access = protocol["confirmation_access_at_registration"]
        self.assertFalse(access["confirmation_predictions_computed"])
        self.assertFalse(access["confirmation_metrics_computed"])
        self.assertFalse(access["confirmation_labels_summarised"])

    def test_frozen_model_hashes_and_sizes_match_local_artifacts(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        for key in ("selected_candidate", "comparison_baseline"):
            record = protocol["frozen_inputs"][key]
            path = Path(record["artifact"])
            self.assertEqual(path.stat().st_size, record["artifact_bytes"])
            self.assertEqual(sha256_file(path), record["sha256"])

    def test_locked_development_provenance_matches(self) -> None:
        verify_locked_provenance(load_protocol(PROTOCOL_PATH))

    def test_one_time_marker_blocks_repeat_execution(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "confirmation"
            claim_one_time_execution(output_dir, protocol)
            marker = output_dir / "execution_started.json"
            self.assertTrue(marker.exists())
            with self.assertRaises(FileExistsError):
                claim_one_time_execution(output_dir, protocol)

    def test_runner_contains_no_fit_or_threshold_selection_path(self) -> None:
        source = inspect.getsource(run_fresh_confirmation_evaluation)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("select_threshold", source)
        self.assertNotIn("target_calibration", source)

    def test_paired_bootstrap_uses_same_iteration(self) -> None:
        samples = pd.DataFrame(
            [
                {"dataset": "sealed", "model": "candidate", "iteration": 0, "metric": "mcc", "value": 0.7},
                {"dataset": "sealed", "model": "baseline", "iteration": 0, "metric": "mcc", "value": 0.2},
                {"dataset": "sealed", "model": "candidate", "iteration": 1, "metric": "mcc", "value": 0.6},
                {"dataset": "sealed", "model": "baseline", "iteration": 1, "metric": "mcc", "value": 0.3},
            ]
        )
        differences = paired_difference_samples(samples, "candidate", "baseline")
        observed = differences["candidate_minus_baseline"].tolist()
        self.assertAlmostEqual(observed[0], 0.5)
        self.assertAlmostEqual(observed[1], 0.3)
        intervals = paired_difference_intervals(differences)
        self.assertEqual(intervals.iloc[0]["metric"], "mcc")

    def test_decision_rule_keeps_aggregate_and_coverage_separate(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        metric_values = {
            "mcc": 0.70,
            "balanced_accuracy": 0.90,
            "average_precision_lift": 5.0,
            "false_positive_rate": 0.04,
        }
        metrics = pd.DataFrame(
            [
                {"model": "target_rf_all_features", "metric": metric, "value": value}
                for metric, value in metric_values.items()
            ]
        )
        paired = pd.DataFrame(
            [
                {"dataset": "sealed", "metric": "mcc", "estimate": 0.5, "ci_lower": 0.4, "ci_upper": 0.6},
                {"dataset": "sealed", "metric": "balanced_accuracy", "estimate": 0.3, "ci_lower": 0.2, "ci_upper": 0.4},
            ]
        )
        attacks = pd.DataFrame(
            [
                {"model": "target_rf_all_features", "attack": "large", "positive_rows": 1000, "recall": 0.90},
                {"model": "target_rf_all_features", "attack": "weak", "positive_rows": 200, "recall": 0.20},
            ]
        )
        decision = apply_decision_rules(metrics, paired, attacks, protocol)
        self.assertTrue(decision["aggregate_improvement_supported"])
        self.assertFalse(decision["broad_attack_coverage_supported"])
        self.assertIn("coverage not supported", decision["verdict"])

    def test_protocol_json_is_strict(self) -> None:
        with PROTOCOL_PATH.open(encoding="utf-8") as handle:
            json.load(handle)


if __name__ == "__main__":
    unittest.main()
