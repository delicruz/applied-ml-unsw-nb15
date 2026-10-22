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
    load_frozen_models,
    load_protocol,
    paired_difference_intervals,
    paired_difference_samples,
    verify_locked_provenance,
)


PROTOCOL_PATH = Path("protocols/fresh_confirmation_protocol.json")
AMENDMENT_PATH = Path("protocols/fresh_confirmation_amendment_001.json")
HASH_AMENDMENT_PATH = Path("protocols/fresh_confirmation_amendment_002.json")
DEVELOPMENT_SUMMARY_PATH = Path("artifacts/fresh_target_development/summary.json")
ORIGINAL_PROTOCOL_SHA256 = (
    "1723cc40e789df2fbcc439b4025c3ad0756db0a3e7fbc4d0a2d96f089da2b58a"
)
REGISTERED_HASH_TYPO = (
    "ed3387646fda24084889f07788cd22da2978b860f5b3d2d6f1210a4e3d79998e0"
)
CORRECTED_DATASET_SHA256 = (
    "ed3387646fda24084889f07788cd22da2978b860f5b3d2d6f1210a4e3d7998e0"
)


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

    def test_frozen_models_load_with_legacy_transform_reference(self) -> None:
        models = load_frozen_models(load_protocol(PROTOCOL_PATH))
        self.assertEqual(
            set(models), {"target_rf_all_features", "frozen_source_rf"}
        )
        for record in models.values():
            self.assertTrue(hasattr(record["pipeline"], "predict_proba"))

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

    def test_original_protocol_is_immutable_and_retains_documented_typo(self) -> None:
        with PROTOCOL_PATH.open(encoding="utf-8") as handle:
            original = json.load(handle)
        self.assertEqual(sha256_file(PROTOCOL_PATH), ORIGINAL_PROTOCOL_SHA256)
        self.assertEqual(
            original["dataset"]["expected_csv_sha256"], REGISTERED_HASH_TYPO
        )

    def test_hash_amendment_is_authorised_and_changes_no_design(self) -> None:
        with HASH_AMENDMENT_PATH.open(encoding="utf-8") as handle:
            amendment = json.load(handle)
        self.assertEqual(amendment["parent_protocol_sha256"], ORIGINAL_PROTOCOL_SHA256)
        self.assertTrue(amendment["retry_authorisation"]["authorised_by_user"])
        changed = [
            key
            for key, value in amendment["correction"].items()
            if key.endswith("_changed") and value
        ]
        self.assertEqual(changed, [])
        failed = amendment["failed_attempt"]
        self.assertFalse(failed["confirmation_csv_parsed"])
        self.assertFalse(failed["confirmation_rows_read"])
        self.assertFalse(failed["predictions_computed"])
        self.assertFalse(failed["metrics_computed"])
        self.assertFalse(failed["execution_marker_created"])

    def test_load_protocol_applies_only_documented_hash_correction(self) -> None:
        with DEVELOPMENT_SUMMARY_PATH.open(encoding="utf-8") as handle:
            development_summary = json.load(handle)
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(
            protocol["dataset"]["expected_csv_sha256"], CORRECTED_DATASET_SHA256
        )
        self.assertEqual(
            protocol["dataset"]["expected_csv_sha256"],
            development_summary["data_provenance"]["csv_sha256"],
        )
        self.assertEqual(len(protocol["_technical_amendments"]), 2)

    def test_preflight_amendment_preserves_registered_design(self) -> None:
        with AMENDMENT_PATH.open(encoding="utf-8") as handle:
            amendment = json.load(handle)
        self.assertEqual(
            amendment["parent_protocol_sha256"], sha256_file(PROTOCOL_PATH)
        )
        self.assertFalse(amendment["failed_attempt"]["confirmation_dataset_opened"])
        self.assertFalse(amendment["failed_attempt"]["execution_marker_created"])
        changed = [
            key
            for key, value in amendment["correction"].items()
            if key.endswith("_changed") and value
        ]
        self.assertEqual(changed, [])
        self.assertTrue(amendment["retry_authorisation"]["authorised_by_user"])


if __name__ == "__main__":
    unittest.main()
