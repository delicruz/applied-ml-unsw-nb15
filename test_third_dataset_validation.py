"""Integrity tests for the sealed third-dataset protocol and runner."""

from __future__ import annotations

import json
import inspect
import unittest
from pathlib import Path

import pandas as pd

from run_external_netflow_validation import sha256_file
import run_third_dataset_validation
from run_third_dataset_validation import apply_decision_rule, load_protocol


PROTOCOL_PATH = Path("protocols/third_dataset_preregistration.json")


class ThirdDatasetProtocolTests(unittest.TestCase):
    def test_protocol_is_sealed_before_data_access(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        access = protocol["data_access_at_registration"]
        self.assertFalse(access["archive_downloaded"])
        self.assertFalse(access["row_level_data_viewed"])
        self.assertFalse(access["labels_or_predictions_computed"])
        self.assertFalse(protocol["analysis"]["external_threshold_tuning"])
        self.assertFalse(protocol["analysis"]["model_refitting"])

    def test_frozen_model_hashes_match_local_artifacts(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        for record in protocol["frozen_inputs"]["models"].values():
            self.assertEqual(sha256_file(Path(record["artifact"])), record["sha256"])

    def test_decision_rule_requires_every_condition(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        metrics = pd.DataFrame(
            [
                {"model": "random_forest", "metric": "balanced_accuracy", "value": 0.60},
                {"model": "random_forest", "metric": "average_precision_lift", "value": 1.10},
                {"model": "random_forest", "metric": "mcc", "value": 0.20},
            ]
        )
        intervals = pd.DataFrame(
            [
                {
                    "dataset": "NF-BoT-IoT",
                    "model": "random_forest",
                    "metric": "mcc",
                    "estimate": 0.20,
                    "ci_lower": 0.01,
                    "ci_upper": 0.30,
                }
            ]
        )
        supported = apply_decision_rule(metrics, intervals, protocol)
        self.assertEqual(supported["verdict"], "useful transfer supported")
        intervals.loc[0, "ci_lower"] = -0.01
        unsupported = apply_decision_rule(metrics, intervals, protocol)
        self.assertEqual(unsupported["verdict"], "useful transfer not supported")

    def test_runner_contains_no_model_fit_path(self) -> None:
        source = inspect.getsource(run_third_dataset_validation)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("select_threshold", source)


if __name__ == "__main__":
    unittest.main()
