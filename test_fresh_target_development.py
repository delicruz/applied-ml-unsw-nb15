"""Integrity tests for the fresh target-development runner."""

from __future__ import annotations

import inspect
import json
import unittest

import numpy as np
import pandas as pd

import run_fresh_target_development
from run_fresh_target_development import (
    MODEL_FEATURES,
    REDUCED_FEATURES,
    assign_partitions,
    canonical_group_hashes,
    evaluate_candidates,
    json_safe,
    normalise_attack_counts,
    round_to_significant,
)


class FreshTargetDevelopmentTests(unittest.TestCase):
    def test_reduced_features_are_a_strict_subset(self) -> None:
        self.assertTrue(set(REDUCED_FEATURES) < set(MODEL_FEATURES))

    def test_significant_rounding(self) -> None:
        values = np.array([0.0, 0.001234567, 123456.7, np.nan])
        observed = round_to_significant(values, digits=5)
        self.assertEqual(observed[0], 0.0)
        self.assertAlmostEqual(observed[1], 0.0012346)
        self.assertEqual(observed[2], 123460.0)
        self.assertTrue(np.isnan(observed[3]))

    def test_near_duplicate_rows_share_hash_and_partition(self) -> None:
        base = {
            "PROTOCOL": [6.0, 6.0],
            "L7_PROTO": [7.0, 7.0],
            "IN_BYTES": [12345.11, 12345.12],
            "OUT_BYTES": [80.0, 80.0],
            "IN_PKTS": [10.0, 10.0],
            "OUT_PKTS": [9.0, 9.0],
            "TCP_FLAGS": [27.0, 27.0],
            "FLOW_DURATION_MILLISECONDS": [101.0, 101.0],
        }
        first, second = canonical_group_hashes(pd.DataFrame(base))
        self.assertEqual(int(first[0]), int(first[1]))
        self.assertEqual(int(second[0]), int(second[1]))
        self.assertEqual(assign_partitions(first)[0], assign_partitions(first)[1])

    def test_evaluator_has_no_confirmation_argument(self) -> None:
        parameters = inspect.signature(evaluate_candidates).parameters
        self.assertEqual(
            list(parameters), ["training", "calibration", "validation", "output_dir"]
        )

    def test_runner_does_not_read_prior_external_datasets(self) -> None:
        source = inspect.getsource(run_fresh_target_development)
        self.assertNotIn("NF-ToN-IoT.csv", source)
        self.assertNotIn("NF-BoT-IoT.csv", source)

    def test_json_safe_rejects_nonstandard_nan(self) -> None:
        safe = json_safe({"missing": float("nan"), "value": np.float64(1.5)})
        self.assertIsNone(safe["missing"])
        json.dumps(safe, allow_nan=False)

    def test_raw_attack_labels_normalise_to_official_parents(self) -> None:
        raw = {
            "Benign": 10,
            "Bot": 1,
            "Infilteration": 2,
            "FTP-BruteForce": 3,
            "SSH-Bruteforce": 4,
            "Brute Force -Web": 5,
            "Brute Force -XSS": 6,
            "SQL Injection": 7,
            "DoS attacks-Hulk": 8,
            "DoS attacks-SlowHTTPTest": 9,
            "DoS attacks-GoldenEye": 10,
            "DoS attacks-Slowloris": 11,
            "DDOS attack-HOIC": 12,
            "DDoS attacks-LOIC-HTTP": 13,
            "DDOS attack-LOIC-UDP": 14,
        }
        self.assertEqual(
            normalise_attack_counts(raw),
            {
                "Benign": 10,
                "Bot": 1,
                "BruteForce": 7,
                "DDoS": 39,
                "DoS": 38,
                "Infiltration": 2,
                "Web Attacks": 18,
            },
        )


if __name__ == "__main__":
    unittest.main()
