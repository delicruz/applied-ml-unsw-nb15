"""Integrity tests for the post-hoc fresh generalisation-failure audit."""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_fresh_generalisation_failure_audit
from run_fresh_generalisation_failure_audit import (
    attack_comparison,
    group_size_analysis,
    load_audit_protocol,
)


PROTOCOL_PATH = Path("protocols/fresh_generalisation_failure_audit.json")


class FreshGeneralisationFailureAuditTests(unittest.TestCase):
    def test_protocol_is_posthoc_and_prohibits_adaptation(self) -> None:
        protocol = load_audit_protocol(PROTOCOL_PATH)
        self.assertEqual(
            protocol["status"], "REGISTERED_POSTHOC_SCOPE_BEFORE_AUDIT_RUN"
        )
        self.assertFalse(any(protocol["scope_boundaries"].values()))

    def test_runner_has_no_fit_or_selection_path(self) -> None:
        source = inspect.getsource(run_fresh_generalisation_failure_audit)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("select_threshold", source)
        self.assertNotIn("make_target_model", source)

    def test_attack_comparison_exposes_missing_validation_attack(self) -> None:
        validation = pd.DataFrame(
            {"Attack": ["Benign", "Seen"], "Label": [0, 1]}
        )
        confirmation = pd.DataFrame(
            {"Attack": ["Benign", "New", "New"], "Label": [0, 1, 1]}
        )
        probabilities = {
            "target_validation": np.array([0.1, 0.9]),
            "sealed_confirmation": np.array([0.1, 0.2, 0.8]),
        }
        result = attack_comparison(
            validation, confirmation, probabilities, threshold=0.5
        ).set_index("attack")
        self.assertEqual(result.loc["New", "target_validation_positive_rows"], 0)
        self.assertEqual(result.loc["New", "sealed_confirmation_positive_rows"], 2)
        self.assertAlmostEqual(result.loc["New", "sealed_confirmation_recall"], 0.5)

    def test_group_size_analysis_reports_concentration(self) -> None:
        frame = pd.DataFrame(
            {
                "_group_key": ["a", "b", "b", "b"],
                "Label": [0, 1, 1, 1],
            }
        )
        profile, summary = group_size_analysis(
            "scope", frame, np.array([0.1, 0.9, 0.9, 0.9]), threshold=0.5
        )
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["maximum_rows_per_group"], 3)
        self.assertEqual(int(profile["rows"].sum()), 4)

    def test_protocol_json_is_strict(self) -> None:
        with PROTOCOL_PATH.open(encoding="utf-8") as handle:
            json.load(handle)


if __name__ == "__main__":
    unittest.main()
