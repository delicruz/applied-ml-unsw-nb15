"""Integrity checks for the consolidated final research-prototype package."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "deliverables" / "final_research_prototype"
MANIFEST = ROOT / "artifacts" / "final_research_prototype" / "evidence_manifest.json"


class FinalResearchPrototypePackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with (PACKAGE / "Key_Results.csv").open(encoding="utf-8", newline="") as handle:
            cls.results = list(csv.DictReader(handle))
        cls.status = (PACKAGE / "Project_Final_Status.md").read_text(encoding="utf-8")

    def test_evidence_hashes_match(self) -> None:
        for item in self.manifest["evidence"].values():
            path = ROOT / item["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"], item["path"])

    def test_final_boundary_is_explicit(self) -> None:
        self.assertEqual(self.manifest["project_status"], "RESEARCH_PROTOTYPE_COMPLETE")
        self.assertEqual(
            self.manifest["model_status"], "NO_CROSS_DOMAIN_CANDIDATE_SELECTED"
        )
        self.assertFalse(self.manifest["deployment_ready"])
        self.assertIn("not deployment-ready", self.status)

    def test_key_results_cover_internal_and_external_progression(self) -> None:
        self.assertEqual(len(self.results), 7)
        stages = {row["stage"] for row in self.results}
        self.assertTrue(
            {
                "internal_validation",
                "internal_holdout",
                "external_transfer",
                "sealed_confirmation",
                "version_2",
                "version_3",
            }.issubset(stages)
        )

    def test_v2_and_v3_verdicts_match_machine_results(self) -> None:
        rows = {row["stage"]: row for row in self.results}
        v2 = json.loads(
            (ROOT / "artifacts" / "v2_generalisation_development" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        v3 = json.loads(
            (ROOT / "artifacts" / "v3_group_balanced_development" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(rows["version_2"]["verdict"], v2["promotion_verdict"])
        self.assertEqual(rows["version_3"]["verdict"], v3["promotion_verdict"])
        self.assertAlmostEqual(float(rows["version_3"]["mcc"]), 0.025611, places=6)
        self.assertAlmostEqual(
            float(rows["version_3"]["worst_group_balanced_mcc"]),
            -0.162441,
            places=6,
        )

    def test_internal_validation_row_matches_ttl_removed_result(self) -> None:
        internal = next(row for row in self.results if row["stage"] == "internal_validation")
        with (
            ROOT / "artifacts" / "robustness_audit" / "robustness_metrics.csv"
        ).open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        source = next(
            row
            for row in source_rows
            if row["evaluation_scope"] == "current_validation"
            and row["variant"] == "without_all_ttl"
        )
        for column in ("average_precision", "mcc", "recall", "false_positive_rate"):
            self.assertAlmostEqual(
                float(internal[column]), float(source[column]), places=6
            )

    def test_ai_use_and_graded_methodology_boundaries_are_recorded(self) -> None:
        self.assertIn("## AI-use acknowledgement", self.status)
        self.assertIn("graded methodology files were not modified", self.status)


if __name__ == "__main__":
    unittest.main()
