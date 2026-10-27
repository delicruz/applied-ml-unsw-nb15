import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "v3_group_balanced_development_protocol.json"


class V3GroupBalancedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_is_registered_before_implementation(self) -> None:
        self.assertEqual(
            self.protocol["status"],
            "REGISTERED_BEFORE_V3_IMPLEMENTATION_OR_MODEL_FITTING",
        )
        self.assertEqual(
            self.protocol["repository_parent_commit"],
            "c4fb98d7ad2425bb0aea543b47d423d0e8a7efdd",
        )

    def test_all_opened_domains_remain_development_only(self) -> None:
        self.assertEqual(len(self.protocol["development_domains"]), 4)
        self.assertTrue(
            self.protocol["development_status"]["all_domains_already_opened"]
        )
        self.assertFalse(
            self.protocol["independent_confirmation_boundary"][
                "included_in_this_protocol"
            ]
        )

    def test_group_cap_and_full_evaluation_are_fixed(self) -> None:
        sampling = self.protocol["group_and_sampling_protocol"]
        self.assertEqual(sampling["maximum_retained_rows_per_group_per_partition"], 64)
        self.assertIn("every row", sampling["held_out_domain_evaluation"])
        self.assertIn("never the original complete-group row count", sampling["group_selection"])

    def test_candidate_set_is_narrow_and_fixed(self) -> None:
        candidate_ids = {
            candidate["id"] for candidate in self.protocol["candidate_experiments"]
        }
        self.assertEqual(
            candidate_ids,
            {
                "group_capped_lr_base_features",
                "group_capped_lr_derived_features",
                "group_capped_rf_derived_features",
            },
        )
        self.assertEqual(
            self.protocol["promotion_rule"]["reference"],
            "group_capped_lr_base_features",
        )
        self.assertIn(
            "no Version 3 candidate selected",
            self.protocol["promotion_rule"]["failure_outcome"],
        )

    def test_derived_features_use_only_registered_base_features(self) -> None:
        features = self.protocol["features"]
        self.assertEqual(len(features["base_features"]), 8)
        self.assertEqual(len(features["derived_features"]), 8)
        self.assertIn("BYTES_PER_MILLISECOND", features["derived_features"])
        self.assertIn("BYTE_ASYMMETRY", features["derived_features"])

    def test_shortcut_and_endpoint_features_are_excluded(self) -> None:
        features = self.protocol["features"]
        self.assertIn("sttl", features["excluded_shortcut_features"])
        self.assertIn("dttl", features["excluded_shortcut_features"])
        for prohibited in (
            features["excluded_shortcut_features"]
            + features["excluded_endpoint_features"]
        ):
            self.assertNotIn(prohibited, features["base_features"])

    def test_group_balanced_metric_is_a_promotion_gate(self) -> None:
        evaluation = self.protocol["evaluation_metrics"]
        self.assertIn("equal total weight", evaluation["group_balanced_definition"])
        self.assertTrue(
            any(
                "group-balanced held-out MCC" in requirement
                for requirement in self.protocol["promotion_rule"][
                    "eligibility_requirements"
                ]
            )
        )

    def test_locked_v2_evidence_hashes_match(self) -> None:
        for item in self.protocol["locked_inputs"].values():
            path = ROOT / item["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"], item["path"])


if __name__ == "__main__":
    unittest.main()
