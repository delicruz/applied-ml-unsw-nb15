import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "protocols" / "v2_generalisation_development_protocol.json"


class V2GeneralisationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_is_registered_before_implementation(self) -> None:
        self.assertEqual(
            self.protocol["status"],
            "REGISTERED_BEFORE_V2_IMPLEMENTATION_OR_MODEL_FITTING",
        )
        self.assertEqual(
            self.protocol["repository_parent_commit"],
            "0406151faca65b5218db2cefd887361bb4b38446",
        )

    def test_every_opened_dataset_is_development_only(self) -> None:
        domains = self.protocol["development_domains"]
        self.assertEqual(len(domains), 4)
        self.assertTrue(
            self.protocol["known_at_registration"][
                "all_listed_datasets_have_already_been_opened"
            ]
        )
        self.assertFalse(
            self.protocol["independent_confirmation_boundary"][
                "included_in_this_protocol"
            ]
        )

    def test_shortcut_and_endpoint_features_are_excluded(self) -> None:
        features = self.protocol["features"]
        self.assertIn("sttl", features["excluded_shortcut_features"])
        self.assertIn("dttl", features["excluded_shortcut_features"])
        self.assertEqual(len(features["excluded_endpoint_features"]), 4)
        for prohibited in (
            features["excluded_shortcut_features"]
            + features["excluded_endpoint_features"]
        ):
            self.assertNotIn(prohibited, features["all_common_endpoint_free_features"])

    def test_candidate_set_and_promotion_failure_are_fixed(self) -> None:
        candidate_ids = {
            candidate["id"] for candidate in self.protocol["candidate_experiments"]
        }
        self.assertEqual(
            candidate_ids,
            {
                "pooled_rf_unweighted_all_features",
                "pooled_lr_domain_family_weighted_all_features",
                "pooled_rf_domain_family_weighted_all_features",
                "pooled_rf_domain_family_weighted_reduced_features",
            },
        )
        self.assertIn(
            "no Version 2 candidate selected",
            self.protocol["promotion_rule"]["failure_outcome"],
        )

    def test_locked_evidence_hashes_match(self) -> None:
        for item in self.protocol["locked_inputs"].values():
            path = ROOT / item["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["sha256"], item["path"])


if __name__ == "__main__":
    unittest.main()
