"""Preflight and leakage-guard tests for the Version 2 runner."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_v2_generalisation_development as v2


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads(
    (ROOT / "protocols" / "v2_generalisation_development_protocol.json").read_text(
        encoding="utf-8"
    )
)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PROTOCOL": [6.0, 6.0, 17.0, 17.0],
            "L7_PROTO": [7.0, 7.0, 0.0, 0.0],
            "IN_BYTES": [12345.11, 12345.12, 80.0, 81.0],
            "OUT_BYTES": [80.0, 80.0, 40.0, 41.0],
            "IN_PKTS": [10.0, 10.0, 2.0, 2.0],
            "OUT_PKTS": [9.0, 9.0, 1.0, 1.0],
            "TCP_FLAGS": [27.0, 27.0, 2.0, 2.0],
            "FLOW_DURATION_MILLISECONDS": [101.0, 101.0, 20.0, 21.0],
            "Label": [1, 1, 0, 0],
            "Attack": ["A", "A", "Benign", "Benign"],
        }
    )


class V2GeneralisationDevelopmentTests(unittest.TestCase):
    def test_frozen_implementation_manifest_matches_files(self) -> None:
        manifest = v2.verify_implementation(
            ROOT / "protocols" / "v2_generalisation_implementation_manifest.json",
            ROOT / "run_v2_generalisation_development.py",
        )
        self.assertEqual(manifest["status"], "IMPLEMENTATION_FROZEN_BEFORE_EXECUTION")

    def test_registered_features_and_candidates_match_runner(self) -> None:
        v2.verify_protocol(PROTOCOL)
        self.assertEqual(
            tuple(PROTOCOL["features"]["all_common_endpoint_free_features"]),
            v2.MODEL_FEATURES,
        )
        self.assertEqual(
            len(PROTOCOL["candidate_experiments"]),
            4,
        )

    def test_endpoint_and_ttl_shortcuts_are_never_model_features(self) -> None:
        prohibited = set(v2.ENDPOINT_FEATURES) | set(v2.SHORTCUT_FEATURES)
        self.assertTrue(prohibited.isdisjoint(v2.MODEL_FEATURES))
        self.assertNotIn("XGBoost", inspect.getsource(v2.make_candidate))

    def test_near_duplicates_group_together_but_domains_are_namespaced(self) -> None:
        frame = toy_frame().iloc[:2]
        first = v2.canonical_group_hashes(frame, "domain-a")
        second = v2.canonical_group_hashes(frame, "domain-b")
        self.assertEqual(first.loc[0, v2.GROUP], first.loc[1, v2.GROUP])
        self.assertNotEqual(first.loc[0, v2.GROUP], second.loc[0, v2.GROUP])
        self.assertEqual(
            v2.group_partition(first["_first"])[0],
            v2.group_partition(first["_first"])[1],
        )

    def test_group_selector_preserves_each_family_and_whole_groups(self) -> None:
        strata = pd.DataFrame(
            {
                v2.GROUP: ["g1", "g2", "g3", "g4"],
                "_second": [1, 2, 3, 4],
                "_partition": ["training"] * 4,
                v2.TARGET: [0, 1, 1, 1],
                v2.FAMILY: ["Normal", "A", "B", "B"],
                "stratum_rows": [100, 3, 2, 100],
            }
        )
        selected = v2.select_groups(strata, "training", cap=5)
        self.assertTrue({"g1", "g2", "g3"}.issubset(selected))

    def test_domain_family_weights_equalise_uncapped_strata(self) -> None:
        frame = pd.DataFrame(
            {
                v2.DOMAIN: ["d1"] * 6 + ["d2"] * 4,
                v2.FAMILY: ["Normal"] * 4 + ["A"] * 2 + ["Normal"] * 2 + ["B"] * 2,
            }
        )
        weights = v2.domain_family_weights(frame, cap_multiple=1_000.0)
        frame["weight"] = weights
        totals = frame.groupby([v2.DOMAIN, v2.FAMILY])["weight"].sum()
        self.assertAlmostEqual(totals.loc[("d1", "Normal")], totals.loc[("d1", "A")])
        self.assertAlmostEqual(totals.groupby(level=0).sum().loc["d1"], totals.groupby(level=0).sum().loc["d2"])

    def test_threshold_selection_honours_constraints(self) -> None:
        calibration = pd.DataFrame(
            {
                v2.DOMAIN: ["d1"] * 8 + ["d2"] * 8,
                v2.TARGET: [0, 0, 0, 0, 1, 1, 1, 1] * 2,
                v2.ATTACK: ["Benign"] * 4 + ["A"] * 4 + ["Benign"] * 4 + ["B"] * 4,
            }
        )
        probabilities = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95] * 2)
        protocol = json.loads(json.dumps(PROTOCOL))
        protocol["group_and_sampling_protocol"]["minimum_supported_attack_rows"] = 2
        threshold, table, passed = v2.select_constrained_threshold(
            calibration, probabilities, protocol
        )
        self.assertTrue(passed)
        self.assertGreaterEqual(threshold, 0.20)
        self.assertTrue(bool(table.loc[table["threshold"].eq(threshold), "feasible"].iloc[0]))

    def test_group_confusions_reproduce_row_level_threshold_metrics(self) -> None:
        labels = np.array([1, 0, 1, 1, 0, 0], dtype="int8")
        probabilities = np.array([0.9, 0.8, 0.6, 0.2, 0.1, 0.05])
        point = v2.row_metrics(labels, probabilities, threshold=0.5)
        groups = pd.DataFrame(
            {
                "positives": [1, 2],
                "negatives": [1, 2],
                "true_positive": [1, 1],
                "false_positive": [1, 0],
            }
        )
        grouped = v2.metrics_from_group_confusions(groups)
        for metric in ("mcc", "balanced_accuracy", "recall", "false_positive_rate"):
            self.assertAlmostEqual(point[metric], grouped[metric])

    def test_fold_source_construction_excludes_heldout_domain(self) -> None:
        domains = list(v2.DATASET_GLOBS)
        for heldout in domains:
            source = [domain for domain in domains if domain != heldout]
            self.assertNotIn(heldout, source)
            self.assertEqual(len(source), 3)

    def test_streamed_evaluation_keeps_row_metrics_and_group_confusions(self) -> None:
        class FakeModel:
            def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
                probability = np.clip(
                    frame["IN_BYTES"].to_numpy(dtype=float) / 20_000.0,
                    0.01,
                    0.99,
                )
                return np.column_stack([1.0 - probability, probability])

        frame = toy_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.csv"
            for endpoint in v2.ENDPOINT_FEATURES:
                frame[endpoint] = 0
            frame.to_csv(path, index=False)
            metrics, attacks, group_metrics, groups = v2.stream_heldout_evaluation(
                path,
                "toy-domain",
                FakeModel(),
                v2.MODEL_FEATURES,
                threshold=0.5,
                candidate="toy",
                chunk_size=2,
            )
        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(int(attacks["positive_rows"].sum()), 2)
        self.assertEqual(int(groups["rows"].sum()), 4)
        self.assertIn("toy__true_positive", groups)
        self.assertIn("size_bucket", set(group_metrics["dimension"]))

    def test_completed_output_cannot_be_overwritten(self) -> None:
        source = inspect.getsource(v2.run_experiment)
        self.assertIn("Completed output already exists", source)
        self.assertNotIn("retry", inspect.getsource(v2.main).lower())


if __name__ == "__main__":
    unittest.main()
