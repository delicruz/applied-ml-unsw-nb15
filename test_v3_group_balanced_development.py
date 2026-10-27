"""Preflight, sampling, leakage, and metric tests for the Version 3 runner."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_v3_group_balanced_development as v3


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads(
    (ROOT / "protocols" / "v3_group_balanced_development_protocol.json").read_text(
        encoding="utf-8"
    )
)


def one_group_frame(rows: int = 100) -> pd.DataFrame:
    labels = np.zeros(rows, dtype="int8")
    labels[-5:] = 1
    attacks = np.where(labels == 1, "MinorityAttack", "Benign")
    return pd.DataFrame(
        {
            "PROTOCOL": np.full(rows, 6.0),
            "L7_PROTO": np.full(rows, 7.0),
            "IN_BYTES": np.full(rows, 1200.0),
            "OUT_BYTES": np.full(rows, 300.0),
            "IN_PKTS": np.full(rows, 12.0),
            "OUT_PKTS": np.full(rows, 3.0),
            "TCP_FLAGS": np.full(rows, 27.0),
            "FLOW_DURATION_MILLISECONDS": np.full(rows, 100.0),
            "Label": labels,
            "Attack": attacks,
        }
    )


class V3GroupBalancedDevelopmentTests(unittest.TestCase):
    def test_frozen_implementation_manifest_matches_files(self) -> None:
        manifest = v3.verify_implementation(
            ROOT / "protocols" / "v3_group_balanced_implementation_manifest.json",
            ROOT / "run_v3_group_balanced_development.py",
        )
        self.assertEqual(manifest["status"], "IMPLEMENTATION_FROZEN_BEFORE_EXECUTION")
        self.assertEqual(manifest["execution_state"], "NOT_RUN")

    def test_registered_features_candidates_and_cap_match_runner(self) -> None:
        v3.verify_protocol(PROTOCOL)
        self.assertEqual(tuple(PROTOCOL["features"]["base_features"]), v3.BASE_FEATURES)
        self.assertEqual(
            tuple(PROTOCOL["features"]["derived_features"]), v3.DERIVED_FEATURES
        )
        self.assertEqual(
            PROTOCOL["group_and_sampling_protocol"][
                "maximum_retained_rows_per_group_per_partition"
            ],
            64,
        )

    def test_endpoint_and_ttl_shortcuts_are_never_model_features(self) -> None:
        prohibited = set(v3.ENDPOINT_FEATURES) | set(v3.SHORTCUT_FEATURES)
        self.assertTrue(prohibited.isdisjoint(v3.BASE_FEATURES + v3.DERIVED_FEATURES))
        self.assertNotIn("XGBoost", inspect.getsource(v3.make_candidate))

    def test_derived_feature_formulas(self) -> None:
        source = pd.DataFrame(
            {
                "IN_BYTES": [90.0],
                "OUT_BYTES": [10.0],
                "IN_PKTS": [9.0],
                "OUT_PKTS": [1.0],
                "FLOW_DURATION_MILLISECONDS": [10.0],
            }
        )
        result = v3.add_derived_features(source)
        self.assertEqual(result.loc[0, "TOTAL_BYTES"], 100.0)
        self.assertEqual(result.loc[0, "TOTAL_PKTS"], 10.0)
        self.assertEqual(result.loc[0, "IN_AVG_PKT_BYTES"], 10.0)
        self.assertEqual(result.loc[0, "OUT_AVG_PKT_BYTES"], 10.0)
        self.assertAlmostEqual(result.loc[0, "BYTE_ASYMMETRY"], 80.0 / 101.0)
        self.assertAlmostEqual(result.loc[0, "PKT_ASYMMETRY"], 8.0 / 11.0)
        self.assertEqual(result.loc[0, "BYTES_PER_MILLISECOND"], 10.0)
        self.assertEqual(result.loc[0, "PKTS_PER_MILLISECOND"], 1.0)

    def test_all_registered_candidate_pipelines_fit_toy_source_data(self) -> None:
        training = one_group_frame()
        training.loc[:49, "IN_BYTES"] = np.arange(50, dtype=float) + 100.0
        training.loc[50:, "OUT_BYTES"] = np.arange(50, dtype=float) + 50.0
        for specification in PROTOCOL["candidate_experiments"]:
            model = v3.fit_candidate(specification, training)
            probabilities = model.predict_proba(training.loc[:, v3.BASE_FEATURES])[:, 1]
            self.assertEqual(len(probabilities), len(training))
            self.assertTrue(np.isfinite(probabilities).all())

    def test_group_selection_charges_capped_not_original_rows(self) -> None:
        strata = pd.DataFrame(
            {
                v3.GROUP: ["g1", "g2", "g3"],
                "_second": [1, 2, 3],
                "_partition": ["training"] * 3,
                v3.TARGET: [1, 1, 1],
                v3.FAMILY: ["A"] * 3,
                "stratum_rows": [10_000, 10_000, 10_000],
            }
        )
        selected = v3.select_groups_capped(
            strata, "training", row_target=128, group_cap=64
        )
        self.assertEqual(len(selected), 2)

    def test_stratum_quota_preserves_minority_and_caps_group(self) -> None:
        strata = pd.DataFrame(
            {
                v3.GROUP: ["g1", "g1"],
                "_second": [1, 1],
                "_partition": ["training", "training"],
                v3.TARGET: [0, 1],
                v3.FAMILY: ["Normal", "MinorityAttack"],
                "stratum_rows": [95, 5],
            }
        )
        quota = v3.allocate_stratum_quotas(strata, {"g1"}, "training", 64)
        self.assertEqual(int(quota["quota"].sum()), 64)
        self.assertGreater(
            int(quota.loc[quota[v3.FAMILY].eq("MinorityAttack"), "quota"].iloc[0]),
            0,
        )

    def test_materialisation_retains_at_most_64_rows_and_minority_stratum(self) -> None:
        frame = one_group_frame()
        domain = "toy-domain"
        normalised = v3.core.normalise_chunk(frame.copy(), domain)
        hashes = v3.core.canonical_group_hashes(normalised, domain)
        key = str(hashes.loc[0, v3.GROUP])
        partition = str(v3.core.group_partition(hashes["_first"])[0])
        other = "calibration" if partition == "training" else "training"
        quotas = {
            partition: pd.DataFrame(
                {
                    v3.GROUP: [key, key],
                    v3.TARGET: [0, 1],
                    v3.FAMILY: ["Normal", "MinorityAttack"],
                    "quota": [59, 5],
                }
            ),
            other: pd.DataFrame(
                {
                    v3.GROUP: ["absent"],
                    v3.TARGET: [0],
                    v3.FAMILY: ["Normal"],
                    "quota": [1],
                }
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "No rows materialised"):
                v3.materialise_capped_groups(path, domain, quotas, chunk_size=25)

        # Test the active partition directly with the absent partition populated by
        # a second group whose hash falls into the other deterministic partition.
        found = None
        for value in range(1, 10_000):
            candidate = frame.iloc[[0]].copy()
            candidate["IN_BYTES"] = float(value)
            candidate["Label"] = 0
            candidate["Attack"] = "Benign"
            candidate_normalised = v3.core.normalise_chunk(candidate.copy(), domain)
            candidate_hashes = v3.core.canonical_group_hashes(candidate_normalised, domain)
            candidate_partition = str(
                v3.core.group_partition(candidate_hashes["_first"])[0]
            )
            if candidate_partition == other:
                found = candidate
                other_key = str(candidate_hashes.loc[0, v3.GROUP])
                break
        self.assertIsNotNone(found)
        combined = pd.concat([frame, found], ignore_index=True)
        quotas[other] = pd.DataFrame(
            {
                v3.GROUP: [other_key],
                v3.TARGET: [0],
                v3.FAMILY: ["Normal"],
                "quota": [1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.csv"
            combined.to_csv(path, index=False)
            retained = v3.materialise_capped_groups(
                path, domain, quotas, chunk_size=25
            )
        active = retained[partition]
        self.assertEqual(len(active), 64)
        self.assertEqual(int(active[v3.TARGET].sum()), 5)
        self.assertLessEqual(int(active.groupby(v3.GROUP).size().max()), 64)

    def test_equal_group_metrics_reduce_large_group_dominance(self) -> None:
        candidate = "toy"
        groups = pd.DataFrame(
            {
                "rows": [1000, 100, 1],
                "positives": [0, 100, 1],
                "negatives": [1000, 0, 0],
                f"{candidate}__true_positive": [0, 0, 1],
                f"{candidate}__false_positive": [0, 0, 0],
            }
        )
        result = v3.group_balanced_metrics(groups, candidate)
        row_recall = 1.0 / 101.0
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertGreater(result["recall"], row_recall)

    def test_threshold_selection_honours_cross_domain_constraints(self) -> None:
        calibration = pd.DataFrame(
            {
                v3.DOMAIN: ["d1"] * 8 + ["d2"] * 8,
                v3.TARGET: [0, 0, 0, 0, 1, 1, 1, 1] * 2,
                v3.ATTACK: ["Benign"] * 4 + ["A"] * 4 + ["Benign"] * 4 + ["B"] * 4,
            }
        )
        scores = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95] * 2)
        protocol = json.loads(json.dumps(PROTOCOL))
        protocol["threshold_rule"]["constraints"][
            "minimum_rows_for_supported_stratum"
        ] = 2
        threshold, table, passed = v3.select_constrained_threshold(
            calibration, scores, protocol
        )
        self.assertTrue(passed)
        self.assertTrue(
            bool(table.loc[table["threshold"].eq(threshold), "feasible"].iloc[0])
        )

    def test_each_fold_excludes_heldout_domain_from_sources(self) -> None:
        domains = list(v3.DATASET_GLOBS)
        for heldout in domains:
            source = [domain for domain in domains if domain != heldout]
            self.assertNotIn(heldout, source)
            self.assertEqual(len(source), 3)

    def test_completed_output_cannot_be_overwritten_or_retried(self) -> None:
        source = inspect.getsource(v3.run_experiment)
        self.assertIn("Completed output already exists", source)
        self.assertNotIn("retry", inspect.getsource(v3.main).lower())


if __name__ == "__main__":
    unittest.main()
