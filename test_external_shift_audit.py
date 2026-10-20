"""Focused integrity tests for the external shift audit."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_external_netflow_validation import make_models
from run_external_shift_audit import binned_feature, population_stability_index


class ExternalShiftAuditTests(unittest.TestCase):
    def test_identical_distribution_has_zero_psi(self) -> None:
        values = pd.Series(np.arange(1, 101, dtype=float))
        psi, bins = population_stability_index(values, values.copy())
        self.assertGreaterEqual(bins, 2)
        self.assertAlmostEqual(psi, 0.0, places=12)

    def test_shifted_distribution_has_positive_psi(self) -> None:
        source = pd.Series(np.arange(1, 1001, dtype=float))
        external = pd.Series(np.arange(1001, 2001, dtype=float))
        psi, _ = population_stability_index(source, external)
        self.assertGreater(psi, 1.0)

    def test_external_numeric_bins_use_source_edges(self) -> None:
        source = pd.Series(np.arange(100, dtype=float))
        external = pd.Series([-100.0, 10.0, 500.0])
        source_bins, external_bins = binned_feature(source, external, categorical=False)
        self.assertEqual(source_bins.nunique(), 10)
        self.assertEqual(len(external_bins), 3)
        self.assertFalse(external_bins.isna().any())

    def test_new_pipelines_use_portable_numpy_transform(self) -> None:
        pipeline = make_models(20)["logistic_regression"]
        numeric_pipeline = pipeline.named_steps["preprocess"].transformers[1][1]
        self.assertIs(numeric_pipeline.named_steps["log1p"].func, np.log1p)


if __name__ == "__main__":
    unittest.main()
