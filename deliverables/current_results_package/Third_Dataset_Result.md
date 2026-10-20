# Third-Dataset Validation Result

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-20
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: nf-bot-iot-v1-third-holdout-2026-10-20
- **Type**: analysis
- **Status**: completed
- **Command**: `.\.venv\Scripts\python.exe run_third_dataset_validation.py --dataset-path <ignored local NF-BoT-IoT.csv>`
- **Working Directory**: repository root
- **Duration**: 16.7 seconds
- **Exit Code**: 0

### Integrity checks

- The preregistration was committed before the archive was downloaded or row-level data was opened.
- The official CSV matched the preregistered 600,100 rows, comprising 586,241 attacks and 13,859 benign records.
- The evaluated data contained 39,901 exact groups across the eight frozen features.
- The CSV SHA-256 was `0dcaeb12918dccee4127e537863366958a2fdc222fda06e0b1b0458c31a429ff`.
- The downloaded archive SHA-256 was `091b10ae174eed345e5e8695273948dd85304266dbc1cea4c119b3e20bf0808c`.
- The Logistic Regression and Random Forest artifact hashes matched the preregistration.
- No model fitting, feature selection, threshold selection, or external tuning occurred.

### Primary result

The preregistered verdict was **useful transfer not supported**. The primary
Random Forest failed all three required conditions:

| Requirement | Observed result | Passed |
|---|---:|:---:|
| MCC 95% group-bootstrap lower bound > 0 | -0.02348 | No |
| Point balanced accuracy > 0.50 | 0.49811 | No |
| Point Average Precision lift over prevalence > 1.00 | 0.96766 | No |

| Model | AP | AP lift | ROC AUC | MCC | Balanced accuracy | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.97258 | 0.99558 | 0.49746 | -0.03627 | 0.46624 | 0.91292 | 0.01955 |
| Random Forest | 0.94531 | 0.96766 | 0.09381 | -0.01170 | 0.49811 | 0.00228 | 0.99394 |

The dataset attack prevalence was 0.97691. Therefore, both apparently high AP
values were below the no-skill prevalence baseline. At its frozen threshold of
0.64, the Random Forest detected 1,335 of 586,241 attacks and missed 584,906.
Its group-bootstrap MCC interval was [-0.02348, 0.00029].

### Attack-family result

| Attack family | Rows | Random Forest recall |
|---|---:|---:|
| DDoS | 56,844 | 0.00255 |
| DoS | 56,833 | 0.00255 |
| Reconnaissance | 470,655 | 0.00216 |
| Theft | 1,909 | 0.01467 |

### Interpretation boundary

This result is evidence that the frozen eight-feature NetFlow-v1 model does not
transfer usefully to NF-BoT-IoT. Together with the earlier NF-ToN-IoT failure,
it strengthens the diagnosis of dataset-specific fitting and distribution
shift. It does not isolate a single causal mechanism, validate the separate
44-feature UNSW-NB15 model, or establish deployment readiness.

### Post-evaluation shift audit

The frozen-model audit re-verified the source and third-dataset hashes, the
protocol hash, both fitted-model hashes, and the thresholds. No training or
selection occurred. Its strongest descriptive findings were:

- `FLOW_DURATION_MILLISECONDS` PSI 11.402, with medians 7 in the source and
  4,294,966 in NF-BoT-IoT;
- `IN_PKTS` PSI 9.062 and `OUT_PKTS` PSI 6.541;
- `L7_PROTO` Jensen-Shannon divergence 0.352 bits;
- prevalence-adjusted association drift of 2.269 for `TCP_FLAGS`, 1.761 for
  `IN_BYTES`, and 1.067 for `IN_PKTS`; and
- Random Forest mean attack-minus-benign score margin changing from +0.847 on
  source validation to -0.193 on NF-BoT-IoT.

Both models ranked benign traffic above attacks on average on NF-BoT-IoT, as
they had on NF-ToN-IoT. This repeated score-ordering reversal shows that the
failure is not repairable by moving the decision threshold alone.

![Cross-dataset frozen score ordering](../../artifacts/third_dataset_shift_audit/plots/cross_external_score_ordering.png)

### Cross-external error-pattern synthesis

The saved NF-ToN-IoT and NF-BoT-IoT feature-bin profiles were aligned without
re-running or retuning either model. With at least 100 class examples per
dataset as the support rule, Random Forest had 32 feature-bin rows with recall
at or below 20% in both external datasets and no recurring high-false-positive
rows. Logistic Regression had the complementary failure: no recurring
low-recall rows but 25 feature-bin rows with a false-positive rate at or above
20% in both datasets.

For Random Forest, recurring low-recall bins covered at least 94% of attacks in
both datasets for `IN_BYTES`, `OUT_PKTS`, `PROTOCOL`, and `L7_PROTO`. For
Logistic Regression, recurring high-false-positive bins covered at least 94%
of benign rows in both datasets for `IN_BYTES`, `OUT_BYTES`, `PROTOCOL`, and
`TCP_FLAGS`. Shares are feature-specific and overlap, so they are not additive.
The result describes a broad, model-specific failure pattern rather than a
single causal feature.

![Cross-external recurring feature-bin failures](../../artifacts/cross_external_error_synthesis/plots/consistent_feature_bin_failures.png)

### Output files

| File | Purpose |
|---|---|
| `artifacts/third_dataset_validation/summary.json` | Protocol, hashes, counts, and preregistered verdict |
| `artifacts/third_dataset_validation/metrics_long.csv` | Point metrics for both frozen models |
| `artifacts/third_dataset_validation/group_bootstrap_intervals.csv` | Exact-feature-group 95% intervals |
| `artifacts/third_dataset_validation/group_bootstrap_samples.csv` | 200 bootstrap replicates |
| `artifacts/third_dataset_validation/per_attack_recall.csv` | Attack-family recall |
| `artifacts/third_dataset_shift_audit/summary.json` | Verified descriptive shift-audit summary |
| `artifacts/third_dataset_shift_audit/cross_external_score_comparison.csv` | Source and two-external-dataset score ordering |
| `artifacts/cross_external_error_synthesis/summary.json` | Aligned recurring error-bin rules and counts |
| `artifacts/cross_external_error_synthesis/feature_consistency_summary.csv` | Per-feature coverage of recurring errors in both datasets |

### Anomalies detected

The first invocation used the system Python and stopped at import time because
`joblib` was absent. It did not open the dataset or generate predictions. After
explicit user approval, the same runner was executed with the repository's
existing virtual environment and completed successfully.
