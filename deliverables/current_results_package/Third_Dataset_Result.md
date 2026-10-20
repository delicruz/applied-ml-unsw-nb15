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

### Output files

| File | Purpose |
|---|---|
| `artifacts/third_dataset_validation/summary.json` | Protocol, hashes, counts, and preregistered verdict |
| `artifacts/third_dataset_validation/metrics_long.csv` | Point metrics for both frozen models |
| `artifacts/third_dataset_validation/group_bootstrap_intervals.csv` | Exact-feature-group 95% intervals |
| `artifacts/third_dataset_validation/group_bootstrap_samples.csv` | 200 bootstrap replicates |
| `artifacts/third_dataset_validation/per_attack_recall.csv` | Attack-family recall |

### Anomalies detected

The first invocation used the system Python and stopped at import time because
`joblib` was absent. It did not open the dataset or generate predictions. After
explicit user approval, the same runner was executed with the repository's
existing virtual environment and completed successfully.
