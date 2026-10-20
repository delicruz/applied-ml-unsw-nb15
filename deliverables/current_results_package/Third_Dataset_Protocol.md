# Third-Dataset Preregistered Validation Protocol

## Material Passport

- Material ID: `nf-bot-iot-v1-third-holdout-2026-10-20`
- Type: code experiment plan
- Status: `PREREGISTERED_BEFORE_DATA_ACCESS`
- Registration date: 20 October 2026, Australia/Adelaide
- Repository state before registration: `fc730109314ff0ad03bb1e49a327c4e403dc9763`
- Machine-readable protocol: `../../protocols/third_dataset_preregistration.json`
- Data state: official aggregate metadata viewed; archive not downloaded;
  row-level data, labels, and predictions not opened

## Purpose

NF-ToN-IoT has already been examined and is now diagnostic development data.
It cannot provide an unbiased assessment of any future feature-selection,
calibration, or adaptation strategy. NF-BoT-IoT NetFlow-v1 is therefore
reserved as the third untouched external holdout.

The official University of Queensland catalogue reports 600,100 flows:
586,241 attacks and 13,859 benign records. Its NetFlow-v1 representation is
compatible with the already-fitted endpoint-free models.

## Frozen evaluation

No training or tuning will occur. The two existing models will be loaded by
their exact hashes:

| Model | Role | Frozen threshold | SHA-256 |
|---|---|---:|---|
| Logistic Regression | Baseline | 0.69 | `65200507f2fdf43703026d8d8fcd0e577cad670b459247fd6dc944bd50acd746` |
| Random Forest | Primary | 0.64 | `38756c9b09be364b5ff483df82b98183634753ef326faafeaaa8ebcfd498646b` |

Both models use only `PROTOCOL`, `L7_PROTO`, `IN_BYTES`, `OUT_BYTES`,
`IN_PKTS`, `OUT_PKTS`, `TCP_FLAGS`, and `FLOW_DURATION_MILLISECONDS`.
Source/destination IP addresses and ports remain excluded.

## Outcomes and uncertainty

The primary endpoint is Random Forest Matthews Correlation Coefficient. The
secondary outcomes are Average Precision, Average Precision lift over the
dataset prevalence, ROC AUC, F1, precision, recall, specificity, balanced
accuracy, Brier score, and recall for each attack family.

Two hundred 95% bootstrap intervals will resample exact duplicate groups formed
from all eight model features. This preserves the dependence definition used in
the earlier external analysis.

Useful transfer is supported only if all three preregistered requirements hold:

1. the Random Forest MCC group-bootstrap lower bound is greater than zero;
2. Random Forest balanced accuracy is greater than 0.50; and
3. Random Forest Average Precision lift over prevalence is greater than 1.00.

Failure to satisfy all three will be reported as **useful transfer not
supported**. Neither outcome permits a deployment-readiness claim.

## Integrity rules

Before scoring, the runner may verify the archive/CSV hash, schema, numeric
parsability, official row and label counts, and frozen model hashes. Any
mismatch must stop the run. It may not change a threshold, select features,
refit a model, modify the decision rule, or exclude inconvenient rows after the
dataset is opened.

Any amendment must be committed and pushed before the archive is downloaded or
row-level data is viewed. Once NF-BoT-IoT has been scored, it must be described
as evaluated data and cannot later be called untouched.

## Interpretation boundary

This experiment evaluates direct transfer of the already-frozen eight-feature
NetFlow-v1 models. It is not an external validation of the separate 44-feature
UNSW-NB15 pipeline, and it cannot establish operational or deployment readiness.
