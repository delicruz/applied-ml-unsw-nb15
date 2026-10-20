# Explainable ML for Suspicious Network Traffic

This project builds a reproducible binary classifier for the UNSW-NB15 dataset. It compares a logistic-regression baseline with a Random Forest, evaluates precision, recall, F1 and PR-AUC, examines decision thresholds, and explains the Random Forest globally and locally with feature importance and SHAP.

## Design choices

- The supplied test split remains untouched until final evaluation. The supplied
  training split is divided into development-training and validation partitions.
- The development/validation split is approximately 80/20, stratified by attack
  category, and keeps repeated predictor vectors in one partition only.
- `label` is the binary target: `0` is Normal and `1` is Suspicious.
- `id` and `attack_cat` are excluded from model inputs. `attack_cat` directly reveals whether a row is normal, so using it would leak the answer. It is retained only for error analysis.
- Numerical values use median imputation. Categorical values use most-frequent imputation and one-hot encoding with unknown-category handling.
- Logistic Regression is the interpretable baseline; Random Forest is the main nonlinear model.
- The default operating threshold is 0.50, with a separate table showing alternatives from 0.10 to 0.90.

## Setup

From PowerShell in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data\raw
Expand-Archive -LiteralPath 'C:\Users\Dung\Downloads\OneDrive_2_8-20-2026.zip' -DestinationPath data\raw
```

## Run

Create the development and validation partitions without modifying the raw data:

```powershell
.\.venv\Scripts\python.exe prepare_data_splits.py
```

The generated output CSVs are written to `data/processed/`; split summaries are
written to `artifacts/eda/`. The supplied testing CSV remains in `data/raw/`.

Run the complete analysis:

```powershell
.\.venv\Scripts\python.exe run_project.py
```

Run the development-only overfitting diagnostics without opening the protected
test set:

```powershell
.\.venv\Scripts\python.exe run_overfitting_diagnostics.py
```

This diagnostic stage verifies split integrity, audits individual predictors for
leakage signals, compares training, out-of-bag and validation performance, runs
duplicate-aware cross-validation, removes suspected shortcut features, and
builds a grouped learning curve. Use `--rf-trees` or `--cv-folds` to change the
computational budget while developing.

Tune and regularise the Random Forest after the diagnostic stage:

```powershell
.\.venv\Scripts\python.exe tune_random_forest.py
```

The tuning command screens a bounded structural grid with grouped folds, then
confirms the selected structure using 200 and 500 trees. Average Precision is
the primary selection metric, and explicit tolerance rules prefer a shallower,
more regularised, or smaller model when results are practically tied.

Run the paired statistical comparison and probability-calibration analysis:

```powershell
.\.venv\Scripts\python.exe run_statistical_comparison.py
```

This stage compares Logistic Regression with the selected Random Forest using
paired group-cluster bootstrap intervals for Average Precision, F1 and MCC. It
also runs a conventional row-bootstrap sensitivity analysis, calculates Brier
score, log loss and calibration error, and saves row-level validation
predictions for reproducible error analysis.

Freeze and explain the selected model and generate the complete validation error
analysis:

```powershell
.\.venv\Scripts\python.exe run_explainability_analysis.py
```

This command serialises the selected pipeline, records dataset hashes and model
versions, calculates original-feature permutation importance, aggregates SHAP
importance back to original predictors, produces local explanations, and
analyses attack-specific, high-confidence, and cross-model errors.

For a quick development check:

```powershell
.\.venv\Scripts\python.exe run_project.py --train-limit 10000 --test-limit 5000 --rf-trees 50 --shap-samples 50
```

To omit SHAP or run only one stage:

```powershell
.\.venv\Scripts\python.exe run_project.py --skip-shap
.\.venv\Scripts\python.exe run_project.py --mode audit
.\.venv\Scripts\python.exe run_project.py --mode train
```

## Outputs

Outputs are written to `artifacts/`:

- `audit/dataset_summary.csv`, feature types, and attack-category counts
- EDA, confusion matrix, precision-recall, threshold, feature-importance, and SHAP plots
- `metrics.json` and `model_comparison.csv`
- `threshold_analysis.csv`
- `false_positives.csv`, `false_negatives.csv`, and errors by attack category
- three local explanation cases when available
- `random_forest_pipeline.joblib`, containing preprocessing and model together
- `overfitting_diagnostics/summary.json` and CSV tables for the generalisation
  gap, grouped cross-validation, feature ablation, and learning curve
- `overfitting_diagnostics/plots/` containing the corresponding diagnostic plots
- `random_forest_tuning/selected_configuration.json`, fold-level tuning tables,
  the provisional validation comparison, and selection plots
- `statistical_comparison/summary.json`, paired bootstrap samples and intervals,
  calibration tables, validation predictions, and statistical comparison plots
- `explainability_analysis/analysis_summary.json`, the frozen fitted pipeline,
  permutation and SHAP outputs, detailed error tables, and explanation plots
- `robustness_audit/summary.json`, TTL-removal results, near-duplicate overlap,
  grouped robustness-split metrics, paired bootstrap intervals, and plots
- `final_holdout_evaluation/summary.json`, conservative supplied-test metrics,
  group-bootstrap intervals, per-attack recall, predictions, plots, and the
  frozen no-TTL model
- `distribution_shift_audit/summary.json`, train-to-holdout feature-shift
  statistics, false-positive subgroup tables, SHAP drivers, and diagnostic plots
- `operational_burden/summary.json`, development-only cost-sensitive threshold
  policies, validation policy evaluation, frozen-threshold alert burden, and
  operational trade-off plots
- `subgroup_error_analysis/summary.json`, descriptive Fuzzers miss and Normal
  `state=INT` false-positive profiles, numeric contrasts, sampled SHAP
  contrasts, and diagnostic plots
- `external_netflow_validation/summary.json`, endpoint-free matched-representation
  NF-UNSW-NB15 to NF-ToN-IoT results, exact-feature-group bootstrap intervals,
  per-attack recall, locked models, and generalisation plots
- `external_shift_audit/summary.json`, cross-dataset feature-distribution shift,
  prevalence-adjusted feature/label drift, frozen-model score shift, external
  feature-bin error profiles, and diagnostic plots
- `third_dataset_validation/summary.json`, preregistered NF-BoT-IoT verdict,
  point metrics, exact-feature-group bootstrap intervals and samples, and
  attack-family recall
- `third_dataset_shift_audit/summary.json`, NF-BoT-IoT feature-distribution and
  prevalence-adjusted association shift, frozen-score reversal, feature-bin
  errors, and comparison with the earlier NF-ToN-IoT score audit

The generated CSVs and plots are intended to feed directly into the Dataset, EDA, Results, Explainability, Error Analysis, and Discussion sections of the final report.

Run the TTL shortcut and near-duplicate robustness audit without opening the
supplied test set:

```powershell
.\.venv\Scripts\python.exe run_robustness_audit.py
```

After the model and threshold are frozen, run the final supplied-test evaluation:

```powershell
.\.venv\Scripts\python.exe run_final_holdout_evaluation.py
```

Run the descriptive post-evaluation shift and false-positive audit:

```powershell
.\.venv\Scripts\python.exe run_distribution_shift_audit.py
```

This audit verifies the frozen model hash, reproduces the conservative primary
holdout scope, measures numeric and categorical distribution shift, profiles
false-positive subgroups, and contrasts false positives with true negatives
using SHAP. It does not change the model, features, or decision threshold.

Quantify operational alert burden and explore relative cost scenarios without
using holdout performance for threshold selection:

```powershell
.\.venv\Scripts\python.exe run_operational_burden_analysis.py
```

This stage reports false alerts per 10,000 Normal flows and missed attacks per
10,000 Suspicious flows. Candidate policies are selected only on an internal
near-duplicate-grouped development fold and checked on validation. The supplied
test predictions are used only to report the already-frozen threshold of 0.54;
the script does not retune or replace the final model.

Describe the two priority error groups without creating test-derived rules:

```powershell
.\.venv\Scripts\python.exe run_subgroup_error_analysis.py
```

This post-evaluation stage verifies the frozen model hash, reproduces the
conservative holdout, and compares missed versus detected Fuzzers attacks and
false-positive versus true-negative Normal `state=INT` records. Its score,
categorical, numeric, and SHAP contrasts are explanatory diagnostics only; they
do not change the model, features, or threshold.

Run the cross-dataset NetFlow-v1 stress test after downloading the official
open-access NF-UNSW-NB15 and NF-ToN-IoT archives from the University of
Queensland and extracting both CSV files anywhere below `data/external/`:

- [UQ NetFlow dataset catalogue](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
- [NF-UNSW-NB15 NetFlow-v1 file](https://rdm.uq.edu.au/files/c31a9f50-ef99-11ed-ab7b-c7846b13c8a9)
- [NF-ToN-IoT NetFlow-v1 file](https://rdm.uq.edu.au/files/2ad93cd0-ef9c-11ed-827d-e762de186848)

```powershell
.\.venv\Scripts\python.exe run_external_netflow_validation.py
```

The runner excludes source/destination IP addresses and ports, keeps exact
duplicates of the eight remaining shared traffic features in one source fold,
uses separate source folds for training, threshold calibration, and validation,
and never tunes on NF-ToN-IoT. The experiment tests transfer of a common
eight-feature modelling approach; it is not a direct external evaluation of
the frozen 44-feature UNSW-NB15 model. Large external CSVs under
`data/external/` are intentionally ignored by Git.

Explain the completed cross-dataset failure without changing either frozen
model or its threshold:

```powershell
.\.venv\Scripts\python.exe run_external_shift_audit.py
```

This audit verifies the full-run dataset hashes and fitted model artifacts,
reconstructs the source-validation fold, and compares source with external
feature distributions, prevalence-adjusted feature/label relationships, model
scores, and error profiles. Its outputs are descriptive and must not be used to
retune against NF-ToN-IoT.

The third external evaluation was sealed before data access in
`protocols/third_dataset_preregistration.json`. After that protocol was
committed, the official NF-BoT-IoT NetFlow-v1 archive was downloaded and its
CSV was extracted below the ignored `data/external/` directory. Reproduce the
frozen evaluation with:

```powershell
.\.venv\Scripts\python.exe run_third_dataset_validation.py `
  --dataset-path data\external\nf-bot-iot\NF-BoT-IoT.csv
```

The runner has no training or threshold-selection path. It verifies official
row and attack counts plus both frozen model hashes before scoring, then applies
the preregistered group-bootstrap decision rule. The completed run matched all
official counts and evaluated 600,100 rows. Random Forest recall was 0.00228,
balanced accuracy was 0.49811, and MCC was -0.01170 with a 95% exact-feature-
group interval of [-0.02348, 0.00029]. Its Average Precision of 0.94531 was
below the 0.97691 attack-prevalence baseline. It failed all three frozen
requirements, producing the preregistered verdict `useful transfer not
supported`. See
`deliverables/current_results_package/Third_Dataset_Result.md` for the concise
experiment record.

Explain the completed NF-BoT-IoT transfer failure without changing either
frozen model or its threshold:

```powershell
.\.venv\Scripts\python.exe run_third_dataset_shift_audit.py
```

The audit verifies the preregistration, dataset and model hashes, reconstructs
the original source-validation fold, and applies the source-derived bins to
NF-BoT-IoT. It records extreme numeric shift, prevalence-adjusted feature/label
drift, and a frozen-score ordering reversal for both models. It also creates a
single comparison of source, NF-ToN-IoT, and NF-BoT-IoT score ordering. The
analysis is descriptive and does not authorise target-data tuning.
