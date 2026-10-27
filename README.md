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
- `cross_external_error_synthesis/summary.json`, aligned recurring feature-bin
  failures across NF-ToN-IoT and NF-BoT-IoT, with coverage tables and a plot
- `fresh_confirmation_evaluation/summary.json`, the one-time sealed-confirmation
  verdict, frozen-model metrics, paired group-bootstrap intervals, attack-label
  recall, and plots
- `fresh_generalisation_failure_audit/summary.json`, post-hoc validation-to-
  confirmation attack-mix, repeated-group, feature, label-association and
  frozen-score diagnostics

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

Align the two completed external error audits and identify recurring supported
feature-bin failures without fitting or threshold selection:

```powershell
.\.venv\Scripts\python.exe run_cross_external_error_synthesis.py
```

The synthesis flags a bin only when it has at least 100 relevant class rows in
each external dataset. It reports Random Forest's recurring low-recall bins and
Logistic Regression's recurring high-false-positive bins. These are post-hoc
hypotheses from already-opened external data, not validated correction rules.

The next improvement study uses the previously unused NF-CSE-CIC-IDS2018
NetFlow-v1 dataset. Its split, candidate set, development success rule, and
sealed 20% confirmation partition were registered before download or row-level
access in `protocols/fresh_target_development_protocol.json`. See
`deliverables/current_results_package/Fresh_Development_Protocol.md` for the
human-readable protocol.

Run the preregistered development-only comparison with:

```powershell
.\.venv\Scripts\python.exe run_fresh_target_development.py `
  --dataset-path data\external\nf-cse-cic-ids2018\NF-CSE-CIC-IDS2018.csv
```

The development run verified all 8,392,401 official rows, preserved zero grouped
overlap among the three development partitions, and initially left 1,631,543
confirmation rows unscored. The strongest development candidate was the target-trained
eight-feature Random Forest at threshold 0.39, with validation MCC 0.71401 and
balanced accuracy 0.93047. This was not a final generalisation result:
Infiltration recall was only 0.12771 and development-partition prevalences differed. See
`deliverables/current_results_package/Fresh_Development_Result.md`.

The selected model is now frozen and the one-time confirmation analysis is
preregistered in `protocols/fresh_confirmation_protocol.json`. The protocol
locks the model and baseline hashes, feature list, thresholds, full-partition
scope, paired group-bootstrap method, aggregate-improvement rule, and separate
supported-attack coverage rule. `run_fresh_confirmation_evaluation.py` contains
no model-fitting or threshold-selection path.

The one-time confirmation is complete. The frozen target Random Forest achieved
MCC 0.12593, balanced accuracy 0.54261, Average Precision 0.48298, recall
0.15377, and false-positive rate 0.06854. It missed the registered absolute MCC
and balanced-accuracy gates, and the paired balanced-accuracy difference
interval included zero. Broad coverage also failed: recall was zero for both
FTP-BruteForce and SlowHTTPTest and 0.12221 for Infiltration. The exact verdict
is `development improvement not confirmed`; no deployment claim is permitted.
See `deliverables/current_results_package/Fresh_Confirmation_Result.md`.

Explain the completed confirmation failure without fitting or changing the
model:

```powershell
.\.venv\Scripts\python.exe run_fresh_generalisation_failure_audit.py `
  --dataset-path data\external\nf-cse-cic-ids2018\extracted\88a47ba2ab64258e_MOHANAD_A4706\data\NF-CSE-CIC-IDS2018.csv
```

The audit reproduces both saved metric sets before analysis. It shows that
FTP-BruteForce and SlowHTTPTest were absent from sampled validation but formed
81.12% of confirmation attacks, where both had zero recall. It also records
large repeated-group concentration, feature and feature/label shift, and a
collapse in frozen-score class separation. The analysis is post hoc and cannot
validate a correction or authorise confirmation-data tuning. See
`deliverables/current_results_package/Fresh_Generalisation_Failure_Audit.md`.

## Version 2 research prototype

Version 2 is isolated on the `codex/v2-generalisation-improvements` branch.
Its development design is registered in
`protocols/v2_generalisation_development_protocol.json` before implementation
or model fitting. The four already-open NetFlow datasets are development domains
in a leave-one-dataset-out comparison; none is described as untouched
confirmation. See
`deliverables/v2_research_prototype/V2_Generalisation_Development_Protocol.md`.

The Version 2 runner is frozen separately before model fitting. Verify its
protocol, implementation and four local dataset files without training:

```powershell
.\.venv\Scripts\python.exe run_v2_generalisation_development.py --preflight-only
```

After preflight succeeds, omit `--preflight-only` for the registered run. The
runner refuses to overwrite a completed `summary.json` and contains no automatic
retry path.

The registered run completed on 27 October 2026 after one separately recorded
implementation failure and a user-authorised numerical type-safety amendment.
All 16 candidate-fold evaluations completed, but no candidate passed the fixed
promotion rule. Weighted Logistic Regression had the highest macro MCC
(`0.05632`) while reaching a maximum held-out false-positive rate of `0.99812`;
the weighted Random Forest candidates had negative macro MCC. The result is a
development-only rejection, not a selected or deployable model. See
`deliverables/v2_research_prototype/V2_Development_Result.md`.

## Version 3 group-balanced development

Version 3 is isolated on the `codex/v3-group-balanced-derived-features` branch.
Its protocol was registered before implementation or fitting in
`protocols/v3_group_balanced_development_protocol.json`. It addresses the group
concentration exposed by Version 2 by capping each source group at 64 retained
rows for fitting and calibration, while still scoring every held-out row.

The fixed comparison contains a group-capped Logistic Regression reference, a
derived-feature Logistic Regression, and a regularised derived-feature Random
Forest. It adds equal-group-weight metrics to the usual row-weighted metrics and
retains strict cross-domain false-positive and attack-coverage gates. IP, port,
and TTL shortcut features remain prohibited. See
`deliverables/v3_research_prototype/V3_Group_Balanced_Development_Protocol.md`.

The Version 3 implementation is frozen separately from its protocol and result.
Verify its implementation hashes, locked evidence, dataset headers, and four
dataset hashes without scanning rows or fitting models:

```powershell
.\.venv\Scripts\python.exe run_v3_group_balanced_development.py --preflight-only
```

After preflight succeeds, omit `--preflight-only` to perform the one registered
development run. The runner refuses to overwrite a completed `summary.json` and
has no automatic retry path.

The one registered Version 3 run completed on 27 October 2026 with no retry.
The 64-row cap worked: source training samples used 18,907 to 107,392 distinct
groups per domain, and no retained group exceeded 64 rows. The derived Logistic
Regression improved macro MCC by `0.10656` over the matched reference, but its
worst-domain balanced accuracy was `0.47506`, maximum held-out false-positive
rate was `0.85690`, worst equal-group MCC was `-0.16244`, and some supported
attacks still had zero recall. No candidate passed all threshold and eligibility
requirements, so the fixed verdict is `no Version 3 candidate selected`. See
`deliverables/v3_research_prototype/V3_Group_Balanced_Development_Result.md`.
