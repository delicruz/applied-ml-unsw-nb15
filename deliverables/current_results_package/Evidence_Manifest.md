# Evidence Manifest

This manifest maps the main claims in `Results_and_Discussion.md` to the saved
machine-readable artifacts. The graded methodology documents are not part of
this package and were not modified.

| Claim or section | Primary evidence | Supporting output |
|---|---|---|
| Random Forest configuration | `../../artifacts/random_forest_tuning/selected_configuration.json` | `../../artifacts/random_forest_tuning/` |
| RF versus LR validation comparison | `../../artifacts/statistical_comparison/summary.json` | `../../artifacts/statistical_comparison/paired_bootstrap_summary.csv` |
| Training, OOB, validation, grouped CV and learning curve | `../../artifacts/overfitting_diagnostics/summary.json` | `../../artifacts/overfitting_diagnostics/` |
| TTL-feature ablation | `../../artifacts/robustness_audit/robustness_metrics.csv` | `../../artifacts/robustness_audit/ttl_ablation_paired_bootstrap.csv` |
| Near-duplicate overlap and reconstructed split | `../../artifacts/robustness_audit/summary.json` | `../../artifacts/robustness_audit/` |
| Frozen model identity and holdout integrity | `../../artifacts/final_holdout_evaluation/summary.json` | `../../artifacts/final_holdout_evaluation/final_random_forest_no_ttl.joblib` |
| Conservative holdout metrics | `../../artifacts/final_holdout_evaluation/holdout_metrics.csv` | `../../artifacts/final_holdout_evaluation/holdout_bootstrap_intervals.csv` |
| Per-attack recall | `../../artifacts/final_holdout_evaluation/per_attack_recall.csv` | `../../artifacts/final_holdout_evaluation/plots/primary_attack_recall.png` |
| Numeric and categorical distribution shift | `../../artifacts/distribution_shift_audit/numeric_feature_shift.csv` | `../../artifacts/distribution_shift_audit/categorical_feature_shift.csv` |
| False-positive subgroup analysis | `../../artifacts/distribution_shift_audit/categorical_error_groups.csv` | `../../artifacts/distribution_shift_audit/plots/categorical_false_positive_groups.png` |
| False-positive score distribution | `../../artifacts/distribution_shift_audit/normal_probability_bins.csv` | `../../artifacts/distribution_shift_audit/plots/normal_probability_distribution.png` |
| False-positive SHAP drivers | `../../artifacts/distribution_shift_audit/false_positive_shap_drivers.csv` | `../../artifacts/distribution_shift_audit/plots/false_positive_shap_drivers.png` |
| Operational burden at frozen threshold | `../../artifacts/operational_burden/frozen_threshold_burden.csv` | `../../artifacts/operational_burden/plots/frozen_threshold_burden.png` |
| Development-only cost sensitivity | `../../artifacts/operational_burden/selected_cost_policies.csv` | `../../artifacts/operational_burden/plots/cost_sensitivity.png` |
| Validation of candidate policies | `../../artifacts/operational_burden/validation_policy_evaluation.csv` | `../../artifacts/operational_burden/summary.json` |
| Fuzzers and `state=INT` score profiles | `../../artifacts/subgroup_error_analysis/score_summary.csv` | `../../artifacts/subgroup_error_analysis/plots/subgroup_probability_distributions.png` |
| Supported categorical subgroup profiles | `../../artifacts/subgroup_error_analysis/categorical_subgroup_profiles.csv` | `../../artifacts/subgroup_error_analysis/plots/categorical_subgroup_error_rates.png` |
| Numeric outcome contrasts | `../../artifacts/subgroup_error_analysis/numeric_outcome_contrasts.csv` | `../../artifacts/subgroup_error_analysis/plots/numeric_outcome_contrasts.png` |
| Sampled subgroup SHAP contrasts | `../../artifacts/subgroup_error_analysis/subgroup_shap_contrasts.csv` | `../../artifacts/subgroup_error_analysis/plots/subgroup_shap_contrasts.png` |
| External experiment scope, hashes, splits and locked thresholds | `../../artifacts/external_netflow_validation/summary.json` | `../../artifacts/external_netflow_validation/source_split_summary.csv` |
| Source-validation and NF-ToN-IoT metrics | `../../artifacts/external_netflow_validation/metrics_long.csv` | `../../artifacts/external_netflow_validation/plots/generalisation_comparison.png` |
| External exact-feature-group bootstrap intervals | `../../artifacts/external_netflow_validation/group_bootstrap_intervals.csv` | `../../artifacts/external_netflow_validation/group_bootstrap_samples.csv` |
| External attack-family recall | `../../artifacts/external_netflow_validation/per_attack_recall.csv` | `../../artifacts/external_netflow_validation/plots/precision_recall_curves.png` |
| Cross-dataset numeric and categorical feature shift | `../../artifacts/external_shift_audit/numeric_distribution_shift.csv` | `../../artifacts/external_shift_audit/categorical_distribution_shift.csv` |
| Prevalence-adjusted feature/label association drift | `../../artifacts/external_shift_audit/feature_label_association_drift.csv` | `../../artifacts/external_shift_audit/plots/feature_label_association_drift.png` |
| Frozen-model score reversal | `../../artifacts/external_shift_audit/model_score_shift.csv` | `../../artifacts/external_shift_audit/plots/model_score_distribution_shift.png` |
| External feature-bin error profiles | `../../artifacts/external_shift_audit/external_error_profiles_by_feature_bin.csv` | `../../artifacts/external_shift_audit/summary.json` |
| Third external holdout preregistration | `../../protocols/third_dataset_preregistration.json` | `Third_Dataset_Protocol.md` |
| Third external holdout integrity and verdict | `../../artifacts/third_dataset_validation/summary.json` | `Third_Dataset_Result.md` |
| NF-BoT-IoT point metrics and group-bootstrap intervals | `../../artifacts/third_dataset_validation/metrics_long.csv` | `../../artifacts/third_dataset_validation/group_bootstrap_intervals.csv` |
| NF-BoT-IoT attack-family recall | `../../artifacts/third_dataset_validation/per_attack_recall.csv` | `../../artifacts/third_dataset_validation/group_bootstrap_samples.csv` |

## Integrity checks already applied

- The final model SHA-256 was verified before the distribution-shift audit.
- The conservative holdout mask reproduced 69,222 records.
- False-positive subgroup totals reproduce 8,779 false positives separately for
  `proto`, `service`, and `state`.
- Probability-bin counts reproduce all 33,278 Normal records.
- The post-hoc audit records `model_or_threshold_changes: false`.
- Operational policies were selected on an internal grouped development fold;
  holdout predictions only report the unchanged threshold of 0.54.
- The targeted subgroup audit verified the frozen model SHA-256, reproduced the
  69,222-row conservative scope, and records both `model_or_threshold_changes:
  false` and `subgroup_rules_created: false`.
- The external experiment excludes all four endpoint identifiers, reports zero
  exact-feature-group overlap among source training, threshold-calibration and
  validation partitions, and records `external_data_used_for_tuning: false`.
- Full external results use all 1,623,118 NF-UNSW-NB15 rows and all 1,379,274
  NF-ToN-IoT rows; SHA-256 values for both official CSV files are recorded.
- The cross-dataset shift audit re-verifies both dataset hashes, records hashes
  for both fitted pipelines, reconstructs the 325,466-row source-validation
  fold, and records no model, feature, threshold, or tuning changes.
- The NF-BoT-IoT protocol was committed before the official archive was
  downloaded or row-level data was opened. The runner reproduced all official
  row and attack counts and verified both frozen model hashes before scoring.
- The third-dataset result uses all 600,100 rows and 39,901 exact eight-feature
  groups. It records `model_or_threshold_changes: false` and
  `external_tuning: false`.
- The primary Random Forest failed each preregistered transfer requirement:
  MCC interval lower bound above zero, balanced accuracy above 0.50, and
  Average Precision lift above 1.00.

## Interpretation boundary

Numerical performance claims are directly supported by local artifacts.
Statements about likely distribution shift are inferences from those results.
The package does not claim that PSI, Jensen-Shannon divergence, subgroup error
rates, SHAP values, or cross-dataset performance establish a causal mechanism.
The NetFlow-v1 experiments test transfer of a matched modelling approach, not
the previously frozen 44-feature model. The NF-BoT-IoT result supports the
fixed protocol verdict "useful transfer not supported" but does not identify a
single causal mechanism for the failure.
