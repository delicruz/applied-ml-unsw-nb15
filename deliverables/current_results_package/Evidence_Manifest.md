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
| NF-BoT-IoT numeric and categorical feature shift | `../../artifacts/third_dataset_shift_audit/numeric_distribution_shift.csv` | `../../artifacts/third_dataset_shift_audit/categorical_distribution_shift.csv` |
| NF-BoT-IoT prevalence-adjusted feature/label drift | `../../artifacts/third_dataset_shift_audit/feature_label_association_drift.csv` | `../../artifacts/third_dataset_shift_audit/feature_label_association_bins.csv` |
| Source and two-external-dataset score ordering | `../../artifacts/third_dataset_shift_audit/cross_external_score_comparison.csv` | `../../artifacts/third_dataset_shift_audit/plots/cross_external_score_ordering.png` |
| NF-BoT-IoT feature-bin error profiles | `../../artifacts/third_dataset_shift_audit/external_error_profiles_by_feature_bin.csv` | `../../artifacts/third_dataset_shift_audit/summary.json` |
| Aligned recurring external error bins | `../../artifacts/cross_external_error_synthesis/aligned_feature_bin_errors.csv` | `../../artifacts/cross_external_error_synthesis/summary.json` |
| Per-feature coverage of recurring failure modes | `../../artifacts/cross_external_error_synthesis/feature_consistency_summary.csv` | `../../artifacts/cross_external_error_synthesis/plots/consistent_feature_bin_failures.png` |
| Fresh target-development protocol | `../../protocols/fresh_target_development_protocol.json` | `Fresh_Development_Protocol.md` |
| Fresh target-development integrity and candidate selection | `../../artifacts/fresh_target_development/summary.json` | `Fresh_Development_Result.md` |
| Fresh target-validation candidate metrics | `../../artifacts/fresh_target_development/validation_candidate_comparison.csv` | `../../artifacts/fresh_target_development/plots/candidate_validation_comparison.png` |
| Fresh target overfitting gaps and OOB results | `../../artifacts/fresh_target_development/overfitting_gaps.csv` | `../../artifacts/fresh_target_development/plots/target_development_overfitting_gaps.png` |
| Fresh target attack-family recall | `../../artifacts/fresh_target_development/validation_per_attack_recall.csv` | `Fresh_Development_Result.md` |
| Frozen fresh-confirmation inputs and decision rules | `../../protocols/fresh_confirmation_protocol.json` | `Fresh_Confirmation_Protocol.md` |
| Confirmation runner integrity tests | `../../test_fresh_confirmation_evaluation.py` | `../../run_fresh_confirmation_evaluation.py` |
| Confirmation preflight failure and portability amendment | `../../protocols/fresh_confirmation_amendment_001.json` | `Fresh_Confirmation_Protocol.md` |
| Confirmation hash-preflight failure and transcription amendment | `../../protocols/fresh_confirmation_amendment_002.json` | `Fresh_Confirmation_Protocol.md` |
| Sealed-confirmation integrity and preregistered verdict | `../../artifacts/fresh_confirmation_evaluation/summary.json` | `Fresh_Confirmation_Result.md` |
| Confirmation point metrics and group-bootstrap intervals | `../../artifacts/fresh_confirmation_evaluation/metrics_long.csv` | `../../artifacts/fresh_confirmation_evaluation/group_bootstrap_intervals.csv` |
| Paired candidate-minus-baseline uncertainty | `../../artifacts/fresh_confirmation_evaluation/paired_bootstrap_intervals.csv` | `../../artifacts/fresh_confirmation_evaluation/paired_bootstrap_differences.csv` |
| Confirmation attack-label recall and coverage | `../../artifacts/fresh_confirmation_evaluation/per_attack_recall.csv` | `../../artifacts/fresh_confirmation_evaluation/plots/selected_attack_recall.png` |

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
- The NF-BoT-IoT shift audit re-verifies the protocol, source and target data,
  both frozen model hashes, the original source-validation fold, and the two
  source-selected thresholds. It records no model, feature, threshold, or
  external-tuning changes.
- The cross-external score table uses the earlier NF-ToN-IoT audit artifact and
  the newly computed NF-BoT-IoT scores. Both external datasets reverse the
  attack-above-benign mean-score ordering observed on source validation.
- The cross-external error synthesis reads only saved aggregated audit tables,
  re-verifies the shared source fold and model hashes, and records no model,
  feature, threshold, or external-tuning changes. Its support and error-rate
  cutoffs are descriptive post-hoc reporting rules, not selection criteria.
- The fresh target runner verified all 8,392,401 official rows and parent attack
  counts before fitting. Training, calibration, and validation shared zero
  five-significant-digit feature groups.
- Before confirmation, the 1,631,543-row partition was counted and hashed but
  not scored. The completed run then reproduced 1,631,543 rows, 153,825 groups,
  the registered ordered digest, and all official global counts before scoring.
- Candidate models used target training only, thresholds used target
  calibration only, and ranking used target validation only. NF-ToN-IoT and
  NF-BoT-IoT were not reused for tuning.
- The confirmation protocol fixes the selected and baseline model hashes and
  sizes, thresholds, eight-feature set, all-row confirmation scope, paired
  group-bootstrap method, and two-part decision rule before the sealed
  partition is opened. Its runner exposes no model-fitting or threshold-
  selection path.
- Amendments 001 and 002 document two pre-row-access technical failures and
  explicit retry authorisation. The original protocol hash remains unchanged;
  neither amendment changes a model, feature, threshold, partition, metric,
  bootstrap procedure, or decision rule.
- The one-time confirmation records `one_time_evaluation_completed: true` and
  `model_or_threshold_changes: false`. The candidate failed the absolute MCC
  and balanced-accuracy gates, the paired balanced-accuracy gate, and broad
  supported-attack coverage, producing `development improvement not confirmed`.

## Interpretation boundary

Numerical performance claims are directly supported by local artifacts.
Statements about likely distribution shift are inferences from those results.
The package does not claim that PSI, Jensen-Shannon divergence, subgroup error
rates, SHAP values, or cross-dataset performance establish a causal mechanism.
The NetFlow-v1 experiments test transfer of a matched modelling approach, not
the previously frozen 44-feature model. The NF-BoT-IoT result supports the
fixed protocol verdict "useful transfer not supported" but does not identify a
single causal mechanism for the failure. The fresh sealed result supports the
fixed verdict "development improvement not confirmed". It demonstrates
instability of the development claim but does not by itself identify classic
capacity overfitting, composition shift, or residual campaign dependence as
the unique cause, and it does not permit deployment.
