# Fresh Target-Development Result

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-21
- Verification Status: ANALYZED
- Version Label: fresh_target_development_result_v1

## Experiment record

- **Protocol:** `nf-cse-cic-ids2018-v1-target-development-2026-10-20`
- **Dataset:** NF-CSE-CIC-IDS2018 NetFlow v1
- **Status:** completed successfully
- **Scope:** development only
- **Duration:** 780.66 seconds
- **Command:** `.\.venv\Scripts\python.exe run_fresh_target_development.py --dataset-path 'data\external\nf-cse-cic-ids2018\extracted\88a47ba2ab64258e_MOHANAD_A4706\data\NF-CSE-CIC-IDS2018.csv'`
- **CSV SHA-256:** `ed3387646fda24084889f07788cd22da2978b860f5b3d2d6f1210a4e3d79998e0`
- **Protocol SHA-256:** `4280933d0accdc6ed51e0eedc3b09f1685a955ebb9fcfa2bbe2aa276689535f66`

The runner verified all 8,392,401 official rows and the registered parent attack
counts before fitting. Five-significant-digit feature groups had zero overlap
among target training, calibration, and validation. The 1,631,543-row
confirmation partition was counted and hashed, then discarded: no predictions,
metrics, or label summaries were computed for it.

## Candidate comparison

Thresholds for adapted candidates were selected only on target calibration.
The following results use the untouched target-validation development split of
400,087 rows, with attack prevalence 0.06190.

| Candidate | Threshold | Average Precision | MCC | Balanced accuracy | Recall | Specificity | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen source Logistic Regression | 0.690 | 0.05771 | 0.09985 | 0.60116 | 0.79739 | 0.40493 | 0.59507 |
| Frozen source Random Forest | 0.640 | 0.03910 | -0.02726 | 0.49142 | 0.00747 | 0.97537 | 0.02463 |
| Frozen source RF, target-recalibrated threshold | 0.105 | 0.03910 | 0.01033 | 0.50217 | 0.99374 | 0.01059 | 0.98941 |
| Target Logistic Regression, all features | 0.965 | 0.62346 | 0.63042 | 0.77198 | 0.55558 | 0.98839 | 0.01161 |
| **Target Random Forest, all features** | **0.390** | **0.91443** | **0.71401** | **0.93047** | **0.90076** | **0.96019** | **0.03981** |
| Target Random Forest, reduced features | 0.055 | 0.91710 | 0.20014 | 0.70251 | 0.99035 | 0.41467 | 0.58533 |

The registered development rule required both MCC and balanced accuracy to
improve over the frozen source Random Forest by at least 0.05. Four candidates
passed that mechanical rule, including the frozen source Logistic Regression,
because the reference Random Forest was exceptionally weak. The target-trained
all-feature Random Forest was the highest-MCC candidate and improved MCC by
0.74126 and balanced accuracy by 0.43905 over the registered reference. This is
a development-selection result, not confirmation of generalization.

Threshold recalibration alone did not repair the source Random Forest. It
raised recall from 0.00747 to 0.99374 by classifying almost every Normal flow as
an attack, producing a false-positive rate of 0.98941. Removing `L7_PROTO`,
`TCP_FLAGS`, and `FLOW_DURATION_MILLISECONDS` also did not yield the best
operating model: the reduced-feature candidate had slightly higher ranking
Average Precision but a false-positive rate of 0.58533 at its
calibration-selected threshold.

## Overfitting assessment

| Candidate | Train minus validation AP | Train minus validation MCC | Train minus validation balanced accuracy | RF OOB accuracy |
|---|---:|---:|---:|---:|
| Target Logistic Regression | +0.00104 | +0.04353 | +0.02885 | not applicable |
| Target Random Forest, all features | -0.03872 | -0.14904 | -0.01284 | 0.98181 |
| Target Random Forest, reduced features | -0.03283 | -0.00059 | +0.09406 | 0.98525 |

There is no conventional training-to-validation collapse for the selected
full-feature Random Forest. Its validation MCC and balanced accuracy are higher
than its training values, and its false-positive rate is similar across the two
partitions. This argues against simple memorisation within the registered
development data.

It does **not** prove that overfitting has been eliminated. Training prevalence
was 0.02844, calibration prevalence was 0.38492, and validation prevalence was
0.06190 because large grouped duplicate clusters were unevenly distributed.
Average Precision and F1 are prevalence-sensitive, so their raw train-validation
gaps are not directly comparable. OOB accuracy is also optimistic-looking in
an imbalanced dataset and is not interchangeable with MCC. The sealed
confirmation result remains the decisive check.

Attack-level results also prevent an overly broad claim. The selected candidate
recalled 12.77% of the 2,811 Infiltration records and none of the two SQL
Injection records. It achieved at least 99.97% recall for the large Hulk,
Slowloris, GoldenEye, and SSH-Bruteforce groups. The very small Bot, web-attack,
HOIC, UDP-DDoS, and SQL supports make their point estimates unstable.

## Statistical validation and fallacy scan

- **Overall confidence:** CAUTION
- **Coverage:** 11/11 statistical fallacy types checked
- **Reproducibility:** not independently rerun; execution integrity and 27 unit
  tests were checked, so the result is `ANALYZED`, not `VERIFIED`.

| Fallacy | Assessment | Result-specific interpretation |
|---|---|---|
| Simpson's paradox | Caution | Strong aggregate recall coexists with weak Infiltration recall; attack-family results must accompany the aggregate. |
| Ecological fallacy | No detected misuse | Dataset- and group-level results are not used to infer causal properties of individual flows. |
| Berkson's paradox | Caution | The benchmark and deterministic group sampling are selected populations, so deployment-population claims are unsupported. |
| Collider bias | Not applicable | No causal adjustment model or controlled covariate claim is used. |
| Base-rate neglect | Caution | Partition prevalence differs substantially; MCC, balanced accuracy, specificity, false-positive rate, and AP lift are reported alongside AP. |
| Regression to the mean | Not applicable | This is not a pre-post design selected on extreme outcomes. |
| Survivorship bias | Not applicable | No participant attrition or completion filtering is involved. |
| Look-elsewhere effect | Reduced, not eliminated | Six candidates and the success rule were registered before row-level access; the result still requires sealed confirmation. |
| Garden of forking paths | Reduced, not eliminated | Features, splits, thresholds, and metrics were preregistered; later confirmation rules must be registered before opening the seal. |
| Correlation implies causation | No causal claim | Performance differences do not establish why the model succeeds or fails. |
| Reverse causality | Not applicable | The analysis makes no directional causal claim. |

## Decision

Freeze `target_rf_all_features` as the candidate to carry forward, with its
eight registered features and threshold 0.39. Before opening the confirmation
partition, register a confirmation protocol that fixes the model hash, feature
list, threshold, primary metrics, uncertainty method, attack-level reporting,
and accept/reject rule. Until that evaluation is complete, describe this model
only as **development-promising**.
