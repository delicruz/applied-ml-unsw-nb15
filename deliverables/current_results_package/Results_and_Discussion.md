# Current Results and Discussion

## Scope and status

This document consolidates the completed modelling work after the graded
methodology stage. It reports the evidence produced by the current codebase and
does not alter the previously submitted methodology report or slides. The final
model is a Random Forest with 200 trees, maximum depth 24, minimum leaf size 5,
and `sqrt` feature sampling. The final feature set excludes `sttl`, `dttl`, and
`ct_state_ttl`, and the decision threshold is fixed at 0.54. These choices were
made using development data before the supplied test set was opened.

The main result is more nuanced than the validation score initially suggested.
The model retains excellent attack-ranking performance on the conservative
holdout, but the fixed decision threshold produces too many false alarms for a
deployment claim. Internal overfitting is present but limited; the larger
problem is that the validation distribution does not fully represent the
official holdout distribution.

## 1. Model selection and internal validation

Grouped cross-validation selected a regularised Random Forest structure of 200
trees, depth 24, minimum leaf size 5, and `sqrt` feature sampling. The smaller
200-tree model was retained because its performance was practically equivalent
to the 500-tree alternative while requiring less computation. Logistic
Regression remained the interpretable baseline.

On the held-out validation partition, the Random Forest outperformed Logistic
Regression on every primary metric. Average Precision increased from 0.99165 to
0.99649, F1 increased from 0.95423 to 0.96596, and MCC increased from 0.85183 to
0.89206. Paired bootstrap intervals based on five-significant-digit
near-duplicate groups excluded zero for all three improvements:

| Metric | Logistic Regression | Random Forest | RF minus LR | 95% paired bootstrap interval |
|---|---:|---:|---:|---:|
| Average Precision | 0.99165 | 0.99649 | +0.00484 | [0.00404, 0.00574] |
| F1 | 0.95423 | 0.96596 | +0.01174 | [0.00987, 0.01395] |
| MCC | 0.85183 | 0.89206 | +0.04023 | [0.03419, 0.04706] |

The bootstrap probability that the Random Forest was better was 1.00 for each
metric across 1,000 resamples. This supports the choice of the nonlinear model,
but the absolute differences should not be presented as algorithmic novelty.
Random Forest and Logistic Regression have both been studied extensively on
UNSW-NB15. The contribution of this work is the stress-tested evaluation rather
than a new classifier.

## 2. Overfitting assessment

The original all-feature Random Forest achieved training Average Precision of
0.99985 and validation Average Precision of 0.99672, a gap of 0.00313. The
training-to-validation gaps were larger for threshold-dependent metrics: 0.02586
for F1 and 0.08234 for MCC. These gaps show some fitting to development-specific
patterns, but they do not resemble uncontrolled memorisation.

Three results support that interpretation. First, out-of-bag F1 was 0.97085,
close to validation F1 of 0.96697. Second, five-fold grouped cross-validation
produced mean F1 of 0.96613 with a standard deviation of 0.00165. Third, the
grouped learning curve improved slightly as the available training data
increased and did not show a collapsing holdout score. Therefore, the high
validation result was not explained solely by conventional model overfitting.

The more important evidence emerged when the frozen model was evaluated on the
official holdout. For the final no-TTL model, validation and conservative
holdout results differed as follows:

| Metric | Validation | Conservative holdout | Change |
|---|---:|---:|---:|
| Average Precision | 0.99645 | 0.98207 | -0.01438 |
| F1 | 0.96558 | 0.88226 | -0.08332 |
| MCC | 0.89078 | 0.74624 | -0.14454 |
| False-positive rate | 0.08938 | 0.26381 | +0.17443 |

This pattern changes the conclusion. Ranking quality remained strong, but the
operating threshold transferred poorly. The evidence is consistent with a
combination of limited internal overfitting, dataset shift, and threshold
miscalibration. The descriptive audit cannot establish which mechanism is
causal, so the result should not be labelled as pure overfitting or pure concept
drift.

## 3. Shortcut-feature and near-duplicate robustness

The TTL-related variables required a direct stress test because `sttl` showed a
strong association with the target and could reflect the benchmark-generation
setup. Removing `sttl`, `dttl`, and `ct_state_ttl` produced almost no practical
loss on the near-duplicate-grouped robustness split:

| Variant | Average Precision | F1 | MCC |
|---|---:|---:|---:|
| All features | 0.996834 | 0.968233 | 0.899071 |
| Without TTL-related features | 0.996773 | 0.968408 | 0.898988 |
| Difference, no-TTL minus all | -0.000060 | +0.000176 | -0.000083 |

The paired 95% interval for the Average Precision difference was
[-0.000108, -0.000021], showing a statistically detectable but operationally
tiny reduction. The intervals for F1 and MCC crossed zero. This result supports
using the more conservative no-TTL model: nearly all validation performance was
retained without relying on the disputed TTL signals.

The initial exact-row grouping did not remove every near duplicate. Under the
stricter definition, where categorical values remained exact and numeric values
were rounded to five significant digits, 20 groups crossed the original
development-validation boundary and affected 500 validation rows (1.43%). All
500 were Normal and label-consistent. A reconstructed split placed each such
group entirely in one partition. Performance remained high and similar, so the
validation score was not driven by these identified near duplicates. This does
not prove the absence of all semantic or temporal duplication; it tests one
explicit, reproducible definition.

## 4. Frozen holdout performance

The final model was refitted on all 175,341 supplied training records and then
evaluated without further model or threshold changes. Because an older workflow
had already examined a deterministic 5,000-row test sample, the primary scope
excluded those records. It also excluded 8,647 test records whose coarsened
final-feature fingerprints overlapped training. The conservative primary scope
therefore contained 69,222 records.

| Metric | Conservative holdout result | 95% group-bootstrap interval |
|---|---:|---:|
| Average Precision | 0.98207 | [0.98091, 0.98322] |
| Accuracy | 0.86389 | not bootstrapped |
| Precision | 0.80084 | not bootstrapped |
| Recall | 0.98211 | not bootstrapped |
| F1 | 0.88226 | [0.87875, 0.88593] |
| MCC | 0.74624 | [0.74086, 0.75123] |
| False-positive rate | 0.26381 | not bootstrapped |
| Brier score | 0.08965 | not bootstrapped |
| Log loss | 0.25825 | not bootstrapped |

The confusion matrix contained 35,301 true positives, 643 false negatives,
24,499 true negatives, and 8,779 false positives. High recall was therefore
obtained at a substantial alert burden: approximately one in four Normal
records was classified as suspicious. Average Precision of 0.98207 shows that
the model still ranks suspicious records effectively across possible
thresholds. It does not mean that the frozen 0.54 operating point is
deployment-ready.

## 5. Attack-specific performance

Attack recall was high for most categories, but aggregate recall concealed a
clear weakness for Fuzzers:

| Attack category | Support | Detected | Recall |
|---|---:|---:|---:|
| Fuzzers | 5,454 | 4,919 | 0.90191 |
| Exploits | 8,759 | 8,670 | 0.98984 |
| Shellcode | 351 | 348 | 0.99145 |
| DoS | 1,953 | 1,942 | 0.99437 |
| Reconnaissance | 2,226 | 2,222 | 0.99820 |
| Generic | 16,091 | 16,090 | 0.99994 |
| Analysis | 523 | 523 | 1.00000 |
| Backdoor | 546 | 546 | 1.00000 |
| Worms | 41 | 41 | 1.00000 |

Perfect recall for Worms should be interpreted cautiously because it is based
on only 41 observations. The 535 missed Fuzzers attacks account for 83.2% of all
643 false negatives, making this category the principal detection weakness in
the conservative holdout.

## 6. Distribution shift and false-positive analysis

The post-evaluation audit was descriptive and did not retune the model. The
largest numeric Population Stability Index values were observed for `rate`
(0.133), `ct_src_ltm` (0.123), `ct_src_dport_ltm` (0.121), `sbytes` (0.110),
and `sload` (0.106). Among categorical predictors, `proto` showed the greatest
Jensen-Shannon divergence (0.032 bits). TCP prevalence increased from 45.6% in
training to 58.4% in the conservative holdout. The prevalence of `state=INT`
decreased from 46.9% to 35.0%. These measurements show that the input
distribution changed, although they do not prove that any single change caused
the performance gap.

False positives were strongly concentrated in particular traffic patterns.
Among Normal records, `state=INT` had a false-positive rate of 77.0%
(2,173/2,821), `service=ftp` had a rate of 46.9% (333/710), and UDP traffic had
a rate of 30.6% (2,161/7,056). `state=FIN` generated the largest absolute number
of false positives (6,584) because it was much more common, although its
within-group rate was lower at 29.0%. Risk and volume therefore need to be
reported separately.

The score distribution also matters. The mean false-positive probability was
0.709 and the median was 0.702. Only 322 false positives, or 3.67%, had a score
of at least 0.90. Approximately half of the false positives fell between 0.54
and 0.70, indicating that many errors were not extreme predictions. A future
threshold or calibration study could potentially reduce the alert rate, but it
must use new calibration data rather than optimising against this test result.

On a balanced sample of 500 false positives and 500 true negatives, SHAP
identified `ackdat`, `tcprtt`, `synack`, `dload`, `rate`, and `dinpkt` as the
strongest average positive contributors to false alarms. The corresponding raw
feature contrasts were substantial: for example, false positives had median
`ackdat` of 0.0608 compared with 0.000137 for true negatives, and median
`dinpkt` of 95.79 compared with 0.98. These are model-behaviour explanations,
not evidence that the variables cause malicious activity.

![Validation-to-holdout numeric shift](../../artifacts/distribution_shift_audit/plots/numeric_feature_shift_psi.png)

![False-positive SHAP drivers](../../artifacts/distribution_shift_audit/plots/false_positive_shap_drivers.png)

## 7. What is new in this work

The novelty is evaluative rather than algorithmic. A claim that Logistic
Regression and Random Forest have not previously been compared on UNSW-NB15
would be incorrect. The distinctive contribution is the combined evidence
chain:

1. model selection and preprocessing are separated from the final holdout;
2. threshold selection is fixed before holdout evaluation;
3. bootstrap resampling respects near-duplicate groups rather than treating
   correlated rows as independent;
4. the suspected TTL shortcut is removed and tested directly;
5. both exact and five-significant-digit near-duplicate overlap are audited;
6. results include per-attack recall and uncertainty intervals; and
7. the validation-to-holdout gap is investigated through distribution-shift,
   subgroup-error, and SHAP analyses.

This contribution is best described as a reproducible robustness and
generalisation study of a conventional detector. The strongest substantive
finding is that near-perfect internal Average Precision can coexist with an
operationally poor false-positive rate after the threshold is transferred to a
different holdout distribution.

## 8. Limitations

- The official holdout is not completely pristine because a deterministic
  5,000-row sample had been evaluated by an older workflow. Excluding those
  records reduces the risk but cannot undo historical exposure.
- Near-duplicate detection uses a transparent rounding rule, not a semantic or
  temporal similarity model. Other forms of related traffic may remain across
  partitions.
- Bootstrap intervals cluster the selected coarsened fingerprints. They account
  for that defined dependence structure but not every possible dependency.
- Distribution-shift statistics and SHAP values are post-hoc descriptions. They
  do not establish causality or identify a guaranteed correction.
- Attack-specific point estimates are uncertain for rare categories, especially
  the 41 Worms records.
- UNSW-NB15 is a controlled benchmark and does not establish performance on
  contemporary production traffic.
- The binary target hides differences between attack families and does not
  evaluate multiclass diagnosis.
- The frozen threshold maximises a development-stage objective rather than an
  explicitly measured operational cost function.

## 9. Supported conclusion

The selected Random Forest is superior to the Logistic Regression baseline on
the internal validation data, and its high validation performance survives both
TTL removal and a stricter near-duplicate grouping rule. This makes severe
classic overfitting or simple duplicate leakage unlikely as the sole
explanation for the original score. Nevertheless, the conservative holdout
reveals a substantial generalisation gap. Average Precision remains high, but
F1 and MCC decline and the false-positive rate rises to 26.38%.

The defensible conclusion is therefore that the model is a strong benchmark
ranker but not yet a deployment-ready detector at the frozen threshold. The
results support further evaluation on fresh, representative calibration data;
they do not support tuning the model on the official test set or claiming that
the existing threshold is operationally acceptable.

## 10. Next targets

1. Obtain or construct a genuinely fresh calibration partition that reflects
   the intended deployment environment.
2. Evaluate probability calibration and cost-sensitive threshold choices on
   that calibration data, leaving the official test results unchanged.
3. Repeat evaluation using a temporal or source-aware split if timestamps or
   capture-session identifiers can be recovered.
4. Investigate Fuzzers errors and the high-risk `state=INT` subgroup without
   creating subgroup rules from the current test set.
5. Report alert burden using an operational unit such as false alerts per
   10,000 Normal flows and specify the acceptable cost trade-off.
6. Validate the complete pipeline on a newer or external intrusion-detection
   dataset before making any deployment claim.

## Draft AI-use acknowledgement

Generative AI assistance was used to support code development, result
organisation, and language editing. All analyses were executed in the local
project environment, and numerical claims in this document were checked against
saved experiment artifacts. The student remains responsible for the analysis,
interpretation, and final submitted text. This wording should be adapted to the
course or venue's required disclosure format before submission.
