# Final Research-Prototype Status

## Material Passport

- Origin: academic-research-suite, evidence consolidation
- Date: 27 October 2026
- Verification status: ANALYZED
- Version: final-research-prototype-status-v1

## Final outcome

The project is complete as a **research prototype**, but it did not produce a
model that can be claimed to generalise reliably to real network environments.
The fixed final development verdict is:

> **No cross-domain candidate was selected. The model is not deployment-ready.**

This is not a failed project. The project implemented a reproducible intrusion-
detection workflow, detected that initially strong internal scores were
misleading, identified several sources of optimism, and tested two registered
attempts to improve generalisation without weakening their decision rules.

## Completed project features

| Feature | Status | Evidence |
|---|---|---|
| Data ingestion and audit | Complete | Dataset summaries, feature typing, class and attack counts |
| Duplicate-aware development splitting | Complete | Exact and five-significant-digit group checks |
| Leakage-controlled preprocessing | Complete | Imputation, encoding and scaling fitted on source training only |
| Logistic Regression baseline | Complete | Internal and multidomain comparisons |
| Random Forest model | Complete | Regularised, tuned and evaluated variants |
| Threshold analysis | Complete | Source calibration only; fixed feasibility constraints |
| Explainability | Complete | Permutation importance, SHAP and local error cases |
| Overfitting diagnostics | Complete | Train/OOB/validation gaps, grouped CV and learning curves |
| Shortcut-feature audit | Complete | TTL removal and endpoint-feature exclusion |
| Near-duplicate robustness | Complete | Grouped splits, group bootstrap and group-size analysis |
| Statistical uncertainty | Complete | Paired and group-cluster bootstrap intervals |
| Operational burden analysis | Complete | False alerts and missed attacks per 10,000 flows |
| Subgroup error analysis | Complete | Attack-family and Normal subgroup failures |
| External transfer testing | Complete | NF-ToN-IoT and NF-BoT-IoT |
| Fresh target development and confirmation | Complete | NF-CSE-CIC-IDS2018 sealed confirmation |
| Version 2 multidomain development | Complete | Fixed four-domain leave-one-dataset-out comparison |
| Version 3 group-balanced development | Complete | 64-row group cap, derived features and equal-group metrics |

## Evidence progression

The internal UNSW-NB15 results initially looked excellent. The no-TTL Random
Forest achieved validation Average Precision 0.99645 and MCC 0.89078. Even the
conservative supplied holdout remained strong, with Average Precision 0.98207
and MCC 0.74624, although its false-positive rate was already high at 0.26381.

The apparent performance did not transfer:

| Evaluation | Main model or candidate | Key result | Registered outcome |
|---|---|---|---|
| NF-ToN-IoT external | Frozen Random Forest | MCC -0.15243; recall 0.04297 | Transfer unsupported |
| NF-BoT-IoT external | Frozen Random Forest | MCC -0.01170; recall 0.00228 | Useful transfer not supported |
| NF-CSE sealed confirmation | Target-trained Random Forest | MCC 0.12593; recall 0.15377 | Development improvement not confirmed |
| Version 2 multidomain | Weighted Logistic Regression | Macro MCC 0.05632; maximum FPR 0.99812 | No V2 candidate selected |
| Version 3 group-balanced | Derived Logistic Regression | Macro MCC 0.02561; worst BA 0.47506; maximum FPR 0.85690 | No V3 candidate selected |

## Overfitting and generalisation conclusion

The early high score cannot be trusted as evidence of real-world
generalisation. TTL shortcuts were removed, endpoint identifiers were excluded,
and repeated groups were capped, but external and leave-one-domain-out
performance remained weak.

Version 3 provides the clearest diagnosis. The largest retained source group
was limited to 64 rows, source samples used thousands of distinct groups, and
training-to-calibration MCC gaps were near zero. Nevertheless, mean
calibration-to-held-out MCC gaps ranged from 0.724 to 0.918. This indicates that
the main remaining problem is cross-domain and attack-composition shift rather
than simple memorisation of the source training rows.

The derived Logistic Regression improved relative to the matched V3 reference,
but it failed the absolute requirements: worst-domain balanced accuracy was
below 0.50, maximum FPR was far above 0.25, worst-domain equal-group MCC was
negative, and supported attacks still had zero recall. Relative improvement is
therefore not enough to justify model selection.

## What can be claimed

- The project provides a reproducible and extensively audited research
  pipeline for studying intrusion-detection generalisation.
- Duplicate-aware splitting, shortcut removal, group-capped sampling and
  equal-group evaluation reduce several sources of optimistic bias.
- Deterministic derived traffic features improved the matched linear reference,
  but not enough to satisfy the fixed reliability gates.
- Cross-dataset shift and attack-family coverage remain unresolved.

## What cannot be claimed

- The model is ready for real-world or production deployment.
- High UNSW-NB15 validation accuracy establishes generalisation.
- Version 2 or Version 3 selected a successful replacement model.
- Additional tuning on the same four known datasets would provide independent
  confirmation.

## Final delivery position

The existing frozen UNSW-NB15 model may be retained as a demonstration artifact
for the original dataset. It must not be represented as a generally usable
network detector. V2 and V3 correctly produce no selected model and therefore
do not create a replacement production artifact.

With the current deadline, the defensible completion path is documentation and
presentation of the result, not another tuning cycle. Future model development
should begin only when genuinely new capture data and a deployment-specific
false-positive budget are available.

## AI-use acknowledgement

OpenAI ChatGPT/Codex assisted with code drafting, test generation, experiment
orchestration, result organisation and prose editing. Experiments were executed
locally against the project datasets; no project dataset was uploaded to an
external AI service during this work. The student remains responsible for
reviewing the artifacts, interpreting the results and complying with the
institution's disclosure requirements before submission.

The previously graded methodology files were not modified during this
consolidation.
