# Version 2 Generalisation Development Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-26
- Verification Status: UNVERIFIED
- Version Label: v2_generalisation_development_protocol_v1

## Purpose

Version 2 is a bounded research prototype intended to test whether development
across several network domains is more robust than fitting to one dataset. It
does not attempt to repair the failed Version 1 confirmation by tuning on that
confirmation partition.

All four available NetFlow datasets have already been inspected. They are
therefore treated only as Version 2 development domains:

1. NF-UNSW-NB15;
2. NF-ToN-IoT;
3. NF-BoT-IoT; and
4. NF-CSE-CIC-IDS2018, including its consumed former confirmation partition.

No result from this protocol will be called an untouched external test.

## Registered research question

Does equalising the contribution of datasets and supported attack families,
with stronger Random Forest regularisation, improve leave-one-dataset-out MCC,
worst-domain balanced accuracy, and supported attack-family recall without an
unacceptable increase in false positives?

## What changes from Version 1

- Development uses four dataset domains rather than one random or hash-based
  validation mixture.
- Every evaluation fold withholds one complete dataset from fitting,
  preprocessing, weighting, and threshold selection.
- Weighted candidates balance source datasets and raw attack labels.
- Model selection includes worst-domain and per-attack requirements, not only
  aggregate MCC or accuracy.
- A more regularised Random Forest is tested against a matched unweighted
  Random Forest and a weighted Logistic Regression.
- A pre-specified five-feature Random Forest provides a stability ablation.
- Endpoint identifiers and TTL-related features, including `sttl` and `dttl`,
  are prohibited.

## Fixed candidate set

| Candidate | Purpose |
|---|---|
| Unweighted pooled Random Forest, eight features | Matched baseline |
| Domain/family-weighted Logistic Regression, eight features | Linear control |
| Domain/family-weighted regularised Random Forest, eight features | Primary hypothesis |
| Domain/family-weighted regularised Random Forest, five features | Stability ablation |

No XGBoost candidate is included. This keeps the experiment focused on the
identified generalisation problem rather than expanding into a broad algorithm
benchmark.

## Evaluation design

Four leave-one-dataset-out folds are fixed. In each fold, three datasets supply
group-separated training and calibration data, while the fourth dataset is
used only for that fold's development evaluation. Training is capped
deterministically at 250,000 rows per source domain and calibration at 100,000
rows per source domain. The complete held-out dataset is scored in chunks.

Thresholds are selected using only calibration data from the three source
domains. The registered threshold constraints require:

- false-positive rate no greater than 0.20 in each source domain; and
- recall of at least 0.20 for every source attack label with at least 100 rows.

If no threshold satisfies these constraints, the candidate is marked as having
failed the constraint and cannot pass the promotion rule.

## Promotion rule

The primary measure is macro-average held-out-domain MCC across the four folds.
Compared with the matched unweighted Random Forest, a promotable candidate must:

- improve macro-average MCC by at least 0.05;
- improve worst-domain balanced accuracy by at least 0.02;
- achieve balanced accuracy above 0.50 in every domain;
- keep false-positive rate at or below 0.25 in every domain;
- recall every held-out attack label with at least 100 rows at 0.20 or better;
  and
- avoid increasing maximum domain false-positive rate by more than 0.02.

If no candidate satisfies every condition, the required result is **no Version
2 candidate selected**. The conditions will not be weakened after results are
seen.

## Overfitting checks

The run must report training-to-calibration and calibration-to-held-out gaps,
per-domain metrics, every supported attack-label recall, performance by
repeated-group size and mixed-label status, Random Forest out-of-bag results,
and frozen score separation. High aggregate accuracy cannot override a failed
domain or attack-family requirement.

## Independent confirmation boundary

This protocol contains no final confirmation dataset. A selected Version 2
candidate remains a development-only research prototype until a genuinely new
dataset or capture is separately preregistered before inspection or scoring.
If no such data can be obtained, the project will report that limitation rather
than claim real-world readiness.

## Registered machine-readable protocol

The authoritative specification is
`../../protocols/v2_generalisation_development_protocol.json`.
