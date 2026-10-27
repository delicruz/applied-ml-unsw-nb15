# Version 3 Group-Balanced Development Protocol

**Registered:** 27 October 2026, Australia/Adelaide

**Status:** registered before Version 3 implementation or model fitting

**Purpose:** bounded research-prototype development, not deployment validation

## Why Version 3 exists

Version 2 did not produce an acceptable model. Its best macro MCC was only
0.05632 and its maximum held-out false-positive rate was 0.99812. Some source
samples were also dominated by very large repeated groups; for example, one
NF-CSE-CIC-IDS2018 training sample contained 299,372 rows from only 13 groups.

Version 3 tests one focused correction: limit the influence of each repeated
group during fitting and calibration. It also tests a small, fixed set of
traffic-rate and directionality features derived from the same eight common
endpoint-free NetFlow fields. This is not another broad hyperparameter search.

## Development design

The four already-opened datasets remain development domains:

- NF-UNSW-NB15
- NF-ToN-IoT
- NF-BoT-IoT
- NF-CSE-CIC-IDS2018

The experiment uses four leave-one-dataset-out folds. In every fold, three
source domains may be used for training and calibration. The held-out domain
cannot influence preprocessing, sampling, fitting, or threshold selection.
Every held-out row is scored. Because all four datasets have already been
inspected, none is an untouched test set or independent confirmation dataset.

## Group-balanced sampling

Groups use the same dataset-namespaced, five-significant-digit definition as
Version 2. Groups remain intact when source data are divided 75/25 into training
and calibration partitions. Within each partition, at most 64 rows from a group
may contribute. Deterministic selection first preserves available class and
attack-family strata, then fills the remaining allowance by hash order.

Groups are selected in a deterministic round-robin across domain and attack
family. The row budget is charged by the capped retained contribution, not the
original size of a group. Target budgets remain 250,000 training rows and
100,000 calibration rows per source domain, but Version 3 should draw those rows
from many more distinct groups. The cap never applies to held-out evaluation.

## Fixed feature representations

All candidates exclude IP addresses, ports, and TTL-related shortcut fields such
as `sttl` and `dttl`. The base representation contains:

`PROTOCOL`, `L7_PROTO`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`,
`TCP_FLAGS`, and `FLOW_DURATION_MILLISECONDS`.

The derived representation adds fixed totals, average packet sizes,
in/out asymmetry, bytes per millisecond, and packets per millisecond. These
features use only the eight base fields. Any learned clipping, imputation,
encoding, or scaling is fitted on source-training rows only.

## Fixed candidates

1. `group_capped_lr_base_features` — matched Logistic Regression reference.
2. `group_capped_lr_derived_features` — linear derived-feature candidate.
3. `group_capped_rf_derived_features` — regularised nonlinear candidate.

XGBoost and additional parameter searches are outside this protocol. This keeps
the experiment focused on group influence and representation rather than model
shopping.

## Threshold and promotion rules

Thresholds from 0.05 to 0.95 are evaluated only on source calibration data. A
feasible threshold must keep false-positive rate at or below 0.20 in every
source domain and recall at or above 0.20 for every supported attack/domain
stratum. Among feasible thresholds, the rule maximises minimum source-domain
MCC using the registered tie breaks.

A derived-feature candidate can be selected only if all four folds pass the
source threshold constraints and every held-out domain has:

- row-weighted balanced accuracy above 0.50;
- row-weighted false-positive rate at most 0.25;
- recall of at least 0.20 for every supported attack/domain stratum; and
- equal-group-weight MCC above zero.

It must also improve macro row-weighted MCC over the matched reference by at
least 0.03 and worst-domain balanced accuracy by at least 0.02, without raising
maximum false-positive rate by more than 0.02. Otherwise the fixed verdict is
`no Version 3 candidate selected`.

## Overfitting and claim boundary

Version 3 will compare training, calibration, and held-out results; row-weighted
and equal-group-weight metrics; per-attack recall; predicted-positive rates;
score separation; group-size performance; and concentration before and after
the cap. These checks are designed to expose apparent gains caused by repeated
rows or near-universal attack predictions.

Even if a candidate passes, it remains a development-promising research
prototype. A real-world reliability claim requires a genuinely new dataset or
capture plus operational validation. The machine-readable source of truth is
`protocols/v3_group_balanced_development_protocol.json`.
