# Fresh Sealed-Confirmation Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-21
- Verification Status: UNVERIFIED
- Version Label: fresh_confirmation_protocol_v1

## Purpose

This protocol freezes the candidate selected on NF-CSE-CIC-IDS2018 development
data and defines the one-time evaluation of the dataset's untouched confirmation
partition. It was written after development results were known but before any
confirmation prediction, metric, or label summary was computed.

The confirmation asks two separate questions:

1. Does the aggregate improvement over the unchanged source Random Forest
   replicate on the sealed partition?
2. Does the selected model achieve broad recall across supported attack labels
   without an excessive Normal-traffic false-positive rate?

## Frozen inputs

| Item | Frozen value |
|---|---|
| Selected candidate | Target-trained Random Forest, all eight features |
| Candidate SHA-256 | `a39a4c1355f638ed526bbe0004c6d9bfdc4cab6976081d73b3bd9b89bba54743` |
| Candidate threshold | 0.39 |
| Comparison baseline | Unchanged source-trained Random Forest |
| Baseline SHA-256 | `38756c9b09be364b5ff483df82b98183634753ef326faafeaaa8ebcfd498646b` |
| Baseline threshold | 0.64 |
| Features | `PROTOCOL`, `L7_PROTO`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS` |
| Confirmation rows | 1,631,543 |
| Confirmation digest | `2eb47d7a15380c5d9e18c9544ca099a0a50d2482128ea3e74a196e88a1bf40b2` |

No refitting, threshold adjustment, feature change, candidate reselection, row
exclusion, or confirmation sampling is permitted.

## Analysis

The runner evaluates every sealed-confirmation row. Uncertainty uses 1,000
paired percentile bootstrap samples of the registered five-significant-digit
feature groups. Candidate and baseline are evaluated on the same resampled
groups. Primary metrics are MCC and balanced accuracy. Average Precision and
its lift over prevalence, F1, precision, recall, specificity, false-positive
rate, Brier score, and every observed attack-label recall are also reported.

## Decision rules

Aggregate improvement is supported only when all conditions hold:

- candidate MCC is at least 0.60;
- candidate balanced accuracy is at least 0.80;
- candidate Average Precision lift is above 1.00;
- the paired 95% interval for candidate-minus-baseline MCC is above zero; and
- the paired 95% interval for candidate-minus-baseline balanced accuracy is
  above zero.

Broad supported-attack coverage is a separate verdict. It requires:

- candidate false-positive rate no greater than 0.10; and
- recall of at least 0.50 for every attack label represented by at least 100
  positive confirmation rows.

This separation prevents a strong aggregate result from hiding a repeat of the
development-stage Infiltration weakness. The registered final labels are:

- `aggregate improvement confirmed with broad supported-attack coverage`;
- `aggregate improvement confirmed; broad supported-attack coverage not supported`; or
- `development improvement not confirmed`.

No outcome permits a production-deployment claim.

## Integrity boundary

Before scoring, the runner must verify the complete dataset hash and official
counts, reconstruct confirmation membership, reproduce its row count and
ordered group digest, verify the earlier protocol and result hashes, and verify
both frozen model hashes and byte sizes. It writes a durable one-time execution
marker before reading the sealed rows and refuses to start when prior output is
present. Results must be reported regardless of outcome. A technical failure
must not be silently retried.

## Technical preflight amendment

The first execution command stopped during frozen-model deserialisation before
the confirmation dataset was opened. The development model had stored its
existing `log1p_nonnegative` transformer under Python's `__main__` namespace.
The confirmation runner now exposes that same imported function under the
legacy name before loading the unchanged artifact. A regression test loads both
frozen pipelines successfully. No model, hash, feature, threshold, partition,
metric, bootstrap procedure, or decision rule changed. The user explicitly
authorised a retry, recorded in
`../../protocols/fresh_confirmation_amendment_001.json`.

The authorised retry then stopped during the dataset-byte-hash preflight. No
CSV parsing, confirmation-row access, prediction, metric computation, marker,
or output occurred. The registered SHA-256 contained one extra `9`; the value
already recorded in the locked development summary and independently observed
for the unchanged 620,014,585-byte CSV is
`ed3387646fda24084889f07788cd22da2978b860f5b3d2d6f1210a4e3d7998e0`.
Amendment 002 corrects only that transcription error. All frozen inputs,
analysis choices, and decision rules remain unchanged, and the user explicitly
authorised the corrected retry. The complete audit record is stored in
`../../protocols/fresh_confirmation_amendment_002.json`.
