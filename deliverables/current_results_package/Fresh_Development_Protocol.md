# Fresh Target-Development Protocol

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-20
- Verification Status: UNVERIFIED
- Version Label: target_dev_protocol_v1

## Purpose

The two completed external evaluations show that the source-trained models do
not transfer reliably. NF-ToN-IoT and NF-BoT-IoT must therefore remain
evaluation-only datasets and cannot be reused to select a repair. This protocol
designates the previously unused NF-CSE-CIC-IDS2018 NetFlow-v1 dataset for
target-development work.

The University of Queensland catalogue reports 8,392,401 flows, comprising
7,373,198 benign flows and 1,019,203 attacks. Its twelve-field NetFlow-v1
representation contains the same eight endpoint-free traffic features used by
the existing external models.

- Official catalogue: <https://staff.itee.uq.edu.au/marius/NIDS_datasets/>
- Official file: <https://rdm.uq.edu.au/files/650f1fa0-ef9c-11ed-b5f6-b1a04f482c13>
- Machine-readable protocol: `../../protocols/fresh_target_development_protocol.json`

## Leakage-safe split

Rows are grouped using exact categorical values and five-significant-digit
numeric values across the eight model features. A deterministic SHA-256 group
hash assigns complete groups to four partitions:

| Partition | Hash interval | Permitted use |
|---|---:|---|
| Target training | [0.00, 0.48) | Fit development candidates |
| Target calibration | [0.48, 0.64) | Select thresholds only |
| Target validation | [0.64, 0.80) | Compare the fixed candidates |
| Sealed confirmation | [0.80, 1.00) | No scoring in this development step |

The processing runner may count and hash confirmation groups while constructing
the split, but it must not compute confirmation predictions, class-specific
summaries, metrics, or plots. Development partitions may be deterministically
subsampled by whole group to cap local computation.

## Bounded candidates

The development comparison includes the unchanged source-trained Logistic
Regression and Random Forest, threshold-only recalibration of the frozen Random
Forest, target-developed Logistic Regression and Random Forest models using all
eight features, and one reduced Random Forest using `PROTOCOL`, `IN_BYTES`,
`OUT_BYTES`, `IN_PKTS`, and `OUT_PKTS`.

The reduced set is a pre-specified hypothesis derived from the completed
cross-external audit. It does not assume that the removed variables caused the
earlier failures. Model fitting is restricted to target training, threshold
selection to target calibration, and candidate comparison to target
validation. NF-ToN-IoT and NF-BoT-IoT are not used for selection.

## Development interpretation rule

A candidate is called development-promising only if its target-validation MCC
and balanced accuracy each exceed the unchanged frozen-source Random Forest by
at least 0.05. Even then, the result remains development evidence. It cannot be
described as externally confirmed until a later protocol freezes one candidate
and evaluates the still-sealed confirmation partition.
