# Fresh Generalisation-Failure Audit Protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-26
- Verification Status: UNVERIFIED
- Version Label: fresh_generalisation_failure_audit_protocol_v1

## Purpose

This is a post-hoc diagnostic audit of the failed NF-CSE-CIC-IDS2018 sealed
confirmation. It asks why the frozen target Random Forest's performance changed
between the registered target-validation sample and the full confirmation
partition. Because confirmation results are already known, all findings are
hypothesis-generating and cannot validate a correction.

## Frozen scope

The audit reconstructs the exact 400,087-row sampled target-validation
partition and the complete 1,631,543-row confirmation partition. It reloads
the unchanged target Random Forest and threshold 0.39, then verifies that the
saved point metrics are reproduced before producing new diagnostics.

The registered diagnostic families are:

1. raw attack-label composition and recall change;
2. numeric and categorical feature-distribution shift;
3. prevalence-adjusted feature/label association drift;
4. frozen-model score-distribution change;
5. confirmation error rates in validation-derived feature bins; and
6. repeated-group size, concentration, label conflict, recall and false-positive
   profiles.

## Prohibited actions

The audit does not fit a model, select a feature, select a threshold, compare
new candidates, change a prediction rule, or support a deployment claim. No
diagnostic result may be presented as independent evidence that a version-2
correction works.

## Interpretation boundary

Observed differences may be consistent with attack-composition shift,
feature-distribution shift, feature/label association drift, repeated-group
concentration, or model-specific instability. They cannot establish that any
one mechanism is causal. A correction derived from this audit requires a new
development protocol and genuinely new confirmation data.
