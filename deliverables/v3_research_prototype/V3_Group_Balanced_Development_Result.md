# Version 3 Group-Balanced Development Result

**Run date:** 27 October 2026, Australia/Adelaide

**Status:** completed once with exit code 0; no retry

**Fixed verdict:** `no Version 3 candidate selected`

## What was tested

Version 3 capped every source training and calibration group at 64 retained
rows, added deterministic traffic-rate and directionality features, and compared
three fixed candidates across four leave-one-dataset-out development folds. The
held-out domain never contributed to preprocessing, fitting, or threshold
selection, and every held-out row was scored.

All four domains were previously opened. This is development evidence, not an
untouched test or independent confirmation result.

## Did the group cap work?

Yes. The cap changed the sampling concentration as intended:

| Domain | Training rows | Training groups | Largest original group | Largest retained group |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15 | 250,009 | 107,392 | 37,156 | 64 |
| NF-ToN-IoT | 250,001 | 79,751 | 9,341 | 64 |
| NF-BoT-IoT | 99,257 | 18,907 | 326,232 | 64 |
| NF-CSE-CIC-IDS2018 | 250,019 | 32,845 | 432,393 | 64 |

NF-BoT-IoT did not contain enough distinct retained group capacity to reach the
nominal 250,000-row target. The runner used all available capped capacity rather
than allowing large groups to dominate.

## Aggregate results

| Candidate | Macro MCC | Worst BA | Maximum FPR | Worst equal-group MCC | Threshold constraints in all folds | Eligible |
|---|---:|---:|---:|---:|---|---|
| Base Logistic Regression | -0.0810 | 0.3077 | 0.9586 | -0.4714 | No | No |
| Derived Logistic Regression | 0.0256 | 0.4751 | 0.8569 | -0.1624 | No | No |
| Derived Random Forest | -0.0330 | 0.2963 | 0.5292 | -0.2347 | No | No |

The derived Logistic Regression improved macro MCC by 0.1066 and worst-domain
balanced accuracy by 0.1674 over the matched base-feature reference. It also
reduced maximum FPR by 0.1017. It therefore passed the relative improvement
part of the rule, but it failed the absolute eligibility gates.

## Why no model was selected

The derived Logistic Regression still had:

- worst-domain balanced accuracy of 0.4751, below the required 0.50;
- maximum held-out FPR of 0.8569, far above the required 0.25;
- worst-domain equal-group MCC of -0.1624, below the required zero;
- zero recall for multiple supported NF-CSE attack families; and
- source threshold constraints that did not pass in all four folds.

No candidate passed source threshold constraints in all folds. Only the derived
Random Forest in the NF-CSE-held-out fold found a feasible source threshold, but
that fold still had held-out MCC -0.2663 and balanced accuracy 0.2963.

## Overfitting interpretation

The cap successfully removed massive repeated-group dominance, but it did not
solve cross-domain generalisation. Training-to-calibration MCC gaps were near
zero for all three candidates, while mean calibration-to-held-out MCC gaps were
0.8286 for the base Logistic Regression, 0.7238 for the derived Logistic
Regression, and 0.9180 for the Random Forest.

This pattern is not mainly ordinary memorisation of the source training rows:
performance transfers from source training to source calibration, then collapses
on another dataset. The remaining problem is substantial domain and attack-family
shift. Adding more tuning on these same four domains would risk adapting to the
known development domains without establishing real-world reliability.

## Conclusion

Version 3 produced a useful negative result. Group-capped sampling is a stronger
and less duplicate-dominated evaluation design, and the derived representation
improved the matched linear reference. However, the absolute performance and
attack coverage remain unacceptable. No V3 model should be presented as usable
or deployment-ready.

The machine-readable result is in
`artifacts/v3_group_balanced_development/summary.json`.
