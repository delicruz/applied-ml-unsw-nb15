# Final Research-Prototype Evidence Manifest

This manifest connects the consolidated status to the primary machine-readable
evidence. The SHA-256 values are also stored in
`artifacts/final_research_prototype/evidence_manifest.json` and checked by the
repository test suite.

| Evidence | Purpose | SHA-256 |
|---|---|---|
| `artifacts/robustness_audit/robustness_metrics.csv` | TTL-removed internal validation | `70f60ee08b1a270cfc1accb6a861396affbe664d6c6d694350f7497613188197` |
| `artifacts/final_holdout_evaluation/summary.json` | Conservative UNSW-NB15 holdout | `83e82e96c5c65a99ba776b7b1f7528c8b9397b3cb53bea61c64ff5638fa8ef72` |
| `artifacts/external_netflow_validation/summary.json` | NF-ToN-IoT transfer | `f7aa07d7b8ea618e0b93471bc7273055fec1c9e7ae4e3b7ec702e636ac744579` |
| `artifacts/third_dataset_validation/summary.json` | NF-BoT-IoT transfer | `52f93435c59c9babad2081ced911a2ad30447638c9975d04d3d3c2bc5da16e2c` |
| `artifacts/fresh_confirmation_evaluation/summary.json` | NF-CSE sealed confirmation | `23f4bee33042209a9126e6ae516c2c9ccf9fab4f7c2a73f8ef361f2d317cab96` |
| `artifacts/v2_generalisation_development/summary.json` | V2 multidomain verdict | `ce1ca9c9e8cb58c672803889bf7df50b2a1ce69ee035f61c63c7df6956d25e4f` |
| `artifacts/v3_group_balanced_development/summary.json` | V3 group-balanced verdict | `eac836ff6bdde20c8dccfb577a2bfff4c89f40cf2ba1637e8bfa33a90d8c45ef` |
| `protocols/v2_generalisation_development_protocol.json` | Frozen V2 design | `ee3aa81854b74da7ed52eaea8ee43c3aa113a64a5745d6ba720e147bf0670bea` |
| `protocols/v3_group_balanced_development_protocol.json` | Frozen V3 design | `0d0f089b811aea1fcce629eae0ac285d63db4a22daaf032cd7a59d214595b410` |

## Interpretation boundary

The numerical claims in the final status are supported by local artifacts.
The conclusion that cross-domain and attack-composition shift is the primary
remaining problem is an inference from the repeated transfer failures, score
reversals, per-attack results and V3 calibration-to-held-out gaps. It is not a
claim that one unique causal mechanism has been proven.

All four V2/V3 domains were already opened before those protocols. Their results
are development evidence, not independent confirmation. The package therefore
uses `ANALYZED`, not `VERIFIED`, as its verification status.
