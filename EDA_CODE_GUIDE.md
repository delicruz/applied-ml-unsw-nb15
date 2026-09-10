# EDA code guide

The milestone code is in `eda_analysis.py`. It deliberately stops before model training.

## What each section demonstrates

1. **Load and validate** checks that both files exist, confirms the binary target, and verifies that `label` agrees with `attack_cat`.
2. **Data-quality audit** measures missing values, non-finite numbers, exact duplicate rows, repeated predictor vectors, feature types, and category shift between training and testing.
3. **Target analysis** compares Normal and Suspicious traffic across the supplied splits.
4. **Attack coverage** shows whether the nine attack types have enough observations and reveals rare classes.
5. **Categorical analysis** examines the most common protocol, service, and connection-state values.
6. **Numerical analysis** compares selected traffic measurements by class after a display-only `log(1+x)` transform. The raw data remains unchanged.
7. **Outlier analysis** uses the IQR rule as a diagnostic. It does not automatically delete extreme network traffic because attacks may produce legitimate extremes.
8. **Correlation analysis** identifies candidate relationships and redundancy. Correlation is not used as proof of feature importance.

## Run it

```powershell
.\.venv\Scripts\python.exe eda_analysis.py
```

Results are written to `artifacts/eda/` as CSV tables and PNG figures.

## Preparation decisions supported by the EDA

- Exclude `id` because it identifies records rather than traffic behaviour.
- Exclude `attack_cat` from future model inputs because it directly reveals the binary target.
- Keep `attack_cat` for descriptive and error analysis.
- Keep `service = "-"` as an explicit category rather than interpreting it as a missing value.
- Use one-hot encoding with unknown-category handling because `ACC` and `CLO` appear only in testing.
- Fit any future preprocessing only on training data.
- Investigate repeated predictor vectors with a sensitivity analysis rather than deleting them without evidence.
- Avoid PCA at this stage because feature-level interpretation is central to the project.
