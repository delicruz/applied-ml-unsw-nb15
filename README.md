# Explainable ML for Suspicious Network Traffic

This project builds a reproducible binary classifier for the UNSW-NB15 dataset. It compares a logistic-regression baseline with a Random Forest, evaluates precision, recall, F1 and PR-AUC, examines decision thresholds, and explains the Random Forest globally and locally with feature importance and SHAP.

## Design choices

- The supplied training split is used only for fitting; the supplied test split remains untouched until evaluation.
- `label` is the binary target: `0` is Normal and `1` is Suspicious.
- `id` and `attack_cat` are excluded from model inputs. `attack_cat` directly reveals whether a row is normal, so using it would leak the answer. It is retained only for error analysis.
- Numerical values use median imputation. Categorical values use most-frequent imputation and one-hot encoding with unknown-category handling.
- Logistic Regression is the interpretable baseline; Random Forest is the main nonlinear model.
- The default operating threshold is 0.50, with a separate table showing alternatives from 0.10 to 0.90.

## Setup

From PowerShell in this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data\raw
Expand-Archive -LiteralPath 'C:\Users\Dung\Downloads\OneDrive_2_8-20-2026.zip' -DestinationPath data\raw
```

## Run

Run the complete analysis:

```powershell
.\.venv\Scripts\python.exe run_project.py
```

For a quick development check:

```powershell
.\.venv\Scripts\python.exe run_project.py --train-limit 10000 --test-limit 5000 --rf-trees 50 --shap-samples 50
```

To omit SHAP or run only one stage:

```powershell
.\.venv\Scripts\python.exe run_project.py --skip-shap
.\.venv\Scripts\python.exe run_project.py --mode audit
.\.venv\Scripts\python.exe run_project.py --mode train
```

## Outputs

Outputs are written to `artifacts/`:

- `audit/dataset_summary.csv`, feature types, and attack-category counts
- EDA, confusion matrix, precision-recall, threshold, feature-importance, and SHAP plots
- `metrics.json` and `model_comparison.csv`
- `threshold_analysis.csv`
- `false_positives.csv`, `false_negatives.csv`, and errors by attack category
- three local explanation cases when available
- `random_forest_pipeline.joblib`, containing preprocessing and model together

The generated CSVs and plots are intended to feed directly into the Dataset, EDA, Results, Explainability, Error Analysis, and Discussion sections of the final report.
