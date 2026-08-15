# AI-Enabled Predictive Maintenance: Predicting Industrial Machine Failure

MBA Working with AI (WAI) assignment — Data Mining & Predictive Analytics, IIM Ranchi.

Predicts machine failure risk from operating conditions (air/process temperature, rotational speed,
torque, tool wear, machine type) to help maintenance managers prioritize proactive interventions.

## Repository contents

| File | Purpose |
|---|---|
| `core.py` | Shared analytical engine (inspection, preprocessing, models, synthetic-data generation, risk logic). Used identically by the notebook and the app — no duplicated logic. |
| `PM_Analytics_Workflow.ipynb` | Full 35-section analysis notebook: EDA → 4 models → evaluation → model selection → synthetic prediction dataset → risk categorization → dashboard. Pre-executed; outputs saved under `outputs/`. |
| `app.py` | Streamlit application (10 tabs) — deployable directly on Streamlit Community Cloud. |
| `training_dataset.csv` | Historical training data (AI4I2020-style: UDI, Product ID, Type, Air/Process temperature, Rotational speed, Torque, Tool wear, Machine failure, TWF/HDF/PWF/OSF/RNF). |
| `requirements.txt` | Python dependencies. |
| `outputs/` | All saved figures, tables, and CSVs from the notebook run. |

## Only one CSV is required

No separate prediction file is needed. After model training/selection, the app and notebook
automatically generate a **synthetic unseen prediction dataset** (~2,000 rows) from the statistical
characteristics of the training data, score it with the finalized model, and produce risk-tiered,
recommendation-ready output — clearly labelled as synthetic/demonstration data (see Limitations).

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (include `app.py`, `core.py`, `training_dataset.csv`, `requirements.txt`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. No secrets/config needed — the app loads `training_dataset.csv` from the repo automatically.

## Running the notebook locally

```bash
pip install -r requirements.txt jupyter
jupyter nbconvert --to notebook --execute --inplace PM_Analytics_Workflow.ipynb
```

## Methodology at a glance

- **Target:** `Machine failure` (binary). **Excluded as leakage:** `TWF, HDF, PWF, OSF, RNF` (failure-mode
  diagnostics, not independent pre-failure conditions). **Excluded as identifiers:** `UDI, Product ID`.
- **Engineered features:** `Temperature_Diff_K` (Process − Air temperature), `Mechanical_Power_W`
  (Torque × angular velocity).
- **Models:** Logistic Regression, Decision Tree, Random Forest, ANN (MLP) — all inside a leakage-safe
  `Pipeline`, tuned via `GridSearchCV` (5-fold stratified CV, ROC-AUC scoring), class imbalance handled
  via `class_weight='balanced'`.
- **Model selection:** business-oriented — ranked by **Recall** first (false negatives are the costliest
  error in predictive maintenance), then ROC-AUC, then F1. The winning model is *not* assumed in advance;
  see `outputs/model_results/model_selection_summary.json` for the actual result of this run.
- **Risk tiers:** Low (<30% predicted failure probability), Medium (30–60%), High (≥60%), each mapped to
  a managerial (not engineering-safety) recommendation.

## Limitations

The synthetic prediction dataset is a demonstration of model application, generated from training-data
statistics — it is not independently collected, real-world future data. See Section 32 of the notebook
/ Tab 10 of the app for the full discussion of data, model, bias, privacy and responsible-AI limitations.
