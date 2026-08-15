"""
core.py
=======
AI-Enabled Predictive Maintenance — shared analytical engine.

This module contains every reusable function used by:
  1. The Jupyter Notebook (PM_Analytics_Workflow.ipynb)
  2. The Streamlit application (app.py)

Keeping the logic in one place guarantees that the notebook and the deployed
app use IDENTICAL preprocessing, modelling and synthetic-data-generation
logic — no duplicated / drifting code, no fabricated numbers.

Nothing in this file is hard-coded from a "known" AI4I2020 result — every
number is (re)computed from whatever CSV is handed to these functions.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------
# 1. DATASET INSPECTION
# --------------------------------------------------------------------------

LIKELY_LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LIKELY_ID_COLS = ["UDI", "Product ID", "ProductID", "Id", "ID"]
LIKELY_TARGET_NAMES = ["machine failure", "target", "failure", "class"]


def _norm(s):
    return str(s).strip().lower().replace("_", " ")


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Automatically profile the uploaded training dataset. Returns a dict
    with everything the rest of the pipeline needs, and nothing fabricated —
    every value is computed directly from `df`."""

    summary = {}
    summary["n_rows"] = int(df.shape[0])
    summary["n_cols"] = int(df.shape[1])
    summary["columns"] = list(df.columns)
    summary["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    summary["missing_values"] = {c: int(v) for c, v in df.isnull().sum().items() if v > 0}
    summary["duplicate_rows"] = int(df.duplicated().sum())
    summary["n_unique"] = {c: int(df[c].nunique()) for c in df.columns}

    # --- identify target column ---
    target_col = None
    for c in df.columns:
        if _norm(c) in LIKELY_TARGET_NAMES:
            target_col = c
            break
    if target_col is None:
        # fallback: any binary 0/1 column not in leakage list
        for c in df.columns:
            vals = set(pd.unique(df[c].dropna()))
            if vals.issubset({0, 1}) and c not in LIKELY_LEAKAGE_COLS:
                target_col = c
                break
    summary["target_col"] = target_col

    # --- identify id columns present ---
    id_cols = [c for c in df.columns if _norm(c).replace(" ", "") in
               [_norm(x).replace(" ", "") for x in LIKELY_ID_COLS]]
    summary["id_cols"] = id_cols

    # --- identify leakage columns present ---
    leak_cols = [c for c in df.columns if c in LIKELY_LEAKAGE_COLS]
    summary["leakage_cols"] = leak_cols

    # --- categorical / numerical predictors ---
    candidate_predictors = [c for c in df.columns
                             if c != target_col and c not in id_cols and c not in leak_cols]
    cat_cols = [c for c in candidate_predictors if df[c].dtype == object or df[c].nunique() <= 6]
    # numeric columns that are actually categorical-coded (e.g. 'Type' as letter) already caught above
    num_cols = [c for c in candidate_predictors if c not in cat_cols]

    summary["categorical_predictors"] = cat_cols
    summary["numerical_predictors"] = num_cols
    summary["final_predictors"] = cat_cols + num_cols

    # --- class distribution ---
    if target_col is not None:
        vc = df[target_col].value_counts()
        summary["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
        summary["failure_pct"] = float(df[target_col].mean() * 100)

    return summary


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, air_col=None, process_col=None,
                       torque_col=None, speed_col=None) -> pd.DataFrame:
    """Adds business-justified engineered features:
       - Temperature_Diff_K       = Process Temperature - Air Temperature
       - Mechanical_Power_W       = Torque [Nm] * Rotational speed [rad/s]
    Both are computed identically for training data AND later for the
    synthetic prediction dataset, guaranteeing pipeline reproducibility.
    """
    df = df.copy()

    def find(colnames, keys):
        for c in colnames:
            cl = _norm(c)
            if all(k in cl for k in keys):
                return c
        return None

    air_col = air_col or find(df.columns, ["air", "temp"])
    process_col = process_col or find(df.columns, ["process", "temp"])
    torque_col = torque_col or find(df.columns, ["torque"])
    speed_col = speed_col or find(df.columns, ["rotational", "speed"])

    if air_col and process_col:
        df["Temperature_Diff_K"] = df[process_col] - df[air_col]
    if torque_col and speed_col:
        # Power (Watts) = Torque (Nm) x Angular velocity (rad/s); rpm -> rad/s = rpm * 2*pi/60
        df["Mechanical_Power_W"] = df[torque_col] * (df[speed_col] * 2 * np.pi / 60)

    return df


ENGINEERED_FEATURES_DOC = {
    "Temperature_Diff_K": "Process Temperature minus Air Temperature (Kelvin). A larger "
                           "gap indicates the process is generating more heat than it is "
                           "dissipating to ambient air, which can be an early indicator of "
                           "thermal stress on the machine. Derived purely from operating "
                           "conditions, no leakage risk.",
    "Mechanical_Power_W": "Approximate mechanical power delivered by the spindle "
                           "(Torque x Angular Velocity, in Watts). Captures the combined "
                           "mechanical loading of the tool instead of torque and speed "
                           "separately, which is operationally meaningful (very low or very "
                           "high power draw both signal abnormal operation). No leakage risk."
}


# --------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# --------------------------------------------------------------------------

def build_preprocessor(numerical_cols, categorical_cols):
    numeric_transform = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transform = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transform, numerical_cols),
        ("cat", categorical_transform, categorical_cols),
    ])
    return preprocessor


# --------------------------------------------------------------------------
# 4. MODEL DEFINITIONS
# --------------------------------------------------------------------------

def get_model_definitions():
    """Returns dict of {name: (estimator, param_grid)} — modest search
    spaces to keep tuning cost reasonable."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
            {"clf__C": [0.01, 0.1, 1, 10]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
            {"clf__max_depth": [4, 6, 8, 10], "clf__min_samples_leaf": [5, 10, 20]}
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [8, 12, None],
             "clf__min_samples_leaf": [1, 5]}
        ),
        "Artificial Neural Network": (
            MLPClassifier(random_state=RANDOM_SEED, max_iter=500, early_stopping=True,
                           hidden_layer_sizes=(32, 16)),
            {"clf__alpha": [0.0001, 0.001, 0.01]}
        ),
    }
    return models


def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor, cv_folds=5):
    """Trains all four models with light GridSearchCV tuning (ROC-AUC scoring),
    evaluates on the held-out test set, and returns a results dict.
    NOTHING here touches synthetic data."""

    results = {}
    fitted_pipelines = {}
    cv_scores_record = {}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    for name, (estimator, grid) in get_model_definitions().items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=skf, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        fitted_pipelines[name] = best_pipe

        # cross-val scores of the best estimator (re-scored for reporting)
        cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_scores_record[name] = {"mean": float(cv_scores.mean()), "std": float(cv_scores.std()),
                                   "scores": [float(s) for s in cv_scores]}

        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "best_params": gs.best_params_,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    return results, fitted_pipelines, cv_scores_record


def select_best_model(results: dict) -> str:
    """Business-oriented selection: rank primarily by Recall (missed
    failures are the costliest error in predictive maintenance), using
    ROC-AUC and F1 as tie-breakers. This is NOT a fixed model name — it is
    computed from whatever `results` contains."""
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["recall"], kv[1]["roc_auc"], kv[1]["f1"]),
        reverse=True
    )
    return ranked[0][0]


def get_feature_importance(pipeline, model_name, numerical_cols, categorical_cols):
    """Returns a DataFrame of feature -> importance/coefficient for models
    that support it (Random Forest, Decision Tree, Logistic Regression)."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    try:
        cat_features = list(prep.named_transformers_["cat"].named_steps["onehot"]
                             .get_feature_names_out(categorical_cols))
    except Exception:
        cat_features = []
    feature_names = list(numerical_cols) + cat_features

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
    else:
        return None

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 5. SYNTHETIC UNSEEN PREDICTION DATASET
# --------------------------------------------------------------------------

def generate_synthetic_prediction_data(train_df: pd.DataFrame, numerical_cols, categorical_cols,
                                        id_col_name="Product ID", n_samples=2000, seed=RANDOM_SEED):
    """Generates a NEW synthetic 'unseen' prediction dataset using an
    empirical / joint bootstrap-with-jitter approach: rows of the raw
    numerical predictors are jointly resampled (with replacement) from the
    training data to preserve realistic correlations (e.g. torque vs speed),
    then a small amount of Gaussian jitter is added to each numeric value so
    that observations are NEW rather than exact copies of training rows.
    Categorical variables are sampled independently, respecting training
    proportions. No target / leakage columns are ever touched."""

    rng = np.random.default_rng(seed)

    # 1. Jointly resample numeric rows (preserves relationships among numeric vars)
    idx = rng.integers(0, len(train_df), size=n_samples)
    base_numeric = train_df[numerical_cols].iloc[idx].reset_index(drop=True)

    jittered = base_numeric.copy()
    for col in numerical_cols:
        std = train_df[col].std()
        jitter = rng.normal(loc=0, scale=0.02 * std, size=n_samples)  # 2% of std -> small realistic noise
        jittered[col] = base_numeric[col].values + jitter
        # clip to realistic training-data min/max to avoid impossible values
        jittered[col] = jittered[col].clip(train_df[col].min(), train_df[col].max())

    # 2. Sample categorical vars independently respecting training proportions
    synth = jittered.copy()
    for col in categorical_cols:
        probs = train_df[col].value_counts(normalize=True)
        synth[col] = rng.choice(probs.index, size=n_samples, p=probs.values)

    # 3. Generate identifiers
    synth.insert(0, "Synthetic_Machine_ID", [f"SYN-{i+1:05d}" for i in range(n_samples)])
    if id_col_name:
        letters = synth[categorical_cols[0]] if categorical_cols else "X"
        synth.insert(1, id_col_name, [f"{letters.iloc[i] if categorical_cols else 'X'}{rng.integers(10000,99999)}"
                                       for i in range(n_samples)])

    return synth


# --------------------------------------------------------------------------
# 6. RISK CATEGORIZATION + RECOMMENDATIONS
# --------------------------------------------------------------------------

def categorize_risk(prob, low_thresh=0.30, high_thresh=0.60):
    if prob >= high_thresh:
        return "High"
    elif prob >= low_thresh:
        return "Medium"
    else:
        return "Low"


RECOMMENDATIONS = {
    "High": "Priority inspection recommended. Schedule a maintenance assessment, "
            "review current operating conditions, and consider proactive intervention "
            "before the next production run.",
    "Medium": "Increase monitoring frequency. Schedule a routine inspection and keep "
               "watch on key operating variables (torque, tool wear, temperature differential).",
    "Low": "Continue routine monitoring. Follow the standard preventive-maintenance schedule; "
           "no immediate action indicated by current operating conditions.",
}
