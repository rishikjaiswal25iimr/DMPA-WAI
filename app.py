"""
app.py — AI-Enabled Predictive Maintenance
Streamlit application (deployable via GitHub + Streamlit Community Cloud).

Automatically loads training_dataset.csv from the repo, runs the full
analytical workflow (via core.py — the SAME engine used by the notebook),
generates a synthetic unseen prediction dataset, scores it, and displays
everything through interactive tabs. No second CSV upload required.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import core

st.set_page_config(page_title="AI-Enabled Predictive Maintenance", layout="wide", page_icon="🛠️")
sns.set_style("whitegrid")

TRAIN_FILE = "training_dataset.csv"
N_SYNTHETIC = 2000


# --------------------------------------------------------------------------
# Cached pipeline — runs once per deployment session, not on every click
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(TRAIN_FILE)
    return df


@st.cache_resource(show_spinner="Training and evaluating models (Logistic Regression, "
                                 "Decision Tree, Random Forest, ANN) — this runs once...")
def run_full_workflow():
    df_raw = load_data()
    summary = core.inspect_dataset(df_raw)
    df_fe = core.engineer_features(df_raw)

    num_cols = summary["numerical_predictors"] + ["Temperature_Diff_K", "Mechanical_Power_W"]
    cat_cols = summary["categorical_predictors"]
    target = summary["target_col"]

    X = df_fe[num_cols + cat_cols]
    y = df_fe[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=core.RANDOM_SEED)

    preprocessor = core.build_preprocessor(num_cols, cat_cols)
    results, fitted_pipelines, cv_scores_record = core.train_and_evaluate_models(
        X_train, y_train, X_test, y_test, preprocessor, cv_folds=5)

    best_model_name = core.select_best_model(results)
    final_pipeline = fitted_pipelines[best_model_name]
    fi = core.get_feature_importance(final_pipeline, best_model_name, num_cols, cat_cols)

    # Synthetic prediction data — generated ONLY after model selection
    synthetic_df = core.generate_synthetic_prediction_data(
        df_fe, summary["numerical_predictors"], cat_cols,
        id_col_name="Product ID", n_samples=N_SYNTHETIC, seed=core.RANDOM_SEED)
    synthetic_df_fe = core.engineer_features(synthetic_df)
    X_synth = synthetic_df_fe[num_cols + cat_cols]
    synthetic_df_fe["Failure_Probability"] = final_pipeline.predict_proba(X_synth)[:, 1]
    synthetic_df_fe["Predicted_Class"] = np.where(
        synthetic_df_fe["Failure_Probability"] >= 0.5, "Failure", "No Failure")
    synthetic_df_fe["Risk_Category"] = synthetic_df_fe["Failure_Probability"].apply(core.categorize_risk)
    synthetic_df_fe["Maintenance_Recommendation"] = synthetic_df_fe["Risk_Category"].map(core.RECOMMENDATIONS)

    return {
        "df_raw": df_raw, "df_fe": df_fe, "summary": summary,
        "num_cols": num_cols, "cat_cols": cat_cols, "target": target,
        "X_test": X_test, "y_test": y_test,
        "results": results, "cv_scores_record": cv_scores_record,
        "best_model_name": best_model_name, "final_pipeline": final_pipeline,
        "feature_importance": fi, "synthetic_df": synthetic_df_fe,
    }


with st.spinner("Loading data and running the analytical workflow..."):
    W = run_full_workflow()

df_raw, df_fe, summary = W["df_raw"], W["df_fe"], W["summary"]
results, best_model_name = W["results"], W["best_model_name"]
synthetic_df = W["synthetic_df"]
best = results[best_model_name]

st.title("🛠️ AI-Enabled Predictive Maintenance Dashboard")
st.caption("Predicting industrial machine failure to support proactive maintenance decisions "
           "— MBA Working with AI (WAI) Project, Data Mining & Predictive Analytics")

tabs = st.tabs([
    "1. Overview", "2. Dataset & Quality", "3. EDA", "4. Model Development",
    "5. Model Evaluation", "6. Feature Importance", "7. Synthetic Prediction & Risk",
    "8. Maintenance Recommendations", "9. Executive Dashboard", "10. Methodology & Limitations",
])

# ---------------------------------------------------------------- 1
with tabs[0]:
    st.header("Project Overview")
    st.markdown("""
**Business question:** Can machine operating conditions predict the probability of machine failure,
and how can these predictions help maintenance managers prioritize proactive maintenance interventions?

**Workflow:** Training data → Data preparation → EDA → Feature engineering → Four ML models →
Model evaluation → Model selection (business/recall-oriented) → Synthetic unseen prediction dataset →
Final prediction → Risk categorization → Maintenance recommendations → Dashboard.

Only the training dataset (`training_dataset.csv`) is used to train and select the model. A synthetic
"unseen" prediction dataset is generated automatically **after** model selection is finalized — no
second CSV upload is required.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training observations", f"{summary['n_rows']:,}")
    c2.metric("Historical failure rate", f"{summary['failure_pct']:.2f}%")
    c3.metric("Recommended model", best_model_name)

# ---------------------------------------------------------------- 2
with tabs[1]:
    st.header("Dataset & Data Quality")
    st.write(f"**Target column detected:** `{summary['target_col']}`")
    st.write(f"**Identifier columns excluded from modelling:** {summary['id_cols']}")
    st.write(f"**Leakage columns excluded from modelling:** {summary['leakage_cols']} "
             f"(specific failure-mode diagnostics — not independent pre-failure operating conditions)")
    st.write(f"**Final predictors used:** {summary['final_predictors']} "
             f"+ engineered: `Temperature_Diff_K`, `Mechanical_Power_W`")

    dq = pd.DataFrame({
        "dtype": df_raw.dtypes.astype(str),
        "n_missing": df_raw.isnull().sum(),
        "n_unique": df_raw.nunique(),
    })
    st.dataframe(dq, use_container_width=True)
    st.write(f"Duplicate rows: {df_raw.duplicated().sum()}")
    st.dataframe(df_raw.head(20), use_container_width=True)

# ---------------------------------------------------------------- 3
with tabs[2]:
    st.header("Exploratory Data Analysis")
    target = summary["target_col"]

    col1, col2 = st.columns(2)
    with col1:
        vc = df_fe[target].value_counts().sort_index()
        fig = px.bar(x=["No Failure", "Failure"], y=vc.values,
                     labels={"x": "Outcome", "y": "Count"}, title="Target Distribution",
                     color=["No Failure", "Failure"],
                     color_discrete_map={"No Failure": "#4C72B0", "Failure": "#C44E52"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if summary["categorical_predictors"]:
            rate = df_fe.groupby(summary["categorical_predictors"][0])[target].mean() * 100
            fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Type", "y": "Failure rate (%)"},
                         title="Failure Rate by Machine Type")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df_fe, x=target, y="Tool wear [min]", title="Tool Wear by Outcome")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(df_fe, x=target, y="Torque [Nm]", title="Torque by Outcome")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df_fe[W["num_cols"] + [target]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------------- 4
with tabs[3]:
    st.header("Model Development & Comparison")
    st.markdown("Four classifiers trained inside a leakage-safe `Pipeline` "
                "(preprocessing fit on training data only), tuned with `GridSearchCV` "
                "(5-fold stratified CV, ROC-AUC scoring), class imbalance handled via "
                "`class_weight='balanced'`.")
    comparison = pd.DataFrame({
        name: {"Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
               "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
        for name, r in results.items()
    }).T.round(3)
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Cross-Validation (ROC-AUC)")
    cv_df = pd.DataFrame({name: {"CV Mean": v["mean"], "CV Std": v["std"]}
                           for name, v in W["cv_scores_record"].items()}).T.round(3)
    st.dataframe(cv_df, use_container_width=True)

    with st.expander("Best hyperparameters per model"):
        for name, r in results.items():
            st.write(f"**{name}**: {r['best_params']}")

# ---------------------------------------------------------------- 5
with tabs[4]:
    st.header("Model Evaluation")
    st.markdown("""In predictive maintenance a **False Negative** (predicted *No Failure*, machine
actually fails) is typically far costlier than a **False Positive**. Model selection therefore
prioritizes **Recall**, then ROC-AUC, then F1 — not accuracy alone.""")

    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(W["y_test"], r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Confusion Matrices")
    cols = st.columns(4)
    for c, (name, r) in zip(cols, results.items()):
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred:No", "Pred:Yes"], yticklabels=["True:No", "True:Yes"])
        ax.set_title(name, fontsize=8)
        c.pyplot(fig)

    st.success(f"**Selected model: {best_model_name}**  "
               f"(Recall={best['recall']:.3f}, ROC-AUC={best['roc_auc']:.3f}, F1={best['f1']:.3f}) "
               "— chosen by a business-oriented, recall-first rule computed from the results above.")

# ---------------------------------------------------------------- 6
with tabs[5]:
    st.header("Feature Importance")
    fi = W["feature_importance"]
    if fi is not None:
        fig = px.bar(fi.head(10), x="importance", y="feature", orientation="h",
                     title=f"Top Predictors — {best_model_name}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The model identifies these variables as important predictors of failure risk. "
                   "This reflects predictive association, not causation.")
    else:
        st.info("Feature importance is not available for this model type.")

# ---------------------------------------------------------------- 7
with tabs[6]:
    st.header("Synthetic Prediction & Risk Analysis")
    st.warning("⚠️ The dataset below is a **synthetic, demonstration** 'unseen' prediction set, "
               "generated programmatically from the statistical characteristics of the training data "
               "(joint bootstrap + jitter) — it is NOT independently collected real-world data. "
               "Predictions on it demonstrate model application, not validated future outcomes.")

    n_high = int((synthetic_df["Risk_Category"] == "High").sum())
    n_med = int((synthetic_df["Risk_Category"] == "Medium").sum())
    n_low = int((synthetic_df["Risk_Category"] == "Low").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total machines assessed", f"{len(synthetic_df):,}")
    c2.metric("High risk", n_high)
    c3.metric("Medium risk", n_med)
    c4.metric("Low risk", n_low)

    risk_type_filter = st.multiselect("Filter by risk category", ["Low", "Medium", "High"],
                                       default=["Low", "Medium", "High"])
    filtered = synthetic_df[synthetic_df["Risk_Category"].isin(risk_type_filter)]

    display_cols = (["Synthetic_Machine_ID", "Product ID", "Type"] + summary["numerical_predictors"] +
                     ["Failure_Probability", "Predicted_Class", "Risk_Category"])
    st.dataframe(filtered[display_cols].sort_values("Failure_Probability", ascending=False),
                 use_container_width=True, height=350)

    fig = px.histogram(synthetic_df, x="Failure_Probability", nbins=30,
                        title="Predicted Failure Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇️ Download synthetic prediction dataset (input, no target)",
                        data=synthetic_df.drop(columns=["Failure_Probability", "Predicted_Class",
                                                          "Risk_Category", "Maintenance_Recommendation"]
                                                ).to_csv(index=False),
                        file_name="synthetic_prediction_dataset.csv", mime="text/csv")

    full_output_cols = (["Synthetic_Machine_ID", "Product ID", "Type"] + summary["numerical_predictors"] +
                         ["Temperature_Diff_K", "Mechanical_Power_W", "Failure_Probability",
                          "Predicted_Class", "Risk_Category", "Maintenance_Recommendation"])
    st.download_button("⬇️ Download full predictions",
                        data=synthetic_df[full_output_cols].sort_values(
                            "Failure_Probability", ascending=False).to_csv(index=False),
                        file_name="final_machine_failure_predictions.csv", mime="text/csv")

    high_risk_df = synthetic_df[synthetic_df["Risk_Category"] == "High"][full_output_cols].sort_values(
        "Failure_Probability", ascending=False)
    st.download_button("⬇️ Download high-risk machine list",
                        data=high_risk_df.to_csv(index=False),
                        file_name="high_risk_machine_list.csv", mime="text/csv")

# ---------------------------------------------------------------- 8
with tabs[7]:
    st.header("Maintenance Recommendations")
    for tier, rec in core.RECOMMENDATIONS.items():
        color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[tier]
        st.markdown(f"### {color} {tier} Risk")
        st.write(rec)

    st.subheader("Top 15 Highest-Risk Machines")
    top15 = synthetic_df.sort_values("Failure_Probability", ascending=False).head(15)
    st.dataframe(top15[["Synthetic_Machine_ID", "Product ID", "Type", "Failure_Probability",
                         "Risk_Category", "Maintenance_Recommendation"]], use_container_width=True)

# ---------------------------------------------------------------- 9
with tabs[8]:
    st.header("Executive Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total machines assessed", f"{len(synthetic_df):,}")
    c2.metric("Predicted failure %", f"{(synthetic_df['Predicted_Class']=='Failure').mean()*100:.1f}%")
    c3.metric(f"Best model ROC-AUC ({best_model_name})", f"{best['roc_auc']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = synthetic_df["Risk_Category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     title="Predicted Risk Category Distribution",
                     color=risk_counts.index,
                     color_discrete_map={"Low": "#55A868", "Medium": "#DD8452", "High": "#C44E52"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        comp = pd.DataFrame({name: {"Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
                              for name, r in results.items()}).T.reset_index()
        comp = comp.melt(id_vars="index", var_name="Metric", value_name="Score")
        fig = px.bar(comp, x="index", y="Score", color="Metric", barmode="group",
                     title="Model Performance Comparison", labels={"index": "Model"})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 10
with tabs[9]:
    st.header("AI / Project Methodology & Limitations")
    st.markdown("""
**Methodology summary:** Stratified 80/20 train-test split; 5-fold stratified cross-validation;
`GridSearchCV` hyperparameter tuning (ROC-AUC scoring); class imbalance handled via
`class_weight='balanced'`; model selection ranked by Recall → ROC-AUC → F1 (business priority:
minimize missed failures). The synthetic prediction dataset was generated **only after** model
training, tuning, evaluation and selection were complete, and never influenced any of those steps.

**Data limitations:** single historical snapshot; representativeness of other plants/environments
cannot be verified from this dataset alone.

**Synthetic prediction-data limitation:** generated from training-data statistics (joint bootstrap +
jitter) — demonstrates model application, not independently observed real-world future failures.

**AI limitations:** AI-generated code may contain errors and requires human validation before
production use.

**Model limitations:** predictions are probabilistic; feature importance is correlational, not causal;
performance may not transfer to different machines/sites without retraining.

**Bias:** class imbalance and the synthetic-sampling approach may under/over-represent certain
operating regimes.

**Privacy:** no personally identifiable information is present — only machine/product IDs and sensor
readings.

**Responsible AI:** this is a decision-support tool. Maintenance actions should remain subject to
human engineering judgement and validation.
""")

st.divider()
st.caption("AI-Enabled Predictive Maintenance — MBA WAI Project, IIM Ranchi. "
           "Built with scikit-learn, pandas, Streamlit and Plotly. "
           "Notebook and app share identical logic via `core.py`.")
