"""
app.py — AI-Enabled Predictive Maintenance
Streamlit application (deployable via GitHub + Streamlit Community Cloud).

Automatically loads training_dataset.csv from the repo, runs the full
analytical workflow (via core.py — the SAME engine used by the notebook),
generates a synthetic unseen prediction dataset, scores it, and displays
everything through interactive tabs. No second CSV upload required.

NOTE: This file contains UI/presentation code only. All analytical logic
(model training, feature engineering, risk classification, recommendations)
lives in core.py and is untouched — this file only changes how results look.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import core

st.set_page_config(page_title="AI-Enabled Predictive Maintenance", layout="wide", page_icon="🛠️")
sns.set_style("whitegrid")

TRAIN_FILE = "training_dataset.csv"
N_SYNTHETIC = 2000

# ==========================================================================
# THEME — semantic colour palette used consistently across the whole app
# ==========================================================================
NAVY = "#0B2545"
NAVY_LIGHT = "#13315C"
BLUE = "#2E6FE0"
TEAL = "#0FA3A3"
GREEN = "#1E9E6B"
AMBER = "#E8A013"
RED = "#D64545"
PURPLE = "#6A5ACD"
GREY_BG = "#F4F6F9"
GREY_BORDER = "#E2E6ED"
TEXT_MUTED = "#5B6472"

RISK_COLOR = {"Low": GREEN, "Medium": AMBER, "High": RED}
OUTCOME_COLOR = {"No Failure": GREEN, "Failure": RED}

PLOTLY_TEMPLATE = "plotly_white"


# ==========================================================================
# GLOBAL CSS
# ==========================================================================
def inject_css():
    st.markdown(f"""
    <style>
        /* ---- General page ---- */
        .stApp {{
            background-color: {GREY_BG};
        }}
        html, body, [class*="css"] {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* ---- App header banner ---- */
        .app-header {{
            background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 55%, {BLUE} 130%);
            padding: 2.1rem 2.4rem;
            border-radius: 14px;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(11, 37, 69, 0.25);
        }}
        .app-header h1 {{
            color: #FFFFFF;
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            letter-spacing: 0.2px;
        }}
        .app-header h2 {{
            color: #CFE0FF;
            font-size: 1.08rem;
            font-weight: 500;
            margin: 0 0 0.6rem 0;
        }}
        .app-header p {{
            color: #9FB6DC;
            font-size: 0.87rem;
            margin: 0;
            font-style: italic;
        }}

        /* ---- KPI cards ---- */
        .kpi-card {{
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.05rem 1.2rem 0.95rem 1.2rem;
            box-shadow: 0 2px 10px rgba(11, 37, 69, 0.07);
            border-left: 5px solid var(--accent, {BLUE});
            height: 100%;
            min-height: 118px;
        }}
        .kpi-label {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {NAVY};
            line-height: 1.15;
        }}
        .kpi-sub {{
            font-size: 0.76rem;
            color: {TEXT_MUTED};
            margin-top: 0.25rem;
        }}

        /* ---- Section header card ---- */
        .section-card {{
            background: #FFFFFF;
            border-top: 4px solid var(--accent, {BLUE});
            border-radius: 10px;
            padding: 0.85rem 1.2rem;
            margin: 0.4rem 0 1.1rem 0;
            box-shadow: 0 1px 6px rgba(11, 37, 69, 0.06);
        }}
        .section-card .icon {{
            font-size: 1.1rem;
            margin-right: 0.4rem;
        }}
        .section-card h3 {{
            margin: 0 0 0.15rem 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: {NAVY};
        }}
        .section-card p {{
            margin: 0;
            font-size: 0.85rem;
            color: {TEXT_MUTED};
        }}

        /* ---- Highlight / "recommended model" panel ---- */
        .highlight-panel {{
            background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 160%);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            color: #FFFFFF;
            box-shadow: 0 8px 20px rgba(11, 37, 69, 0.2);
        }}
        .highlight-panel .tag {{
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #BFE8E8;
            text-transform: uppercase;
        }}
        .highlight-panel .model-name {{
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0.15rem 0 0.6rem 0;
        }}
        .highlight-panel .metric-row span {{
            display: inline-block;
            background: rgba(255,255,255,0.13);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
            margin: 0 0.4rem 0.4rem 0;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .highlight-panel .note {{
            font-size: 0.78rem;
            color: #D8ECEC;
            margin-top: 0.6rem;
        }}

        /* ---- Risk badges (used via markdown in tables/text) ---- */
        .badge {{
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #fff;
        }}

        /* ---- Tabs styling ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: #FFFFFF;
            padding: 0.4rem 0.4rem;
            border-radius: 12px;
            box-shadow: 0 1px 6px rgba(11, 37, 69, 0.06);
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 42px;
            border-radius: 8px;
            padding: 0 14px;
            font-weight: 600;
            font-size: 0.85rem;
            color: {NAVY};
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {GREY_BG};
            color: {BLUE};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {NAVY} !important;
            color: #FFFFFF !important;
        }}

        /* ---- Dataframe/table polish ---- */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ---- Footer ---- */
        .app-footer {{
            text-align: center;
            padding: 1rem 0 0.4rem 0;
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            border-top: 1px solid {GREY_BORDER};
            margin-top: 1.5rem;
        }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E7EEFB !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15);
        }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent=BLUE):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle="", icon="📊", accent=BLUE):
    st.markdown(f"""
    <div class="section-card" style="--accent:{accent};">
        <h3><span class="icon">{icon}</span>{title}</h3>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def risk_badge(category):
    color = RISK_COLOR.get(category, TEXT_MUTED)
    return f'<span class="badge" style="background-color:{color};">{category}</span>'


def style_risk_table(df, risk_col="Risk_Category"):
    """Return a pandas Styler that colour-codes the risk column, without
    altering any underlying values."""
    def _row_style(row):
        color = RISK_COLOR.get(row.get(risk_col, ""), None)
        if color:
            return [f"background-color: {color}22" for _ in row]
        return ["" for _ in row]
    return df.style.apply(_row_style, axis=1)


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


# ==========================================================================
# RENDER
# ==========================================================================
inject_css()

with st.spinner("Loading data and running the analytical workflow..."):
    W = run_full_workflow()

df_raw, df_fe, summary = W["df_raw"], W["df_fe"], W["summary"]
results, best_model_name = W["results"], W["best_model_name"]
synthetic_df = W["synthetic_df"]
best = results[best_model_name]

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🛠️ Predictive Maintenance")
    st.caption("MBA Working with AI (WAI) Project")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.write(f"{summary['n_rows']:,} machines · {df_raw.shape[1]} variables")
    st.markdown("**Target**")
    st.write(f"`{summary['target_col']}`")
    st.markdown("**Model in use**")
    st.write(best_model_name)
    st.markdown("---")
    st.caption("Data Mining & Predictive Analytics")

# ---- Header banner ----
st.markdown(f"""
<div class="app-header">
    <h1>🛠️ AI-Enabled Predictive Maintenance</h1>
    <h2>Predicting Industrial Machine Failure for Proactive Maintenance Decisions</h2>
    <p>Machine-learning driven decision support for maintenance risk prioritization.</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Project Overview", "Data Quality", "Exploratory Analysis", "Model Development",
    "Model Evaluation", "Feature Importance", "Prediction & Risk",
    "Maintenance Recommendations", "Executive Dashboard", "Methodology & Limitations",
])

# ---------------------------------------------------------------- 1
with tabs[0]:
    section_header("Project Overview", "Business context and end-to-end analytical workflow",
                    icon="🧭", accent=BLUE)
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
    with c1:
        kpi_card("Training Observations", f"{summary['n_rows']:,}", "Historical dataset", accent=BLUE)
    with c2:
        kpi_card("Historical Failure Rate", f"{summary['failure_pct']:.2f}%", "Class imbalance present", accent=AMBER)
    with c3:
        kpi_card("Recommended Model", best_model_name, "Selected on business-first criteria", accent=PURPLE)

# ---------------------------------------------------------------- 2
with tabs[1]:
    section_header("Dataset & Data Quality", "Structure, completeness and predictor selection",
                    icon="🧪", accent=TEAL)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Machines", f"{summary['n_rows']:,}", "Rows in training data", accent=BLUE)
    with c2:
        kpi_card("Total Variables", f"{df_raw.shape[1]}", "Raw columns", accent=BLUE)
    with c3:
        kpi_card("Duplicate Rows", f"{df_raw.duplicated().sum()}", "Exact duplicates", accent=AMBER)
    with c4:
        kpi_card("Missing Values", f"{int(df_raw.isnull().sum().sum())}", "Across all columns", accent=AMBER)

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
    st.dataframe(dq, use_container_width=True, key="dq_table")
    st.dataframe(df_raw.head(20), use_container_width=True, key="raw_head_table")

# ---------------------------------------------------------------- 3
with tabs[2]:
    section_header("Exploratory Data Analysis", "Distributions and relationships driving failure risk",
                    icon="📈", accent=BLUE)
    target = summary["target_col"]

    col1, col2 = st.columns(2)
    with col1:
        vc = df_fe[target].value_counts().sort_index()
        fig = px.bar(x=["No Failure", "Failure"], y=vc.values,
                     labels={"x": "Outcome", "y": "Count"}, title="Target Distribution",
                     color=["No Failure", "Failure"],
                     color_discrete_map=OUTCOME_COLOR, template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False, title_font_size=15)
        st.plotly_chart(fig, use_container_width=True, key="eda_target_dist")
    with col2:
        if summary["categorical_predictors"]:
            rate = df_fe.groupby(summary["categorical_predictors"][0])[target].mean() * 100
            fig = px.bar(x=rate.index, y=rate.values, labels={"x": "Type", "y": "Failure rate (%)"},
                         title="Failure Rate by Machine Type", template=PLOTLY_TEMPLATE,
                         color_discrete_sequence=[TEAL])
            fig.update_layout(title_font_size=15)
            st.plotly_chart(fig, use_container_width=True, key="eda_failure_rate_by_type")

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df_fe, x=target, y="Tool wear [min]", title="Tool Wear by Outcome",
                     color=target, color_discrete_map={0: GREEN, 1: RED}, template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False, title_font_size=15)
        st.plotly_chart(fig, use_container_width=True, key="eda_tool_wear_box")
    with col4:
        fig = px.box(df_fe, x=target, y="Torque [Nm]", title="Torque by Outcome",
                     color=target, color_discrete_map={0: GREEN, 1: RED}, template=PLOTLY_TEMPLATE)
        fig.update_layout(showlegend=False, title_font_size=15)
        st.plotly_chart(fig, use_container_width=True, key="eda_torque_box")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    corr = df_fe[W["num_cols"] + [target]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.tick_params(labelsize=8)
    st.pyplot(fig, key="eda_corr_heatmap")

# ---------------------------------------------------------------- 4
with tabs[3]:
    section_header("Model Development & Comparison",
                    "Four classifiers trained inside a leakage-safe Pipeline, tuned with GridSearchCV",
                    icon="⚙️", accent=PURPLE)
    st.markdown("Four classifiers trained inside a leakage-safe `Pipeline` "
                "(preprocessing fit on training data only), tuned with `GridSearchCV` "
                "(5-fold stratified CV, ROC-AUC scoring), class imbalance handled via "
                "`class_weight='balanced'`.")
    comparison = pd.DataFrame({
        name: {"Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
               "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
        for name, r in results.items()
    }).T.round(3)
    st.dataframe(
        comparison.style.format("{:.3f}").highlight_max(axis=0, color=f"{GREEN}33"),
        use_container_width=True, key="model_comparison_table")

    st.subheader("Cross-Validation (ROC-AUC)")
    cv_df = pd.DataFrame({name: {"CV Mean": v["mean"], "CV Std": v["std"]}
                           for name, v in W["cv_scores_record"].items()}).T.round(3)
    st.dataframe(cv_df.style.format("{:.3f}"), use_container_width=True, key="cv_scores_table")

    with st.expander("Best hyperparameters per model"):
        for name, r in results.items():
            st.write(f"**{name}**: {r['best_params']}")

# ---------------------------------------------------------------- 5
with tabs[4]:
    section_header("Model Evaluation", "Recall-first model selection to minimize missed failures",
                    icon="🎯", accent=RED)
    st.markdown("""In predictive maintenance a **False Negative** (predicted *No Failure*, machine
actually fails) is typically far costlier than a **False Positive**. Model selection therefore
prioritizes **Recall**, then ROC-AUC, then F1 — not accuracy alone.""")

    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        st.subheader("ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(6, 5.3))
        palette = [BLUE, TEAL, AMBER, PURPLE]
        for i, (name, r) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(W["y_test"], r["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})",
                    color=palette[i % len(palette)], linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.set_title("ROC Curve — All Models", fontsize=11, fontweight="bold")
        st.pyplot(fig, key="eval_roc_curve")

    with col_b:
        st.markdown('<div class="highlight-panel">', unsafe_allow_html=True)
        st.markdown(f"""
            <span class="tag">Recommended Model</span>
            <div class="model-name">{best_model_name}</div>
            <div class="metric-row">
                <span>Recall: {best['recall']*100:.1f}%</span>
                <span>ROC-AUC: {best['roc_auc']*100:.1f}%</span>
                <span>F1: {best['f1']*100:.1f}%</span>
            </div>
            <div class="note">Selected based on business-first, recall-oriented model evaluation
            computed from the results above.</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Confusion Matrices")
    cols = st.columns(4)
    for c, (name, r) in zip(cols, results.items()):
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred:No", "Pred:Yes"], yticklabels=["True:No", "True:Yes"])
        ax.set_title(name, fontsize=8)
        c.pyplot(fig, key=f"eval_confmat_{name}")

# ---------------------------------------------------------------- 6
with tabs[5]:
    section_header("Feature Importance", "Which operating variables the model weighs most heavily",
                    icon="🔑", accent=PURPLE)
    fi = W["feature_importance"]
    if fi is not None:
        fig = px.bar(fi.head(10), x="importance", y="feature", orientation="h",
                     title=f"Top Predictors — {best_model_name}", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[PURPLE])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, title_font_size=15, height=460)
        st.plotly_chart(fig, use_container_width=True, key="fi_top10_chart")
        st.caption("The model identifies these variables as important predictors of failure risk. "
                   "This reflects predictive association, not causation.")
    else:
        st.info("Feature importance is not available for this model type.")

# ---------------------------------------------------------------- 7
with tabs[6]:
    section_header("Synthetic Prediction & Risk Analysis",
                    "Scoring a demonstration 'unseen' machine population",
                    icon="🧮", accent=BLUE)
    st.warning("⚠️ The dataset below is a **synthetic, demonstration** 'unseen' prediction set, "
               "generated programmatically from the statistical characteristics of the training data "
               "(joint bootstrap + jitter) — it is NOT independently collected real-world data. "
               "Predictions on it demonstrate model application, not validated future outcomes.")

    n_high = int((synthetic_df["Risk_Category"] == "High").sum())
    n_med = int((synthetic_df["Risk_Category"] == "Medium").sum())
    n_low = int((synthetic_df["Risk_Category"] == "Low").sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Machines Assessed", f"{len(synthetic_df):,}", "Synthetic prediction set", accent=BLUE)
    with c2:
        kpi_card("High Risk", f"{n_high:,}", "Immediate attention", accent=RED)
    with c3:
        kpi_card("Medium Risk", f"{n_med:,}", "Monitor closely", accent=AMBER)
    with c4:
        kpi_card("Low Risk", f"{n_low:,}", "Routine schedule", accent=GREEN)

    st.write("")
    risk_type_filter = st.multiselect("Filter by risk category", ["Low", "Medium", "High"],
                                       default=["Low", "Medium", "High"], key="risk_filter_multiselect")
    filtered = synthetic_df[synthetic_df["Risk_Category"].isin(risk_type_filter)]

    display_cols = (["Synthetic_Machine_ID", "Product ID", "Type"] + summary["numerical_predictors"] +
                     ["Failure_Probability", "Predicted_Class", "Risk_Category"])
    table_df = filtered[display_cols].sort_values("Failure_Probability", ascending=False)
    st.dataframe(
        style_risk_table(table_df).format({"Failure_Probability": "{:.3f}"}),
        use_container_width=True, height=350, key="prediction_risk_table")

    fig = px.histogram(synthetic_df, x="Failure_Probability", nbins=30,
                        title="Predicted Failure Probability Distribution",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=[BLUE])
    fig.update_layout(title_font_size=15)
    st.plotly_chart(fig, use_container_width=True, key="pred_failure_prob_hist")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("⬇️ Synthetic prediction dataset (input, no target)",
                            data=synthetic_df.drop(columns=["Failure_Probability", "Predicted_Class",
                                                              "Risk_Category", "Maintenance_Recommendation"]
                                                    ).to_csv(index=False),
                            file_name="synthetic_prediction_dataset.csv", mime="text/csv",
                            key="dl_synthetic_input")

    full_output_cols = (["Synthetic_Machine_ID", "Product ID", "Type"] + summary["numerical_predictors"] +
                         ["Temperature_Diff_K", "Mechanical_Power_W", "Failure_Probability",
                          "Predicted_Class", "Risk_Category", "Maintenance_Recommendation"])
    with dl2:
        st.download_button("⬇️ Full predictions",
                            data=synthetic_df[full_output_cols].sort_values(
                                "Failure_Probability", ascending=False).to_csv(index=False),
                            file_name="final_machine_failure_predictions.csv", mime="text/csv",
                            key="dl_full_predictions")

    high_risk_df = synthetic_df[synthetic_df["Risk_Category"] == "High"][full_output_cols].sort_values(
        "Failure_Probability", ascending=False)
    with dl3:
        st.download_button("⬇️ High-risk machine list",
                            data=high_risk_df.to_csv(index=False),
                            file_name="high_risk_machine_list.csv", mime="text/csv",
                            key="dl_high_risk_list")

# ---------------------------------------------------------------- 8
with tabs[7]:
    section_header("Maintenance Recommendations", "Action guidance by predicted risk tier",
                    icon="🧰", accent=TEAL)

    rec_cols = st.columns(3)
    tier_order = ["High", "Medium", "Low"]
    tier_icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    tier_accent = {"High": RED, "Medium": AMBER, "Low": GREEN}
    for col, tier in zip(rec_cols, tier_order):
        rec = core.RECOMMENDATIONS.get(tier, "")
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{tier_accent[tier]}; min-height:160px;">
                <div class="kpi-label">{tier_icons[tier]} {tier} Risk</div>
                <div style="font-size:0.88rem; color:#333; margin-top:0.4rem; line-height:1.4;">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Top 15 Highest-Risk Machines")
    top15 = synthetic_df.sort_values("Failure_Probability", ascending=False).head(15)
    top15_display = top15[["Synthetic_Machine_ID", "Product ID", "Type", "Failure_Probability",
                            "Risk_Category", "Maintenance_Recommendation"]]
    st.dataframe(
        style_risk_table(top15_display).format({"Failure_Probability": "{:.3f}"}),
        use_container_width=True, key="maint_top15_table")

# ---------------------------------------------------------------- 9
with tabs[8]:
    section_header("Executive Dashboard", "Context → Evidence → Model → Prediction → Decision, at a glance",
                    icon="📊", accent=NAVY)

    # SECTION 1 — Executive KPIs
    st.markdown("##### Executive KPIs")
    n_high = int((synthetic_df["Risk_Category"] == "High").sum())
    n_med = int((synthetic_df["Risk_Category"] == "Medium").sum())
    n_low = int((synthetic_df["Risk_Category"] == "Low").sum())
    pred_fail_pct = (synthetic_df['Predicted_Class'] == 'Failure').mean() * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Machines Assessed", f"{len(synthetic_df):,}", "Synthetic set", accent=BLUE)
    with k2:
        kpi_card("High Risk", f"{n_high:,}", "Act now", accent=RED)
    with k3:
        kpi_card("Medium Risk", f"{n_med:,}", "Monitor", accent=AMBER)
    with k4:
        kpi_card("Low Risk", f"{n_low:,}", "Routine", accent=GREEN)
    with k5:
        kpi_card("Predicted Failure %", f"{pred_fail_pct:.1f}%", "Of assessed machines", accent=RED)
    with k6:
        kpi_card(f"Best Model ROC-AUC", f"{best['roc_auc']:.3f}", best_model_name, accent=PURPLE)

    st.write("")
    # SECTION 2 — Risk overview
    st.markdown("##### Risk Overview")
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = synthetic_df["Risk_Category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     title="Predicted Risk Category Distribution", hole=0.45,
                     color=risk_counts.index, color_discrete_map=RISK_COLOR,
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title_font_size=15)
        st.plotly_chart(fig, use_container_width=True, key="exec_risk_pie")
    with col2:
        fig = px.histogram(synthetic_df, x="Failure_Probability", nbins=30,
                            title="Predicted Failure Probability Distribution",
                            template=PLOTLY_TEMPLATE, color_discrete_sequence=[BLUE])
        fig.update_layout(title_font_size=15)
        st.plotly_chart(fig, use_container_width=True, key="exec_failure_prob_hist")

    # SECTION 3 — Model performance
    st.markdown("##### Model Performance")
    comp = pd.DataFrame({name: {"Recall": r["recall"], "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
                          for name, r in results.items()}).T.reset_index()
    comp = comp.melt(id_vars="index", var_name="Metric", value_name="Score")
    fig = px.bar(comp, x="index", y="Score", color="Metric", barmode="group",
                 title="Model Performance Comparison", labels={"index": "Model"},
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=[BLUE, TEAL, AMBER])
    fig.update_layout(title_font_size=15)
    st.plotly_chart(fig, use_container_width=True, key="exec_model_perf_bar")

    # SECTION 4 — Key predictors
    st.markdown("##### Key Predictors")
    fi = W["feature_importance"]
    if fi is not None:
        fig = px.bar(fi.head(8), x="importance", y="feature", orientation="h",
                     title=f"Top Predictors — {best_model_name}", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[PURPLE])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, title_font_size=15, height=380)
        st.plotly_chart(fig, use_container_width=True, key="exec_feature_importance_chart")
    else:
        st.info("Feature importance is not available for this model type.")

    # SECTION 5 — Maintenance priorities
    st.markdown("##### Maintenance Priorities")
    priority_cols = (["Synthetic_Machine_ID", "Product ID", "Type"] + summary["numerical_predictors"] +
                      ["Failure_Probability", "Risk_Category"])
    high_priority = synthetic_df[synthetic_df["Risk_Category"] == "High"][priority_cols].sort_values(
        "Failure_Probability", ascending=False).head(20)
    st.dataframe(
        style_risk_table(high_priority).format({"Failure_Probability": "{:.3f}"}),
        use_container_width=True, height=380, key="exec_maintenance_priority_table")

# ---------------------------------------------------------------- 10
with tabs[9]:
    section_header("AI / Project Methodology & Limitations",
                    "Transparency on approach, assumptions and responsible-use boundaries",
                    icon="📋", accent=NAVY)
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

st.markdown(f"""
<div class="app-footer">
    AI-Enabled Predictive Maintenance &nbsp;|&nbsp; Data Mining &amp; Predictive Analytics &nbsp;|&nbsp; WAI Project
    <br>Built with scikit-learn, pandas, Streamlit and Plotly. Notebook and app share identical logic via <code>core.py</code>.
</div>
""", unsafe_allow_html=True)
