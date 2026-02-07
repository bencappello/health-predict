#!/usr/bin/env python3
"""
Health Predict - Comprehensive Monitoring Dashboard

Streamlit dashboard pulling real data from MLflow and the live API,
covering drift detection, model performance, training history,
and deployment status.

Usage:
    streamlit run scripts/drift_dashboard.py --server.port=8501
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
import requests
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Predict - Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DRIFT_THRESHOLD = 0.30
REGRESSION_THRESHOLD = -0.02
MODEL_NAME = "HealthPredictModel"
DRIFT_EXPERIMENT = "HealthPredict_Drift_Monitoring"
TRAINING_EXPERIMENT = "HealthPredict_Training_HPO_Airflow"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_drift_data(tracking_uri: str) -> pd.DataFrame:
    """Load drift monitoring runs from MLflow."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(DRIFT_EXPERIMENT)
        if experiment is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100,
        )
        if runs.empty:
            return pd.DataFrame()
        return runs
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_training_data(tracking_uri: str) -> pd.DataFrame:
    """Load quality-gate runs (those with new_auc_test metric) from the training experiment."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(TRAINING_EXPERIMENT)
        if experiment is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.new_auc_test > 0",
            order_by=["start_time DESC"],
            max_results=100,
        )
        if runs.empty:
            return pd.DataFrame()
        return runs
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_hpo_trials(tracking_uri: str, parent_run_id: str) -> pd.DataFrame:
    """Load child (HPO trial) runs for a given parent run."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(TRAINING_EXPERIMENT)
        if experiment is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.val_roc_auc_score DESC"],
            max_results=50,
        )
        return runs
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_model_registry(tracking_uri: str) -> pd.DataFrame:
    """Load model version history from the MLflow model registry."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            return pd.DataFrame()
        rows = []
        for v in versions:
            rows.append(
                {
                    "version": int(v.version),
                    "stage": v.current_stage,
                    "created": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "run_id": v.run_id,
                    "status": v.status,
                    "description": v.description or "",
                }
            )
        return pd.DataFrame(rows).sort_values("version", ascending=False)
    except Exception:
        return pd.DataFrame()


def check_api_health(api_url: str) -> dict:
    """Query the live prediction API for health and model info."""
    result = {"api_status": "unavailable", "model_version": "N/A", "model_stage": "N/A", "loaded_at": "N/A"}
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            result["api_status"] = "healthy"
    except Exception:
        pass
    try:
        r = requests.get(f"{api_url}/model-info", timeout=3)
        if r.status_code == 200:
            info = r.json()
            result["model_version"] = info.get("model_version", "N/A")
            result["model_stage"] = info.get("model_stage", "N/A")
            result["loaded_at"] = info.get("loaded_at", "N/A")
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Helper: extract param/metric from MLflow DataFrame
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, prefix: str, name: str):
    """Return column values, trying prefixed and unprefixed names."""
    for candidate in [f"{prefix}{name}", name]:
        if candidate in df.columns:
            return df[candidate]
    return pd.Series([None] * len(df))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")
mlflow_uri = st.sidebar.text_input(
    "MLflow Tracking URI",
    value=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
)
api_url = st.sidebar.text_input(
    "Prediction API URL",
    value=os.getenv("API_URL", "http://health-predict-api:8000"),
    help="The base URL for the deployed prediction API (K8s service or minikube IP).",
)
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Drift threshold:** {DRIFT_THRESHOLD}  \n"
    f"**Regression threshold:** {REGRESSION_THRESHOLD}"
)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Health Predict - Monitoring Dashboard")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
drift_df = load_drift_data(mlflow_uri)
training_df = load_training_data(mlflow_uri)
registry_df = load_model_registry(mlflow_uri)
api_info = check_api_health(api_url)

# ---------------------------------------------------------------------------
# 1. System Status Bar
# ---------------------------------------------------------------------------
st.header("System Status")
c1, c2, c3, c4 = st.columns(4)

with c1:
    is_healthy = api_info["api_status"] == "healthy"
    st.metric("API Status", "Healthy" if is_healthy else "Unavailable")

with c2:
    st.metric("Current Model", f"v{api_info['model_version']} ({api_info['model_stage']})")

with c3:
    if not drift_df.empty:
        latest_drift_share = _col(drift_df, "metrics.", "drift_share").iloc[0]
        try:
            latest_drift_share = float(latest_drift_share)
            st.metric("Latest Drift Share", f"{latest_drift_share:.3f}")
        except (TypeError, ValueError):
            st.metric("Latest Drift Share", "N/A")
    else:
        st.metric("Latest Drift Share", "N/A")

with c4:
    if not drift_df.empty:
        try:
            gate = "RETRAIN" if float(latest_drift_share) >= DRIFT_THRESHOLD else "SKIP"
        except Exception:
            gate = "N/A"
        st.metric("Drift Gate", gate)
    else:
        st.metric("Drift Gate", "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_drift, tab_perf, tab_training, tab_registry = st.tabs(
    ["Drift Monitoring", "Model Performance", "Training History", "Model Registry"]
)

# ---------------------------------------------------------------------------
# 2. Drift Monitoring
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Drift Monitoring")
    if drift_df.empty:
        st.info("No drift monitoring data available. Run the pipeline to generate drift metrics.")
    else:
        # Extract relevant columns
        batch_col = _col(drift_df, "params.", "batch_number")
        drift_share_col = _col(drift_df, "metrics.", "drift_share")
        n_drifted_col = _col(drift_df, "metrics.", "n_drifted_columns")
        dataset_drift_col = _col(drift_df, "metrics.", "dataset_drift")

        plot_df = pd.DataFrame(
            {
                "batch": pd.to_numeric(batch_col, errors="coerce"),
                "drift_share": pd.to_numeric(drift_share_col, errors="coerce"),
                "n_drifted": pd.to_numeric(n_drifted_col, errors="coerce"),
                "dataset_drift": dataset_drift_col,
                "timestamp": drift_df["start_time"] if "start_time" in drift_df.columns else None,
            }
        ).dropna(subset=["batch", "drift_share"]).sort_values("batch")

        if not plot_df.empty:
            # Drift share bar chart
            st.markdown("#### Drift Share by Batch")
            colors = ["red" if v >= DRIFT_THRESHOLD else "green" for v in plot_df["drift_share"]]
            fig_share = go.Figure()
            fig_share.add_trace(
                go.Bar(
                    x=plot_df["batch"].astype(int).astype(str),
                    y=plot_df["drift_share"],
                    marker_color=colors,
                    name="Drift Share",
                )
            )
            fig_share.add_hline(
                y=DRIFT_THRESHOLD,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Threshold ({DRIFT_THRESHOLD})",
            )
            fig_share.update_layout(
                xaxis_title="Batch",
                yaxis_title="Drift Share",
                height=400,
            )
            st.plotly_chart(fig_share, use_container_width=True)

            # Drifted features count
            st.markdown("#### Drifted Features Count by Batch")
            fig_count = go.Figure()
            fig_count.add_trace(
                go.Bar(
                    x=plot_df["batch"].astype(int).astype(str),
                    y=plot_df["n_drifted"],
                    marker_color="steelblue",
                    name="Drifted Columns",
                )
            )
            fig_count.update_layout(
                xaxis_title="Batch",
                yaxis_title="Drifted Columns",
                height=350,
            )
            st.plotly_chart(fig_count, use_container_width=True)

            # Detail table
            st.markdown("#### Drift Detail Table")
            detail = plot_df[["batch", "drift_share", "n_drifted", "dataset_drift", "timestamp"]].copy()
            detail["batch"] = detail["batch"].astype(int)
            detail.columns = ["Batch", "Drift Share", "Drifted Columns", "Dataset Drift", "Timestamp"]
            st.dataframe(detail, use_container_width=True, hide_index=True)
        else:
            st.warning("Drift data could not be parsed.")

# ---------------------------------------------------------------------------
# 3. Model Performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader("Model Performance")
    if training_df.empty:
        st.info("No quality-gate evaluation data available. Run the pipeline with deployment to generate metrics.")
    else:
        batch_col = _col(training_df, "params.", "batch_number")
        new_auc = _col(training_df, "metrics.", "new_auc_test")
        new_f1 = _col(training_df, "metrics.", "new_f1_test")
        prod_auc = _col(training_df, "metrics.", "prod_auc_test")
        auc_imp = _col(training_df, "metrics.", "auc_improvement")
        f1_imp = _col(training_df, "metrics.", "f1_improvement")

        perf_df = pd.DataFrame(
            {
                "batch": pd.to_numeric(batch_col, errors="coerce"),
                "new_auc": pd.to_numeric(new_auc, errors="coerce"),
                "new_f1": pd.to_numeric(new_f1, errors="coerce"),
                "prod_auc": pd.to_numeric(prod_auc, errors="coerce"),
                "auc_improvement": pd.to_numeric(auc_imp, errors="coerce"),
                "f1_improvement": pd.to_numeric(f1_imp, errors="coerce"),
            }
        ).dropna(subset=["batch", "new_auc"]).sort_values("batch")

        if not perf_df.empty:
            # AUC trend
            st.markdown("#### AUC Trend Across Batches")
            fig_auc = go.Figure()
            fig_auc.add_trace(
                go.Scatter(
                    x=perf_df["batch"].astype(int).astype(str),
                    y=perf_df["new_auc"],
                    mode="lines+markers",
                    name="New Model AUC",
                    line=dict(color="blue"),
                )
            )
            if perf_df["prod_auc"].notna().any():
                fig_auc.add_trace(
                    go.Scatter(
                        x=perf_df["batch"].astype(int).astype(str),
                        y=perf_df["prod_auc"],
                        mode="lines+markers",
                        name="Production AUC",
                        line=dict(color="gray", dash="dot"),
                    )
                )
            fig_auc.update_layout(xaxis_title="Batch", yaxis_title="AUC", height=400)
            st.plotly_chart(fig_auc, use_container_width=True)

            # AUC improvement bar chart
            st.markdown("#### AUC Improvement per Batch")
            imp_colors = [
                "green" if v >= 0 else ("red" if v < REGRESSION_THRESHOLD else "orange")
                for v in perf_df["auc_improvement"].fillna(0)
            ]
            fig_imp = go.Figure()
            fig_imp.add_trace(
                go.Bar(
                    x=perf_df["batch"].astype(int).astype(str),
                    y=perf_df["auc_improvement"],
                    marker_color=imp_colors,
                    name="AUC Improvement",
                )
            )
            fig_imp.add_hline(
                y=REGRESSION_THRESHOLD,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Regression Threshold ({REGRESSION_THRESHOLD})",
            )
            fig_imp.add_hline(y=0, line_color="black", line_width=0.5)
            fig_imp.update_layout(xaxis_title="Batch", yaxis_title="AUC Improvement", height=350)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Metrics comparison table
            st.markdown("#### Metrics Comparison Table")
            comp = perf_df[["batch", "new_auc", "new_f1", "prod_auc", "auc_improvement", "f1_improvement"]].copy()
            comp["batch"] = comp["batch"].astype(int)
            comp.columns = [
                "Batch",
                "New AUC",
                "New F1",
                "Prod AUC",
                "AUC Improvement",
                "F1 Improvement",
            ]
            st.dataframe(comp, use_container_width=True, hide_index=True)
        else:
            st.warning("Performance data could not be parsed.")

# ---------------------------------------------------------------------------
# 4. Training History
# ---------------------------------------------------------------------------
with tab_training:
    st.subheader("Training History")
    if training_df.empty:
        st.info("No training data available.")
    else:
        batch_col = _col(training_df, "params.", "batch_number")
        batches = pd.to_numeric(batch_col, errors="coerce").dropna().unique()
        batches = sorted([int(b) for b in batches])

        if batches:
            selected_batch = st.selectbox("Select batch", batches, index=len(batches) - 1)

            # Find the parent run for this batch
            batch_mask = _col(training_df, "params.", "batch_number").astype(str) == str(selected_batch)
            batch_runs = training_df[batch_mask]

            if not batch_runs.empty:
                parent_run_id = batch_runs.iloc[0]["run_id"]
                st.markdown(f"**Parent run ID:** `{parent_run_id}`")

                trials_df = load_hpo_trials(mlflow_uri, parent_run_id)
                if not trials_df.empty:
                    st.markdown("#### HPO Trials")
                    # Select useful columns
                    display_cols = []
                    for col in trials_df.columns:
                        if col.startswith("params.") and any(
                            k in col
                            for k in [
                                "max_depth",
                                "learning_rate",
                                "n_estimators",
                                "subsample",
                                "colsample_bytree",
                                "reg_alpha",
                                "reg_lambda",
                                "gamma",
                                "min_child_weight",
                                "model_type",
                            ]
                        ):
                            display_cols.append(col)
                    metric_cols = [
                        c
                        for c in trials_df.columns
                        if c.startswith("metrics.") and "val_" in c
                    ]
                    show_cols = display_cols + metric_cols
                    if show_cols:
                        t = trials_df[show_cols].copy()
                        t.columns = [c.replace("params.", "").replace("metrics.", "") for c in t.columns]
                        st.dataframe(t, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(trials_df.head(20), use_container_width=True, hide_index=True)
                else:
                    st.info("No HPO trial (child) runs found for this batch.")

            # Best model per batch summary
            st.markdown("#### Best Model per Batch")
            summary_rows = []
            for b in batches:
                mask = _col(training_df, "params.", "batch_number").astype(str) == str(b)
                b_runs = training_df[mask]
                if not b_runs.empty:
                    best_auc_col = _col(b_runs, "metrics.", "new_auc_test")
                    try:
                        best_auc = float(best_auc_col.iloc[0])
                    except (TypeError, ValueError):
                        best_auc = None
                    model_type = _col(b_runs, "params.", "model_type")
                    mt = model_type.iloc[0] if not model_type.empty else "N/A"
                    summary_rows.append({"Batch": int(b), "Best AUC (test)": best_auc, "Model Type": mt})
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No batches found in training data.")

# ---------------------------------------------------------------------------
# 5. Model Registry
# ---------------------------------------------------------------------------
with tab_registry:
    st.subheader("Model Registry")
    if registry_df.empty:
        st.info("No model versions found in the registry.")
    else:
        st.markdown(f"**Registered model:** `{MODEL_NAME}`")

        # Highlight production row
        prod = registry_df[registry_df["stage"] == "Production"]
        if not prod.empty:
            st.markdown("#### Current Production Model")
            st.dataframe(prod, use_container_width=True, hide_index=True)

        st.markdown("#### Version History")
        st.dataframe(registry_df, use_container_width=True, hide_index=True)
