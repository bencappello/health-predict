#!/usr/bin/env python3
"""
Health Predict — Continuous Improvement Experiment Report

A vertical-scroll dashboard that tells the story of feeding 5 batches of data
through an automated drift-detection and retraining pipeline.

Usage:
    streamlit run scripts/drift_dashboard.py --server.port=8501
"""

import json
import os
import logging
import warnings

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Predict — Experiment Report",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DRIFT_THRESHOLD = 0.30
MODEL_NAME = "HealthPredictModel"
DRIFT_EXPERIMENT = "HealthPredict_Drift_Monitoring"
TRAINING_EXPERIMENT = "HealthPredict_Training_HPO_Airflow"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame, prefix: str, name: str, default=None):
    """Return column values, trying prefixed and unprefixed names."""
    for candidate in [f"{prefix}{name}", name]:
        if candidate in df.columns:
            return df[candidate]
    if default is not None:
        return pd.Series([default] * len(df), index=df.index)
    return pd.Series([None] * len(df), index=df.index)


@st.cache_data(ttl=30)
def get_latest_per_batch(tracking_uri: str, experiment_name: str, filter_string: str = None) -> pd.DataFrame:
    """Query MLflow and return only the most recent run per batch number."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=200,
        )
        if runs.empty:
            return runs

        runs["batch"] = pd.to_numeric(
            runs.get("params.batch_number", pd.Series(dtype=float)), errors="coerce"
        )
        runs = runs.dropna(subset=["batch"])
        runs = runs.sort_values("start_time", ascending=False).drop_duplicates(
            subset=["batch"], keep="first"
        )
        return runs.sort_values("batch").reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to load runs from '{experiment_name}': {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_model_registry(tracking_uri: str) -> dict:
    """Return info about the current Production model."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri)
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return {}
        v = versions[0]
        return {
            "version": int(v.version),
            "stage": v.current_stage,
            "run_id": v.run_id,
        }
    except Exception as e:
        logger.error(f"Failed to load model registry: {e}")
        return {}


def check_api_health(api_url: str) -> dict:
    """Quick API health check."""
    result = {"status": "not_configured", "model_version": None}
    if not api_url:
        return result
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        result["status"] = "healthy" if r.status_code == 200 else f"HTTP {r.status_code}"
    except requests.Timeout:
        result["status"] = "timeout"
        return result
    except requests.ConnectionError:
        result["status"] = "unreachable"
        return result
    except Exception as e:
        result["status"] = str(e)[:50]
        return result

    try:
        r = requests.get(f"{api_url}/model-info", timeout=5)
        if r.status_code == 200:
            info = r.json()
            result["model_version"] = info.get("model_version")
    except Exception:
        pass
    return result


def load_roc_curves(tracking_uri: str, run_id: str) -> dict:
    """Download ROC curve JSON artifacts for a run. Returns dict with 'production' and/or 'new' keys."""
    curves = {}
    try:
        client = MlflowClient(tracking_uri)
        artifact_dir = client.download_artifacts(run_id, "roc_curves")
        for name in ["production", "new"]:
            path = os.path.join(artifact_dir, f"roc_curve_{name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    curves[name] = json.load(f)
    except Exception as e:
        logger.debug(f"Could not load ROC curves for {run_id}: {e}")
    return curves


# ---------------------------------------------------------------------------
# Sidebar (minimal)
# ---------------------------------------------------------------------------
mlflow_uri = st.sidebar.text_input(
    "MLflow Tracking URI",
    value=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
)
api_url = st.sidebar.text_input(
    "Prediction API URL",
    value=os.getenv("API_URL", ""),
)
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Drift threshold:** {DRIFT_THRESHOLD}  \n"
    f"[MLflow UI]({mlflow_uri.replace('mlflow:5000','localhost:5000')})  \n"
    f"[Airflow UI](http://localhost:8080)"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
drift_runs = get_latest_per_batch(mlflow_uri, DRIFT_EXPERIMENT)
quality_runs = get_latest_per_batch(
    mlflow_uri, TRAINING_EXPERIMENT, filter_string="metrics.new_auc_test > 0"
)
prod_model = load_model_registry(mlflow_uri)
api_info = check_api_health(api_url)

# Build a unified per-batch dataframe
batches = set()
if not drift_runs.empty:
    batches.update(drift_runs["batch"].astype(int).tolist())
if not quality_runs.empty:
    batches.update(quality_runs["batch"].astype(int).tolist())
batches = sorted(batches)

rows = []
for b in batches:
    row = {"batch": b}

    # Drift info
    if not drift_runs.empty:
        d = drift_runs[drift_runs["batch"] == b]
        if not d.empty:
            row["drift_share"] = pd.to_numeric(_col(d, "metrics.", "drift_share").iloc[0], errors="coerce")
            row["n_drifted"] = pd.to_numeric(_col(d, "metrics.", "n_drifted_columns").iloc[0], errors="coerce")

    # Quality gate info
    if not quality_runs.empty:
        q = quality_runs[quality_runs["batch"] == b]
        if not q.empty:
            row["run_id"] = q.iloc[0].get("run_id")
            row["decision"] = _col(q, "tags.", "decision").iloc[0]
            row["new_auc"] = pd.to_numeric(_col(q, "metrics.", "new_auc_test").iloc[0], errors="coerce")
            row["new_f1"] = pd.to_numeric(_col(q, "metrics.", "new_f1_test").iloc[0], errors="coerce")
            row["new_precision"] = pd.to_numeric(_col(q, "metrics.", "new_precision_test").iloc[0], errors="coerce")
            row["new_recall"] = pd.to_numeric(_col(q, "metrics.", "new_recall_test").iloc[0], errors="coerce")
            row["prod_auc"] = pd.to_numeric(_col(q, "metrics.", "prod_auc_test").iloc[0], errors="coerce")
            row["prod_f1"] = pd.to_numeric(_col(q, "metrics.", "prod_f1_test").iloc[0], errors="coerce")
            row["prod_precision"] = pd.to_numeric(_col(q, "metrics.", "prod_precision_test").iloc[0], errors="coerce")
            row["prod_recall"] = pd.to_numeric(_col(q, "metrics.", "prod_recall_test").iloc[0], errors="coerce")
            row["auc_improvement"] = pd.to_numeric(_col(q, "metrics.", "auc_improvement").iloc[0], errors="coerce")
            row["f1_improvement"] = pd.to_numeric(_col(q, "metrics.", "f1_improvement").iloc[0], errors="coerce")

    # Infer decision when tag is missing (retrain-path runs don't set it)
    decision = row.get("decision")
    if decision is None or (isinstance(decision, float) and pd.isna(decision)):
        auc_imp = row.get("auc_improvement")
        new_a = row.get("new_auc")
        prod_a = row.get("prod_auc")
        if pd.notna(auc_imp) and abs(auc_imp) > 0.001:
            row["decision"] = "DEPLOY" if auc_imp >= 0 else "DEPLOY_REFRESH"
        elif pd.notna(new_a) and pd.notna(prod_a) and abs(new_a - prod_a) > 0.001:
            row["decision"] = "DEPLOY" if new_a >= prod_a else "DEPLOY_REFRESH"

    rows.append(row)

report_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------------------------
st.markdown("# Health Predict — Continuous Improvement Experiment")
st.markdown(
    "An automated MLOps system that monitors incoming patient data for distribution drift, "
    "retrains models when drift is detected, and deploys improved models — all without manual intervention."
)

c1, c2, c3 = st.columns(3)
with c1:
    if prod_model:
        st.metric("Production Model", f"v{prod_model['version']}")
    elif api_info.get("model_version"):
        st.metric("Production Model", f"v{api_info['model_version']}")
    else:
        st.metric("Production Model", "—")
with c2:
    st.metric("Batches Processed", len(batches))
with c3:
    retrain_count = 0
    if not report_df.empty and "decision" in report_df.columns:
        retrain_count = report_df["decision"].isin(["DEPLOY", "DEPLOY_REFRESH"]).sum()
    st.metric("Retraining Events", int(retrain_count))

st.markdown("---")

# ---------------------------------------------------------------------------
# EXPERIMENT SUMMARY
# ---------------------------------------------------------------------------
if report_df.empty:
    st.info("No experiment data found. Run the pipeline and/or the baseline script first.")
    st.stop()

st.markdown("## Experiment Summary")

# Drift Progression Chart
if "drift_share" in report_df.columns and report_df["drift_share"].notna().any():
    drift_plot = report_df.dropna(subset=["drift_share"])
    colors = ["#e74c3c" if v >= DRIFT_THRESHOLD else "#27ae60" for v in drift_plot["drift_share"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=drift_plot["batch"].astype(int).astype(str),
            y=drift_plot["drift_share"],
            marker_color=colors,
            text=[f"{v:.2f}" for v in drift_plot["drift_share"]],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=DRIFT_THRESHOLD,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Retrain Threshold ({DRIFT_THRESHOLD})",
    )
    fig.update_layout(
        title="Drift Progression Across Batches",
        xaxis_title="Batch",
        yaxis_title="Drift Share",
        yaxis_range=[0, max(drift_plot["drift_share"].max() * 1.3, 0.5)],
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# Metrics Summary Table
st.markdown("### Metrics Summary")

table_rows = []
for _, r in report_df.iterrows():
    b = int(r["batch"])
    decision = r.get("decision", "—") or "—"
    drift = r.get("drift_share")
    drift_str = f"{drift:.2f}" if pd.notna(drift) else "—"

    retrained = "—"
    if decision in ("DEPLOY", "DEPLOY_REFRESH"):
        retrained = "Yes"
    elif decision == "SKIP_NO_DRIFT":
        retrained = "No"
    elif decision == "BASELINE":
        retrained = "—"
    elif decision == "SKIP":
        retrained = "Skipped"

    prod_auc_val = r.get("prod_auc")
    prod_f1_val = r.get("prod_f1")
    new_auc_val = r.get("new_auc")
    new_f1_val = r.get("new_f1")

    # For non-retrained batches, new_* == prod_* (same model), so only show once
    show_new = decision in ("DEPLOY", "DEPLOY_REFRESH")

    table_rows.append({
        "Batch": b,
        "Drift Share": drift_str,
        "Retrained?": retrained,
        "Decision": decision,
        "Prod AUC": f"{prod_auc_val:.3f}" if pd.notna(prod_auc_val) else "—",
        "Prod F1": f"{prod_f1_val:.3f}" if pd.notna(prod_f1_val) else "—",
        "New AUC": f"{new_auc_val:.3f}" if show_new and pd.notna(new_auc_val) else "—",
        "New F1": f"{new_f1_val:.3f}" if show_new and pd.notna(new_f1_val) else "—",
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# PER-BATCH JOURNEY
# ---------------------------------------------------------------------------
st.markdown("## Batch-by-Batch Journey")

for _, r in report_df.iterrows():
    b = int(r["batch"])
    decision = r.get("decision", None) or "—"
    drift = r.get("drift_share")
    n_drifted = r.get("n_drifted")
    run_id = r.get("run_id")

    # Header with badge
    if decision == "BASELINE":
        badge = ":blue[Baseline]"
        header_suffix = "Initial Training — Baseline"
    elif decision == "SKIP_NO_DRIFT":
        badge = ":green[No Drift]"
        header_suffix = "No Drift Detected"
    elif decision in ("DEPLOY", "DEPLOY_REFRESH"):
        badge = ":red[Drift Detected → Retrained]"
        header_suffix = "Drift Detected — Retrained"
    elif decision == "SKIP":
        badge = ":orange[Drift Detected — Regression]"
        header_suffix = "Drift Detected — Deployment Skipped"
    else:
        badge = ""
        header_suffix = ""

    st.markdown(f"### Batch {b}: {header_suffix} {badge}")

    # Drift info (not for baseline)
    if b > 0 and pd.notna(drift):
        above = "above" if drift >= DRIFT_THRESHOLD else "below"
        st.markdown(
            f"**Drift share:** {drift:.3f} ({above} {DRIFT_THRESHOLD} threshold) "
            f"&nbsp;|&nbsp; **Drifted features:** {int(n_drifted) if pd.notna(n_drifted) else '—'}"
        )
    elif b == 0:
        st.markdown("*Baseline — no drift analysis (this is the reference point)*")

    # Decision
    if decision not in ("BASELINE", "—"):
        st.markdown(f"**Decision:** `{decision}`")

    # Metrics
    prod_auc_val = r.get("prod_auc")
    prod_f1_val = r.get("prod_f1")
    prod_prec = r.get("prod_precision")
    prod_rec = r.get("prod_recall")
    new_auc_val = r.get("new_auc")
    new_f1_val = r.get("new_f1")
    new_prec = r.get("new_precision")
    new_rec = r.get("new_recall")
    auc_imp = r.get("auc_improvement")

    retrained = decision in ("DEPLOY", "DEPLOY_REFRESH")

    if retrained:
        # Show both prod and new model side by side
        left, right = st.columns(2)
        with left:
            st.markdown("**Production Model (before)**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AUC", f"{prod_auc_val:.3f}" if pd.notna(prod_auc_val) else "—")
            m2.metric("F1", f"{prod_f1_val:.3f}" if pd.notna(prod_f1_val) else "—")
            m3.metric("Precision", f"{prod_prec:.3f}" if pd.notna(prod_prec) else "—")
            m4.metric("Recall", f"{prod_rec:.3f}" if pd.notna(prod_rec) else "—")
        with right:
            st.markdown("**New Model (after)**")
            m1, m2, m3, m4 = st.columns(4)
            delta_auc = f"{auc_imp:+.3f}" if pd.notna(auc_imp) else None
            m1.metric("AUC", f"{new_auc_val:.3f}" if pd.notna(new_auc_val) else "—", delta=delta_auc)
            m2.metric("F1", f"{new_f1_val:.3f}" if pd.notna(new_f1_val) else "—")
            m3.metric("Precision", f"{new_prec:.3f}" if pd.notna(new_prec) else "—")
            m4.metric("Recall", f"{new_rec:.3f}" if pd.notna(new_rec) else "—")
    else:
        # Show prod metrics only (or baseline metrics)
        label = "Baseline Metrics" if decision == "BASELINE" else "Production Model on This Batch"
        st.markdown(f"**{label}**")
        m1, m2, m3, m4 = st.columns(4)
        # For non-retrained batches, new_* == prod_* (same model evaluated)
        auc_show = prod_auc_val if pd.notna(prod_auc_val) else new_auc_val
        f1_show = prod_f1_val if pd.notna(prod_f1_val) else new_f1_val
        prec_show = prod_prec if pd.notna(prod_prec) else new_prec
        rec_show = prod_rec if pd.notna(prod_rec) else new_rec
        m1.metric("AUC", f"{auc_show:.3f}" if pd.notna(auc_show) else "—")
        m2.metric("F1", f"{f1_show:.3f}" if pd.notna(f1_show) else "—")
        m3.metric("Precision", f"{prec_show:.3f}" if pd.notna(prec_show) else "—")
        m4.metric("Recall", f"{rec_show:.3f}" if pd.notna(rec_show) else "—")

    # ROC Curve (expandable)
    if run_id and pd.notna(run_id):
        with st.expander("View ROC Curve"):
            curves = load_roc_curves(mlflow_uri, run_id)
            if curves:
                fig_roc = go.Figure()
                if "production" in curves:
                    roc = curves["production"]
                    fig_roc.add_trace(go.Scatter(
                        x=roc["fpr"], y=roc["tpr"], mode="lines",
                        name=f"Production (AUC={roc['auc']:.3f})",
                        line=dict(color="blue", width=2),
                    ))
                if "new" in curves:
                    roc = curves["new"]
                    fig_roc.add_trace(go.Scatter(
                        x=roc["fpr"], y=roc["tpr"], mode="lines",
                        name=f"New Model (AUC={roc['auc']:.3f})",
                        line=dict(color="green", width=2),
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines",
                    name="Random",
                    line=dict(color="gray", width=1, dash="dash"),
                ))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    legend=dict(x=0.55, y=0.1),
                    height=450,
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.caption("No ROC curve artifacts available for this batch.")

    st.markdown("---")

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
st.markdown("## Final Summary")

starting_auc = None
ending_auc = None
if not report_df.empty and "new_auc" in report_df.columns:
    first = report_df.iloc[0]
    last = report_df.iloc[-1]
    starting_auc = first.get("prod_auc") if pd.notna(first.get("prod_auc")) else first.get("new_auc")
    ending_auc = last.get("new_auc") if pd.notna(last.get("new_auc")) else last.get("prod_auc")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Starting AUC (Batch 0)", f"{starting_auc:.3f}" if pd.notna(starting_auc) else "—")
with c2:
    st.metric(
        "Final AUC",
        f"{ending_auc:.3f}" if pd.notna(ending_auc) else "—",
        delta=f"{ending_auc - starting_auc:+.3f}" if pd.notna(starting_auc) and pd.notna(ending_auc) else None,
    )
with c3:
    st.metric("Retraining Events", f"{int(retrain_count)} of {max(len(batches) - 1, 0)} batches")

if pd.notna(starting_auc) and pd.notna(ending_auc) and retrain_count is not None:
    drift_count = 0
    if "drift_share" in report_df.columns:
        drift_count = (report_df["drift_share"].fillna(0) >= DRIFT_THRESHOLD).sum()
    st.markdown(
        f"The system detected significant drift in **{int(drift_count)}** of "
        f"**{max(len(batches) - 1, 0)}** batches and triggered **{int(retrain_count)}** retraining events, "
        f"moving AUC from **{starting_auc:.3f}** to **{ending_auc:.3f}** "
        f"({ending_auc - starting_auc:+.3f})."
    )
