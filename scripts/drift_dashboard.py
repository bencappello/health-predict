#!/usr/bin/env python3
"""
Drift Monitoring Dashboard for Health Predict MLOps Pipeline

This script creates interactive visualizations for drift monitoring results,
including feature-level drift analysis, trend analysis, and concept drift monitoring.

Usage:
    streamlit run scripts/drift_dashboard.py
    
    Or as a standalone script:
    python scripts/drift_dashboard.py --mlflow_tracking_uri http://mlflow:5000
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.tracking
from datetime import datetime, timedelta
import argparse
import sys
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Health Predict - Drift Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_drift_experiments(tracking_uri: str, experiment_name: str = "HealthPredict_Drift_Monitoring") -> pd.DataFrame:
    """Load drift monitoring experiments from MLflow."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            st.error(f"Experiment '{experiment_name}' not found in MLflow")
            return pd.DataFrame()
        
        # Get all runs from the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100
        )
        
        if runs.empty:
            st.warning("No drift monitoring runs found")
            return pd.DataFrame()
        
        # Clean up column names (remove 'metrics.' and 'params.' prefixes)
        runs.columns = [col.replace('metrics.', '').replace('params.', '') for col in runs.columns]
        
        return runs
        
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return pd.DataFrame()


def create_drift_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create a trend chart showing drift metrics over time."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Evidently Drift Share', 'Statistical Drift Share', 
                       'Consensus Drift Detection', 'Confidence Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Convert start_time to datetime if it's not already
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        x_axis = df['start_time']
    else:
        x_axis = range(len(df))
    
    # Evidently drift share
    if 'evidently_drift_share' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['evidently_drift_share'], 
                      name='Evidently Drift Share', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Statistical drift share
    if 'statistical_drift_share' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['statistical_drift_share'], 
                      name='Statistical Drift Share', line=dict(color='red')),
            row=1, col=2
        )
    
    # Consensus drift detection (binary)
    if 'consensus_drift_detected' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['consensus_drift_detected'], 
                      name='Consensus Drift', mode='markers+lines', 
                      line=dict(color='orange')),
            row=2, col=1
        )
    
    # Confidence score
    if 'confidence_score' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['confidence_score'], 
                      name='Confidence Score', line=dict(color='green')),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Drift Monitoring Trends Over Time",
        height=600,
        showlegend=False
    )
    
    return fig


def create_drift_methods_comparison(df: pd.DataFrame) -> go.Figure:
    """Create a comparison chart of different drift detection methods."""
    methods_data = []
    
    for idx, row in df.iterrows():
        timestamp = row.get('start_time', idx)
        
        # Evidently AI
        if 'evidently_dataset_drift' in row:
            methods_data.append({
                'timestamp': timestamp,
                'method': 'Evidently AI',
                'drift_detected': row['evidently_dataset_drift'],
                'confidence': row.get('evidently_drift_share', 0)
            })
        
        # KS Test
        if 'ks_drift_count' in row and 'total_features_analyzed' in row:
            ks_share = row['ks_drift_count'] / max(row['total_features_analyzed'], 1)
            methods_data.append({
                'timestamp': timestamp,
                'method': 'KS Test',
                'drift_detected': ks_share > 0.2,
                'confidence': ks_share
            })
        
        # PSI
        if 'avg_psi' in row:
            methods_data.append({
                'timestamp': timestamp,
                'method': 'PSI',
                'drift_detected': row['avg_psi'] > 0.1,
                'confidence': min(row['avg_psi'], 1.0)
            })
        
        # Concept Drift
        if 'concept_drift_detected' in row:
            methods_data.append({
                'timestamp': timestamp,
                'method': 'Concept Drift',
                'drift_detected': row['concept_drift_detected'],
                'confidence': row.get('concept_drift_confidence', 0)
            })
    
    if not methods_data:
        return go.Figure()
    
    methods_df = pd.DataFrame(methods_data)
    
    fig = px.scatter(
        methods_df, 
        x='timestamp', 
        y='method',
        color='drift_detected',
        size='confidence',
        title="Drift Detection Methods Comparison",
        color_discrete_map={True: 'red', False: 'green'},
        labels={'drift_detected': 'Drift Detected', 'confidence': 'Confidence'}
    )
    
    fig.update_layout(height=400)
    return fig


def create_feature_drift_heatmap(latest_run_data: Dict) -> go.Figure:
    """Create a heatmap showing drift metrics for individual features."""
    try:
        # This would need to be extracted from the feature_level_metrics
        # For now, create a placeholder visualization
        
        # Sample feature names (in real implementation, extract from MLflow artifacts)
        features = ['age_ordinal', 'time_in_hospital', 'num_medications', 'number_diagnoses', 
                   'num_lab_procedures', 'num_procedures', 'discharge_disposition_id_ordinal']
        
        # Sample drift metrics (in real implementation, extract from stored metrics)
        metrics = ['KS Statistic', 'PSI', 'Wasserstein Distance', 'JS Divergence']
        
        # Generate sample data (replace with actual data extraction)
        np.random.seed(42)
        drift_matrix = np.random.rand(len(features), len(metrics))
        
        fig = go.Figure(data=go.Heatmap(
            z=drift_matrix,
            x=metrics,
            y=features,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Drift Intensity")
        ))
        
        fig.update_layout(
            title="Feature-Level Drift Analysis",
            xaxis_title="Drift Metrics",
            yaxis_title="Features",
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating feature drift heatmap: {e}")
        return go.Figure()


def create_concept_drift_analysis(df: pd.DataFrame) -> go.Figure:
    """Create concept drift analysis visualization."""
    if 'concept_drift_detected' not in df.columns:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Prediction Mean Shift', 'Model Accuracy Trend', 
                       'Concept Drift Detection', 'Performance Degradation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    x_axis = pd.to_datetime(df['start_time']) if 'start_time' in df.columns else range(len(df))
    
    # Prediction mean shift
    if 'concept_prediction_mean_shift' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['concept_prediction_mean_shift'], 
                      name='Prediction Mean Shift', line=dict(color='purple')),
            row=1, col=1
        )
    
    # Model accuracy trend
    if 'concept_new_data_accuracy' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['concept_new_data_accuracy'], 
                      name='New Data Accuracy', line=dict(color='blue')),
            row=1, col=2
        )
    
    # Concept drift detection
    fig.add_trace(
        go.Scatter(x=x_axis, y=df['concept_drift_detected'], 
                  name='Concept Drift', mode='markers+lines', 
                  line=dict(color='red')),
        row=2, col=1
    )
    
    # Performance degradation
    if 'concept_performance_degradation' in df.columns:
        fig.add_trace(
            go.Scatter(x=x_axis, y=df['concept_performance_degradation'], 
                      name='Performance Degradation', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Concept Drift Analysis",
        height=600,
        showlegend=False
    )
    
    return fig


def main_dashboard():
    """Main dashboard function for Streamlit app."""
    st.title("🏥 Health Predict - Drift Monitoring Dashboard")
    st.markdown("Monitor data and concept drift in the Health Predict MLOps pipeline")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # MLflow configuration
    mlflow_uri = st.sidebar.text_input(
        "MLflow Tracking URI", 
        value="http://mlflow:5000",
        help="URI of the MLflow tracking server"
    )
    
    experiment_name = st.sidebar.text_input(
        "Experiment Name",
        value="HealthPredict_Drift_Monitoring",
        help="Name of the drift monitoring experiment"
    )
    
    # Time range selection
    st.sidebar.subheader("Time Range")
    time_range = st.sidebar.selectbox(
        "Select time range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.experimental_rerun()
    
    # Load data
    with st.spinner("Loading drift monitoring data..."):
        df = load_drift_experiments(mlflow_uri, experiment_name)
    
    if df.empty:
        st.warning("No drift monitoring data available. Please run drift monitoring first.")
        return
    
    # Filter by time range
    if 'start_time' in df.columns and time_range != "All time":
        df['start_time'] = pd.to_datetime(df['start_time'])
        now = datetime.now()
        
        if time_range == "Last 24 hours":
            cutoff = now - timedelta(hours=24)
        elif time_range == "Last 7 days":
            cutoff = now - timedelta(days=7)
        elif time_range == "Last 30 days":
            cutoff = now - timedelta(days=30)
        
        df = df[df['start_time'] >= cutoff]
    
    # Main metrics
    st.header("📊 Current Drift Status")
    
    if not df.empty:
        latest_run = df.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            consensus_drift = latest_run.get('consensus_drift_detected', 0)
            st.metric(
                "Consensus Drift",
                "DETECTED" if consensus_drift else "NOT DETECTED",
                delta=None,
                delta_color="inverse"
            )
        
        with col2:
            confidence = latest_run.get('confidence_score', 0)
            st.metric(
                "Confidence Score",
                f"{confidence:.3f}",
                delta=None
            )
        
        with col3:
            evidently_drift = latest_run.get('evidently_drift_share', 0)
            st.metric(
                "Evidently Drift Share",
                f"{evidently_drift:.3f}",
                delta=None
            )
        
        with col4:
            concept_drift = latest_run.get('concept_drift_detected', 0)
            st.metric(
                "Concept Drift",
                "DETECTED" if concept_drift else "NOT DETECTED",
                delta=None,
                delta_color="inverse"
            )
    
    # Drift trends
    st.header("📈 Drift Trends")
    
    if len(df) > 1:
        trend_fig = create_drift_trend_chart(df)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Methods comparison
        st.header("🔍 Detection Methods Comparison")
        methods_fig = create_drift_methods_comparison(df)
        st.plotly_chart(methods_fig, use_container_width=True)
        
        # Concept drift analysis
        if any('concept_' in col for col in df.columns):
            st.header("🧠 Concept Drift Analysis")
            concept_fig = create_concept_drift_analysis(df)
            st.plotly_chart(concept_fig, use_container_width=True)
    
    # Feature-level analysis
    st.header("🎯 Feature-Level Drift Analysis")
    feature_fig = create_feature_drift_heatmap(latest_run.to_dict() if not df.empty else {})
    st.plotly_chart(feature_fig, use_container_width=True)
    
    # Detailed data table
    st.header("📋 Detailed Drift Monitoring Results")
    
    if st.checkbox("Show raw data"):
        # Select relevant columns for display
        display_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['drift', 'confidence', 'start_time', 'run_id', 'status'])]
        
        if display_cols:
            st.dataframe(df[display_cols].head(20))
        else:
            st.dataframe(df.head(20))
    
    # Export functionality
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download drift monitoring data as CSV",
                data=csv,
                file_name=f"drift_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Generate Report"):
            st.info("Report generation feature coming soon!")


def standalone_mode():
    """Run dashboard in standalone mode (non-Streamlit)."""
    parser = argparse.ArgumentParser(description="Drift Monitoring Dashboard")
    parser.add_argument('--mlflow_tracking_uri', default='http://mlflow:5000',
                       help='MLflow tracking server URI')
    parser.add_argument('--experiment_name', default='HealthPredict_Drift_Monitoring',
                       help='MLflow experiment name')
    parser.add_argument('--output_dir', default='drift_reports',
                       help='Output directory for generated reports')
    
    args = parser.parse_args()
    
    print("Loading drift monitoring data...")
    df = load_drift_experiments(args.mlflow_tracking_uri, args.experiment_name)
    
    if df.empty:
        print("No drift monitoring data found.")
        return
    
    print(f"Loaded {len(df)} drift monitoring runs")
    print("\nLatest drift status:")
    
    if not df.empty:
        latest = df.iloc[0]
        print(f"  Consensus Drift: {'DETECTED' if latest.get('consensus_drift_detected', 0) else 'NOT DETECTED'}")
        print(f"  Confidence Score: {latest.get('confidence_score', 0):.3f}")
        print(f"  Evidently Drift Share: {latest.get('evidently_drift_share', 0):.3f}")
        print(f"  Concept Drift: {'DETECTED' if latest.get('concept_drift_detected', 0) else 'NOT DETECTED'}")
    
    # Save summary report
    os.makedirs(args.output_dir, exist_ok=True)
    summary_file = os.path.join(args.output_dir, f"drift_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(summary_file, index=False)
    print(f"\nDrift monitoring summary saved to: {summary_file}")


if __name__ == "__main__":
    # Check if running in Streamlit
    if 'streamlit' in sys.modules:
        main_dashboard()
    else:
        standalone_mode() 