#!/usr/bin/env python3
"""
Data Drift Monitoring Script for Health Predict MLOps Pipeline

This script uses Evidently AI to detect data and concept drift by comparing 
new data batches against a reference dataset. It integrates with MLflow for 
logging drift metrics and reports, and outputs drift status for Airflow DAG integration.

Usage:
    python monitor_drift.py \
        --s3_new_data_path s3://bucket/new_batch.csv \
        --s3_reference_data_path s3://bucket/reference_data.csv \
        --mlflow_model_uri models:/HealthPredict_Model/Production \
        --s3_evidently_reports_path s3://bucket/drift_reports/run_id \
        --mlflow_experiment_name HealthPredict_Drift_Monitoring
"""

import argparse
import os
import sys
import tempfile
import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import numpy as np
import boto3
import mlflow
import mlflow.pyfunc
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Ensure src directory is in Python path to import feature_engineering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.feature_engineering import clean_data, engineer_features, get_preprocessor
except ImportError:
    print("Error: Unable to import from src.feature_engineering. Ensure it's in the Python path.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_df_from_s3(s3_uri: str, s3_client: boto3.client) -> pd.DataFrame:
    """Loads a CSV file from S3 into a pandas DataFrame."""
    # Parse S3 URI (s3://bucket/key)
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)  # Remove 's3://' prefix
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    bucket, key = parts
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"Successfully loaded '{key}' from S3 bucket '{bucket}'. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading '{key}' from S3: {e}")
        raise


def upload_file_to_s3(local_file_path: str, s3_uri: str, s3_client: boto3.client):
    """Upload a local file to S3."""
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    bucket, key = parts
    
    try:
        s3_client.upload_file(local_file_path, bucket, key)
        logger.info(f"Successfully uploaded '{local_file_path}' to '{s3_uri}'")
    except Exception as e:
        logger.error(f"Error uploading '{local_file_path}' to S3: {e}")
        raise


def prepare_data_for_drift_detection(df: pd.DataFrame, target_column: str = 'readmitted_binary') -> pd.DataFrame:
    """
    Apply the same data cleaning and feature engineering used in training.
    Remove target column if present for drift detection.
    """
    # Apply cleaning and feature engineering
    df_cleaned = clean_data(df)
    df_featured = engineer_features(df_cleaned)
    
    # Drop columns that should not be used for drift detection
    columns_to_drop = [target_column, 'readmitted', 'age']  # Drop original versions
    df_for_drift = df_featured.drop(columns=columns_to_drop, errors='ignore')
    
    logger.info(f"Data prepared for drift detection. Shape: {df_for_drift.shape}")
    logger.info(f"Columns: {list(df_for_drift.columns)}")
    
    return df_for_drift


def calculate_psi(reference_data: pd.Series, new_data: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for a feature.
    
    Args:
        reference_data: Reference dataset feature values
        new_data: New dataset feature values  
        bins: Number of bins for discretization
        
    Returns:
        PSI value (higher values indicate more drift)
    """
    try:
        # Ensure we have numeric data
        if not pd.api.types.is_numeric_dtype(reference_data):
            return np.nan
            
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference_data.dropna(), bins=bins)
        
        # Calculate bin proportions for both datasets
        ref_counts, _ = np.histogram(reference_data.dropna(), bins=bin_edges)
        new_counts, _ = np.histogram(new_data.dropna(), bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / np.sum(ref_counts)
        new_props = new_counts / np.sum(new_counts)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        ref_props = np.maximum(ref_props, epsilon)
        new_props = np.maximum(new_props, epsilon)
        
        # Calculate PSI
        psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        
        return psi
        
    except Exception as e:
        logger.warning(f"Error calculating PSI: {e}")
        return np.nan


def detect_concept_drift_with_predictions(
    reference_data: pd.DataFrame, 
    new_data: pd.DataFrame,
    model_uri: str,
    target_column: str = 'readmitted_binary'
) -> Dict[str, float]:
    """
    Detect concept drift by comparing model predictions on reference vs new data.
    
    Args:
        reference_data: Reference dataset with actual labels
        new_data: New dataset (may or may not have labels)
        model_uri: MLflow model URI to load for predictions
        target_column: Target column name
        
    Returns:
        Dictionary with concept drift metrics
    """
    logger.info(f"Starting concept drift detection using model: {model_uri}")
    
    concept_metrics = {}
    
    try:
        # Load the model
        logger.info("Loading model for concept drift detection...")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Prepare data for prediction (exclude target)
        ref_features = reference_data.drop(columns=[target_column], errors='ignore')
        new_features = new_data.drop(columns=[target_column], errors='ignore')
        
        # Align columns between reference and new data
        common_columns = list(set(ref_features.columns) & set(new_features.columns))
        ref_features_aligned = ref_features[common_columns]
        new_features_aligned = new_features[common_columns]
        
        logger.info(f"Making predictions on {len(ref_features_aligned)} reference and {len(new_features_aligned)} new samples...")
        
        # Get predictions
        ref_predictions = model.predict(ref_features_aligned)
        new_predictions = model.predict(new_features_aligned)
        
        # Convert predictions to consistent format
        if hasattr(ref_predictions, 'flatten'):
            ref_predictions = ref_predictions.flatten()
        if hasattr(new_predictions, 'flatten'):
            new_predictions = new_predictions.flatten()
        
        # Calculate prediction distribution drift
        ref_pred_mean = np.mean(ref_predictions)
        new_pred_mean = np.mean(new_predictions)
        ref_pred_std = np.std(ref_predictions)
        new_pred_std = np.std(new_predictions)
        
        concept_metrics['ref_prediction_mean'] = ref_pred_mean
        concept_metrics['new_prediction_mean'] = new_pred_mean
        concept_metrics['ref_prediction_std'] = ref_pred_std
        concept_metrics['new_prediction_std'] = new_pred_std
        concept_metrics['prediction_mean_shift'] = abs(new_pred_mean - ref_pred_mean)
        concept_metrics['prediction_std_ratio'] = new_pred_std / ref_pred_std if ref_pred_std > 0 else 1.0
        
        # Statistical tests on predictions
        try:
            # KS test on predictions
            ks_stat, ks_pval = ks_2samp(ref_predictions, new_predictions)
            concept_metrics['prediction_ks_statistic'] = ks_stat
            concept_metrics['prediction_ks_pvalue'] = ks_pval
            concept_metrics['prediction_ks_drift'] = ks_pval < 0.05
            
            # Wasserstein distance on predictions
            pred_wasserstein = wasserstein_distance(ref_predictions, new_predictions)
            concept_metrics['prediction_wasserstein'] = pred_wasserstein
            
        except Exception as e:
            logger.warning(f"Error in prediction drift statistical tests: {e}")
        
        # If we have actual labels in new data, calculate performance drift
        if target_column in new_data.columns:
            new_labels = new_data[target_column]
            ref_labels = reference_data[target_column] if target_column in reference_data.columns else None
            
            try:
                # Convert predictions to binary if needed (assuming threshold of 0.5)
                if len(np.unique(ref_predictions)) > 2:  # Probability predictions
                    new_pred_binary = (new_predictions > 0.5).astype(int)
                    ref_pred_binary = (ref_predictions > 0.5).astype(int) if ref_labels is not None else None
                else:  # Already binary
                    new_pred_binary = new_predictions.astype(int)
                    ref_pred_binary = ref_predictions.astype(int) if ref_labels is not None else None
                
                # Calculate accuracy on new data
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                
                new_accuracy = accuracy_score(new_labels, new_pred_binary)
                concept_metrics['new_data_accuracy'] = new_accuracy
                
                new_f1 = f1_score(new_labels, new_pred_binary, average='binary', zero_division=0)
                concept_metrics['new_data_f1'] = new_f1
                
                # Calculate AUC if we have probabilities
                if len(np.unique(new_predictions)) > 2:
                    try:
                        new_auc = roc_auc_score(new_labels, new_predictions)
                        concept_metrics['new_data_auc'] = new_auc
                    except Exception:
                        pass
                
                # Compare with reference performance if available
                if ref_labels is not None and len(ref_labels) > 0:
                    ref_accuracy = accuracy_score(ref_labels, ref_pred_binary)
                    ref_f1 = f1_score(ref_labels, ref_pred_binary, average='binary', zero_division=0)
                    
                    concept_metrics['ref_data_accuracy'] = ref_accuracy
                    concept_metrics['ref_data_f1'] = ref_f1
                    concept_metrics['accuracy_drift'] = abs(new_accuracy - ref_accuracy)
                    concept_metrics['f1_drift'] = abs(new_f1 - ref_f1)
                    concept_metrics['performance_degradation'] = ref_accuracy - new_accuracy
                    
                    # Performance-based concept drift flag
                    concept_metrics['performance_concept_drift'] = (
                        concept_metrics['accuracy_drift'] > 0.05 or 
                        concept_metrics['f1_drift'] > 0.05
                    )
                    
                    if len(np.unique(new_predictions)) > 2 and len(np.unique(ref_predictions)) > 2:
                        try:
                            ref_auc = roc_auc_score(ref_labels, ref_predictions)
                            concept_metrics['ref_data_auc'] = ref_auc
                            concept_metrics['auc_drift'] = abs(new_auc - ref_auc) if 'new_data_auc' in concept_metrics else 0
                        except Exception:
                            pass
                
            except Exception as e:
                logger.warning(f"Error calculating performance metrics: {e}")
        
        # Overall concept drift assessment
        concept_drift_indicators = []
        
        # Check prediction distribution drift
        if concept_metrics.get('prediction_ks_drift', False):
            concept_drift_indicators.append('prediction_distribution')
        
        # Check prediction mean shift (relative to prediction scale)
        pred_scale = max(abs(ref_pred_mean), abs(new_pred_mean), 0.1)
        if concept_metrics['prediction_mean_shift'] / pred_scale > 0.1:
            concept_drift_indicators.append('prediction_mean_shift')
        
        # Check performance degradation
        if concept_metrics.get('performance_concept_drift', False):
            concept_drift_indicators.append('performance_degradation')
        
        concept_metrics['concept_drift_detected'] = len(concept_drift_indicators) > 0
        concept_metrics['concept_drift_indicators'] = concept_drift_indicators
        concept_metrics['concept_drift_confidence'] = len(concept_drift_indicators) / 3  # Normalized
        
        logger.info(f"Concept drift analysis completed. Drift detected: {concept_metrics['concept_drift_detected']}")
        logger.info(f"Concept drift indicators: {concept_drift_indicators}")
        
    except Exception as e:
        logger.error(f"Error in concept drift detection: {e}")
        concept_metrics = {
            'concept_drift_detected': False,
            'concept_drift_error': str(e),
            'concept_drift_confidence': 0.0
        }
    
    return concept_metrics


def calculate_feature_drift_metrics(reference_data: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive drift metrics for each feature using multiple methods.
    
    Args:
        reference_data: Reference dataset
        new_data: New dataset
        
    Returns:
        Dictionary with drift metrics for each feature
    """
    logger.info("Calculating advanced drift metrics for all features...")
    
    feature_metrics = {}
    
    for feature in reference_data.columns:
        if feature not in new_data.columns:
            continue
            
        ref_values = reference_data[feature].dropna()
        new_values = new_data[feature].dropna()
        
        if len(ref_values) == 0 or len(new_values) == 0:
            continue
            
        metrics = {}
        
        # Only calculate advanced metrics for numeric features
        if pd.api.types.is_numeric_dtype(ref_values):
            try:
                # Kolmogorov-Smirnov test
                ks_statistic, ks_pvalue = ks_2samp(ref_values, new_values)
                metrics['ks_statistic'] = ks_statistic
                metrics['ks_pvalue'] = ks_pvalue
                metrics['ks_drift'] = ks_pvalue < 0.05  # Significant drift if p < 0.05
                
                # Wasserstein distance (Earth Mover's Distance)
                wasserstein_dist = wasserstein_distance(ref_values, new_values)
                metrics['wasserstein_distance'] = wasserstein_dist
                
                # Population Stability Index
                psi_value = calculate_psi(ref_values, new_values)
                metrics['psi'] = psi_value
                metrics['psi_drift'] = psi_value > 0.1  # PSI > 0.1 indicates drift
                
                # Statistical summary differences
                ref_mean, new_mean = ref_values.mean(), new_values.mean()
                ref_std, new_std = ref_values.std(), new_values.std()
                
                metrics['mean_shift'] = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0
                metrics['std_ratio'] = new_std / ref_std if ref_std > 0 else 1
                
                # Jensen-Shannon divergence (for distributions)
                try:
                    # Create histograms with same bins
                    bins = np.linspace(
                        min(ref_values.min(), new_values.min()),
                        max(ref_values.max(), new_values.max()),
                        20
                    )
                    ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
                    new_hist, _ = np.histogram(new_values, bins=bins, density=True)
                    
                    # Normalize to get probabilities
                    ref_hist = ref_hist / np.sum(ref_hist)
                    new_hist = new_hist / np.sum(new_hist)
                    
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    ref_hist = ref_hist + epsilon
                    new_hist = new_hist + epsilon
                    
                    # Calculate Jensen-Shannon divergence
                    m = 0.5 * (ref_hist + new_hist)
                    js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(new_hist, m)
                    metrics['js_divergence'] = js_div
                    
                except Exception as e:
                    logger.warning(f"Error calculating JS divergence for {feature}: {e}")
                    metrics['js_divergence'] = np.nan
                
            except Exception as e:
                logger.warning(f"Error calculating drift metrics for {feature}: {e}")
                
        else:
            # For categorical features, use different metrics
            try:
                # Chi-square test for categorical data
                ref_counts = ref_values.value_counts()
                new_counts = new_values.value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(new_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                new_aligned = [new_counts.get(cat, 0) for cat in all_categories]
                
                if sum(ref_aligned) > 0 and sum(new_aligned) > 0:
                    chi2_stat, chi2_pvalue = stats.chisquare(new_aligned, ref_aligned)
                    metrics['chi2_statistic'] = chi2_stat
                    metrics['chi2_pvalue'] = chi2_pvalue
                    metrics['chi2_drift'] = chi2_pvalue < 0.05
                    
            except Exception as e:
                logger.warning(f"Error calculating categorical drift metrics for {feature}: {e}")
        
        feature_metrics[feature] = metrics
    
    logger.info(f"Calculated drift metrics for {len(feature_metrics)} features")
    return feature_metrics


def detect_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame, model_uri: str = None, target_column: str = 'readmitted_binary') -> dict:
    """
    Use multiple methods to detect data drift between reference and new data.
    Combines Evidently AI with statistical tests and custom metrics.
    """
    logger.info("Starting comprehensive data drift detection...")
    
    # 1. Evidently AI Analysis
    logger.info("Running Evidently AI drift analysis...")
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    try:
        # Run the drift analysis
        report.run(reference_data=reference_data, current_data=new_data)
        
        # Extract drift results
        drift_results = report.as_dict()
        
        # Extract key metrics
        dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
        drift_share = drift_results['metrics'][0]['result']['drift_share']
        number_of_drifted_columns = drift_results['metrics'][0]['result']['number_of_drifted_columns']
        
        evidently_summary = {
            'dataset_drift_detected': dataset_drift,
            'drift_share': drift_share,
            'number_of_drifted_columns': number_of_drifted_columns,
            'total_columns': len(reference_data.columns),
        }
        
        logger.info(f"Evidently AI analysis completed. Dataset drift detected: {dataset_drift}")
        logger.info(f"Drift share: {drift_share:.3f}, Drifted columns: {number_of_drifted_columns}")
        
    except Exception as e:
        logger.error(f"Error during Evidently AI analysis: {e}")
        evidently_summary = {
            'dataset_drift_detected': True,  # Conservative assumption
            'drift_share': 1.0,
            'number_of_drifted_columns': len(reference_data.columns),
            'total_columns': len(reference_data.columns),
        }
        report = None
    
    # 2. Advanced Statistical Methods
    logger.info("Running advanced statistical drift analysis...")
    feature_drift_metrics = calculate_feature_drift_metrics(reference_data, new_data)
    
    # 3. Concept Drift Detection (if model provided)
    concept_drift_metrics = {}
    if model_uri:
        logger.info("Running concept drift detection with model predictions...")
        concept_drift_metrics = detect_concept_drift_with_predictions(
            reference_data, new_data, model_uri, target_column
        )
    
    # 4. Ensemble Drift Decision
    logger.info("Computing ensemble drift decision...")
    
    # Count features with drift according to different methods
    ks_drift_count = sum(1 for metrics in feature_drift_metrics.values() 
                        if metrics.get('ks_drift', False))
    psi_drift_count = sum(1 for metrics in feature_drift_metrics.values() 
                         if metrics.get('psi_drift', False))
    chi2_drift_count = sum(1 for metrics in feature_drift_metrics.values() 
                          if metrics.get('chi2_drift', False))
    
    # Calculate average drift scores
    avg_ks_statistic = np.nanmean([metrics.get('ks_statistic', 0) 
                                  for metrics in feature_drift_metrics.values()])
    avg_psi = np.nanmean([metrics.get('psi', 0) 
                         for metrics in feature_drift_metrics.values()])
    avg_wasserstein = np.nanmean([metrics.get('wasserstein_distance', 0) 
                                 for metrics in feature_drift_metrics.values()])
    avg_js_divergence = np.nanmean([metrics.get('js_divergence', 0) 
                                   for metrics in feature_drift_metrics.values()])
    
    # Create comprehensive drift summary
    comprehensive_summary = {
        # Evidently AI results
        'evidently_dataset_drift': evidently_summary['dataset_drift_detected'],
        'evidently_drift_share': evidently_summary['drift_share'],
        'evidently_drifted_columns': evidently_summary['number_of_drifted_columns'],
        
        # Statistical method results
        'ks_drift_count': ks_drift_count,
        'psi_drift_count': psi_drift_count,
        'chi2_drift_count': chi2_drift_count,
        'total_features_analyzed': len(feature_drift_metrics),
        
        # Average drift scores
        'avg_ks_statistic': avg_ks_statistic,
        'avg_psi': avg_psi,
        'avg_wasserstein_distance': avg_wasserstein,
        'avg_js_divergence': avg_js_divergence,
        
        # Ensemble decision metrics
        'statistical_drift_share': (ks_drift_count + psi_drift_count + chi2_drift_count) / (3 * len(feature_drift_metrics)) if feature_drift_metrics else 0,
        'consensus_drift_detected': None,  # To be calculated
        
        # Concept drift metrics (if available)
        'concept_drift_detected': concept_drift_metrics.get('concept_drift_detected', False),
        'concept_drift_confidence': concept_drift_metrics.get('concept_drift_confidence', 0.0),
        'concept_drift_indicators': concept_drift_metrics.get('concept_drift_indicators', []),
        
        # Metadata
        'drift_timestamp': datetime.now().isoformat(),
        'feature_level_metrics': feature_drift_metrics,
        'concept_drift_metrics': concept_drift_metrics
    }
    
    # Ensemble drift decision using multiple criteria
    drift_indicators = []
    
    # Evidently AI says drift
    if evidently_summary['dataset_drift_detected']:
        drift_indicators.append('evidently')
    
    # High statistical drift share (>25% of features show drift)
    if comprehensive_summary['statistical_drift_share'] > 0.25:
        drift_indicators.append('statistical_share')
    
    # High average drift scores
    if avg_psi > 0.1:  # PSI threshold
        drift_indicators.append('psi')
    if avg_ks_statistic > 0.3:  # KS threshold
        drift_indicators.append('ks')
    if avg_js_divergence > 0.1:  # JS divergence threshold
        drift_indicators.append('js_divergence')
    
    # Concept drift detected
    if concept_drift_metrics.get('concept_drift_detected', False):
        drift_indicators.append('concept_drift')
    
    # Consensus: drift detected if multiple methods agree
    consensus_drift = len(drift_indicators) >= 2
    comprehensive_summary['consensus_drift_detected'] = consensus_drift
    comprehensive_summary['drift_indicators'] = drift_indicators
    comprehensive_summary['confidence_score'] = len(drift_indicators) / 5  # Normalized confidence
    
    logger.info(f"Comprehensive drift analysis completed.")
    logger.info(f"Consensus drift detected: {consensus_drift}")
    logger.info(f"Drift indicators: {drift_indicators}")
    logger.info(f"Confidence score: {comprehensive_summary['confidence_score']:.3f}")
    
    return comprehensive_summary, report


def main():
    parser = argparse.ArgumentParser(description="Data Drift Monitoring for Health Predict")
    
    # Required arguments
    parser.add_argument('--s3_new_data_path', required=True,
                       help='S3 URI for the new data batch to analyze (e.g., s3://bucket/new_batch.csv)')
    parser.add_argument('--s3_reference_data_path', required=True,
                       help='S3 URI for the reference dataset (e.g., s3://bucket/reference_data.csv)')
    parser.add_argument('--s3_evidently_reports_path', required=True,
                       help='S3 prefix to save Evidently HTML reports (e.g., s3://bucket/drift_reports/run_id)')
    
    # MLflow configuration
    parser.add_argument('--mlflow_tracking_uri', 
                       default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
                       help='MLflow tracking server URI')
    parser.add_argument('--mlflow_experiment_name', 
                       default="HealthPredict_Drift_Monitoring",
                       help='MLflow experiment name for logging drift results')
    
    # Optional arguments
    parser.add_argument('--target_column', default='readmitted_binary',
                       help='Target column name to exclude from drift detection')
    parser.add_argument('--drift_threshold', type=float, default=0.1,
                       help='Threshold for dataset drift detection (0.0-1.0)')
    parser.add_argument('--mlflow_model_uri', 
                       help='MLflow model URI for concept drift detection (e.g., models:/ModelName/Production)')
    parser.add_argument('--enable_concept_drift', action='store_true',
                       help='Enable concept drift detection using model predictions')
    
    args = parser.parse_args()
    
    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"drift_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_param("s3_new_data_path", args.s3_new_data_path)
            mlflow.log_param("s3_reference_data_path", args.s3_reference_data_path)
            mlflow.log_param("drift_threshold", args.drift_threshold)
            mlflow.log_param("target_column", args.target_column)
            mlflow.log_param("mlflow_model_uri", args.mlflow_model_uri)
            mlflow.log_param("enable_concept_drift", args.enable_concept_drift)
            
            # Load data from S3
            logger.info("Loading reference data...")
            reference_df = load_df_from_s3(args.s3_reference_data_path, s3_client)
            
            logger.info("Loading new batch data...")
            new_batch_df = load_df_from_s3(args.s3_new_data_path, s3_client)
            
            # Prepare data for drift detection
            logger.info("Preparing reference data for drift detection...")
            reference_prepared = prepare_data_for_drift_detection(reference_df, args.target_column)
            
            logger.info("Preparing new batch data for drift detection...")
            new_batch_prepared = prepare_data_for_drift_detection(new_batch_df, args.target_column)
            
            # Ensure column alignment
            common_columns = list(set(reference_prepared.columns) & set(new_batch_prepared.columns))
            if len(common_columns) != len(reference_prepared.columns):
                logger.warning(f"Column mismatch. Reference: {len(reference_prepared.columns)}, "
                              f"New batch: {len(new_batch_prepared.columns)}, Common: {len(common_columns)}")
            
            # Use only common columns for drift detection
            reference_aligned = reference_prepared[common_columns]
            new_batch_aligned = new_batch_prepared[common_columns]
            
            # Detect drift (with optional concept drift detection)
            model_uri = args.mlflow_model_uri if args.enable_concept_drift else None
            drift_summary, drift_report = detect_data_drift(
                reference_aligned, new_batch_aligned, model_uri, args.target_column
            )
            
            # Log comprehensive drift metrics to MLflow
            # Evidently AI metrics
            mlflow.log_metric("evidently_dataset_drift", int(drift_summary['evidently_dataset_drift']))
            mlflow.log_metric("evidently_drift_share", drift_summary['evidently_drift_share'])
            mlflow.log_metric("evidently_drifted_columns", drift_summary['evidently_drifted_columns'])
            
            # Statistical method metrics
            mlflow.log_metric("ks_drift_count", drift_summary['ks_drift_count'])
            mlflow.log_metric("psi_drift_count", drift_summary['psi_drift_count'])
            mlflow.log_metric("chi2_drift_count", drift_summary['chi2_drift_count'])
            mlflow.log_metric("total_features_analyzed", drift_summary['total_features_analyzed'])
            
            # Average drift scores
            mlflow.log_metric("avg_ks_statistic", drift_summary['avg_ks_statistic'])
            mlflow.log_metric("avg_psi", drift_summary['avg_psi'])
            mlflow.log_metric("avg_wasserstein_distance", drift_summary['avg_wasserstein_distance'])
            mlflow.log_metric("avg_js_divergence", drift_summary['avg_js_divergence'])
            
            # Ensemble metrics
            mlflow.log_metric("statistical_drift_share", drift_summary['statistical_drift_share'])
            mlflow.log_metric("consensus_drift_detected", int(drift_summary['consensus_drift_detected']))
            mlflow.log_metric("confidence_score", drift_summary['confidence_score'])
            
            # Log drift indicators as parameter
            mlflow.log_param("drift_indicators", ",".join(drift_summary['drift_indicators']))
            
            # Log concept drift metrics if available
            if drift_summary['concept_drift_metrics']:
                concept_metrics = drift_summary['concept_drift_metrics']
                mlflow.log_metric("concept_drift_detected", int(concept_metrics.get('concept_drift_detected', False)))
                mlflow.log_metric("concept_drift_confidence", concept_metrics.get('concept_drift_confidence', 0.0))
                mlflow.log_param("concept_drift_indicators", ",".join(concept_metrics.get('concept_drift_indicators', [])))
                
                # Log prediction metrics if available
                for metric_name in ['ref_prediction_mean', 'new_prediction_mean', 'prediction_mean_shift',
                                   'prediction_ks_statistic', 'prediction_wasserstein', 'new_data_accuracy',
                                   'new_data_f1', 'accuracy_drift', 'f1_drift', 'performance_degradation']:
                    if metric_name in concept_metrics:
                        mlflow.log_metric(f"concept_{metric_name}", concept_metrics[metric_name])
            
            # Save and upload Evidently report
            with tempfile.TemporaryDirectory() as temp_dir:
                report_filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                local_report_path = os.path.join(temp_dir, report_filename)
                
                # Save HTML report
                drift_report.save_html(local_report_path)
                logger.info(f"Drift report saved locally: {local_report_path}")
                
                # Upload report to S3
                s3_report_uri = f"{args.s3_evidently_reports_path.rstrip('/')}/{report_filename}"
                upload_file_to_s3(local_report_path, s3_report_uri, s3_client)
                
                # Log report as MLflow artifact
                mlflow.log_artifact(local_report_path, artifact_path="drift_reports")
                
            # Log drift summary as MLflow artifact
            drift_summary_path = "drift_summary.txt"
            with open(drift_summary_path, 'w') as f:
                for key, value in drift_summary.items():
                    f.write(f"{key}: {value}\n")
            mlflow.log_artifact(drift_summary_path, artifact_path="drift_summary")
            os.remove(drift_summary_path)  # Clean up local file
            
            # Determine drift status using consensus decision
            is_drift_detected = drift_summary['consensus_drift_detected']
            
            # Log final drift decision
            mlflow.log_metric("drift_decision", int(is_drift_detected))
            mlflow.set_tag("drift_status", "DRIFT_DETECTED" if is_drift_detected else "NO_DRIFT")
            mlflow.set_tag("confidence_score", f"{drift_summary['confidence_score']:.3f}")
            
            # Output for Airflow (captured by XCom)
            drift_status = "DRIFT_DETECTED" if is_drift_detected else "NO_DRIFT"
            print(drift_status)  # This will be captured by Airflow
            
            logger.info(f"Drift monitoring completed. Status: {drift_status}")
            logger.info(f"Consensus drift: {is_drift_detected}, Confidence: {drift_summary['confidence_score']:.3f}")
            logger.info(f"Evidently drift share: {drift_summary['evidently_drift_share']:.3f} (threshold: {args.drift_threshold})")
            logger.info(f"Statistical drift share: {drift_summary['statistical_drift_share']:.3f}")
            
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        print("DRIFT_MONITORING_ERROR")  # Output for Airflow
        raise


if __name__ == "__main__":
    main() 