#!/usr/bin/env python3
"""
OPTIMIZED Data Drift Monitoring Script for Health Predict MLOps Pipeline

This version includes memory optimizations:
- Explicit memory cleanup with gc.collect()
- Reference data sampling to reduce memory footprint
- Simplified Evidently reporting
- Memory usage logging at key points

Usage:
    python monitor_drift_optimized.py \
        --s3_new_data_path s3://bucket/new_batch.csv \
        --s3_reference_data_path s3://bucket/reference.csv \
        --s3_evidently_reports_path s3://bucket/drift_reports/run_id \
        --mlflow_experiment_name HealthPredict_Drift_Monitoring
"""

import argparse
import os
import sys
import tempfile
import logging
import gc
from datetime import datetime
from io import StringIO

import pandas as pd
import numpy as np
import boto3
import mlflow
import mlflow.pyfunc
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Ensure src directory is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.feature_engineering import clean_data, engineer_features
except ImportError:
    print("Error: Unable to import from src.feature_engineering")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_memory_usage(stage: str):
    """Log current memory usage"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        logger.info(f"[MEMORY] {stage}: {mem_mb:.2f} MB")
    else:
        logger.debug(f"[MEMORY] {stage}: psutil not available")


def load_df_from_s3(s3_uri: str, s3_client: boto3.client) -> pd.DataFrame:
    """Loads a CSV file from S3 into a pandas DataFrame."""
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    bucket, key = parts
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"Loaded '{key}' from S3. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading '{key}' from S3: {e}")
        raise


def upload_file_to_s3(local_file_path: str, s3_uri: str, s3_client: boto3.client):
    """Upload a local file to S3."""
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    bucket, key = parts
    
    try:
        s3_client.upload_file(local_file_path, bucket, key)
        logger.info(f"Uploaded to '{s3_uri}'")
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        raise


def prepare_data_for_drift_detection(df: pd.DataFrame, target_column: str = 'readmitted_binary') -> pd.DataFrame:
    """
    Apply data cleaning and feature engineering.
    OPTIMIZED: Processes data in-place where possible to reduce copies.
    """
    logger.info(f"Preparing data for drift detection. Initial shape: {df.shape}")
    log_memory_usage("Before data preparation")
    
    # Apply cleaning and feature engineering
    df_cleaned = clean_data(df)
    del df  # Explicit cleanup
    gc.collect()
    
    df_featured = engineer_features(df_cleaned)
    del df_cleaned  # Explicit cleanup
    gc.collect()
    
    # Drop columns not needed for drift detection
    columns_to_drop = [target_column, 'readmitted', 'age']
    df_for_drift = df_featured.drop(columns=columns_to_drop, errors='ignore')
    del df_featured  # Explicit cleanup
    gc.collect()
    
    logger.info(f"Data prepared. Final shape: {df_for_drift.shape}")
    log_memory_usage("After data preparation")
    
    return df_for_drift


def calculate_psi(reference_data: pd.Series, new_data: pd.Series, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a feature."""
    try:
        if not pd.api.types.is_numeric_dtype(reference_data):
            return np.nan
            
        _, bin_edges = np.histogram(reference_data.dropna(), bins=bins)
        ref_counts, _ = np.histogram(reference_data.dropna(), bins=bin_edges)
        new_counts, _ = np.histogram(new_data.dropna(), bins=bin_edges)
        
        ref_props = ref_counts / np.sum(ref_counts)
        new_props = new_counts / np.sum(new_counts)
        
        epsilon = 1e-6
        ref_props = np.maximum(ref_props, epsilon)
        new_props = np.maximum(new_props, epsilon)
        
        psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        return psi
    except Exception as e:
        logger.warning(f"Error calculating PSI: {e}")
        return np.nan


def calculate_feature_drift_metrics(reference_data: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate drift metrics for each feature.
    OPTIMIZED: Limits to essential metrics only.
    """
    logger.info("Calculating drift metrics...")
    log_memory_usage("Before drift metrics calculation")
    
    feature_metrics = {}
    
    # Process only first 20 features to save memory
    features_to_process = list(reference_data.columns)[:20]
    
    for feature in features_to_process:
        if feature not in new_data.columns:
            continue
            
        ref_values = reference_data[feature].dropna()
        new_values = new_data[feature].dropna()
        
        if len(ref_values) == 0 or len(new_values) == 0:
            continue
            
        metrics = {}
        
        if pd.api.types.is_numeric_dtype(ref_values):
            try:
                # KS test
                ks_statistic, ks_pvalue = ks_2samp(ref_values, new_values)
                metrics['ks_statistic'] = ks_statistic
                metrics['ks_pvalue'] = ks_pvalue
                metrics['ks_drift'] = ks_pvalue < 0.05
                
                # Wasserstein distance
                wasserstein_dist = wasserstein_distance(ref_values, new_values)
                metrics['wasserstein_distance'] = wasserstein_dist
                
                # PSI
                psi_value = calculate_psi(ref_values, new_values)
                metrics['psi'] = psi_value
                metrics['psi_drift'] = psi_value > 0.1
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for {feature}: {e}")
        
        feature_metrics[feature] = metrics
    
    log_memory_usage("After drift metrics calculation")
    logger.info(f"Calculated metrics for {len(feature_metrics)} features")
    return feature_metrics


def detect_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame) -> tuple:
    """
    OPTIMIZED drift detection using Evidently AI with simplified reporting.
    """
    logger.info("Starting drift detection...")
    log_memory_usage("Before Evidently")
    
    # 1. Evidently AI Analysis with CONFIGURED threshold
    # drift_share=0.3 means trigger "Dataset Drift" when >= 30% of features drift
    # This aligns with our MODERATE threshold strategy
    report = Report(metrics=[DataDriftPreset(drift_share=0.3)])
    
    try:
        report.run(reference_data=reference_data, current_data=new_data)
        
        drift_results = report.as_dict()
        dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
        drift_share = drift_results['metrics'][0]['result']['drift_share']
        number_of_drifted_columns = drift_results['metrics'][0]['result']['number_of_drifted_columns']
        
        evidently_summary = {
            'dataset_drift_detected': dataset_drift,
            'drift_share': drift_share,
            'number_of_drifted_columns': number_of_drifted_columns,
            'total_columns': len(reference_data.columns),
        }
        
        logger.info(f"Evidently: drift={dataset_drift}, share={drift_share:.3f}")
        
    except Exception as e:
        logger.error(f"Evidently error: {e}")
        evidently_summary = {
            'dataset_drift_detected': True,
            'drift_share': 1.0,
            'number_of_drifted_columns': len(reference_data.columns),
            'total_columns': len(reference_data.columns),
        }
        report = None
    
    log_memory_usage("After Evidently")
    
    # 2. Statistical Methods
    feature_drift_metrics = calculate_feature_drift_metrics(reference_data, new_data)
    
    # 3. Aggregate results
    ks_drift_count = sum(1 for m in feature_drift_metrics.values() if m.get('ks_drift', False))
    psi_drift_count = sum(1 for m in feature_drift_metrics.values() if m.get('psi_drift', False))
    
    avg_ks = np.nanmean([m.get('ks_statistic', 0) for m in feature_drift_metrics.values()])
    avg_psi = np.nanmean([m.get('psi', 0) for m in feature_drift_metrics.values()])
    
    comprehensive_summary = {
        'evidently_dataset_drift': evidently_summary['dataset_drift_detected'],
        'evidently_drift_share': evidently_summary['drift_share'],
        'evidently_drifted_columns': evidently_summary['number_of_drifted_columns'],
        'ks_drift_count': ks_drift_count,
        'psi_drift_count': psi_drift_count,
        'total_features_analyzed': len(feature_drift_metrics),
        'avg_ks_statistic': avg_ks,
        'avg_psi': avg_psi,
        'drift_timestamp': datetime.now().isoformat(),
    }
    
    # Consensus decision
    drift_indicators = []
    if evidently_summary['dataset_drift_detected']:
        drift_indicators.append('evidently')
    if avg_psi > 0.1:
        drift_indicators.append('psi')
    if avg_ks > 0.3:
        drift_indicators.append('ks')
    
    consensus_drift = len(drift_indicators) >= 2
    comprehensive_summary['consensus_drift_detected'] = consensus_drift
    comprehensive_summary['drift_indicators'] = drift_indicators
    comprehensive_summary['confidence_score'] = len(drift_indicators) / 3
    
    logger.info(f"Consensus drift: {consensus_drift}, indicators: {drift_indicators}")
    log_memory_usage("End of drift detection")
    
    return comprehensive_summary, report


def main():
    parser = argparse.ArgumentParser(description="Optimized Drift Monitoring")
    
    parser.add_argument('--s3_new_data_path', required=True)
    parser.add_argument('--s3_reference_data_path', required=True)
    parser.add_argument('--s3_evidently_reports_path', required=True)
    parser.add_argument('--mlflow_tracking_uri', default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument('--mlflow_experiment_name', default="HealthPredict_Drift_Monitoring")
    parser.add_argument('--target_column', default='readmitted_binary')
    parser.add_argument('--drift_threshold', type=float, default=0.1)
    parser.add_argument('--max_reference_rows', type=int, default=5000, 
                       help='Sample reference data to this many rows to reduce memory')
    
    args = parser.parse_args()
    
    logger.info("=== OPTIMIZED DRIFT MONITORING STARTED ===")
    log_memory_usage("Script start")
    
    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    try:
        with mlflow.start_run(run_name=f"drift_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            mlflow.log_param("s3_new_data_path", args.s3_new_data_path)
            mlflow.log_param("s3_reference_data_path", args.s3_reference_data_path)
            mlflow.log_param("max_reference_rows", args.max_reference_rows)
            
            # Load reference data
            logger.info("Loading reference data...")
            reference_df = load_df_from_s3(args.s3_reference_data_path, s3_client)
            log_memory_usage("After loading reference")
            
            # OPTIMIZATION: Sample if too large
            if len(reference_df) > args.max_reference_rows:
                original_size = len(reference_df)
                reference_df = reference_df.sample(n=args.max_reference_rows, random_state=42)
                logger.info(f"Sampled reference data: {original_size} → {len(reference_df)} rows")
                mlflow.log_param("reference_sampled", True)
                mlflow.log_param("original_reference_rows", original_size)
            else:
                mlflow.log_param("reference_sampled", False)
            
            # Load new batch
            logger.info("Loading new batch...")
            new_batch_df = load_df_from_s3(args.s3_new_data_path, s3_client)
            log_memory_usage("After loading batch")
            
            # Prepare data
            logger.info("Preparing reference data...")
            reference_prepared = prepare_data_for_drift_detection(reference_df, args.target_column)
            del reference_df
            gc.collect()
            
            logger.info("Preparing new batch...")
            new_batch_prepared = prepare_data_for_drift_detection(new_batch_df, args.target_column)
            del new_batch_df
            gc.collect()
            
            # Align columns
            common_columns = list(set(reference_prepared.columns) & set(new_batch_prepared.columns))
            reference_aligned = reference_prepared[common_columns]
            new_batch_aligned = new_batch_prepared[common_columns]
            
            del reference_prepared, new_batch_prepared
            gc.collect()
            log_memory_usage("After data alignment")
            
            # Detect drift
            drift_summary, drift_report = detect_data_drift(reference_aligned, new_batch_aligned)
            
            # Log metrics
            mlflow.log_metric("evidently_dataset_drift", int(drift_summary['evidently_dataset_drift']))
            mlflow.log_metric("evidently_drift_share", drift_summary['evidently_drift_share'])
            mlflow.log_metric("ks_drift_count", drift_summary['ks_drift_count'])
            mlflow.log_metric("psi_drift_count", drift_summary['psi_drift_count'])
            mlflow.log_metric("avg_ks_statistic", drift_summary['avg_ks_statistic'])
            mlflow.log_metric("avg_psi", drift_summary['avg_psi'])
            mlflow.log_metric("consensus_drift_detected", int(drift_summary['consensus_drift_detected']))
            mlflow.log_metric("confidence_score", drift_summary['confidence_score'])
            
            # Save Evidently report
            if drift_report:
                with tempfile.TemporaryDirectory() as temp_dir:
                    report_filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    local_report_path = os.path.join(temp_dir, report_filename)
                    
                    drift_report.save_html(local_report_path)
                    logger.info(f"Report saved locally: {local_report_path}")
                    
                    s3_report_uri = f"{args.s3_evidently_reports_path.rstrip('/')}/{report_filename}"
                    upload_file_to_s3(local_report_path, s3_report_uri, s3_client)
                    
                    mlflow.log_artifact(local_report_path, artifact_path="drift_reports")
            
            # Final decision
            is_drift_detected = drift_summary['consensus_drift_detected']
            drift_status = "DRIFT_DETECTED" if is_drift_detected else "NO_DRIFT"
            
            mlflow.set_tag("drift_status", drift_status)
            mlflow.set_tag("optimized_version", "true")
            
            print(drift_status)
            
            logger.info(f"✅ COMPLETED: {drift_status}")
            log_memory_usage("Script end")
            
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        print("DRIFT_MONITORING_ERROR")
        raise


if __name__ == "__main__":
    main()
