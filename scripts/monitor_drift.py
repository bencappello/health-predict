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


def detect_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame) -> dict:
    """
    Use Evidently AI to detect data drift between reference and new data.
    Returns drift results and generates HTML report.
    """
    logger.info("Starting data drift detection with Evidently AI...")
    
    # Create Evidently report with DataDriftPreset
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
        
        drift_summary = {
            'dataset_drift_detected': dataset_drift,
            'drift_share': drift_share,
            'number_of_drifted_columns': number_of_drifted_columns,
            'total_columns': len(reference_data.columns),
            'drift_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Drift detection completed. Dataset drift detected: {dataset_drift}")
        logger.info(f"Drift share: {drift_share:.3f}, Drifted columns: {number_of_drifted_columns}")
        
        return drift_summary, report
        
    except Exception as e:
        logger.error(f"Error during drift detection: {e}")
        raise


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
            
            # Detect drift
            drift_summary, drift_report = detect_data_drift(reference_aligned, new_batch_aligned)
            
            # Log drift metrics to MLflow
            mlflow.log_metric("dataset_drift_detected", int(drift_summary['dataset_drift_detected']))
            mlflow.log_metric("drift_share", drift_summary['drift_share'])
            mlflow.log_metric("number_of_drifted_columns", drift_summary['number_of_drifted_columns'])
            mlflow.log_metric("total_columns", drift_summary['total_columns'])
            
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
            
            # Determine drift status based on threshold
            is_drift_detected = drift_summary['drift_share'] > args.drift_threshold
            
            # Log final drift decision
            mlflow.log_metric("drift_decision", int(is_drift_detected))
            mlflow.set_tag("drift_status", "DRIFT_DETECTED" if is_drift_detected else "NO_DRIFT")
            
            # Output for Airflow (captured by XCom)
            drift_status = "DRIFT_DETECTED" if is_drift_detected else "NO_DRIFT"
            print(drift_status)  # This will be captured by Airflow
            
            logger.info(f"Drift monitoring completed. Status: {drift_status}")
            logger.info(f"Drift share: {drift_summary['drift_share']:.3f} (threshold: {args.drift_threshold})")
            
    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")
        print("DRIFT_MONITORING_ERROR")  # Output for Airflow
        raise


if __name__ == "__main__":
    main() 