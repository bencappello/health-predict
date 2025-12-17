#!/usr/bin/env python3
"""
Lightweight drift detection script that focuses on generating reports
without consuming excessive memory.
"""

import pandas as pd
import boto3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
from datetime import datetime
import sys

def main():
    s3 = boto3.client('s3')
    bucket = 'health-predict-mlops-f9ac6509'
    
    print("Loading reference data...")
    # Load reference data (sample if needed to reduce memory)
    ref_obj = s3.get_object(Bucket=bucket, Key='drift_monitoring/reference_data/reference.csv')
    ref_df = pd.read_csv(ref_obj['Body'])
    print(f"Reference data shape: {ref_df.shape}")
    
    # Sample if too large
    if len(ref_df) > 5000:
        ref_df = ref_df.sample(n=5000, random_state=42)
        print(f"Sampled reference data to {len(ref_df)} rows")
    
    # Get batch to process
    batch_key = sys.argv[1] if len(sys.argv) > 1 else 'drift_monitoring/batch_data/batch_1.csv'
    print(f"Loading batch data: {batch_key}...")
    
    batch_obj = s3.get_object(Bucket=bucket, Key=batch_key)
    batch_df = pd.read_csv(batch_obj['Body'])
    print(f"Batch data shape: {batch_df.shape}")
    
    # Simple preprocessing - just drop target if present
    for col in ['readmitted_binary', 'readmitted']:
        ref_df = ref_df.drop(columns=[col], errors='ignore')
        batch_df = batch_df.drop(columns=[col], errors='ignore')
    
    # Align columns
    common_cols = list(set(ref_df.columns) & set(batch_df.columns))
    ref_df = ref_df[common_cols]
    batch_df = batch_df[common_cols]
    
    # Select only numeric columns for simplicity
    numeric_cols = ref_df.select_dtypes(include=['number']).columns.tolist()
    ref_df = ref_df[numeric_cols]
    batch_df = batch_df[numeric_cols]
    
    print(f"Using {len(numeric_cols)} numeric columns for drift detection")
    
    # Generate Evidently report
    print("Generating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=batch_df)
    
    # Save HTML report
    report_name = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(f"/tmp/{report_name}")
    print(f"Report saved to /tmp/{report_name}")
    
    # Upload to S3
    batch_name = batch_key.split('/')[-1].replace('.csv', '')
    s3_key = f"drift_monitoring/reports/{batch_name}/{report_name}"
    s3.upload_file(f"/tmp/{report_name}", bucket, s3_key)
    print(f"Report uploaded to s3://{bucket}/{s3_key}")
    
    # Extract drift metrics
    drift_dict = report.as_dict()
    dataset_drift = drift_dict['metrics'][0]['result']['dataset_drift']
    drift_share = drift_dict['metrics'][0]['result']['drift_share']
    
    print(f"\n=== DRIFT RESULTS ===")
    print(f"Dataset drift detected: {dataset_drift}")
    print(f"Drift share: {drift_share:.3f}")
    print(f"Report S3 location: s3://{bucket}/{s3_key}")
    
    # Log to MLflow (simplified)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("HealthPredict_Drift_Monitoring")
    
    with mlflow.start_run(run_name=f"drift_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_metric("dataset_drift", int(dataset_drift))
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_param("batch_key", batch_key)
        mlflow.log_param("report_s3_uri", f"s3://{bucket}/{s3_key}")
        print("Metrics logged to MLflow")
    
    return 0 if not dataset_drift else 1

if __name__ == "__main__":
    exit(main())
