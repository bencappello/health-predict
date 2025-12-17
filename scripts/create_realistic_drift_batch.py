#!/usr/bin/env python3
"""
Create realistic drift batch by stratified sampling from future_data.csv.

Strategy: Oversample patients with high number_diagnoses (>=8) to create
natural feature drift that will trigger drift detection while preserving
real target relationships.
"""

import pandas as pd
import boto3
from io import StringIO
import sys

def main():
    print("=== Creating Realistic Drift Batch (Stratified Sampling) ===\n")
    
    # Configuration
    S3_BUCKET = "health-predict-mlops-f9ac6509"
    FUTURE_KEY = "processed_data/future_data.csv"
    TARGET_KEY = "drift_monitoring/batch_data/batch_8_natural_drift.csv"
    SAMPLE_SIZE = 2000
    
    s3 = boto3.client('s3')
    
    # Load future data
    print(f"Loading {FUTURE_KEY} from S3...")
    response = s3.get_object(Bucket=S3_BUCKET, Key=FUTURE_KEY)
    future = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    print(f"✓ Loaded {len(future):,} rows")
    
    # Create readmitted_binary if needed
    if 'readmitted_binary' not in future.columns and 'readmitted' in future.columns:
        future['readmitted_binary'] = (future['readmitted'] == '<30').astype(int)
        print("✓ Created readmitted_binary column")
    
    # Stratified sampling: prioritize high-diagnoses patients
    print(f"\nStratified sampling (target: {SAMPLE_SIZE} rows)...")
    
    # Filter for high-diagnoses patients (natural drift subgroup)
    high_diag = future[future['number_diagnoses'] >= 8]
    print(f"  High diagnoses (>=8): {len(high_diag):,} available")
    
    # Sample from high-diagnoses group
    if len(high_diag) >= SAMPLE_SIZE:
        batch_8 = high_diag.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        # Take all high-diag and fill rest from remaining
        batch_8 = high_diag.copy()
        remaining = SAMPLE_SIZE - len(batch_8)
        other = future[future['number_diagnoses'] < 8].sample(n=remaining, random_state=42)
        batch_8 = pd.concat([batch_8, other])
    
    print(f"✓ Created batch with {len(batch_8):,} rows")
    
    # Verify drift characteristics
    print("\n=== Drift Characteristics (vs initial_train baseline) ===")
    # Load initial train for comparison
    response = s3.get_object(Bucket=S3_BUCKET, Key='processed_data/initial_train.csv')
    train = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    
    for col in ['number_diagnoses', 'num_medications', 'time_in_hospital', 'num_lab_procedures']:
        train_mean, train_std = train[col].mean(), train[col].std()
        batch_mean = batch_8[col].mean()
        shift = (batch_mean - train_mean) / train_std
        symbol = "↑" if shift > 0 else "↓"
        print(f"  {col}: {batch_mean:.2f} (shift: {symbol}{abs(shift):.2f} std)")
    
    # Check readmission rate
    print(f"\n  Readmission rate: {batch_8['readmitted_binary'].mean():.4f}")
    
    # Save locally
    local_path = "/tmp/batch_8_natural_drift.csv"
    batch_8.to_csv(local_path, index=False)
    print(f"\n✓ Saved to {local_path}")
    
    # Upload to S3
    print(f"Uploading to S3: s3://{S3_BUCKET}/{TARGET_KEY}...")
    s3.upload_file(local_path, S3_BUCKET, TARGET_KEY)
    print(f"✓ Upload complete!")
    
    print("\n=== Summary ===")
    print(f"Source: future_data.csv (stratified: number_diagnoses >= 8)")
    print(f"Target: s3://{S3_BUCKET}/{TARGET_KEY}")
    print(f"Size: {len(batch_8):,} rows")
    print(f"Expected: Should trigger DRIFT_DETECTED (>0.15 drift share)")
    print(f"\n✅ Realistic drift batch created successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
