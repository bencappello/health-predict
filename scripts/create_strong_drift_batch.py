#!/usr/bin/env python3
"""
Create STRONGER synthetic drift batch to force drift detection.

Strategy: Increase intensity to 0.40 and drift ALL numeric features
"""

import pandas as pd
import numpy as np
import boto3
from io import StringIO

def inject_strong_drift(data, intensity=0.40, seed=42):
    """Inject strong drift on ALL numeric features"""
    np.random.seed(seed)
    drifted_data = data.copy()
    
    print(f"Injecting STRONG drift (intensity={intensity}) on ALL numeric features...\n")
    
    drifted_count = 0
    for col in data.columns:
        if col in ['patient_nbr', 'encounter_id', 'readmitted', 'readmitted_binary']:
            continue  # Skip IDs and targets
        
        if pd.api.types.is_numeric_dtype(data[col]):
            original = data[col].copy()
            std = original.std()
            if std > 0:
                # Strong shift
                shift = intensity * std
                drifted_data[col] = original + shift
                print(f"✓ {col}: {original.mean():.2f} → {drifted_data[col].mean():.2f} (shift: +{shift:.2f})")
                drifted_count += 1
    
    print(f"\n✓ Drifted {drifted_count} numeric features")
    return drifted_data

def main():
    print("=== Creating STRONG Synthetic Drift Batch ===\n")
    
    S3_BUCKET = "health-predict-mlops-f9ac6509"
    SOURCE_KEY = "drift_monitoring/batch_data/batch_1.csv"
    TARGET_KEY = "drift_monitoring/batch_data/batch_7_strong_drift.csv"
    DRIFT_INTENSITY = 0.40  # 40% - very strong drift
    
    s3 = boto3.client('s3')
    
    # Load batch_1
    print(f"Loading {SOURCE_KEY} from S3...")
    response = s3.get_object(Bucket=S3_BUCKET, Key=SOURCE_KEY)
    batch_1 = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    print(f"✓ Loaded {len(batch_1)} rows, {len(batch_1.columns)} columns\n")
    
    # Inject STRONG drift
    batch_7_strong = inject_strong_drift(batch_1, intensity=DRIFT_INTENSITY)
    
    # Save locally
    local_path = "/tmp/batch_7_strong_drift.csv"
    batch_7_strong.to_csv(local_path, index=False)
    print(f"\n✓ Saved to {local_path}")
    
    # Upload to S3
    print(f"Uploading to S3: s3://{S3_BUCKET}/{TARGET_KEY}...")
    s3.upload_file(local_path, S3_BUCKET, TARGET_KEY)
    print(f"✓ Upload complete!")
    
    print("\n=== Summary ===")
    print(f"Drift intensity: {DRIFT_INTENSITY} (40% - STRONG)")
    print(f"Expected: This SHOULD trigger DRIFT_DETECTED")
    print(f"\n✅ Strong drift batch created!")

if __name__ == "__main__":
    exit(main())
