#!/usr/bin/env python3
"""
Create synthetic drift batch for testing drift detection and automated retraining.

This script:
1. Loads batch_1.csv from S3
2. Injects controlled covariate drift (intensity 0.25)
3. Saves as batch_6_drifted.csv
4. Uploads to S3
"""

import pandas as pd
import numpy as np
import boto3
from io import StringIO
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def inject_covariate_drift(data, features, intensity=0.3, drift_type="shift", seed=None):
    """
    Inject synthetic covariate drift by shifting feature distributions.
    
    Args:
        data: DataFrame to inject drift into
        features: List of feature names to drift
        intensity: Drift magnitude (0.0 to 1.0)
        drift_type: Type of drift ('shift', 'scale', 'both')
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with drifted features
    """
    if seed is not None:
        np.random.seed(seed)
    
    drifted_data = data.copy()
    
    for feature in features:
        if feature not in data.columns:
            print(f"Warning: Feature '{feature}' not found in data")
            continue
        
        original_values = data[feature].copy()
        
        # Only drift numeric features
        if not pd.api.types.is_numeric_dtype(original_values):
            print(f"Skipping non-numeric feature: {feature}")
            continue
        
        if drift_type == "shift":
            # Shift mean by intensity * std
            feature_std = original_values.std()
            if feature_std > 0:
                shift_amount = intensity * feature_std
                drifted_data[feature] = original_values + shift_amount
                print(f"✓ Shifted {feature} by {shift_amount:.4f} (mean: {original_values.mean():.2f} → {drifted_data[feature].mean():.2f})")
        
        elif drift_type == "scale":
            # Scale variance by intensity
            feature_mean = original_values.mean()
            centered = original_values - feature_mean
            scaled = centered * (1 + intensity)
            drifted_data[feature] = scaled + feature_mean
            print(f"✓ Scaled {feature} by factor {1+intensity} (std: {original_values.std():.2f} → {drifted_data[feature].std():.2f})")
        
        elif drift_type == "both":
            # Apply both shift and scale
            feature_mean = original_values.mean()
            feature_std = original_values.std()
            if feature_std > 0:
                # First scale
                centered = original_values - feature_mean
                scaled = centered * (1 + intensity * 0.5)
                # Then shift
                shift_amount = intensity * 0.5 * feature_std
                drifted_data[feature] = scaled + feature_mean + shift_amount
                print(f"✓ Shifted+Scaled {feature}")
    
    return drifted_data


def main():
    print("=== Creating Synthetic Drift Batch ===\n")
    
    # Configuration
    S3_BUCKET = "health-predict-mlops-f9ac6509"
    SOURCE_KEY = "drift_monitoring/batch_data/batch_1.csv"
    TARGET_KEY = "drift_monitoring/batch_data/batch_6_drifted.csv"
    DRIFT_INTENSITY = 0.25  # 25% drift - should exceed MODERATE threshold (0.15)
    
    # Features to drift (select important numerical features)
    FEATURES_TO_DRIFT = [
        'time_in_hospital',
        'num_medications',
        'num_lab_procedures',
        'number_diagnoses',
        'num_procedures',
        'number_emergency',
        'number_inpatient',
        'number_outpatient'
    ]
    
    # Initialize S3
    s3 = boto3.client('s3')
    
    # Load batch_1 from S3
    print(f"Loading {SOURCE_KEY} from S3...")
    response = s3.get_object(Bucket=S3_BUCKET, Key=SOURCE_KEY)
    batch_1 = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    print(f"✓ Loaded {len(batch_1)} rows, {len(batch_1.columns)} columns\n")
    
    # Inject drift
    print(f"Injecting covariate drift (intensity={DRIFT_INTENSITY})...")
    print(f"Drifting {len(FEATURES_TO_DRIFT)} features: {', '.join(FEATURES_TO_DRIFT)}\n")
    
    batch_6_drifted = inject_covariate_drift(
        data=batch_1,
        features=FEATURES_TO_DRIFT,
        intensity=DRIFT_INTENSITY,
        drift_type="shift",
        seed=42
    )
    
    print(f"\n✓ Drift injection complete!")
    
    # Save locally
    local_path = "/tmp/batch_6_drifted.csv"
    batch_6_drifted.to_csv(local_path, index=False)
    print(f"✓ Saved to {local_path}")
    
    # Upload to S3
    print(f"\nUploading to S3: s3://{S3_BUCKET}/{TARGET_KEY}...")
    s3.upload_file(local_path, S3_BUCKET, TARGET_KEY)
    print(f"✓ Upload complete!")
    
    print("\n=== Summary ===")
    print(f"Source: s3://{S3_BUCKET}/{SOURCE_KEY}")
    print(f"Target: s3://{S3_BUCKET}/{TARGET_KEY}")
    print(f"Drift intensity: {DRIFT_INTENSITY} (25%)")
    print(f"Features drifted: {len(FEATURES_TO_DRIFT)}")
    print(f"Expected drift threshold: MODERATE (> 0.15)")
    print(f"\n✅ Synthetic drift batch created successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
