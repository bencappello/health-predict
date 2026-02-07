#!/usr/bin/env python3
"""
Create drift-aware batches from future_data.csv for the Health Predict pipeline.

Batches 1-2: No drift (random samples matching initial_train distribution)
Batch 3: Gradual covariate drift (stratified oversample of older/sicker patients)
Batch 4: Strong covariate drift (aggressive oversampling + numeric mean shifts)
Batch 5: Concept drift (changed feature-target relationships + demographic shift)

Usage:
    python scripts/create_drift_aware_batches.py --bucket-name health-predict-mlops-f9ac6509
    python scripts/create_drift_aware_batches.py --bucket-name health-predict-mlops-f9ac6509 --dry-run
"""

import argparse
import logging
from io import StringIO

import boto3
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

S3_BUCKET_DEFAULT = "health-predict-mlops-f9ac6509"
FUTURE_DATA_KEY = "processed_data/future_data.csv"
REFERENCE_KEY = "processed_data/initial_train.csv"
BATCH_PREFIX = "drift_monitoring/batch_data"
DEFAULT_BATCH_SIZE = 2000


def load_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """Download CSV from S3 into DataFrame."""
    logger.info(f"Loading s3://{bucket}/{key}...")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def upload_to_s3(s3_client, bucket: str, key: str, df: pd.DataFrame):
    """Upload DataFrame as CSV to S3."""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    logger.info(f"  Uploaded to s3://{bucket}/{key}")


def ensure_readmitted_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Create readmitted_binary column if missing."""
    if 'readmitted_binary' not in df.columns and 'readmitted' in df.columns:
        df = df.copy()
        df['readmitted_binary'] = (df['readmitted'] != 'NO').astype(int)
        logger.info("  Created readmitted_binary column")
    return df


def create_no_drift_batch(future_data: pd.DataFrame, batch_size: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Create a batch with NO drift by random sampling from future_data.
    A random sample from the same population should closely match the
    initial_train distribution.
    """
    rng = np.random.RandomState(seed)
    n = min(batch_size, len(future_data))
    indices = rng.choice(len(future_data), size=n, replace=False)
    return future_data.iloc[indices].copy().reset_index(drop=True)


def create_gradual_drift_batch(future_data: pd.DataFrame, batch_size: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Batch 3: Gradual covariate drift via stratified oversampling.

    Strategy:
    - Oversample patients with age >= [70-80) (age_ordinal >= 7) to ~50% of batch
    - Oversample patients with number_diagnoses >= 8 to ~60% of batch
    - This shifts age and diagnosis distributions noticeably but not drastically
    """
    rng = np.random.RandomState(seed)

    # Identify subpopulations
    has_age = 'age' in future_data.columns
    if has_age:
        old_patients = future_data[future_data['age'].isin(['[70-80)', '[80-90)', '[90-100)'])]
        young_patients = future_data[~future_data['age'].isin(['[70-80)', '[80-90)', '[90-100)'])]
    else:
        # Fallback to age_ordinal if age column was already encoded
        old_patients = future_data[future_data.get('age_ordinal', pd.Series(dtype=float)) >= 7]
        young_patients = future_data[future_data.get('age_ordinal', pd.Series(dtype=float)) < 7]

    high_diag = future_data[future_data['number_diagnoses'] >= 8] if 'number_diagnoses' in future_data.columns else pd.DataFrame()

    # Target: 50% old patients, 60% high diagnoses (overlapping)
    n_old = min(int(batch_size * 0.50), len(old_patients))
    n_young = batch_size - n_old

    old_sample = old_patients.sample(n=n_old, random_state=rng, replace=len(old_patients) < n_old)
    young_sample = young_patients.sample(n=n_young, random_state=rng, replace=len(young_patients) < n_young)

    batch = pd.concat([old_sample, young_sample], ignore_index=True)

    # Further boost high-diagnosis patients by swapping some low-diag rows
    if len(high_diag) > 0 and 'number_diagnoses' in batch.columns:
        low_diag_mask = batch['number_diagnoses'] < 8
        n_swap = min(int(batch_size * 0.15), low_diag_mask.sum(), len(high_diag))
        if n_swap > 0:
            swap_indices = batch[low_diag_mask].sample(n=n_swap, random_state=rng).index
            replacements = high_diag.sample(n=n_swap, random_state=rng, replace=True)
            batch.loc[swap_indices] = replacements.values[:n_swap]

    return batch.sample(frac=1, random_state=rng).reset_index(drop=True)


def create_strong_drift_batch(future_data: pd.DataFrame, batch_size: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Batch 4: Strong covariate drift via aggressive oversampling + numeric shifts.

    Strategy:
    - Oversample age 80+ patients to ~40% of batch
    - Oversample number_diagnoses >= 9 to ~50% of batch
    - Add +0.3*std shift to time_in_hospital, num_medications, num_lab_procedures
    """
    rng = np.random.RandomState(seed)

    has_age = 'age' in future_data.columns
    if has_age:
        very_old = future_data[future_data['age'].isin(['[80-90)', '[90-100)'])]
        others = future_data[~future_data['age'].isin(['[80-90)', '[90-100)'])]
    else:
        very_old = future_data[future_data.get('age_ordinal', pd.Series(dtype=float)) >= 8]
        others = future_data[future_data.get('age_ordinal', pd.Series(dtype=float)) < 8]

    # 40% very old, 60% others
    n_very_old = min(int(batch_size * 0.40), max(len(very_old), 1))
    n_others = batch_size - n_very_old

    very_old_sample = very_old.sample(n=n_very_old, random_state=rng, replace=len(very_old) < n_very_old)
    others_sample = others.sample(n=n_others, random_state=rng, replace=len(others) < n_others)

    batch = pd.concat([very_old_sample, others_sample], ignore_index=True)

    # Boost high-diagnosis patients
    if 'number_diagnoses' in batch.columns:
        high_diag = future_data[future_data['number_diagnoses'] >= 9]
        low_diag_mask = batch['number_diagnoses'] < 9
        n_swap = min(int(batch_size * 0.20), low_diag_mask.sum(), len(high_diag))
        if n_swap > 0:
            swap_indices = batch[low_diag_mask].sample(n=n_swap, random_state=rng).index
            replacements = high_diag.sample(n=n_swap, random_state=rng, replace=True)
            batch.loc[swap_indices] = replacements.values[:n_swap]

    # Apply numeric mean shifts (+0.3 * std)
    shift_features = ['time_in_hospital', 'num_medications', 'num_lab_procedures']
    for feat in shift_features:
        if feat in batch.columns and pd.api.types.is_numeric_dtype(batch[feat]):
            std = future_data[feat].std()
            if std > 0:
                shift = 0.3 * std
                batch[feat] = batch[feat] + shift
                logger.info(f"    Shifted {feat} by +{shift:.2f} (0.3 * std={std:.2f})")

    return batch.sample(frac=1, random_state=rng).reset_index(drop=True)


def create_concept_drift_batch(future_data: pd.DataFrame, batch_size: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Batch 5: Concept drift — the relationship between features and target changes.

    Strategy:
    - Take a random sample of future_data
    - For patients with num_medications > 15 AND time_in_hospital > 5,
      flip readmitted_binary for ~40% of them
    - Also oversample number_emergency > 0 patients (different subpopulation)
    - Add slight covariate shift to ensure drift detector triggers
    """
    rng = np.random.RandomState(seed)

    # Oversample emergency patients
    if 'number_emergency' in future_data.columns:
        emergency = future_data[future_data['number_emergency'] > 0]
        non_emergency = future_data[future_data['number_emergency'] == 0]
        n_emergency = min(int(batch_size * 0.35), len(emergency))
        n_non_emergency = batch_size - n_emergency
        emergency_sample = emergency.sample(n=n_emergency, random_state=rng, replace=len(emergency) < n_emergency)
        non_emergency_sample = non_emergency.sample(n=n_non_emergency, random_state=rng, replace=False)
        batch = pd.concat([emergency_sample, non_emergency_sample], ignore_index=True)
    else:
        batch = future_data.sample(n=min(batch_size, len(future_data)), random_state=rng).copy()

    batch = batch.reset_index(drop=True)

    # Flip labels for a subpopulation to create concept drift
    if 'readmitted_binary' in batch.columns:
        target_mask = (
            (batch.get('num_medications', pd.Series(dtype=float)) > 15) &
            (batch.get('time_in_hospital', pd.Series(dtype=float)) > 5)
        )
        target_indices = batch[target_mask].index
        n_flip = int(len(target_indices) * 0.40)
        if n_flip > 0:
            flip_indices = rng.choice(target_indices, size=n_flip, replace=False)
            batch.loc[flip_indices, 'readmitted_binary'] = 1 - batch.loc[flip_indices, 'readmitted_binary']
            logger.info(f"    Flipped {n_flip} labels (concept drift) out of {len(target_indices)} eligible")

    # Add slight covariate shift to ensure drift detector triggers
    for feat in ['num_medications', 'number_emergency', 'number_inpatient']:
        if feat in batch.columns and pd.api.types.is_numeric_dtype(batch[feat]):
            std = future_data[feat].std()
            if std > 0:
                batch[feat] = batch[feat] + 0.2 * std

    return batch.sample(frac=1, random_state=rng).reset_index(drop=True)


def verify_drift_characteristics(reference: pd.DataFrame, batch: pd.DataFrame, batch_name: str):
    """Print summary statistics comparing batch to reference for key features."""
    print(f"\n{'='*60}")
    print(f"  {batch_name} — Drift Characteristics")
    print(f"{'='*60}")
    print(f"  Rows: {len(batch):,}")

    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                        'number_diagnoses', 'num_procedures', 'number_emergency',
                        'number_inpatient', 'number_outpatient']

    for feat in numeric_features:
        if feat in reference.columns and feat in batch.columns:
            ref_mean = reference[feat].mean()
            ref_std = reference[feat].std()
            batch_mean = batch[feat].mean()
            if ref_std > 0:
                shift = (batch_mean - ref_mean) / ref_std
                arrow = "^" if shift > 0.1 else ("v" if shift < -0.1 else "=")
                print(f"  {feat:25s}: ref={ref_mean:7.2f}  batch={batch_mean:7.2f}  shift={shift:+.2f} std {arrow}")

    # Age distribution comparison
    if 'age' in reference.columns and 'age' in batch.columns:
        old_bins = ['[70-80)', '[80-90)', '[90-100)']
        ref_old_pct = reference['age'].isin(old_bins).mean() * 100
        batch_old_pct = batch['age'].isin(old_bins).mean() * 100
        print(f"  {'age >= 70':25s}: ref={ref_old_pct:5.1f}%    batch={batch_old_pct:5.1f}%")

    # Readmission rate
    if 'readmitted_binary' in reference.columns and 'readmitted_binary' in batch.columns:
        ref_rate = reference['readmitted_binary'].mean() * 100
        batch_rate = batch['readmitted_binary'].mean() * 100
        print(f"  {'readmission_rate':25s}: ref={ref_rate:5.1f}%    batch={batch_rate:5.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Create 5 drift-aware batches from future_data.csv for the Health Predict pipeline."
    )
    parser.add_argument("--bucket-name", default=S3_BUCKET_DEFAULT, help="S3 bucket name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rows per batch (default 2000)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without uploading to S3")
    args = parser.parse_args()

    s3 = boto3.client('s3')

    # Load source data
    logger.info("=== Creating Drift-Aware Batches ===\n")
    future_data = load_from_s3(s3, args.bucket_name, FUTURE_DATA_KEY)
    reference_data = load_from_s3(s3, args.bucket_name, REFERENCE_KEY)
    future_data = ensure_readmitted_binary(future_data)
    reference_data = ensure_readmitted_binary(reference_data)

    logger.info(f"\nBatch size: {args.batch_size} rows each\n")

    # Create 5 batches
    batch_configs = [
        (1, "Batch 1 (no drift)", lambda: create_no_drift_batch(future_data, args.batch_size, seed=42)),
        (2, "Batch 2 (no drift)", lambda: create_no_drift_batch(future_data, args.batch_size, seed=99)),
        (3, "Batch 3 (gradual drift)", lambda: create_gradual_drift_batch(future_data, args.batch_size, seed=42)),
        (4, "Batch 4 (strong drift)", lambda: create_strong_drift_batch(future_data, args.batch_size, seed=42)),
        (5, "Batch 5 (concept drift)", lambda: create_concept_drift_batch(future_data, args.batch_size, seed=42)),
    ]

    for batch_num, name, create_fn in batch_configs:
        logger.info(f"Creating {name}...")
        batch_df = create_fn()
        verify_drift_characteristics(reference_data, batch_df, name)

        if not args.dry_run:
            key = f"{BATCH_PREFIX}/batch_{batch_num}.csv"
            upload_to_s3(s3, args.bucket_name, key, batch_df)

    if args.dry_run:
        logger.info("DRY RUN: No data uploaded to S3")
    else:
        logger.info(f"\nAll 5 drift-aware batches uploaded to s3://{args.bucket_name}/{BATCH_PREFIX}/")

    logger.info("Done.")


if __name__ == "__main__":
    main()
