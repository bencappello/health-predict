import argparse
import logging
import os
import math
from io import StringIO

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def split_data(bucket_name, raw_data_key, 
               initial_train_key, initial_validation_key, initial_test_key, 
               future_data_key, initial_data_fraction=0.2, 
               test_size=0.15, validation_size=0.15, random_state=42):
    """Downloads data, separates initial data, splits it, and uploads results.

    Separates the first `initial_data_fraction` of the data.
    Splits this initial data into train/validation/test sets.
    Saves the initial train/validation/test sets and the remaining 'future' data to S3.

    Args:
        bucket_name (str): The name of the S3 bucket.
        raw_data_key (str): Key of the raw data file.
        initial_train_key (str): Key for the initial training data.
        initial_validation_key (str): Key for the initial validation data.
        initial_test_key (str): Key for the initial test data.
        future_data_key (str): Key for the remaining future data.
        initial_data_fraction (float): Fraction of data to use for initial split (0.0 to 1.0).
        test_size (float): Proportion of initial data for the test split.
        validation_size (float): Proportion of initial data for the validation split.
        random_state (int): Random state for splitting.
    """
    s3_client = boto3.client('s3')

    logging.info(f"Downloading raw data from s3://{bucket_name}/{raw_data_key}")
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=raw_data_key)
        raw_data_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(raw_data_content))
        logging.info(f"Successfully loaded data. Shape: {df.shape}")

    except Exception as e:
        logging.error(f"Error downloading or reading data from S3: {e}")
        return

    # Ensure fractions/sizes are valid
    if not (0 < initial_data_fraction <= 1.0):
        logging.error("initial_data_fraction must be between 0 (exclusive) and 1 (inclusive).")
        return
    if not (0 < test_size < 1) or not (0 < validation_size < 1):
        logging.error("test_size and validation_size must be between 0 and 1.")
        return
    if test_size + validation_size >= 1.0:
        logging.error("Combined test_size and validation_size (for initial data) must be less than 1.")
        return

    # Calculate split point for initial vs future data
    n_total = len(df)
    n_initial = math.floor(n_total * initial_data_fraction)
    if n_initial == 0:
        logging.error(f"initial_data_fraction ({initial_data_fraction}) is too small for dataset size ({n_total}). No initial data selected.")
        return
    if n_initial == n_total and initial_data_fraction < 1.0:
        logging.warning("initial_data_fraction resulted in selecting the entire dataset as initial data.")
        # Adjust to leave at least one row for future data if possible
        if n_total > 1:
            n_initial -= 1
        else:
            logging.error("Cannot split dataset of size 1 into initial and future.")
            return
            
    logging.info(f"Separating initial data ({initial_data_fraction*100:.1f}%, {n_initial} rows) from future data ({n_total - n_initial} rows).")
    initial_df = df.iloc[:n_initial].copy()
    future_df = df.iloc[n_initial:].copy()

    if future_df.empty:
        logging.warning("No future data was separated. All data used for initial split.")

    # Calculate adjusted validation size relative to the remainder after test split within the initial data
    validation_size_adjusted = validation_size / (1.0 - test_size)
    if validation_size_adjusted >= 1.0:
        logging.error("Internal check failed: Adjusted validation size is >= 1. Check test/validation sizes.")
        return

    logging.info(f"Splitting initial data ({len(initial_df)} rows): Test size={test_size:.2f}, Validation size (relative)={validation_size_adjusted:.2f}")

    try:
        # First split on initial data: Separate test set
        initial_train_val_df, initial_test_df = train_test_split(
            initial_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=False # Important: Keep chronological order if assuming rows are ordered
            # stratify=initial_df['target_column'] # Stratification might contradict chronological assumption
        )

        # Second split: Separate validation set from the remaining initial data
        initial_train_df, initial_validation_df = train_test_split(
            initial_train_val_df,
            test_size=validation_size_adjusted,
            random_state=random_state,
            shuffle=False # Important: Keep chronological order
            # stratify=initial_train_val_df['target_column']
        )

        logging.info(f"Initial data split complete: Train={initial_train_df.shape}, Validation={initial_validation_df.shape}, Test={initial_test_df.shape}")
        logging.info(f"Future data shape: {future_df.shape}")

        # --- Log dataset lengths before upload --- 
        logging.info(f"Dataset Lengths: Initial Train = {len(initial_train_df)}, Initial Validation = {len(initial_validation_df)}, Initial Test = {len(initial_test_df)}, Future Data = {len(future_df)}")

        # --- Upload splits to S3 ---
        datasets_to_upload = {
            initial_train_key: initial_train_df,
            initial_validation_key: initial_validation_df,
            initial_test_key: initial_test_df,
            future_data_key: future_df
        }

        for key, df_split in datasets_to_upload.items():
            if df_split.empty:
                logging.warning(f"Skipping upload for empty dataframe: {key}")
                continue
            logging.info(f"Uploading {os.path.basename(key)} ({len(df_split)} rows) to s3://{bucket_name}/{key}")
            csv_buffer = StringIO()
            df_split.to_csv(csv_buffer, index=False)
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())

        logging.info("Data partitioning, splitting, and upload complete.")

    except Exception as e:
        logging.error(f"Error during data splitting or uploading: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition dataset into initial/future, split initial set, and upload to S3.")
    parser.add_argument("--bucket-name", required=True, help="S3 bucket name")
    parser.add_argument("--raw-data-key", default="raw_data/diabetic_data.csv", help="S3 key for the raw data CSV file")
    # Initial data splits
    parser.add_argument("--initial-train-key", default="processed_data/initial_train.csv", help="S3 key for initial training data")
    parser.add_argument("--initial-validation-key", default="processed_data/initial_validation.csv", help="S3 key for initial validation data")
    parser.add_argument("--initial-test-key", default="processed_data/initial_test.csv", help="S3 key for initial test data")
    # Future data
    parser.add_argument("--future-data-key", default="processed_data/future_data.csv", help="S3 key for the remaining future data")
    # Fractions/Sizes
    parser.add_argument("--initial-data-fraction", type=float, default=0.2, help="Fraction of data for initial split (0 to 1)")
    parser.add_argument("--test-size", type=float, default=0.15, help="Proportion of initial data for the test set")
    parser.add_argument("--validation-size", type=float, default=0.15, help="Proportion of initial data for the validation set")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for splitting initial data")

    args = parser.parse_args()

    split_data(
        bucket_name=args.bucket_name,
        raw_data_key=args.raw_data_key,
        initial_train_key=args.initial_train_key,
        initial_validation_key=args.initial_validation_key,
        initial_test_key=args.initial_test_key,
        future_data_key=args.future_data_key,
        initial_data_fraction=args.initial_data_fraction,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state
    ) 