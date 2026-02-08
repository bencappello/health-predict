#!/usr/bin/env python3
"""
Create Baseline (Batch 0) — establishes the starting point for the experiment report.

Evaluates the current production model on initial_test.csv and logs:
  1. A quality gate run (HealthPredict_Training_HPO_Airflow) with batch_number=0
  2. A drift monitoring run (HealthPredict_Drift_Monitoring) with batch_number=0, drift_share=0.0

Usage:
    python scripts/create_baseline.py \
        --s3-bucket-name health-predict-mlops-f9ac6509 \
        --mlflow-tracking-uri http://localhost:5000

    # Or inside a container:
    docker exec mlops-services-airflow-scheduler-1 \
        python /opt/airflow/scripts/create_baseline.py \
        --s3-bucket-name health-predict-mlops-f9ac6509 \
        --mlflow-tracking-uri http://mlflow:5000
"""

import argparse
import json
import logging
import sys
import os

import boto3
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

# Allow import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.feature_engineering import clean_data, engineer_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

MODEL_NAME = "HealthPredictModel"
TRAINING_EXPERIMENT = "HealthPredict_Training_HPO_Airflow"
DRIFT_EXPERIMENT = "HealthPredict_Drift_Monitoring"


def load_production_model(client: MlflowClient):
    """Load the current Production model from the MLflow registry."""
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production version found for model '{MODEL_NAME}'")
    mv = versions[0]
    log.info(f"Loading production model v{mv.version} (run {mv.run_id})")
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    # Unwrap to get the raw sklearn/xgboost model for predict_proba
    raw_model = model._model_impl.sklearn_model if hasattr(model._model_impl, 'sklearn_model') else model
    return raw_model, mv


def load_test_data(s3, bucket: str) -> pd.DataFrame:
    """Load initial_test.csv from S3."""
    key = "processed_data/initial_test.csv"
    log.info(f"Loading test data from s3://{bucket}/{key}")
    resp = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(resp['Body'].read().decode('utf-8')))


def prepare_features(df: pd.DataFrame) -> tuple:
    """Apply feature engineering and return (X, y)."""
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)

    # One-hot encode categoricals (matching DAG pattern)
    cat_cols = df_feat.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['readmitted_binary', 'readmitted']]
    if cat_cols:
        df_encoded = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df_feat

    X = df_encoded.drop(columns=['readmitted_binary', 'readmitted', 'age'], errors='ignore')
    y = df_feat['readmitted_binary']
    return X, y


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """Align test features to match what the production model expects."""
    expected = None

    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        expected = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names is not None:
                expected = list(booster.feature_names)
        except Exception:
            pass

    if expected is None and hasattr(model, 'n_features_in_'):
        n = model.n_features_in_
        log.info(f"Aligning by feature count: model expects {n}, data has {len(X.columns)}")
        while len(X.columns) < n:
            X[f'_pad_{len(X.columns)}'] = 0
        if len(X.columns) > n:
            X = X.iloc[:, :n]
        return X

    if expected is None:
        log.warning("Could not determine expected features — using data as-is")
        return X

    log.info(f"Aligning features: model expects {len(expected)}, data has {len(X.columns)}")
    for col in set(expected) - set(X.columns):
        X[col] = 0
    X = X.drop(columns=list(set(X.columns) - set(expected)), errors='ignore')
    return X[expected]


def main():
    parser = argparse.ArgumentParser(description="Create Batch 0 baseline for experiment report")
    parser.add_argument("--s3-bucket-name", required=True)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = MlflowClient(args.mlflow_tracking_uri)
    s3 = boto3.client('s3')

    # 1. Load production model
    model, model_version = load_production_model(client)
    log.info(f"Production model type: {type(model).__name__}")

    # 2. Load initial_test data
    test_df = load_test_data(s3, args.s3_bucket_name)
    log.info(f"Test data shape: {test_df.shape}")

    # 3. Feature engineering + alignment
    X, y = prepare_features(test_df)
    X = align_features(X, model)
    log.info(f"Prepared features: X={X.shape}, y={y.shape}")

    # 4. Predict and compute metrics
    pred_proba = model.predict_proba(X)[:, 1]
    pred = (pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y, pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    fpr, tpr, thresholds = roc_curve(y, pred_proba)

    log.info(f"Baseline metrics — AUC: {auc:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    # 5. Log quality gate run (Batch 0)
    mlflow.set_experiment(TRAINING_EXPERIMENT)
    with mlflow.start_run(run_name="quality_gate_batch_0") as run:
        mlflow.log_param("batch_number", 0)
        mlflow.log_param("test_set_used", "initial_test")
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.set_tag("quality_gate", "True")
        mlflow.set_tag("decision", "BASELINE")

        # Log as both new_* and prod_* (baseline: same model)
        mlflow.log_metric("new_auc_test", auc)
        mlflow.log_metric("new_f1_test", f1)
        mlflow.log_metric("new_precision_test", precision)
        mlflow.log_metric("new_recall_test", recall)
        mlflow.log_metric("prod_auc_test", auc)
        mlflow.log_metric("prod_f1_test", f1)
        mlflow.log_metric("prod_precision_test", precision)
        mlflow.log_metric("prod_recall_test", recall)
        mlflow.log_metric("auc_improvement", 0.0)
        mlflow.log_metric("f1_improvement", 0.0)

        # ROC curve artifact
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': auc,
        }
        roc_path = '/tmp/roc_curve_production.json'
        with open(roc_path, 'w') as f:
            json.dump(roc_data, f)
        mlflow.log_artifact(roc_path, artifact_path='roc_curves')

        log.info(f"Logged quality gate run: {run.info.run_id}")

    # 6. Log drift monitoring run (Batch 0 = no drift by definition)
    mlflow.set_experiment(DRIFT_EXPERIMENT)
    with mlflow.start_run(run_name="drift_batch_0") as run:
        mlflow.log_param("batch_number", 0)
        mlflow.log_metric("drift_share", 0.0)
        mlflow.log_metric("n_drifted_columns", 0)
        mlflow.log_metric("n_total_columns", 0)
        mlflow.log_metric("dataset_drift", 0.0)
        log.info(f"Logged drift run: {run.info.run_id}")

    log.info("Baseline creation complete!")


if __name__ == "__main__":
    main()
