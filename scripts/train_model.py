import argparse
import pandas as pd
import numpy as np
import boto3
from io import StringIO
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure src directory is in Python path to import feature_engineering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.feature_engineering import clean_data, engineer_features, get_preprocessor, save_preprocessor, load_preprocessor, preprocess_data, COLS_FILL_NA_MISSING
except ImportError:
    print("Error: Unable to import from src.feature_engineering. Ensure it's in the Python path.")
    sys.exit(1)

# Ray Tune for HPO
from ray import tune # type: ignore
from ray import train # Import ray.train
from ray.tune.schedulers import ASHAScheduler # type: ignore
from ray.tune.search.hyperopt import HyperOptSearch # type: ignore
from ray.air.config import RunConfig # Import RunConfig
import ray # Import ray


def load_df_from_s3(bucket: str, key: str, s3_client: boto3.client) -> pd.DataFrame:
    """Loads a CSV file from S3 into a pandas DataFrame."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        print(f"Successfully loaded '{key}' from S3 bucket '{bucket}'. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading '{key}' from S3: {e}")
        raise

def evaluate_model(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            metrics["roc_auc_score"] = roc_auc
        except ValueError as e:
            print(f"Could not calculate ROC AUC: {e}") # E.g. if only one class present in y_true
            metrics["roc_auc_score"] = 0.0
    return metrics

def train_model_hpo(config, X_train, y_train, X_val, y_val, model_name):
    """
    Trainable function for Ray Tune.
    Trains a model with given hyperparameters and logs to MLflow.
    """
    # Set tracking URI and experiment in each Ray worker process
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "HealthPredict_Training_HPO_Airflow"))
    
    mlflow.start_run(nested=True) # Start a nested run for each trial
    mlflow.log_params(config)
    mlflow.set_tag("model_name", model_name)
    mlflow.set_tag("hpo_trial", "True")

    if model_name == "LogisticRegression":
        model = LogisticRegression(solver='liblinear', class_weight='balanced', **config)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(class_weight='balanced', **config)
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **config)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.fit(X_train, y_train)
    
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    
    val_metrics = evaluate_model(y_val, y_pred_val, y_proba_val)
    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    
    # Use ray.train.report instead of tune.report
    train.report({
        "accuracy": val_metrics.get("accuracy", 0.0),
        "precision": val_metrics.get("precision", 0.0),
        "recall": val_metrics.get("recall", 0.0),
        "f1_score": val_metrics.get("f1_score", 0.0),
        "roc_auc_score": val_metrics.get("roc_auc_score", 0.0)
    })
    
    mlflow.end_run()
    return val_metrics # Not strictly needed by Tune here, but can be useful


def main(args):
    print(f"Starting training script with args: {args}")
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    s3_client = boto3.client('s3')

    print("Loading data...")
    df_train_raw = load_df_from_s3(args.s3_bucket_name, args.train_key, s3_client)
    df_val_raw = load_df_from_s3(args.s3_bucket_name, args.validation_key, s3_client)

    print("Cleaning data...")
    df_train_clean = clean_data(df_train_raw)
    df_val_clean = clean_data(df_val_raw)

    print("Engineering features...")
    df_train_featured = engineer_features(df_train_clean)
    df_val_featured = engineer_features(df_val_clean)

    # Define features (X) and target (y)
    # Ensure original target and original versions of engineered features are dropped from X
    columns_to_drop_from_features = [args.target_column, 'readmitted', 'age']
    
    X_train_for_preprocessor_fitting = df_train_featured.drop(columns=columns_to_drop_from_features, errors='ignore')
    y_train_for_preprocessor_fitting = df_train_featured[args.target_column]
    
    X_val_for_testing = df_val_featured.drop(columns=columns_to_drop_from_features, errors='ignore')
    y_val_for_testing = df_val_featured[args.target_column]

    # Determine numerical and categorical features from X_train_for_preprocessor_fitting
    num_cols = X_train_for_preprocessor_fitting.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train_for_preprocessor_fitting.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure COLS_FILL_NA_MISSING that are present and object type are in cat_cols,
    # and not in num_cols. This is to be certain about diag_1, diag_2, diag_3.
    for col_to_check in COLS_FILL_NA_MISSING:
        if col_to_check in X_train_for_preprocessor_fitting.columns:
            if X_train_for_preprocessor_fitting[col_to_check].dtype == 'object':
                if col_to_check not in cat_cols:
                    cat_cols.append(col_to_check)
                if col_to_check in num_cols: 
                    num_cols.remove(col_to_check)
            # If it's not object type but somehow ended up in cat_cols, remove it (less likely for these specific cols)
            elif col_to_check in cat_cols:
                 cat_cols.remove(col_to_check)
    
    print(f"Explicitly determined numerical features: {num_cols}")
    print(f"Explicitly determined categorical features: {cat_cols}")

    # Create preprocessor using the explicitly defined feature lists
    preprocessor = get_preprocessor(X_train_for_preprocessor_fitting, numerical_features=num_cols, categorical_features=cat_cols)
    
    print("Fitting preprocessor on the full training feature set (X_train_for_preprocessor_fitting)...")
    preprocessor.fit(X_train_for_preprocessor_fitting) # Fit with X features only

    # Save the fitted preprocessor locally, then log it as an artifact
    local_preprocessor_dir = "preprocessor_files"
    os.makedirs(local_preprocessor_dir, exist_ok=True)
    local_preprocessor_path = os.path.join(local_preprocessor_dir, "preprocessor.joblib")
    save_preprocessor(preprocessor, local_preprocessor_path)
    print(f"Saved fitted preprocessor locally to {local_preprocessor_path}")
    # Log the saved preprocessor to the current MLflow run
    mlflow.log_artifact(local_preprocessor_path, artifact_path="preprocessor")
    print("Logged preprocessor artifact to MLflow run.")

    # TEMP DEBUG: Sample tiny subset of data for ultra-fast training (under 10 seconds)
    # TODO: Remove data sampling after DAG validation
    print("=== TEMP DEBUG: Sampling tiny dataset for speed ===")
    sample_size = min(500, len(X_train_for_preprocessor_fitting))  # Use max 500 rows
    sample_indices = np.random.choice(len(X_train_for_preprocessor_fitting), sample_size, replace=False)
    X_train_sampled = X_train_for_preprocessor_fitting.iloc[sample_indices]
    y_train_sampled = y_train_for_preprocessor_fitting.iloc[sample_indices]
    
    # Use preprocess_data to transform X_train and X_val, with fit_preprocessor=False
    X_train_processed_df = preprocess_data(X_train_sampled, preprocessor, fit_preprocessor=False)
    X_val_processed_df = preprocess_data(X_val_for_testing.head(100), preprocessor, fit_preprocessor=False)  # Sample val too

    # Convert processed data to numpy for Ray/XGBoost if needed, ensure correct dtypes
    X_train_processed = X_train_processed_df.to_numpy()
    X_val_processed = X_val_processed_df.to_numpy()
    y_train_np = y_train_sampled.to_numpy()
    y_val_np = y_val_for_testing.head(100).to_numpy()
    
    # TEMP DEBUG: Skip Ray Tune entirely for ultra-fast training (under 10 seconds)
    # TODO: Restore Ray Tune HPO after DAG validation
    print("=== TEMP DEBUG: Skipping Ray Tune, training simple model directly ===")
    
    # Use simple fixed hyperparameters for fastest training
    model_name = "LogisticRegression"
    simple_hyperparameters = {"C": 1.0, "penalty": "l2"}  # Simple, fast hyperparameters
    
    print(f"Training {model_name} with fixed hyperparameters: {simple_hyperparameters}")

    # Ensure any previous run is ended before starting 
    mlflow.end_run() 

    # Train simple model directly without HPO
    with mlflow.start_run(run_name=f"Best_{model_name}_Model") as final_run:
        mlflow.log_params(simple_hyperparameters)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("best_hpo_model", "False")  # No HPO performed
        mlflow.set_tag("debug_mode", "True")
        
        # Create and train model with minimal configuration
        final_model = LogisticRegression(
            solver='liblinear', 
            class_weight='balanced', 
            max_iter=10,  # Very few iterations for speed
            **simple_hyperparameters
        )
        
        print("Fitting model...")
        final_model.fit(X_train_processed, y_train_np)
                
        # Quick evaluation on validation set
        y_pred_val_final = final_model.predict(X_val_processed)
        y_proba_val_final = final_model.predict_proba(X_val_processed)[:, 1] if hasattr(final_model, "predict_proba") else None
        final_val_metrics = evaluate_model(y_val_np, y_pred_val_final, y_proba_val_final)
        
        # Log minimal metrics
        mlflow.log_metrics({f"val_{k}": v for k,v in final_val_metrics.items()})
        print(f"Validation metrics: {final_val_metrics}")

        # Log the preprocessor artifact used for this model
        if os.path.exists(local_preprocessor_path):
            mlflow.log_artifact(local_preprocessor_path, artifact_path="preprocessor")
            print(f"Logged preprocessor artifact to run {final_run.info.run_id}.")

        # Log the model with minimal example
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=f"Best_{model_name}",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=X_train_processed_df.iloc[:1],  # Just 1 example for speed
            pyfunc_predict_fn="predict_proba"
        )
        print(f"Logged {model_name} model to MLflow.")
            
    if os.path.exists(local_preprocessor_path): # Clean up local preprocessor file
        os.remove(local_preprocessor_path)
        
    # TEMP DEBUG: Ray not used in ultra-fast mode
    # ray.shutdown()  # Commented out since we're not using Ray
        
    print("\nTraining script finished - ultra-fast mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models with HPO.")
    
    # Data arguments
    parser.add_argument("--s3-bucket-name", type=str, required=True, help="S3 bucket name for data.")
    parser.add_argument("--train-key", type=str, required=True, help="S3 key for training data CSV.")
    parser.add_argument("--validation-key", type=str, required=True, help="S3 key for validation data CSV.")
    # parser.add_argument("--preprocessor-output-s3-key", type=str, required=True, help="S3 key to save the fitted preprocessor.")
    parser.add_argument("--target-column", type=str, default="readmitted_binary", help="Name of the target column.")

    # MLflow arguments
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True, help="MLflow tracking server URI.")
    parser.add_argument("--mlflow-experiment-name", type=str, default="HealthPredict_Training", help="MLflow experiment name.")

    # Ray Tune arguments
    parser.add_argument("--ray-num-samples", type=int, default=2, help="Number of HPO trials per model.")
    parser.add_argument("--ray-max-epochs-per-trial", type=int, default=10, help="Max T for ASHAScheduler.") # For tree models, not epochs but can be seen as max iterations
    parser.add_argument("--ray-grace-period", type=int, default=1, help="Grace period for ASHAScheduler.")
    parser.add_argument("--ray-local-dir", type=str, default="~/ray_results", help="Directory for Ray Tune to store results.")


    cli_args = parser.parse_args()
    main(cli_args) 