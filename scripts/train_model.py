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
import ray
from ray import tune # type: ignore
from ray import train # Import ray.train
from ray.tune.schedulers import ASHAScheduler # type: ignore
from ray.tune.search.hyperopt import HyperOptSearch # type: ignore
from ray.air.config import RunConfig # Import RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback # Import MLflowLoggerCallback
from sklearn.pipeline import Pipeline # Import Pipeline
import logging # Import logging

# --- Setup Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# --- End Logger Setup ---

# Ensure src directory is in Python path to import feature_engineering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_engineering import clean_data, engineer_features, get_preprocessor, save_preprocessor, load_preprocessor, preprocess_data, COLS_FILL_NA_MISSING

# Ray Tune for HPO
from ray import tune # type: ignore
from ray import train # Import ray.train
from ray.tune.schedulers import ASHAScheduler # type: ignore
from ray.tune.search.hyperopt import HyperOptSearch # type: ignore
from ray.air.config import RunConfig # Import RunConfig


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

def train_model_hpo(config):
    """
    Trainable function for Ray Tune.
    Trains a model with given hyperparameters and logs to MLflow.
    Accepts a single 'config' dictionary.
    """
    # Extract data and other parameters from config
    X_train = config.get("X_train_processed_data")
    y_train = config.get("y_train_data")
    X_val = config.get("X_val_processed_data")
    y_val = config.get("y_val_data")
    model_name = config.get("model_name")
    model_class = config.get("model_class")
    # preprocessor = config.get("preprocessor") # Preprocessor is not directly used in HPO loop for model training if data is pre-transformed

    # Hyperparameters for the current trial are also in config
    # Filter out the data/fixed params to get only model-specific HPO params
    hpo_params = {k: v for k, v in config.items() if k not in [
        "X_train_processed_data", "y_train_data", 
        "X_val_processed_data", "y_val_data", 
        "model_name", "model_class", "preprocessor",
        "mlflow_tracking_uri", "mlflow_experiment_name" # Ensure MLflow params are not passed to model constructor
    ]}

    mlflow.start_run(nested=True) # Start a nested run for each trial
    mlflow.log_params(hpo_params) # Log only the hyperparameters being tuned
    mlflow.set_tag("model_name", model_name)
    mlflow.set_tag("hpo_trial", "True")

    if model_name == "LogisticRegression":
        model = model_class(solver='liblinear', class_weight='balanced', **hpo_params)
    elif model_name == "RandomForest":
        model = model_class(class_weight='balanced', **hpo_params)
    elif model_name == "XGBoost":
        # For XGBoost, scale_pos_weight is often part of its config directly
        model = model_class(use_label_encoder=False, eval_metric='logloss', **hpo_params)
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
    df_test_raw = load_df_from_s3(args.s3_bucket_name, args.test_key, s3_client)

    print("Cleaning data...")
    df_train_clean = clean_data(df_train_raw)
    df_val_clean = clean_data(df_val_raw)
    df_test_clean = clean_data(df_test_raw)

    print("Engineering features...")
    df_train_featured = engineer_features(df_train_clean)
    df_val_featured = engineer_features(df_val_clean)
    df_test_featured = engineer_features(df_test_clean)

    # Define features (X) and target (y)
    # Ensure original target and original versions of engineered features are dropped from X
    # Also drop identifier columns and high-cardinality/complex diagnostic codes for baseline parity
    columns_to_drop_from_features = [
        args.target_column, 
        'readmitted',             # Original categorical target
        'age',                    # Original categorical age (replaced by age_mapped)
        'A1Cresult',              # Original A1Cresult (replaced by A1Cresult_mapped)
        'max_glu_serum',          # Original max_glu_serum (replaced by max_glu_serum_mapped)
        # 'encounter_id', 'patient_nbr' should have been dropped in clean_data
        # 'diag_1', 'diag_2', 'diag_3' are NOT dropped here anymore, as they are now grouped categorical features.
    ]
    
    # Remove columns that might not exist to prevent errors with errors='ignore' not always working as expected
    # on a list for df.drop. It's safer to check for existence.
    actual_columns_to_drop = [col for col in columns_to_drop_from_features if col in df_train_featured.columns]
    
    X_train_for_preprocessor_fitting = df_train_featured.drop(columns=actual_columns_to_drop, errors='ignore')
    y_train_for_preprocessor_fitting = df_train_featured[args.target_column]
    
    # For val and test, ensure we drop the same actually dropped columns from train, if they exist
    X_val_for_testing = df_val_featured.drop(columns=actual_columns_to_drop, errors='ignore')
    y_val_for_testing = df_val_featured[args.target_column]
    
    X_test_final = df_test_featured.drop(columns=actual_columns_to_drop, errors='ignore')
    y_test_final = df_test_featured[args.target_column]

    # Convert potential ID columns to object type to ensure they are treated as categorical
    potential_categorical_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    for df in [X_train_for_preprocessor_fitting, X_val_for_testing, X_test_final]:
        for col_id in potential_categorical_ids:
            if col_id in df.columns:
                df[col_id] = df[col_id].astype(str) # Using str to ensure it's picked by select_dtypes as object

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
    local_preprocessor_path = "preprocessor.joblib"
    save_preprocessor(preprocessor, local_preprocessor_path)

    with mlflow.start_run(run_name="Preprocessing_Run") as prep_run:
        mlflow.log_artifact(local_preprocessor_path, artifact_path="preprocessor")
        preprocessor_uri = mlflow.get_artifact_uri("preprocessor/preprocessor.joblib")
        mlflow.set_tag("preprocessor_uri", preprocessor_uri)
        print(f"Preprocessor logged to MLflow: {preprocessor_uri}")


    # Use preprocess_data to transform X_train, X_val, and X_test with fit_preprocessor=False
    X_train_processed_df = preprocess_data(X_train_for_preprocessor_fitting, preprocessor, fit_preprocessor=False)
    X_val_processed_df = preprocess_data(X_val_for_testing, preprocessor, fit_preprocessor=False)
    X_test_processed_df = preprocess_data(X_test_final, preprocessor, fit_preprocessor=False)

    # Convert processed data to numpy for Ray/XGBoost if needed, ensure correct dtypes
    X_train_processed = X_train_processed_df.to_numpy()
    X_val_processed = X_val_processed_df.to_numpy()
    X_test_processed = X_test_processed_df.to_numpy()
    
    # Ensure target variables are numpy arrays
    y_train_np = y_train_for_preprocessor_fitting.to_numpy()
    y_val_np = y_val_for_testing.to_numpy()
    y_test_np = y_test_final.to_numpy()

    # Calculate scale_pos_weight for XGBoost
    # Ensure y_train_np is defined and populated before this line
    if len(y_train_np) > 0:
        count_negative = np.sum(y_train_np == 0)
        count_positive = np.sum(y_train_np == 1)
        if count_positive > 0: # Avoid division by zero
            scale_pos_weight_value = count_negative / count_positive
            logger.info(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight_value:.2f} (neg: {count_negative}, pos: {count_positive})")
        else:
            scale_pos_weight_value = 1 # Default if no positive samples
            logger.warning("No positive samples in y_train_np for scale_pos_weight calculation. Defaulting to 1.")
    else:
        scale_pos_weight_value = 1 # Default if y_train_np is empty
        logger.warning("y_train_np is empty. Defaulting scale_pos_weight to 1.")

    # Define models and their HPO settings
    models_to_run = {
        "LogisticRegression": {
            "model_class": LogisticRegression,
            "search_space": {
                "C": tune.loguniform(0.001, 1.0), # Regularization strength
                "penalty": tune.choice(['l1', 'l2'])
            },
            "solver": "liblinear", # Required for L1 penalty
            "class_weight": "balanced"
        },
        "RandomForest": {
            "model_class": RandomForestClassifier,
            "search_space": {
                "n_estimators": tune.randint(50, 200),
                "max_depth": tune.randint(5, 20),
                "min_samples_split": tune.randint(2, 11),
                "min_samples_leaf": tune.randint(1, 11),
                "class_weight": tune.choice(["balanced", "balanced_subsample"])
            },
            # class_weight is in search_space, so no fixed one needed here unless we want a default
            # if not found in HPO config (but it will be from search_space)
        },
        "XGBoost": {
            "model_class": xgb.XGBClassifier,
            "search_space": {
                "n_estimators": tune.randint(50, 200),
                "learning_rate": tune.loguniform(0.01, 0.2),
                "max_depth": tune.randint(3, 10),
                "subsample": tune.uniform(0.6, 1.0),
                "colsample_bytree": tune.uniform(0.6, 1.0),
                "gamma": tune.uniform(0, 5),
            },
            "scale_pos_weight": scale_pos_weight_value, # XGBoost uses scale_pos_weight
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
    }

    best_models = {}
    all_hpo_results = {}

    # Initialize Ray
    if not ray.is_initialized():
        #Shut down Ray if it was previously initialized to clear memory
        #ray.shutdown() #This line might cause issues if Ray is not initialized
        try:
            logger.info("Attempting to initialize Ray...")
            expanded_ray_temp_dir = os.path.expanduser(args.ray_temp_dir)
            ray.init(ignore_reinit_error=True, num_cpus=args.ray_num_cpus, _temp_dir=expanded_ray_temp_dir) # Limit CPUs
            logger.info("Ray initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            logger.info("Running Ray.shutdown() just in case.")
            ray.shutdown() # Attempt shutdown if init failed partially
            logger.info("Re-attempting Ray.init with minimal resources.")
            expanded_ray_temp_dir = os.path.expanduser(args.ray_temp_dir) # Ensure it's expanded here too
            ray.init(ignore_reinit_error=True, num_cpus=1, object_store_memory=10**9, _temp_dir=expanded_ray_temp_dir) # 1 CPU, 1GB object store
            logger.info("Ray re-initialized with minimal resources.")


    for model_name, settings in models_to_run.items():
        logger.info(f"Running HPO for {model_name}...")
        try:
            # Create preprocessor for the current model type
            # Some models (like logistic regression with L1) might need specific handling or scaling
            if model_name == "LogisticRegression":
                current_preprocessor = get_preprocessor(X_train_for_preprocessor_fitting, numerical_features=num_cols, categorical_features=cat_cols)
            else: # RandomForest, XGBoost
                current_preprocessor = get_preprocessor(X_train_for_preprocessor_fitting, numerical_features=num_cols, categorical_features=cat_cols)

            # Fit preprocessor on the subset of training data
            logger.info(f"Fitting preprocessor for {model_name} on X_train_for_preprocessor_fitting (shape: {X_train_for_preprocessor_fitting.shape})")
            current_preprocessor.fit(X_train_for_preprocessor_fitting, y_train_for_preprocessor_fitting)

            # Construct the param_space for Ray Tune
            # This will include both the hyperparameters to search and the static parameters
            current_param_space = settings["search_space"].copy() # Start with the actual HPO params

            # Add static parameters needed by train_model_hpo into the param_space
            # Use tune.grid_search for single, fixed values to ensure they are part of the config
            current_param_space["X_train_processed_data"] = tune.grid_search([X_train_processed])
            current_param_space["y_train_data"] = tune.grid_search([y_train_np])
            current_param_space["X_val_processed_data"] = tune.grid_search([X_val_processed])
            current_param_space["y_val_data"] = tune.grid_search([y_val_np])
            current_param_space["model_name"] = tune.grid_search([model_name])
            current_param_space["model_class"] = tune.grid_search([settings["model_class"]])
            current_param_space["preprocessor"] = tune.grid_search([current_preprocessor]) # Pass preprocessor for context if needed
            current_param_space["mlflow_tracking_uri"] = tune.grid_search([args.mlflow_tracking_uri])
            current_param_space["mlflow_experiment_name"] = tune.grid_search([args.mlflow_experiment_name])
            
            # Add model-specific fixed params like scale_pos_weight or class_weight if they are in settings
            # These are not part of search_space but need to be in the config for model instantiation
            if model_name == "XGBoost" and "scale_pos_weight" in settings:
                 current_param_space["scale_pos_weight"] = tune.grid_search([settings["scale_pos_weight"]])
            # Add other model-specific fixed parameters if necessary (e.g., solver for LogisticRegression)
            if model_name == "LogisticRegression" and "solver" in settings:
                current_param_space["solver"] = tune.grid_search([settings["solver"]])
            # For RandomForest, class_weight is in its search_space. For LogisticRegression, it's fixed.
            if model_name == "LogisticRegression" and "class_weight" in settings: # Ensure fixed class_weight for LR
                 current_param_space["class_weight"] = tune.grid_search([settings["class_weight"]])
            # If RandomForest had a fixed class_weight outside search_space, it would be added like this:
            # elif model_name == "RandomForest" and "class_weight" in settings and "class_weight" not in settings["search_space"]:
            #      current_param_space["class_weight"] = tune.grid_search([settings["class_weight"]])

            analysis = None # Define analysis outside try block
            
            logger.info(f"Starting Ray Tune for {model_name} with num_samples={args.hpo_num_samples}.")
            # Add more specific try-except around tune.run
            try:
                tuner = tune.Tuner(
                    train_model_hpo, # Pass train_model_hpo directly
                    param_space=current_param_space, # Pass the combined param_space
                    tune_config=tune.TuneConfig(
                        metric="f1_score", 
                        mode="max",
                        num_samples=args.hpo_num_samples, # Number of HPO trials
                        #scheduler=ASHAScheduler(metric="val_f1_score", mode="max", grace_period=1, reduction_factor=2) # Optional: ASHA scheduler
                    ),
                    run_config=ray.air.RunConfig(
                        name=f"{model_name}_hpo_parent_run_{prep_run.info.run_id}", # Group trials under parent
                        local_dir=args.ray_tune_local_dir, # Save tune results locally
                        log_to_file=True, # Log trial outputs to files
                        callbacks=[MLflowLoggerCallback(tracking_uri=args.mlflow_tracking_uri, experiment_name=args.mlflow_experiment_name, save_artifact=False,tags={"mlflow.parentRunId": prep_run.info.run_id})]
                    )
                )
                results = tuner.fit()
                analysis = results # Assign results to analysis here
                logger.info(f"Ray Tune for {model_name} COMPLETED.")

            except ray.exceptions.RayTaskError as e:
                logger.error(f"RAY TASK ERROR during HPO for {model_name}: {e}")
                logger.error(f"Failed task: {e.task_name}, Traceback: {e.traceback_str}")
                # Potentially log more details from e if available
                # e.g., e.cause contains the original exception from the task
                if hasattr(e, 'cause') and e.cause:
                    logger.error(f"Original error in Ray task: {e.cause}")
            except ray.exceptions.RayActorError as e:
                logger.error(f"RAY ACTOR ERROR during HPO for {model_name}: {e}")
            except Exception as e:
                logger.error(f"GENERIC EXCEPTION during HPO for {model_name} (tune.run part): {e}", exc_info=True)
                # Force a shutdown of Ray to try and free resources if something went very wrong
                logger.warning(f"Attempting Ray shutdown due to HPO error in {model_name}.")
                ray.shutdown()
                logger.warning(f"Re-initializing Ray after HPO error in {model_name}.")
                ray.init(ignore_reinit_error=True, num_cpus=1, _temp_dir=args.ray_temp_dir) # Minimal re-init


            if analysis and not analysis.errors: # Check if analysis is not None and has no errors
                best_trial = analysis.get_best_result(metric="f1_score", mode="max")
                
                if best_trial and best_trial.config:
                    best_hyperparameters = best_trial.config
                    print(f"Best hyperparameters for {model_name}: {best_hyperparameters}")
                    print(f"Best f1_score for {model_name} (validation): {best_trial.metrics.get('f1_score')}")

                    # Train final best model of this type on full training data and log it
                    with mlflow.start_run(run_name=f"Best_{model_name}_Model") as final_run:
                        mlflow.log_params(best_hyperparameters)
                        mlflow.set_tag("model_name", model_name)
                        mlflow.set_tag("best_hpo_model", "True")
                        mlflow.log_metric(f"best_val_f1_score", best_trial.metrics.get("f1_score", 0.0))

                        # Reconstruct the model with best HPO params, ensuring fixed params are also there
                        final_model_params = best_hyperparameters.copy()
                        if model_name == "LogisticRegression":
                            # Ensure solver and class_weight from 'settings' are used if not in best_hyperparameters (they should be via grid_search)
                            final_model_params['solver'] = best_hyperparameters.get('solver', settings.get('solver', 'liblinear'))
                            final_model_params['class_weight'] = best_hyperparameters.get('class_weight', settings.get('class_weight', 'balanced'))
                            final_model = settings["model_class"](**{k:v for k,v in final_model_params.items() if k in ['C', 'penalty', 'solver', 'class_weight']})
                        elif model_name == "RandomForest":
                            # class_weight will come from best_hyperparameters as it's in the search space
                            final_model_params_rf = {k:v for k,v in best_hyperparameters.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'class_weight']}
                            final_model = settings["model_class"](**final_model_params_rf)
                        elif model_name == "XGBoost":
                            # Ensure fixed params are present
                            final_model_params_xgb = {k:v for k,v in best_hyperparameters.items() if k in settings["search_space"]} # Start with HPO params
                            final_model_params_xgb['use_label_encoder'] = settings.get('use_label_encoder', False)
                            final_model_params_xgb['eval_metric'] = settings.get('eval_metric', 'logloss')
                            final_model_params_xgb['scale_pos_weight'] = settings.get('scale_pos_weight', 1)
                            final_model = settings["model_class"](**final_model_params_xgb)
                        else:
                            # Fallback or raise error
                            logger.error(f"Model type {model_name} not fully handled for final model reconstruction.")
                            final_model = settings["model_class"](**best_hyperparameters) # Default attempt
                        
                        final_model.fit(X_train_processed, y_train_np)
                        
                        # Evaluate on validation set (already done in HPO best trial, but good to confirm)
                        y_pred_val_final = final_model.predict(X_val_processed)
                        y_proba_val_final = final_model.predict_proba(X_val_processed)[:, 1] if hasattr(final_model, "predict_proba") else None
                        final_val_metrics = evaluate_model(y_val_np, y_pred_val_final, y_proba_val_final)
                        mlflow.log_metrics({f"val_{k}": v for k,v in final_val_metrics.items()})
                        print(f"Final {model_name} Validation Metrics: {final_val_metrics}")

                        # Evaluate on TEST set
                        print(f"Evaluating final {model_name} on TEST set...")
                        y_pred_test_final = final_model.predict(X_test_processed)
                        y_proba_test_final = final_model.predict_proba(X_test_processed)[:, 1] if hasattr(final_model, "predict_proba") else None
                        final_test_metrics = evaluate_model(y_test_np, y_pred_test_final, y_proba_test_final)
                        mlflow.log_metrics({f"test_{k}": v for k,v in final_test_metrics.items()})
                        print(f"Final {model_name} TEST Metrics: {final_test_metrics}")

                        # Log the model
                        mlflow.sklearn.log_model(
                            sk_model=final_model,
                            artifact_path=f"models/{model_name}",
                            # Log preprocessor with the model for easier inference later
                            # This requires preprocessor to be available or its URI
                            # For now, we assume preprocessor is logged separately and referenced by URI
                            # registered_model_name=f"Best_{model_name}" # Optionally register
                        )
                        mlflow.log_artifact(local_preprocessor_path, artifact_path=f"models/{model_name}/preprocessor") # Log preprocessor with this model run too
                        print(f"Logged best {model_name} and its preprocessor to MLflow.")
        except Exception as e:
            logger.error(f"GENERIC EXCEPTION during HPO for {model_name}: {e}", exc_info=True)

    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shut down.")

    logger.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models with HPO.")
    
    # Data arguments
    parser.add_argument("--s3-bucket-name", type=str, required=True, help="S3 bucket name for data.")
    parser.add_argument("--train-key", type=str, required=True, help="S3 key for training data CSV.")
    parser.add_argument("--validation-key", type=str, required=True, help="S3 key for validation data CSV.")
    parser.add_argument("--test-key", type=str, default="processed_data/initial_test.csv", help="S3 key for test data CSV.")
    # parser.add_argument("--preprocessor-output-s3-key", type=str, required=True, help="S3 key to save the fitted preprocessor.")
    parser.add_argument("--target-column", type=str, default="readmitted_binary", help="Name of the target column.")

    # MLflow arguments
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True, help="MLflow tracking server URI.")
    parser.add_argument("--mlflow-experiment-name", type=str, default="HealthPredict_Training", help="MLflow experiment name.")

    # Ray Tune arguments
    parser.add_argument("--ray-num-cpus", type=int, default=4, help="Number of CPUs for Ray.")
    parser.add_argument("--ray-temp-dir", type=str, default="~/ray_temp", help="Temporary directory for Ray.")
    parser.add_argument("--ray-tune-local-dir", type=str, default="~/ray_results", help="Directory for Ray Tune to store results.")
    parser.add_argument("--hpo-num-samples", type=int, default=10, help="Number of HPO trials per model.")
    parser.add_argument("--ray-max-epochs-per-trial", type=int, default=10, help="Ray Tune max epochs per trial.")
    parser.add_argument("--ray-grace-period", type=int, default=1, help="Ray Tune grace period for ASHA scheduler.")


    cli_args = parser.parse_args()
    main(cli_args) 