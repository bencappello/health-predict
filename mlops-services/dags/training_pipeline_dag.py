from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
import pandas as pd
import os

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'health_predict_training_hpo',
    default_args=default_args,
    description='Train and tune models for health prediction',
    schedule_interval=None,  # Set to None for manual triggering
    start_date=datetime(2025, 5, 10),
    catchup=False,
    tags=['health-predict', 'training', 'hpo'],
)

# Define environment variables for the training script
env_vars = {
    'S3_BUCKET_NAME': 'health-predict-mlops-f9ac6509',
    'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
    'EXPERIMENT_NAME': 'HealthPredict_Training_HPO_Airflow',
    'TRAIN_KEY': 'processed_data/initial_train.csv',
    'VALIDATION_KEY': 'processed_data/initial_validation.csv',
    'TARGET_COLUMN': 'readmitted_binary',
    'RAY_NUM_SAMPLES': '2',  # Number of HPO trials - reduced to 2 for quick verification
    'RAY_MAX_EPOCHS': '10',
    'RAY_GRACE_PERIOD': '5',
    'RAY_LOCAL_DIR': '/opt/airflow/ray_results_airflow_hpo',
}

# Task 1: Run training and HPO using the train_model.py script
run_training_and_hpo = BashOperator(
    task_id='run_training_and_hpo',
    bash_command='''
    python /home/ubuntu/health-predict/scripts/train_model.py \
        --s3-bucket-name {{ params.s3_bucket_name }} \
        --train-key {{ params.train_key }} \
        --validation-key {{ params.validation_key }} \
        --target-column {{ params.target_column }} \
        --mlflow-tracking-uri {{ params.mlflow_uri }} \
        --mlflow-experiment-name {{ params.experiment_name }} \
        --ray-num-samples {{ params.ray_num_samples }} \
        --ray-max-epochs-per-trial {{ params.ray_max_epochs }} \
        --ray-grace-period {{ params.ray_grace_period }} \
        --ray-local-dir {{ params.ray_local_dir }}
    ''',
    params={
        's3_bucket_name': env_vars['S3_BUCKET_NAME'],
        'train_key': env_vars['TRAIN_KEY'],
        'validation_key': env_vars['VALIDATION_KEY'],
        'target_column': env_vars['TARGET_COLUMN'],
        'mlflow_uri': env_vars['MLFLOW_TRACKING_URI'],
        'experiment_name': env_vars['EXPERIMENT_NAME'],
        'ray_num_samples': env_vars['RAY_NUM_SAMPLES'],
        'ray_max_epochs': env_vars['RAY_MAX_EPOCHS'],
        'ray_grace_period': env_vars['RAY_GRACE_PERIOD'],
        'ray_local_dir': env_vars['RAY_LOCAL_DIR']
    },
    dag=dag,
)

# Function to find and register the best model in MLflow
def find_and_register_best_model(**kwargs):
    mlflow_uri = kwargs['params']['mlflow_uri']
    experiment_name = kwargs['params']['experiment_name']
    target_model_type = "RandomForest"  # Focus on RandomForest for now
    target_registered_model_name = f"HealthPredict_{target_model_type}"

    print(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found.")
        return
    experiment_id = experiment.experiment_id
    print(f"Using experiment ID: {experiment_id}")

    # 1. Archive existing Production models for the target type
    try:
        latest_versions = client.get_latest_versions(target_registered_model_name, stages=["Production"])
        for version in latest_versions:
            print(f"Archiving existing Production model: {version.name} version {version.version}")
            client.transition_model_version_stage(
                name=version.name,
                version=version.version,
                stage="Archived"
            )
    except mlflow.exceptions.RestException as e:
        if e.error_code == 'RESOURCE_DOES_NOT_EXIST':
            print(f"No existing Production versions found for {target_registered_model_name}.")
        else:
            print(f"Error checking/archiving existing models: {e}")
            # Decide if we should continue or raise

    # 2. Find the single best run for the target model type
    print(f"Searching for the best {target_model_type} run...")
    best_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.best_hpo_model = 'True' AND tags.model_name = '{target_model_type}'",
        order_by=["metrics.val_f1_score DESC"],
        max_results=1 # Get only the top one
    )

    if best_runs.empty:
        print(f"No runs found tagged as best_hpo_model for {target_model_type}.")
        return

    best_run = best_runs.iloc[0]
    best_run_id = best_run['run_id']
    f1_score = best_run['metrics.val_f1_score']
    print(f"Found best {target_model_type} run: ID {best_run_id}, F1 Score: {f1_score:.4f}")

    # Check for preprocessor artifact
    preprocessor_artifact_path = "preprocessor/preprocessor.joblib"
    try:
        artifacts = client.list_artifacts(best_run_id, "preprocessor")
        preprocessor_exists = any(artifact.path == "preprocessor/preprocessor.joblib" for artifact in artifacts)
    except Exception as e:
        print(f"Error listing artifacts for run {best_run_id}: {e}")
        preprocessor_exists = False

    if not preprocessor_exists:
        print(f"Preprocessor artifact '{preprocessor_artifact_path}' not found for run {best_run_id}. Skipping registration and promotion.")
        return # Or handle as an error, perhaps by not returning an empty list if that's an issue downstream
    else:
        print(f"Preprocessor artifact '{preprocessor_artifact_path}' found for run {best_run_id}.")

    # 3. Register and transition the best model
    registered_models_info = []
    try:
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Registering model from URI: {model_uri}")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=target_registered_model_name
        )
        print(f"Registered model '{registered_model.name}' version {registered_model.version}.")

        print(f"Transitioning version {registered_model.version} to Production...")
        client.transition_model_version_stage(
            name=registered_model.name,
            version=registered_model.version,
            stage="Production"
        )
        print("Transition complete.")

        registered_models_info.append({
            'model_type': target_model_type,
            'run_id': best_run_id,
            'f1_score': f1_score,
            'registered_model': registered_model.name,
            'version': registered_model.version,
            'stage': 'Production'
        })

    except Exception as e:
        print(f"Error registering or transitioning best {target_model_type} model (Run ID: {best_run_id}): {e}")

    if not registered_models_info:
        print(f"Best {target_model_type} model was not successfully registered and promoted.")

    return registered_models_info

# Task 2: Find and register the best model in MLflow
find_and_register_best_model_task = PythonOperator(
    task_id='find_and_register_best_model',
    python_callable=find_and_register_best_model,
    params={
        'mlflow_uri': env_vars['MLFLOW_TRACKING_URI'],
        'experiment_name': env_vars['EXPERIMENT_NAME']
    },
    dag=dag,
)

# Define the task dependencies
run_training_and_hpo >> find_and_register_best_model_task 

# Define the task dependencies for the training task only (since registration is commented out)
# run_training_and_hpo # This line should be removed or commented out 