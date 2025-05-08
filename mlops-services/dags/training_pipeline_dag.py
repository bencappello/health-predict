from __future__ import annotations

import datetime
import pendulum
import os

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
# from airflow.operators.python import PythonOperator
# from airflow.providers.docker.operators.docker import DockerOperator # Alternative

# Import MLflow if using PythonOperator to find best model
# import mlflow
# from mlflow.tracking import MlflowClient

# --- Configuration ---
# Absolute path to the directory *containing* docker-compose.yml on the HOST
# This assumes the DAG file is in mlops-services/dags/
# Adjust if your structure is different
MLOPS_SERVICES_DIR_ON_HOST = "/home/ubuntu/health-predict/mlops-services" # Use absolute path on host

# Absolute path to the project root inside the jupyterlab container
PROJECT_ROOT_IN_JUPYTER_CONTAINER = "/home/jovyan/work"

# Training Script Arguments
S3_BUCKET = "health-predict-mlops-f9ac6509"
TRAIN_KEY = "processed_data/initial_train.csv"
VALIDATION_KEY = "processed_data/initial_validation.csv"
TEST_KEY = "processed_data/initial_test.csv"
MLFLOW_TRACKING_URI = "http://mlflow:5000" # Use service name from docker network
EXPERIMENT_NAME = "HealthPredict_Training_HPO_Airflow" # Use a distinct name
TARGET_COLUMN = "readmitted_binary"
# Use a unique Ray results directory for Airflow runs to avoid conflicts
RAY_LOCAL_DIR = f"{PROJECT_ROOT_IN_JUPYTER_CONTAINER}/ray_results_airflow_hpo"
# --- End Configuration ---


with DAG(
    dag_id="health_predict_training_hpo",
    schedule=None, # Manual trigger only
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "health_predict", "training", "hpo"],
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': datetime.timedelta(minutes=2),
    }
) as dag:

    # Task to execute the training script within the jupyterlab container
    # Requires docker-compose installed in airflow container & docker socket mounted
    # The bash command navigates to the directory with docker-compose.yml on the host
    # and then executes the script inside the jupyterlab container.
    run_training_script = BashOperator(
        task_id="run_training_and_hpo",
        bash_command=f"""
            echo 'Airflow task running in: $(pwd)'
            echo 'Attempting to execute training script via docker-compose...'
            cd {MLOPS_SERVICES_DIR_ON_HOST} && \
            docker-compose exec -T jupyterlab python3 {PROJECT_ROOT_IN_JUPYTER_CONTAINER}/scripts/train_model.py \
                --s3-bucket-name \"{S3_BUCKET}\" \
                --train-key \"{TRAIN_KEY}\" \
                --validation-key \"{VALIDATION_KEY}\" \
                --test-key \"{TEST_KEY}\" \
                --mlflow-tracking-uri \"{MLFLOW_TRACKING_URI}\" \
                --mlflow-experiment-name \"{EXPERIMENT_NAME}\" \
                --target-column \"{TARGET_COLUMN}\" \
                --ray-num-samples 10 \
                --ray-max-epochs-per-trial 10 \
                --ray-grace-period 1 \
                --ray-local-dir \"{RAY_LOCAL_DIR}\" 
            echo 'Training script execution command finished.'
        """
        # Note: Using -T with docker-compose exec disables pseudo-tty allocation,
        # often needed for non-interactive script execution within containers.
    )

    # TODO: Add Task 2: Find and register the best model using PythonOperator
    # This task would query MLflow runs from the EXPERIMENT_NAME, find the best one
    # based on a metric (e.g., test_f1_score), get its model URI, and register it.

    # Set task dependencies (currently only one task)
    # run_training_script >> find_and_register_model (when Task 2 is added)
    run_training_script 