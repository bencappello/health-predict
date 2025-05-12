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
    python /home/jovyan/work/scripts/train_model.py \
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
"""
def find_and_register_best_model(**kwargs):
    # Set MLflow tracking URI
    mlflow_uri = kwargs['params']['mlflow_uri']
    experiment_name = kwargs['params']['experiment_name']
    
    print(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Creating it.")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    print(f"Using experiment ID: {experiment_id}")
    
    # Search for runs with F1 score metrics
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="metrics.val_f1_score > 0",
        order_by=["metrics.val_f1_score DESC"]
    )
    
    if runs.empty:
        print("No runs found with validation F1 score metrics")
        return
    
    # Group by model type and find the best model of each type
    model_types = runs['tags.model_name'].unique()
    
    registered_models = []
    
    for model_type in model_types:
        if pd.isna(model_type):
            continue
            
        model_runs = runs[runs['tags.model_name'] == model_type]
        if model_runs.empty:
            continue
            
        best_run = model_runs.iloc[0]
        best_run_id = best_run['run_id']
        f1_score = best_run['metrics.val_f1_score']
        
        print(f"Best {model_type} model: Run ID {best_run_id}, F1 Score: {f1_score:.4f}")
        
        # Look up the run to get artifacts URI
        best_run_details = mlflow.get_run(best_run_id)
        
        # Register the model in MLflow Model Registry
        try:
            registered_model = mlflow.register_model(
                model_uri=f"runs:/{best_run_id}/model",
                name=f"HealthPredict_{model_type}"
            )
            registered_models.append({
                'model_type': model_type,
                'run_id': best_run_id,
                'f1_score': f1_score,
                'registered_model': registered_model.name,
                'version': registered_model.version
            })
            
            # Transition model to Production stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=registered_model.name,
                version=registered_model.version,
                stage="Production"
            )
            
            print(f"Registered {model_type} model as '{registered_model.name}' version {registered_model.version} in Production stage")
            
        except Exception as e:
            print(f"Error registering {model_type} model: {str(e)}")
    
    return registered_models
"""

# Task 2: Find and register the best model in MLflow
"""
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
"""

# Define the task dependencies for the training task only (since registration is commented out)
run_training_and_hpo 