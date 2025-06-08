from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.exceptions import AirflowFailException
import mlflow
import pandas as pd
import os
import subprocess
import time
import logging

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

# Function to test API before deployment
def test_api_before_deployment(**kwargs):
    """Test API locally before triggering deployment"""
    logging.info("Starting pre-deployment API testing...")
    
    # Get the newly registered model info
    model_info = kwargs['ti'].xcom_pull(task_ids='find_and_register_best_model')
    if model_info:
        logging.info(f"Testing with newly registered model: {model_info}")
    
    # Use current timestamp for unique container naming
    container_name = f"api-test-{int(time.time())}"
    
    try:
        logging.info(f"Building test Docker image: {container_name}:test")
        
        # Build test image
        build_result = subprocess.run([
            "docker", "build", "-t", f"{container_name}:test", 
            "-f", "Dockerfile", "."
        ], cwd="/home/ubuntu/health-predict", capture_output=True, text=True, check=False)
        
        if build_result.returncode != 0:
            raise AirflowFailException(f"Docker build failed: {build_result.stderr}")
        
        logging.info("Docker image built successfully")
        
        # Run container with test MLflow URI pointing to the MLflow service
        logging.info(f"Starting test container: {container_name}")
        run_result = subprocess.run([
            "docker", "run", "-d", "--name", container_name,
            "--network", "mlops-services_mlops_network",  # Connect to same network as MLflow
            "-p", "8001:8000",  # Use different port to avoid conflicts
            "-e", "MLFLOW_TRACKING_URI=http://mlflow:5000",  # Use service name from same network
            f"{container_name}:test"
        ], capture_output=True, text=True, check=False)
        
        if run_result.returncode != 0:
            raise AirflowFailException(f"Docker run failed: {run_result.stderr}")
        
        logging.info("Test container started, waiting for API to be ready...")
        
        # Wait for container to be ready (with timeout)
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                # Check if container is still running
                status_result = subprocess.run([
                    "docker", "inspect", "--format", "{{.State.Status}}", container_name
                ], capture_output=True, text=True, check=False)
                
                if status_result.returncode == 0 and status_result.stdout.strip() == "running":
                    # Test if API is responding
                    health_result = subprocess.run([
                        "curl", "-f", "http://localhost:8001/health"
                    ], capture_output=True, text=True, check=False)
                    
                    if health_result.returncode == 0:
                        logging.info("API is responding to health checks")
                        break
                else:
                    logging.warning(f"Container status: {status_result.stdout.strip()}")
                    
            except Exception as e:
                logging.warning(f"Health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(2)
        else:
            # Get container logs for debugging
            logs_result = subprocess.run([
                "docker", "logs", container_name
            ], capture_output=True, text=True, check=False)
            
            raise AirflowFailException(f"API failed to become ready after {max_attempts} attempts. Container logs: {logs_result.stdout}\n{logs_result.stderr}")
        
        logging.info("Running comprehensive API tests...")
        
        # Set environment variables for tests
        test_env = os.environ.copy()
        test_env["API_BASE_URL"] = "http://localhost:8001"
        test_env["MINIKUBE_IP"] = "127.0.0.1"
        test_env["K8S_NODE_PORT"] = "8001"
        
        # Run tests against local container
        test_result = subprocess.run([
            "python", "-m", "pytest", 
            "/home/ubuntu/health-predict/tests/api/test_api_endpoints.py", 
            "-v", "--tb=short"
        ], env=test_env, capture_output=True, text=True, check=False)
        
        logging.info(f"Test output:\n{test_result.stdout}")
        if test_result.stderr:
            logging.warning(f"Test stderr:\n{test_result.stderr}")
        
        if test_result.returncode != 0:
            # Get container logs for debugging
            logs_result = subprocess.run([
                "docker", "logs", container_name
            ], capture_output=True, text=True, check=False)
            
            raise AirflowFailException(f"API tests failed (exit code: {test_result.returncode}):\nTest output: {test_result.stdout}\nTest errors: {test_result.stderr}\nContainer logs: {logs_result.stdout}")
            
        logging.info("All API tests passed successfully!")
        
        return {
            "test_status": "passed",
            "container_name": container_name,
            "model_info": model_info
        }
        
    except Exception as e:
        logging.error(f"Pre-deployment API testing failed: {str(e)}")
        raise AirflowFailException(f"Pre-deployment API testing failed: {str(e)}")
        
    finally:
        # Cleanup - always attempt to clean up containers and images
        logging.info("Cleaning up test containers and images...")
        
        # Stop and remove container
        subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
        subprocess.run(["docker", "rm", container_name], capture_output=True, check=False)
        
        # Remove test image
        subprocess.run(["docker", "rmi", f"{container_name}:test"], capture_output=True, check=False)
        
        logging.info("Cleanup completed")

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

# Task 3: Test API before deployment
test_api_before_deployment_task = PythonOperator(
    task_id='test_api_before_deployment',
    python_callable=test_api_before_deployment,
    dag=dag,
    execution_timeout=timedelta(minutes=10),  # Allow sufficient time for testing
)

# Define the task dependencies
run_training_and_hpo >> find_and_register_best_model_task >> test_api_before_deployment_task 