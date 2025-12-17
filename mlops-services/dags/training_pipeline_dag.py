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
    description='Train and tune models for health prediction with drift-aware retraining',
    schedule_interval=None,  # Set to None for manual triggering
    start_date=datetime(2025, 5, 10),
    catchup=False,
    tags=['health-predict', 'training', 'hpo', 'drift-aware'],
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

# Function to prepare drift-aware training data
def prepare_drift_aware_training_data(**kwargs):
    """
    Prepare training data for drift-triggered retraining by combining historical and new data
    """
    import boto3
    import pandas as pd
    from io import StringIO
    
    logging.info("Starting drift-aware training data preparation...")
    
    # Extract drift context from DAG run configuration
    dag_run = kwargs['dag_run']
    drift_context = dag_run.conf if dag_run.conf else {}
    
    s3_bucket = env_vars['S3_BUCKET_NAME']
    s3_client = boto3.client('s3')
    
    # Default data paths
    base_train_key = env_vars['TRAIN_KEY']
    base_validation_key = env_vars['VALIDATION_KEY']
    
    # Check if this is a drift-triggered retraining
    is_drift_triggered = drift_context.get('drift_triggered', False)
    drift_batch_path = drift_context.get('drift_batch_path', '')
    drift_severity = drift_context.get('drift_severity', 'none')
    
    logging.info(f"Drift triggered: {is_drift_triggered}")
    logging.info(f"Drift severity: {drift_severity}")
    logging.info(f"Drift batch path: {drift_batch_path}")
    
    if is_drift_triggered and drift_batch_path:
        logging.info("Preparing cumulative training data for drift-triggered retraining...")
        
        # Load base training data
        logging.info(f"Loading base training data from {base_train_key}")
        train_response = s3_client.get_object(Bucket=s3_bucket, Key=base_train_key)
        base_train_data = pd.read_csv(StringIO(train_response['Body'].read().decode('utf-8')))
        
        # Load validation data for temporal split
        logging.info(f"Loading base validation data from {base_validation_key}")
        val_response = s3_client.get_object(Bucket=s3_bucket, Key=base_validation_key)
        base_val_data = pd.read_csv(StringIO(val_response['Body'].read().decode('utf-8')))
        
        # Combine base training and validation data for cumulative approach
        combined_base_data = pd.concat([base_train_data, base_val_data], ignore_index=True)
        logging.info(f"Combined base data shape: {combined_base_data.shape}")
        
        # Load drift batch data (the new data that triggered drift)
        logging.info(f"Loading drift batch data from {drift_batch_path}")
        batch_response = s3_client.get_object(Bucket=s3_bucket, Key=drift_batch_path)
        drift_batch_data = pd.read_csv(StringIO(batch_response['Body'].read().decode('utf-8')))
        logging.info(f"Drift batch data shape: {drift_batch_data.shape}")
        
        # Load any additional processed batches from drift monitoring
        additional_batches = []
        drift_batch_prefix = 'drift_monitoring/batch_data/'
        
        try:
            # List all batch files to include cumulative data
            response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=drift_batch_prefix)
            if 'Contents' in response:
                batch_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
                batch_keys.sort()  # Process in chronological order
                
                for batch_key in batch_keys:
                    if batch_key != drift_batch_path:  # Don't double-load the triggering batch
                        logging.info(f"Loading additional batch: {batch_key}")
                        batch_response = s3_client.get_object(Bucket=s3_bucket, Key=batch_key)
                        batch_data = pd.read_csv(StringIO(batch_response['Body'].read().decode('utf-8')))
                        additional_batches.append(batch_data)
        except Exception as e:
            logging.warning(f"Error loading additional batches: {e}")
        
        # Combine all data using cumulative approach
        all_data_frames = [combined_base_data, drift_batch_data] + additional_batches
        combined_data = pd.concat(all_data_frames, ignore_index=True)
        
        # Remove duplicates if any
        combined_data = combined_data.drop_duplicates()
        logging.info(f"Combined cumulative data shape after deduplication: {combined_data.shape}")
        
        # Create temporal train/validation split (80/20 with most recent 20% as validation)
        sorted_data = combined_data.copy()  # Use index as time proxy since we don't have timestamps
        split_point = int(len(sorted_data) * 0.8)
        
        new_train_data = sorted_data.iloc[:split_point]
        new_val_data = sorted_data.iloc[split_point:]
        
        logging.info(f"New temporal split - Train: {new_train_data.shape}, Validation: {new_val_data.shape}")
        
        # Upload new training data to S3
        drift_train_key = f'drift_monitoring/retraining_data/drift_train_{int(time.time())}.csv'
        drift_val_key = f'drift_monitoring/retraining_data/drift_val_{int(time.time())}.csv'
        
        # Upload training data
        train_csv_buffer = StringIO()
        new_train_data.to_csv(train_csv_buffer, index=False)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=drift_train_key,
            Body=train_csv_buffer.getvalue()
        )
        
        # Upload validation data
        val_csv_buffer = StringIO()
        new_val_data.to_csv(val_csv_buffer, index=False)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=drift_val_key,
            Body=val_csv_buffer.getvalue()
        )
        
        logging.info(f"Uploaded drift training data to {drift_train_key}")
        logging.info(f"Uploaded drift validation data to {drift_val_key}")
        
        # Return the new data keys and drift context
        return {
            'train_key': drift_train_key,
            'validation_key': drift_val_key,
            'drift_context': drift_context,
            'data_lineage': {
                'base_train_records': len(base_train_data),
                'base_val_records': len(base_val_data),
                'drift_batch_records': len(drift_batch_data),
                'additional_batch_records': sum(len(df) for df in additional_batches),
                'total_combined_records': len(combined_data),
                'final_train_records': len(new_train_data),
                'final_val_records': len(new_val_data)
            }
        }
    else:
        # Regular training (not drift-triggered)
        logging.info("Using standard training data (not drift-triggered)")
        return {
            'train_key': base_train_key,
            'validation_key': base_validation_key,
            'drift_context': {'drift_triggered': False},
            'data_lineage': {'training_type': 'standard'}
        }

# Task 0: Prepare drift-aware training data
prepare_training_data_task = PythonOperator(
    task_id='prepare_drift_aware_training_data',
    python_callable=prepare_drift_aware_training_data,
    dag=dag,
)

# Task 1: Run training and HPO using the train_model.py script (modified to use dynamic data paths)
run_training_and_hpo = BashOperator(
    task_id='run_training_and_hpo',
    bash_command='''
    # Get dynamic data paths from upstream task
    TRAIN_KEY="{{ ti.xcom_pull(task_ids='prepare_drift_aware_training_data')['train_key'] }}"
    VAL_KEY="{{ ti.xcom_pull(task_ids='prepare_drift_aware_training_data')['validation_key'] }}"
    
    python /home/ubuntu/health-predict/scripts/train_model.py \
        --s3-bucket-name {{ params.s3_bucket_name }} \
        --train-key "$TRAIN_KEY" \
        --validation-key "$VAL_KEY" \
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

# Function to find and register the best model in MLflow (enhanced with drift context)
def find_and_register_best_model(**kwargs):
    mlflow_uri = kwargs['params']['mlflow_uri']
    experiment_name = kwargs['params']['experiment_name']
    target_model_type = "XGBoost"  # Use XGBoost as the current production model
    target_registered_model_name = f"HealthPredict_{target_model_type}"

    print(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get drift context from upstream task
    training_data_info = kwargs['ti'].xcom_pull(task_ids='prepare_drift_aware_training_data')
    drift_context = training_data_info.get('drift_context', {})
    data_lineage = training_data_info.get('data_lineage', {})
    
    print(f"Drift context: {drift_context}")
    print(f"Data lineage: {data_lineage}")

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
        return
    else:
        print(f"Preprocessor artifact '{preprocessor_artifact_path}' found for run {best_run_id}.")

    # 3. Register and transition the best model with drift context
    registered_models_info = []
    try:
        model_uri = f"runs:/{best_run_id}/model"
        print(f"Registering model from URI: {model_uri}")
        
        # Add drift context as model description
        model_description = f"Model trained with drift-aware pipeline. "
        if drift_context.get('drift_triggered', False):
            model_description += f"Drift-triggered retraining (severity: {drift_context.get('drift_severity', 'unknown')}). "
            model_description += f"Training data lineage: {data_lineage}."
        else:
            model_description += "Standard training (no drift detected)."
        
        # mlflow.register_model description arg not available in current MLflow version
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=target_registered_model_name
        )
        # Store description as a tag on the model version instead
        client.set_model_version_tag(
            name=registered_model.name,
            version=registered_model.version,
            key="model_description",
            value=model_description
        )
        print(f"Registered model '{registered_model.name}' version {registered_model.version}.")

        # Add drift context tags to the model version
        if drift_context.get('drift_triggered', False):
            client.set_model_version_tag(
                name=registered_model.name,
                version=registered_model.version,
                key="drift_triggered",
                value="true"
            )
            client.set_model_version_tag(
                name=registered_model.name,
                version=registered_model.version,
                key="drift_severity",
                value=drift_context.get('drift_severity', 'unknown')
            )
            client.set_model_version_tag(
                name=registered_model.name,
                version=registered_model.version,
                key="retraining_data_records",
                value=str(data_lineage.get('total_combined_records', 0))
            )

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
            'stage': 'Production',
            'drift_context': drift_context,
            'data_lineage': data_lineage
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
            "-e", "MODEL_NAME=HealthPredictModel",
            "-e", "MODEL_STAGE=Production",
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
                        "curl", "-f", f"http://{container_name}:8000/health"
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
        test_env["API_BASE_URL"] = f"http://{container_name}:8000"
        test_env["MINIKUBE_IP"] = container_name
        test_env["K8S_NODE_PORT"] = "8000"
        
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
prepare_training_data_task >> run_training_and_hpo >> find_and_register_best_model_task >> test_api_before_deployment_task 