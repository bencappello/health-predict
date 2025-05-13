from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
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
    'health_predict_api_deployment',
    default_args=default_args,
    description='Build and deploy the Health Predict API to Kubernetes',
    schedule_interval=None,  # Set to None for manual triggering
    start_date=datetime(2025, 5, 14),
    catchup=False,
    tags=['health-predict', 'deployment', 'k8s'],
)

# Define environment variables
env_vars = {
    'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
    'MODEL_NAME': 'HealthPredict_RandomForest',  # Target registered model name
    'MODEL_STAGE': 'Production',  # Target stage to deploy
    'ECR_REPOSITORY': '536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api',  # ECR repository URI
    'AWS_REGION': 'us-east-1',  # AWS region for ECR
    'K8S_DEPLOYMENT_NAME': 'health-predict-api-deployment',  # Kubernetes deployment name
    'K8S_CONTAINER_NAME': 'health-predict-api-container',  # Container name in Kubernetes deployment
    'EC2_PRIVATE_IP': '10.0.1.99',  # EC2 private IP for MLFLOW_TRACKING_URI in K8s
}

# Task 1: Get latest production model information
def get_production_model_info(**kwargs):
    """
    Fetches information about the latest model in Production stage from MLflow.
    """
    mlflow_uri = kwargs['params']['mlflow_uri']
    model_name = kwargs['params']['model_name']
    model_stage = kwargs['params']['model_stage']
    
    print(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    
    print(f"Searching for {model_name} model in {model_stage} stage")
    try:
        # Get latest model version in Production stage
        latest_versions = client.get_latest_versions(name=model_name, stages=[model_stage])
        
        if not latest_versions:
            raise ValueError(f"No {model_name} model found in {model_stage} stage")
        
        # Get the most recent version
        production_model = latest_versions[0]
        model_version = production_model.version
        model_run_id = production_model.run_id
        model_source = production_model.source
        
        print(f"Found model: {model_name} version {model_version}, run_id: {model_run_id}")
        print(f"Model source: {model_source}")
        
        # Return model information for XCom
        return {
            "model_name": model_name,
            "model_version": model_version,
            "model_run_id": model_run_id,
            "model_source": model_source
        }
        
    except Exception as e:
        print(f"Error getting model information: {e}")
        raise

# Task 2: Define image URI and tag
def define_image_details(**kwargs):
    """
    Defines the full ECR image URI with a unique tag based on model version and timestamp.
    """
    ti = kwargs['ti']
    ecr_repository = kwargs['params']['ecr_repository']
    model_info = ti.xcom_pull(task_ids='get_production_model_info')
    
    if not model_info:
        raise ValueError("Failed to get model information from previous task")
    
    # Create unique tag using model version and timestamp
    model_version = model_info['model_version']
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    image_tag = f"v{model_version}-{timestamp}"
    
    full_image_uri = f"{ecr_repository}:{image_tag}"
    print(f"Defined image URI: {full_image_uri}")
    
    return {
        "image_tag": image_tag,
        "full_image_uri": full_image_uri
    }

# Task 1: Get latest production model information
get_production_model_info_task = PythonOperator(
    task_id='get_production_model_info',
    python_callable=get_production_model_info,
    params={
        'mlflow_uri': env_vars['MLFLOW_TRACKING_URI'],
        'model_name': env_vars['MODEL_NAME'],
        'model_stage': env_vars['MODEL_STAGE']
    },
    dag=dag,
)

# Task 2: Define image URI and tag
define_image_details_task = PythonOperator(
    task_id='define_image_details',
    python_callable=define_image_details,
    params={
        'ecr_repository': env_vars['ECR_REPOSITORY']
    },
    dag=dag,
)

# Task 3: Authenticate Docker with ECR
authenticate_docker_to_ecr = BashOperator(
    task_id='authenticate_docker_to_ecr',
    bash_command=f"""
    aws ecr get-login-password --region {env_vars['AWS_REGION']} | docker login --username AWS --password-stdin {env_vars['ECR_REPOSITORY'].split('/')[0]}
    """,
    dag=dag,
)

# Task 4: Build Docker image
build_api_docker_image = BashOperator(
    task_id='build_api_docker_image',
    bash_command="""
    cd /home/ubuntu/health-predict && \
    docker build -t {{ ti.xcom_pull(task_ids="define_image_details")["full_image_uri"] }} .
    """,
    dag=dag,
)

# Task 5: Push Docker image to ECR
push_image_to_ecr = BashOperator(
    task_id='push_image_to_ecr',
    bash_command="""
    docker push {{ ti.xcom_pull(task_ids="define_image_details")["full_image_uri"] }}
    """,
    dag=dag,
)

# Task 6: Update Kubernetes deployment
update_kubernetes_deployment = BashOperator(
    task_id='update_kubernetes_deployment',
    bash_command=f"""
    kubectl set image deployment/{env_vars['K8S_DEPLOYMENT_NAME']} \
      {env_vars['K8S_CONTAINER_NAME']}={{ ti.xcom_pull(task_ids="define_image_details")["full_image_uri"] }} \
      --record
    
    # Ensure MLFLOW_TRACKING_URI is set correctly in the deployment
    kubectl set env deployment/{env_vars['K8S_DEPLOYMENT_NAME']} \
      MLFLOW_TRACKING_URI=http://{env_vars['EC2_PRIVATE_IP']}:5000
    """,
    dag=dag,
)

# Task 7: Verify deployment rollout
verify_deployment_rollout = BashOperator(
    task_id='verify_deployment_rollout',
    bash_command=f"""
    kubectl rollout status deployment/{env_vars['K8S_DEPLOYMENT_NAME']} --timeout=5m
    
    # Get service URL for user convenience
    echo "Health Predict API service is accessible at:"
    minikube service health-predict-api-service --url
    """,
    dag=dag,
)

# Task 8: Run API tests to ensure the deployed API works correctly
run_api_tests = BashOperator(
    task_id='run_api_tests',
    bash_command="""
    cd /home/ubuntu/health-predict && \
    echo "Running API tests against the newly deployed version..." && \
    python -m pytest tests/api/test_api_endpoints.py -v
    """,
    dag=dag,
)

# Define the task dependencies
get_production_model_info_task >> define_image_details_task >> authenticate_docker_to_ecr >> build_api_docker_image >> push_image_to_ecr >> update_kubernetes_deployment >> verify_deployment_rollout >> run_api_tests 