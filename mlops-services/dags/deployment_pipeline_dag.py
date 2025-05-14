from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowFailException
import boto3
import json
import base64
import subprocess
import mlflow
import os

# Get environment variables
env_vars = {
    'MLFLOW_SERVER': os.getenv('MLFLOW_SERVER', 'mlflow'),
    'ECR_REPOSITORY': os.getenv('ECR_REPOSITORY', '536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api'),
    'AWS_REGION': os.getenv('AWS_REGION', 'us-east-1'),
    'MODEL_NAME': os.getenv('MODEL_NAME', 'HealthPredict_RandomForest'),
    'MODEL_STAGE': os.getenv('MODEL_STAGE', 'Production'),
    'K8S_DEPLOYMENT_NAME': os.getenv('K8S_DEPLOYMENT_NAME', 'health-predict-api-deployment'),
    'K8S_CONTAINER_NAME': os.getenv('K8S_CONTAINER_NAME', 'health-predict-api'),
    'EC2_PRIVATE_IP': os.getenv('EC2_PRIVATE_IP', '10.0.0.123'),
}

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
    start_date=days_ago(1),  # Changed from future date to days_ago(1)
    catchup=False,
    tags=['health-predict', 'deployment', 'k8s'],
)

def get_production_model_info(**kwargs):
    """Get information about the production version of the model from MLflow Model Registry."""
    ti = kwargs['ti']
    mlflow.set_tracking_uri(f"http://{env_vars['MLFLOW_SERVER']}:5000")
    print(f"Setting MLflow tracking URI to: {mlflow.get_tracking_uri()}")
    
    model_name = env_vars['MODEL_NAME']
    model_stage = env_vars['MODEL_STAGE']
    
    print(f"Searching for {model_name} model in {model_stage} stage")
    
    client = mlflow.tracking.MlflowClient()
    
    # Get the latest version of the model in the specified stage
    latest_versions = client.get_latest_versions(name=model_name, stages=[model_stage])
    
    if not latest_versions:
        raise ValueError(f"No model named {model_name} found in stage {model_stage}")
    
    # Get the latest version
    model_version = latest_versions[0]
    
    # Get the run ID and source (uri) of the model
    model_run_id = model_version.run_id
    model_source = mlflow.artifacts.download_artifacts(run_id=model_run_id, artifact_path="model")
    
    print(f"Found model: {model_name} version {model_version.version}, run_id: {model_run_id}")
    print(f"Model source: {model_source}")
    
    # Return model details
    return {
        "model_name": model_name,
        "model_version": model_version.version,
        "model_run_id": model_run_id,
        "model_source": model_source
    }

def define_image_details(**kwargs):
    """Define Docker image tag and full URI."""
    # Get current timestamp for the image tag
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Construct image tag and URI
    image_tag = f"v2-{current_time}"
    full_image_uri = f"{env_vars['ECR_REPOSITORY']}:{image_tag}"
    
    print(f"Defined image URI: {full_image_uri}")
    
    return {
        "image_tag": image_tag,
        "full_image_uri": full_image_uri
    }

def build_and_push_docker_image(**kwargs):
    """Build and push Docker image using values from XCom."""
    # Get the image URI directly from the define_image_details function
    image_details = define_image_details()
    image_uri = image_details['full_image_uri']
    
    print(f"Building Docker image with tag: {image_uri}")
    
    # Build the Docker image
    build_command = f"cd /home/ubuntu/health-predict && docker build -t {image_uri} ."
    print(f"Executing command: {build_command}")
    
    try:
        # First test if we can access the directory
        process = subprocess.run(
            "ls -la /home/ubuntu/health-predict",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Directory listing output: {process.stdout}")
        if process.returncode != 0:
            print(f"Error listing directory: {process.stderr}")
            raise AirflowFailException(f"Failed to access directory: {process.stderr}")
            
        # Now try to build the image
        process = subprocess.run(
            build_command,
            shell=True,
            check=False,  # Don't raise exception so we can log the error
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Error building Docker image: {process.stderr}")
            raise AirflowFailException(f"Failed to build Docker image: {process.stderr}")
        
        print(f"Docker build output: {process.stdout}")
        
        # Push the Docker image to ECR
        print(f"Pushing Docker image: {image_uri}")
        push_command = f"docker push {image_uri}"
        process = subprocess.run(
            push_command,
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Error pushing Docker image: {process.stderr}")
            raise AirflowFailException(f"Failed to push Docker image: {process.stderr}")
            
        print(f"Docker push output: {process.stdout}")
        
        return {
            "image_uri": image_uri,
            "build_success": True,
            "push_success": True
        }
    except Exception as e:
        print(f"Exception during Docker operations: {str(e)}")
        raise AirflowFailException(f"Docker operations failed: {str(e)}")

def update_kubernetes_deployment(**kwargs):
    """Update the Kubernetes deployment with the new image and environment variables."""
    # Call the build_and_push_docker_image function directly to get the image URI
    build_result = kwargs.get('build_result')
    if not build_result:
        # Use a direct call if not passed in
        build_result = build_and_push_docker_image()
    
    image_uri = build_result['image_uri']
    print(f"Updating Kubernetes deployment to use image: {image_uri}")
    
    # Update the Kubernetes deployment
    deployment_name = env_vars['K8S_DEPLOYMENT_NAME']
    container_name = env_vars['K8S_CONTAINER_NAME']
    
    # Update the container image
    update_command = f"kubectl set image deployment/{deployment_name} {container_name}={image_uri} --record"
    process = subprocess.run(
        update_command,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(f"Kubernetes update output: {process.stdout}")
    
    # Set environment variables in the deployment
    env_command = f"kubectl set env deployment/{deployment_name} MLFLOW_TRACKING_URI=http://{env_vars['EC2_PRIVATE_IP']}:5000"
    process = subprocess.run(
        env_command,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(f"Kubernetes env update output: {process.stdout}")
    
    return {
        "deployment_name": deployment_name,
        "container_name": container_name,
        "image_uri": image_uri,
        "update_success": True
    }

def verify_deployment_rollout(**kwargs):
    """Verify that the Kubernetes deployment has been successfully rolled out."""
    # Get deployment info directly
    deployment_info = kwargs.get('deployment_info')
    if not deployment_info:
        # Use direct call if not passed in
        deployment_info = update_kubernetes_deployment()
    
    deployment_name = deployment_info['deployment_name']
    
    # Check the rollout status
    print(f"Verifying rollout of deployment: {deployment_name}")
    rollout_command = f"kubectl rollout status deployment/{deployment_name} --timeout=5m"
    process = subprocess.run(
        rollout_command,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(f"Rollout status: {process.stdout}")
    
    # Get service URL
    service_command = "minikube service health-predict-api-service --url"
    process = subprocess.run(
        service_command,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    service_url = process.stdout.strip()
    print(f"Health Predict API service is accessible at: {service_url}")
    
    return {
        "deployment_name": deployment_name,
        "rollout_success": True,
        "service_url": service_url
    }

# Task 1: Get production model info from MLflow
get_production_model_info_task = PythonOperator(
    task_id='get_production_model_info',
    python_callable=get_production_model_info,
    dag=dag,
)

# Task 2: Define image details (tag, URI)
define_image_details_task = PythonOperator(
    task_id='define_image_details',
    python_callable=define_image_details,
    dag=dag,
)

# Task 3: ECR Login using AWS CLI
ecr_login_task = BashOperator(
    task_id='ecr_login',
    bash_command=(
        "aws ecr get-login-password --region {{ params.region }} | "
        "docker login --username AWS --password-stdin {{ params.registry }}"
    ),
    params={
        "region": env_vars["AWS_REGION"],
        "registry": env_vars["ECR_REPOSITORY"].split("/")[0],
    },
    dag=dag,
)

# Task 4: Build and push Docker image
build_and_push_image = PythonOperator(
    task_id='build_and_push_docker_image',
    python_callable=build_and_push_docker_image,
    dag=dag,
)

# Task 5: Update Kubernetes deployment
update_k8s_deployment = PythonOperator(
    task_id='update_kubernetes_deployment',
    python_callable=update_kubernetes_deployment,
    dag=dag,
)

# Task 6: Verify deployment rollout
verify_k8s_deployment = PythonOperator(
    task_id='verify_deployment_rollout',
    python_callable=verify_deployment_rollout,
    dag=dag,
)

# Task 9: Run API tests to ensure the deployed API works correctly
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
get_production_model_info_task >> define_image_details_task >> ecr_login_task >> build_and_push_image >> update_k8s_deployment >> verify_k8s_deployment >> run_api_tests 