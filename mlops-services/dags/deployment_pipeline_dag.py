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
from kubernetes import client, config, watch
import time

# Get environment variables
env_vars = {
    'MLFLOW_SERVER': os.getenv('MLFLOW_SERVER', 'mlflow'),
    'ECR_REPOSITORY': os.getenv('ECR_REPOSITORY', '536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api'),
    'AWS_REGION': os.getenv('AWS_REGION', 'us-east-1'),
    'MODEL_NAME': os.getenv('MODEL_NAME', 'HealthPredict_RandomForest'),
    'MODEL_STAGE': os.getenv('MODEL_STAGE', 'Production'),
    'K8S_DEPLOYMENT_NAME': os.getenv('K8S_DEPLOYMENT_NAME', 'health-predict-api-deployment'),
    'K8S_CONTAINER_NAME': os.getenv('K8S_CONTAINER_NAME', 'health-predict-api-container'),
    'K8S_SERVICE_NAME': os.getenv('K8S_SERVICE_NAME', 'health-predict-api-service'),
    'EC2_PRIVATE_IP': os.getenv('EC2_PRIVATE_IP', '10.0.0.123'),
}

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
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
    ti = kwargs['ti']
    # Get the image URI from the define_image_details_task via XCom
    image_details = ti.xcom_pull(task_ids='define_image_details')
    if not image_details or 'full_image_uri' not in image_details:
        raise AirflowFailException("Could not pull image_details from XCom task 'define_image_details'")
    image_uri = image_details['full_image_uri']
    
    print(f"Building Docker image with tag: {image_uri}")
    
    docker_build_command_list = ["docker", "build", "-t", image_uri, "."]
    print(f"Executing command list: {docker_build_command_list} in /home/ubuntu/health-predict")
    
    try:
        print(f"Python CWD: {os.getcwd()}")
        print(f"Python root dir listing: {os.listdir('/')}")
        try:
            print(f"Python /home/ubuntu dir listing: {os.listdir('/home/ubuntu')}")
        except Exception as e:
            print(f"Error listing /home/ubuntu: {str(e)}")

        # First test if we can access the directory
        process = subprocess.run(
            "ls -la",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/ubuntu/health-predict"
        )
        print(f"Directory listing output: {process.stdout}")
        if process.returncode != 0:
            print(f"Error listing directory: {process.stderr}")
            raise AirflowFailException(f"Failed to access directory: {process.stderr}")
            
        # Now try to build the image
        process = subprocess.run(
            docker_build_command_list,
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/ubuntu/health-predict"
        )
        
        if process.returncode != 0:
            print(f"Error building Docker image: {process.stderr}")
            raise AirflowFailException(f"Failed to build Docker image: {process.stderr}")
        
        print(f"Docker build output: {process.stdout}")
        
        # Push the Docker image to ECR
        print(f"Pushing Docker image: {image_uri}")
        docker_push_command_list = ["docker", "push", image_uri]
        process = subprocess.run(
            docker_push_command_list,
            shell=False,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/ubuntu/health-predict"
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
    ti = kwargs['ti']
    build_result = ti.xcom_pull(task_ids='build_and_push_docker_image')
    if not build_result or 'image_uri' not in build_result:
        raise AirflowFailException("Missing image_uri in XCom from build_and_push_docker_image")

    image_uri = build_result['image_uri']
    deployment_name = env_vars['K8S_DEPLOYMENT_NAME']
    container_name = env_vars['K8S_CONTAINER_NAME']
    namespace = "default" # Or your target namespace

    print(f"Attempting to update deployment '{deployment_name}' in namespace '{namespace}' to image '{image_uri}'")

    try:
        print("Loading Kubernetes configuration...")
        # config.load_incluster_config() # Use this if Airflow itself runs in K8s
        config.load_kube_config() # Assumes Kubeconfig is in default loc (~/.kube/config) or KUBECONFIG env var is set
        print("Kubernetes configuration loaded.")
    except Exception as e:
        print(f"Error loading Kubeconfig: {str(e)}")
        # Fallback: try specific path if default loading fails (for debugging, less ideal for prod)
        try:
            print("Attempting to load Kubeconfig from /home/airflow/.kube/config")
            config.load_kube_config(config_file="/home/airflow/.kube/config")
            print("Successfully loaded Kubeconfig from /home/airflow/.kube/config")
        except Exception as e2:
            raise AirflowFailException(f"Failed to load kubeconfig from default and specific path /home/airflow/.kube/config. Default error: {str(e)}. Specific path error: {str(e2)}")

    api = client.AppsV1Api()
    print("AppsV1Api client created.")

    # Define the patch for the image update
    image_patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container_name,
                            "image": image_uri
                        }
                    ]
                }
            }
        }
    }

    try:
        print(f"Patching deployment '{deployment_name}' with new image...")
        api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=image_patch
        )
        print(f"Deployment '{deployment_name}' patched successfully with image '{image_uri}'.")
    except client.exceptions.ApiException as e:
        print(f"Kubernetes API Exception when patching image: {e.status} - {e.reason} - {e.body}")
        raise AirflowFailException(f"Failed to patch deployment image: Status {e.status} - {e.reason} - Body: {e.body}")
    except Exception as e:
        raise AirflowFailException(f"An unexpected error occurred when patching deployment image: {str(e)}")

    # Define the patch for the environment variable update
    env_patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container_name,
                            "env": [
                                {"name": "MLFLOW_TRACKING_URI", 
                                 "value": f"http://{env_vars['EC2_PRIVATE_IP']}:5000"}
                            ]
                        }
                    ]
                }
            }
        }
    }

    try:
        print(f"Patching deployment '{deployment_name}' with new environment variables...")
        api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=env_patch
        )
        print(f"Deployment '{deployment_name}' patched successfully with environment variables.")
    except client.exceptions.ApiException as e:
        print(f"Kubernetes API Exception when patching env vars: {e.status} - {e.reason} - {e.body}")
        raise AirflowFailException(f"Failed to patch deployment env vars: Status {e.status} - {e.reason} - Body: {e.body}")
    except Exception as e:
        raise AirflowFailException(f"An unexpected error occurred when patching deployment env vars: {str(e)}")

    return {
        "deployment_name": deployment_name,
        "container_name": container_name,
        "image_uri": image_uri,
        "update_success": True
    }

def verify_deployment_rollout(**kwargs):
    ti = kwargs['ti']
    deployment_name = env_vars['K8S_DEPLOYMENT_NAME']
    namespace = "default"
    max_retries = 10  # Number of times to check for rollout status (e.g., 10 * 30s = 5 minutes)
    retry_delay_seconds = 30

    print(f"Verifying rollout status for deployment '{deployment_name}' in namespace '{namespace}'...")

    try:
        # Ensure Kubeconfig is loaded (same as in update_kubernetes_deployment)
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"Error loading Kubeconfig: {str(e)}")
            try:
                config.load_kube_config(config_file="/home/airflow/.kube/config")
            except Exception as e2:
                raise AirflowFailException(f"Failed to load kubeconfig. Default error: {str(e)}. Specific path error: {str(e2)}")
        
        api = client.AppsV1Api()
        core_v1_api = client.CoreV1Api() # For pod logs

        for i in range(max_retries):
            print(f"Rollout check attempt {i+1}/{max_retries}...")
            deployment = api.read_namespaced_deployment_status(name=deployment_name, namespace=namespace)
            status = deployment.status
            
            desired_replicas = deployment.spec.replicas if deployment.spec.replicas is not None else 1 # Default to 1 if not set
            ready_replicas = status.ready_replicas if status.ready_replicas is not None else 0
            updated_replicas = status.updated_replicas if status.updated_replicas is not None else 0
            available_replicas = status.available_replicas if status.available_replicas is not None else 0

            print(f"  Desired: {desired_replicas}, Updated: {updated_replicas}, Ready: {ready_replicas}, Available: {available_replicas}")

            # Check conditions for successful rollout
            rollout_complete = (
                status.observed_generation == deployment.metadata.generation and
                updated_replicas == desired_replicas and
                available_replicas == desired_replicas and
                ready_replicas == desired_replicas
            )
            
            if rollout_complete:
                print(f"Deployment '{deployment_name}' successfully rolled out.")
                # Optionally: Check service endpoint if applicable after successful rollout
                return {"rollout_status": "success"}

            # If not complete, fetch and print recent pod logs for debugging
            # This helps diagnose issues if pods are crashing or not becoming ready
            print(f"Rollout not yet complete. Fetching pod logs for deployment '{deployment_name}'...")
            pods = core_v1_api.list_namespaced_pod(
                namespace=namespace, 
                label_selector=f"app={deployment.spec.template.metadata.labels.get('app')}" # Assumes a standard 'app' label
            )
            for pod_item in pods.items:
                pod_name = pod_item.metadata.name
                print(f"  Logs for pod: {pod_name} (status: {pod_item.status.phase})")
                try:
                    # Fetch logs for each container in the pod
                    for container in pod_item.spec.containers:
                        container_name = container.name
                        print(f"    Logs for container: {container_name}")
                        api_response = core_v1_api.read_namespaced_pod_log(
                            name=pod_name, 
                            namespace=namespace,
                            container=container_name,
                            tail_lines=20 # Get last 20 lines
                        )
                        print(api_response)
                except client.exceptions.ApiException as log_e:
                    print(f"    Could not retrieve logs for pod {pod_name}: {log_e.reason}")
                except Exception as log_exc:
                    print(f"    Unexpected error retrieving logs for pod {pod_name}: {str(log_exc)}")

            if i < max_retries - 1:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds) # Use time.sleep(), ensure time is imported
            else:
                raise AirflowFailException(f"Deployment '{deployment_name}' failed to roll out after {max_retries} attempts.")

    except client.exceptions.ApiException as e:
        raise AirflowFailException(f"Kubernetes API Error during rollout verification: {e.status} - {e.reason} - Body: {e.body}")
    except Exception as e:
        # If it's an AirflowRetryableException, re-raise it to let Airflow handle the retry
        if isinstance(e, airflow.exceptions.AirflowRetryableException):
            raise e
        # For other exceptions, wrap in AirflowFailException
        raise AirflowFailException(f"Unexpected error during rollout verification: {str(e)}")

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

# Task 5: Update Kubernetes deployment (Now uses Python K8s client)
update_k8s_deployment = PythonOperator(
    task_id='update_kubernetes_deployment',
    python_callable=update_kubernetes_deployment,
    dag=dag,
)

# Task 6: Verify deployment rollout (Now uses Python K8s client)
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