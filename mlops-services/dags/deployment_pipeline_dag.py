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
import logging

# Get environment variables
env_vars = {
    "AWS_ACCOUNT_ID": os.getenv("AWS_ACCOUNT_ID"),
    "AWS_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"), # Standardized to AWS_DEFAULT_REGION
    "ECR_REPO_NAME": "health-predict-api", # Used for tagging image
    "ECR_REPOSITORY": os.getenv("ECR_REPOSITORY", "536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api"), # Used for full image URI
    "MLFLOW_PROD_MODEL_NAME": "HealthPredict_RandomForest",
    "MLFLOW_PROD_MODEL_STAGE": "Production",
    "K8S_NAMESPACE": "default",
    "K8S_DEPLOYMENT_NAME": os.getenv("K8S_DEPLOYMENT_NAME", "health-predict-api-deployment"),
    "K8S_CONTAINER_NAME": os.getenv("K8S_CONTAINER_NAME", "health-predict-api-container"),
    "K8S_SERVICE_NAME": os.getenv("K8S_SERVICE_NAME", "health-predict-api-service"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), # Use standardized env var
    "EC2_PRIVATE_IP": os.getenv("EC2_PRIVATE_IP"), # For K8s pods to connect to MLflow on host IP
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

# Define the local port for kubectl port-forward
# Using a high, less common port to minimize conflicts
K8S_PF_LOCAL_PORT = 19888

# Define the DAG
dag = DAG(
    'health_predict_api_deployment',
    default_args=default_args,
    description='Build and deploy the Health Predict API to Kubernetes',
    schedule_interval=None,  # Set to None for manual triggering
    start_date=days_ago(1),  # Changed from future date to days_ago(1)
    catchup=False,
    tags=['health-predict', 'deployment', 'k8s']
)

def get_production_model_info(**kwargs):
    """Gets the latest production model details from MLflow."""
    # Log environment variables first for debugging
    mlflow_tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI", "http://default-mlflow:5000")
    aws_account_id_env = os.getenv("AWS_ACCOUNT_ID", "not_set_in_env")
    ec2_private_ip_env = os.getenv("EC2_PRIVATE_IP", "not_set_in_env")
    
    logging.info(f"[get_production_model_info] MLFLOW_TRACKING_URI from os.getenv: {mlflow_tracking_uri_env}")
    logging.info(f"[get_production_model_info] AWS_ACCOUNT_ID from os.getenv: {aws_account_id_env}")
    logging.info(f"[get_production_model_info] EC2_PRIVATE_IP from os.getenv: {ec2_private_ip_env}")
    logging.info(f"[get_production_model_info] env_vars dict K8S_NAMESPACE: {env_vars.get('K8S_NAMESPACE')}")
    logging.info(f"[get_production_model_info] env_vars dict MLFLOW_TRACKING_URI: {env_vars.get('MLFLOW_TRACKING_URI')}")

    mlflow_tracking_uri = env_vars['MLFLOW_TRACKING_URI'] # Use value from env_vars dict as intended
    if not mlflow_tracking_uri:
        logging.error("[get_production_model_info] MLFLOW_TRACKING_URI from env_vars dict is missing!")
        raise AirflowFailException("MLFLOW_TRACKING_URI from env_vars dict is not set.")

    logging.info(f"[get_production_model_info] Connecting to MLflow at: {mlflow_tracking_uri}")
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)

    # Diagnostic: List all registered models
    logging.info("[get_production_model_info] Listing all registered models:")
    for rm in client.search_registered_models():
        logging.info(f"  - {rm.name}")
        for mv in rm.latest_versions:
            logging.info(f"    Version: {mv.version}, Stage: {mv.current_stage}, Run ID: {mv.run_id}")

    model_name = env_vars['MLFLOW_PROD_MODEL_NAME']
    model_stage = env_vars['MLFLOW_PROD_MODEL_STAGE']

    logging.info(f"[get_production_model_info] Fetching latest model '{model_name}' from stage '{model_stage}'...")
    
    # Get the latest version of the model in the specified stage
    latest_versions = client.get_latest_versions(name=model_name, stages=[model_stage])
    
    if not latest_versions:
        raise ValueError(f"No model named {model_name} found in stage {model_stage}")
    
    # Get the latest version
    model_version = latest_versions[0]
    model_run_id = model_version.run_id
    model_source = model_version.source # This will be like s3://... or runs:/...
    
    logging.info(f"[get_production_model_info] Found model: {model_name} version {model_version.version}, run_id: {model_run_id}")
    logging.info(f"[get_production_model_info] Model source: {model_source}")
        
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
    
    logging.info(f"[define_image_details] Defined image URI: {full_image_uri}")
    
    return {
        "image_tag": image_tag,
        "full_image_uri": full_image_uri
    }

def create_or_update_k8s_ecr_secret(**kwargs):
    """Creates or updates a Kubernetes secret for ECR authentication."""
    aws_account_id = env_vars["AWS_ACCOUNT_ID"]
    aws_region = env_vars["AWS_REGION"]
    ecr_registry_url = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com"
    secret_name = "ecr-registry-key"
    namespace = "default"

    logging.info(f"[create_or_update_k8s_ecr_secret] Creating/updating K8s secret '{secret_name}' for ECR registry '{ecr_registry_url}' in namespace '{namespace}'.")

    try:
        # Explicitly load Kubernetes configuration
        try:
            logging.info("[create_or_update_k8s_ecr_secret] Loading Kubernetes configuration...")
            # Attempt to load Kubeconfig using default mechanisms first
            # (KUBECONFIG env var, then default path ~/.kube/config)
            if os.getenv('KUBECONFIG'):
                logging.info(f"[create_or_update_k8s_ecr_secret] Attempting to load Kubeconfig from KUBECONFIG env var: {os.getenv('KUBECONFIG')}")
                config.load_kube_config(config_file=os.getenv('KUBECONFIG'))
            else:
                logging.info("[create_or_update_k8s_ecr_secret] Attempting to load Kubeconfig from default path (~/.kube/config).")
                config.load_kube_config()
            logging.info("[create_or_update_k8s_ecr_secret] Kubernetes configuration loaded successfully.")
        except config.ConfigException as e_conf:
            logging.error(f"[create_or_update_k8s_ecr_secret] Kubeconfig loading from default/env var failed (ConfigException): {str(e_conf)}. Trying /home/airflow/.kube/config...")
            try:
                config.load_kube_config(config_file="/home/airflow/.kube/config")
                logging.info("[create_or_update_k8s_ecr_secret] Successfully loaded Kubeconfig from /home/airflow/.kube/config")
            except Exception as e_conf_fallback:
                raise AirflowFailException(f"[create_or_update_k8s_ecr_secret] Failed to load Kubeconfig from default, env var, AND /home/airflow/.kube/config. Last error: {str(e_conf_fallback)}")
        except Exception as e_load_generic:
            raise AirflowFailException(f"[create_or_update_k8s_ecr_secret] An unexpected error occurred loading Kubeconfig: {str(e_load_generic)}")

        ecr_client = boto3.client("ecr", region_name=aws_region)
        auth_data = ecr_client.get_authorization_token()
        auth_token = auth_data["authorizationData"][0]["authorizationToken"]
        # Docker uses username:password base64 encoded, ECR token is 'AWS:<token>'
        # The token from ECR is already base64 encoded user:pass, but dockerconfigjson wants the raw token
        # username, password = base64.b64decode(auth_token).decode('utf-8').split(':')
        # For dockerconfigjson, the auth field is base64(username:password)
        # The ECR token is effectively the password, and username is "AWS"
        
        # The ECR token itself is base64(AWS:actual_token_from_ecr)
        # For .dockerconfigjson, the 'auth' field should be base64('AWS:' + actual_token_from_ecr)
        # The token received from get_authorization_token() is already base64 encoded.
        # No, the token from authorizationData[0]["authorizationToken"] is the base64 encoded string of "username:password".
        # So we just use that directly for the "auth" field in .dockerconfigjson AFTER base64 encoding it AGAIN for the secret data.
        # Correction: The "auth_token" is Base64(username:password).
        # The secret data for .dockerconfigjson's "auth" field needs to be this same base64 string.
        
        docker_config_json_content = {
            "auths": {
                ecr_registry_url: {
                    "auth": auth_token # This is the base64 encoded "AWS:token" string
                }
            }
        }
        
        secret_data_value = base64.b64encode(json.dumps(docker_config_json_content).encode('utf-8')).decode('utf-8')

        kube_api = client.CoreV1Api()
        
        secret_body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": secret_name, "namespace": namespace},
            "type": "kubernetes.io/dockerconfigjson",
            "data": {".dockerconfigjson": secret_data_value},
        }

        try:
            kube_api.read_namespaced_secret(name=secret_name, namespace=namespace)
            logging.info(f"[create_or_update_k8s_ecr_secret] Secret '{secret_name}' already exists. Patching...")
            kube_api.replace_namespaced_secret(name=secret_name, namespace=namespace, body=secret_body)
            logging.info(f"[create_or_update_k8s_ecr_secret] Secret '{secret_name}' patched successfully.")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                logging.info(f"[create_or_update_k8s_ecr_secret] Secret '{secret_name}' does not exist. Creating...")
                kube_api.create_namespaced_secret(namespace=namespace, body=secret_body)
                logging.info(f"[create_or_update_k8s_ecr_secret] Secret '{secret_name}' created successfully.")
            else:
                raise AirflowFailException(f"[create_or_update_k8s_ecr_secret] Failed to read/create K8s secret '{secret_name}': {str(e)}")
                
    except Exception as e:
        logging.error(f"[create_or_update_k8s_ecr_secret] Error creating/updating K8s ECR secret: {str(e)}")
        raise AirflowFailException(f"[create_or_update_k8s_ecr_secret] Failed to create/update K8s ECR secret: {str(e)}")

    return {"k8s_ecr_secret_name": secret_name, "status": "success"}

def build_and_push_docker_image(**kwargs):
    """Build and push Docker image using values from XCom."""
    ti = kwargs['ti']
    # Get the image URI from the define_image_details_task via XCom
    image_details = ti.xcom_pull(task_ids='define_image_details')
    if not image_details or 'full_image_uri' not in image_details:
        raise AirflowFailException("Could not pull image_details from XCom task 'define_image_details'")
    image_uri = image_details['full_image_uri']
    
    logging.info(f"[build_and_push_docker_image] Building Docker image with tag: {image_uri}")
    
    docker_build_command_list = ["docker", "build", "-t", image_uri, "."]
    logging.info(f"[build_and_push_docker_image] Executing command list: {docker_build_command_list} in /home/ubuntu/health-predict")
    
    try:
        logging.info(f"[build_and_push_docker_image] Python CWD: {os.getcwd()}")
        logging.info(f"[build_and_push_docker_image] Python root dir listing: {os.listdir('/')}")
        try:
            logging.info(f"[build_and_push_docker_image] Python /home/ubuntu dir listing: {os.listdir('/home/ubuntu')}")
        except Exception as e:
            logging.warning(f"[build_and_push_docker_image] Error listing /home/ubuntu: {str(e)}")

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
        logging.info(f"[build_and_push_docker_image] Directory listing output: {process.stdout}")
        if process.returncode != 0:
            logging.error(f"[build_and_push_docker_image] Error listing directory: {process.stderr}")
            raise AirflowFailException(f"[build_and_push_docker_image] Failed to access directory: {process.stderr}")
            
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
            logging.error(f"[build_and_push_docker_image] Error building Docker image: {process.stderr}")
            raise AirflowFailException(f"[build_and_push_docker_image] Failed to build Docker image: {process.stderr}")
        
        logging.info(f"[build_and_push_docker_image] Docker build output: {process.stdout}")
        
        # Push the Docker image to ECR
        logging.info(f"[build_and_push_docker_image] Pushing Docker image: {image_uri}")
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
            logging.error(f"[build_and_push_docker_image] Error pushing Docker image: {process.stderr}")
            raise AirflowFailException(f"[build_and_push_docker_image] Failed to push Docker image: {process.stderr}")
            
        logging.info(f"[build_and_push_docker_image] Docker push output: {process.stdout}")
        
        return {
            "image_uri": image_uri,
            "build_success": True,
            "push_success": True
        }
    except Exception as e:
        logging.error(f"[build_and_push_docker_image] Exception during Docker operations: {str(e)}")
        raise AirflowFailException(f"[build_and_push_docker_image] Docker operations failed: {str(e)}")

def update_kubernetes_deployment(**kwargs):
    ti = kwargs['ti']
    build_result = ti.xcom_pull(task_ids='build_and_push_docker_image')
    if not build_result or 'image_uri' not in build_result:
        raise AirflowFailException("Missing image_uri in XCom from build_and_push_docker_image")

    image_uri = build_result['image_uri']
    deployment_name = env_vars['K8S_DEPLOYMENT_NAME']
    container_name = env_vars['K8S_CONTAINER_NAME']
    namespace = "default" # Or your target namespace
    k8s_ecr_secret_name = "ecr-registry-key" # Name of the secret created

    logging.info(f"[update_kubernetes_deployment] Attempting to update deployment '{deployment_name}' in namespace '{namespace}' to image '{image_uri}'")

    try:
        logging.info("[update_kubernetes_deployment] Loading Kubernetes configuration...")
        # config.load_incluster_config() # Use this if Airflow itself runs in K8s
        config.load_kube_config() # Assumes Kubeconfig is in default loc (~/.kube/config) or KUBECONFIG env var is set
        logging.info("[update_kubernetes_deployment] Kubernetes configuration loaded.")
    except Exception as e:
        logging.error(f"[update_kubernetes_deployment] Error loading Kubeconfig: {str(e)}")
        # Fallback: try specific path if default loading fails (for debugging, less ideal for prod)
        try:
            logging.info("[update_kubernetes_deployment] Attempting to load Kubeconfig from /home/airflow/.kube/config")
            config.load_kube_config(config_file="/home/airflow/.kube/config")
            logging.info("[update_kubernetes_deployment] Successfully loaded Kubeconfig from /home/airflow/.kube/config")
        except Exception as e2:
            raise AirflowFailException(f"[update_kubernetes_deployment] Failed to load kubeconfig from default and specific path /home/airflow/.kube/config. Default error: {str(e)}. Specific path error: {str(e2)}")

    api = client.AppsV1Api()
    logging.info("[update_kubernetes_deployment] AppsV1Api client created.")

    try:
        logging.info(f"[update_kubernetes_deployment] Fetching current deployment '{deployment_name}'...")
        current_deployment = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        logging.info("[update_kubernetes_deployment] Current deployment fetched.")

        # Modify the container image
        found_container = False
        for container in current_deployment.spec.template.spec.containers:
            if container.name == container_name:
                logging.info(f"[update_kubernetes_deployment] Updating image for container '{container_name}' to '{image_uri}'.")
                container.image = image_uri
                found_container = True
                break
        if not found_container:
            raise AirflowFailException(f"[update_kubernetes_deployment] Container '{container_name}' not found in deployment '{deployment_name}'.")

        # Add/Update imagePullSecrets
        current_deployment.spec.template.spec.image_pull_secrets = [client.V1LocalObjectReference(name=k8s_ecr_secret_name)]
        logging.info(f"[update_kubernetes_deployment] Set imagePullSecrets to use '{k8s_ecr_secret_name}'.")

        # Update environment variables
        # Ensure EC2_PRIVATE_IP is available
        ec2_private_ip = env_vars.get('EC2_PRIVATE_IP')
        if not ec2_private_ip:
            raise AirflowFailException("EC2_PRIVATE_IP environment variable is not set, cannot update MLFLOW_TRACKING_URI for K8s pod.")
        
        mlflow_tracking_uri_for_pod = f"http://{ec2_private_ip}:5000"
        
        env_var_mlflow = client.V1EnvVar(name="MLFLOW_TRACKING_URI", value=mlflow_tracking_uri_for_pod)
        
        updated_env_vars = False
        for container in current_deployment.spec.template.spec.containers:
            if container.name == container_name:
                if container.env:
                    env_exists = False
                    for i, env_entry in enumerate(container.env):
                        if env_entry.name == "MLFLOW_TRACKING_URI":
                            container.env[i] = env_var_mlflow
                            env_exists = True
                            break
                    if not env_exists:
                        container.env.append(env_var_mlflow)
                else:
                    container.env = [env_var_mlflow]
                updated_env_vars = True
                logging.info(f"[update_kubernetes_deployment] Updated/Set MLFLOW_TRACKING_URI to '{mlflow_tracking_uri_for_pod}' for container '{container_name}'.")
                break
        
        if not updated_env_vars:
             # This case should ideally not be reached if found_container was true earlier for image update.
            raise AirflowFailException(f"[update_kubernetes_deployment] Failed to find container '{container_name}' to update env vars, though it was found for image update.")

        logging.info(f"[update_kubernetes_deployment] Replacing deployment '{deployment_name}' with updated configuration...")
        api.replace_namespaced_deployment(name=deployment_name, namespace=namespace, body=current_deployment)
        logging.info(f"[update_kubernetes_deployment] Deployment '{deployment_name}' replaced successfully.")

    except client.exceptions.ApiException as e:
        logging.error(f"[update_kubernetes_deployment] Kubernetes API Exception: {e.status} - {e.reason} - {e.body}")
        raise AirflowFailException(f"[update_kubernetes_deployment] Failed to update K8s deployment: Status {e.status} - {e.reason} - Body: {e.body}")
    except Exception as e:
        raise AirflowFailException(f"An unexpected error occurred when updating K8s deployment: {str(e)}")

    return {
        "deployment_name": deployment_name,
        "container_name": container_name,
        "image_uri": image_uri,
        "update_success": True
    }

def verify_deployment_rollout(**kwargs):
    """Simplified deployment verification using Kubernetes readiness checks"""
    try:
        if os.getenv('KUBECONFIG'):
            logging.info(f"Loading Kubeconfig from KUBECONFIG env var: {os.getenv('KUBECONFIG')}")
            config.load_kube_config(config_file=os.getenv('KUBECONFIG'))
        else:
            logging.info("Loading Kubeconfig using default mechanisms (e.g., ~/.kube/config or in-cluster).")
            config.load_kube_config()
        logging.info("Kubernetes configuration loaded.")
    except config.ConfigException as e:
        logging.error(f"Error loading Kubeconfig (ConfigException): {e}. Trying specific path /home/airflow/.kube/config")
        try:
            config.load_kube_config(config_file="/home/airflow/.kube/config")
            logging.info("Successfully loaded Kubeconfig from /home/airflow/.kube/config")
        except Exception as e2:
             raise AirflowFailException(f"Failed to load kubeconfig. ConfigException: {e}. Specific path error: {e2}")
    except Exception as e:
        raise AirflowFailException(f"An unexpected error occurred loading Kubeconfig: {str(e)}")

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    namespace = "default"

    K8S_DEPLOYMENT_NAME = env_vars['K8S_DEPLOYMENT_NAME']

    logging.info(f"Verifying rollout status for deployment '{K8S_DEPLOYMENT_NAME}' in namespace '{namespace}'...")

    for i in range(1, 11):  # Poll for up to 5 minutes (10 attempts * 30 seconds interval)
        logging.info(f"Rollout check attempt {i}/10...")
        try:
            deployment = apps_v1.read_namespaced_deployment(name=K8S_DEPLOYMENT_NAME, namespace=namespace)
            desired_replicas = deployment.spec.replicas
            updated_replicas = deployment.status.updated_replicas if deployment.status and deployment.status.updated_replicas is not None else 0
            ready_replicas = deployment.status.ready_replicas if deployment.status and deployment.status.ready_replicas is not None else 0
            available_replicas = deployment.status.available_replicas if deployment.status and deployment.status.available_replicas is not None else 0

            logging.info(
                f"  Desired: {desired_replicas}, Updated: {updated_replicas}, "
                f"Ready: {ready_replicas}, Available: {available_replicas}"
            )

            if desired_replicas == updated_replicas and \
               desired_replicas == ready_replicas and \
               desired_replicas == available_replicas and \
               ready_replicas > 0:
                
                logging.info(f"Deployment '{K8S_DEPLOYMENT_NAME}' successfully rolled out!")
                
                # Verify pod health by checking readiness probes
                healthy_pods = 0
                try:
                    pods = core_v1.list_namespaced_pod(
                        namespace=namespace, 
                        label_selector="app=health-predict-api"
                    )
                    
                    for pod in pods.items:
                        if pod.status.phase == "Running":
                            # Check readiness condition
                            for condition in pod.status.conditions or []:
                                if condition.type == "Ready" and condition.status == "True":
                                    healthy_pods += 1
                                    logging.info(f"  Pod {pod.metadata.name} is healthy and ready")
                                    break
                    
                    if healthy_pods == 0:
                        logging.warning("No healthy pods found despite successful rollout status")
                        continue  # Continue polling
                    
                    logging.info(f"Deployment verification passed: {healthy_pods} healthy pods running")
                    return {
                        "rollout_status": "success",
                        "healthy_pods": healthy_pods,
                        "deployment_name": K8S_DEPLOYMENT_NAME
                    }
                    
                except Exception as pod_check_error:
                    logging.error(f"Error checking pod health: {pod_check_error}")
                    continue  # Continue polling
                
                break  # Exit loop on successful verification
            else:
                # Log pod statuses if rollout is not yet complete
                try:
                    pods = core_v1.list_namespaced_pod(
                        namespace=namespace, label_selector="app=health-predict-api"
                    )
                    if not pods.items:
                        logging.info("  No pods found matching the deployment's selector yet.")
                    for pod_item in pods.items:
                        logging.info(f"  Pod: {pod_item.metadata.name} (Status: {pod_item.status.phase})")
                        if pod_item.status.container_statuses:
                            for cs in pod_item.status.container_statuses:
                                logging.info(f"    Container: {cs.name}, Ready: {cs.ready}, Restart Count: {cs.restart_count}")
                                if cs.state and cs.state.waiting:
                                    logging.info(f"      Waiting: {cs.state.waiting.reason} - {cs.state.waiting.message}")
                                if cs.state and cs.state.terminated:
                                    logging.info(f"      Terminated: {cs.state.terminated.reason} - {cs.state.terminated.message}")
                except Exception as e_pod_log:
                    logging.warning(f"Could not fetch detailed pod statuses: {str(e_pod_log)}")

        except client.ApiException as e:
            logging.error(f"Kubernetes API Exception during rollout check: {e.status} - {e.reason} - {e.body}")
            # Continue to next attempt if it's an API error, might be transient
        except Exception as e_main_loop:
            logging.error(f"Unexpected error during this rollout check attempt: {str(e_main_loop)}")
            if not isinstance(e_main_loop, AirflowFailException):
                raise AirflowFailException(f"Unexpected error in rollout check attempt: {e_main_loop}")
            else:
                raise

        if i < 10:
            logging.info("Retrying rollout check in 30 seconds...")
            time.sleep(30)
    
    # If we get here, the deployment failed
    logging.error(f"Deployment '{K8S_DEPLOYMENT_NAME}' failed to roll out after 10 attempts")
    raise AirflowFailException(f"Deployment failed to roll out successfully within the timeout period")

# Removed complex API testing functions - now handled in training DAG pre-deployment

def post_deployment_health_check(**kwargs):
    """Simple post-deployment health check using Kubernetes service"""
    try:
        if os.getenv('KUBECONFIG'):
            config.load_kube_config(config_file=os.getenv('KUBECONFIG'))
        else:
            config.load_kube_config()
        logging.info("Kubernetes configuration loaded for health check.")
    except Exception as e:
        raise AirflowFailException(f"Failed to load kubeconfig for health check: {str(e)}")

    core_v1 = client.CoreV1Api()
    namespace = "default"
    
    logging.info("Performing basic health check on deployed API...")
    
    # Check if pods are healthy
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace, 
            label_selector="app=health-predict-api"
        )
        
        if not pods.items:
            raise AirflowFailException("No pods found with label app=health-predict-api")
        
        healthy_count = 0
        for pod in pods.items:
            if pod.status.phase == "Running":
                for condition in pod.status.conditions or []:
                    if condition.type == "Ready" and condition.status == "True":
                        healthy_count += 1
                        logging.info(f"Pod {pod.metadata.name} is healthy")
                        break
        
        if healthy_count == 0:
            raise AirflowFailException("No healthy pods found")
        
        logging.info(f"Health check passed: {healthy_count} healthy pod(s) running")
        
        return {
            "health_status": "healthy",
            "healthy_pods": healthy_count,
            "total_pods": len(pods.items)
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise AirflowFailException(f"Post-deployment health check failed: {str(e)}")

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

# New Task 3.5: Create/Update Kubernetes ECR Secret
create_k8s_ecr_secret_task = PythonOperator(
    task_id='create_or_update_k8s_ecr_secret',
    python_callable=create_or_update_k8s_ecr_secret,
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

# Task 6: Verify deployment rollout with simplified health checks
verify_k8s_deployment = PythonOperator(
    task_id='verify_deployment_rollout',
    python_callable=verify_deployment_rollout,
    dag=dag,
)

# Task 7: Post-deployment health check
post_deployment_health_check_task = PythonOperator(
    task_id='post_deployment_health_check',
    python_callable=post_deployment_health_check,
    dag=dag,
)

# Define simplified task dependencies
get_production_model_info_task >> define_image_details_task >> ecr_login_task

ecr_login_task >> build_and_push_image
build_and_push_image >> create_k8s_ecr_secret_task 
create_k8s_ecr_secret_task >> update_k8s_deployment

update_k8s_deployment >> verify_k8s_deployment >> post_deployment_health_check_task