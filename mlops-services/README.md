# MLOps Services (`mlops-services`)

This directory contains the configuration and components for running the core MLOps services (Airflow, MLflow, PostgreSQL, JupyterLab) using Docker Compose.

## Overview

The `docker-compose.yml` file orchestrates the following services:
-   **PostgreSQL (`postgres`):** Backend database for Airflow and MLflow metadata.
-   **Airflow (`airflow-scheduler`, `airflow-webserver`, `airflow-init`):** Workflow management system for orchestrating ML pipelines (training, deployment, monitoring).
    -   Uses `LocalExecutor`.
    -   DAGs are located in the `dags/` subdirectory.
-   **MLflow (`mlflow`):** Platform for managing the end-to-end machine learning lifecycle.
    -   Used for experiment tracking, model versioning, and model registry.
    -   Configured with a PostgreSQL backend and AWS S3 for artifact storage.
-   **JupyterLab (`jupyterlab`):** Interactive development environment for data exploration, notebook execution, and script development.

## Prerequisites
-   Docker installed and running.
-   Docker Compose installed.
-   AWS CLI installed and configured with necessary credentials (if DAGs or scripts interact directly with AWS services outside of what the EC2 instance profile provides).
-   A `.env` file in the project root (`health-predict/`) containing necessary environment variables (see "Key Environment Variables" section).

## Core Workflows

### Starting Services
From the `health-predict/mlops-services/` directory (or specifying the compose file path):
```bash
# Ensure you are in health-predict/mlops-services/ or adjust paths accordingly
docker compose --env-file ../.env up -d
```
To rebuild images (e.g., after Dockerfile changes):
```bash
docker compose --env-file ../.env up -d --build
```

### Stopping Services
```bash
docker compose --env-file ../.env down
```
To remove volumes (e.g., to reset databases - **use with caution**):
```bash
docker compose --env-file ../.env down -v
```

### Accessing UIs
(Assuming you are on the EC2 instance or have appropriate port forwarding if running locally elsewhere)
-   **Airflow UI:** `http://localhost:8080` (or `http://<EC2_Public_IP>:8080`)
-   **MLflow UI:** `http://localhost:5000` (or `http://<EC2_Public_IP>:5000`)
-   **JupyterLab UI:** `http://localhost:8888` (or `http://<EC2_Public_IP>:8888`)

### Running ML Pipelines
1.  **Training & HPO Pipeline:**
    *   Ensure the `health_predict_training_hpo` DAG is unpaused in the Airflow UI.
    *   Trigger it manually from the UI or via Airflow CLI:
        ```bash
        docker compose exec airflow-scheduler airflow dags trigger health_predict_training_hpo
        ```
    *   This DAG will train models, perform HPO, and register the best model (with its preprocessor) to the MLflow Model Registry, promoting it to "Production".

2.  **API Deployment Pipeline:**
    *   Ensure the `health_predict_training_hpo` DAG has successfully run and a model is in the "Production" stage in MLflow.
    *   Ensure the `health_predict_api_deployment` DAG is unpaused in the Airflow UI.
    *   Trigger it manually:
        ```bash
        docker compose exec airflow-scheduler airflow dags trigger health_predict_api_deployment
        ```
    *   This DAG builds the API Docker image, pushes it to ECR, and updates the Kubernetes deployment.

## Debugging Tips

### General Docker Compose
-   **Check service status:** `docker compose ps`
-   **View logs for all services:** `docker compose logs -f`
-   **View logs for a specific service:** `docker compose logs -f <service_name>` (e.g., `airflow-scheduler`)
-   **`.env` file:** Ensure it's correctly located at `health-predict/.env` and sourced using `--env-file ../.env` when running `docker compose` commands from `mlops-services/`.
-   **Port conflicts:** If services fail to start, check for port conflicts on the host.
-   **Volume mounts:** Verify paths in `docker-compose.yml` are correct and host directories have appropriate permissions.

### Airflow
-   **DAG parsing errors:** Check `airflow-scheduler` logs.
-   **Task failures:** View logs directly in the Airflow UI for the specific task instance.
-   **CLI for DAG/Task management:**
    ```bash
    docker compose exec airflow-scheduler airflow dags list
    docker compose exec airflow-scheduler airflow dags state <dag_id>
    docker compose exec airflow-scheduler airflow dags list-runs -d <dag_id>
    docker compose exec airflow-scheduler airflow tasks list <dag_id>
    docker compose exec airflow-scheduler airflow tasks states-for-dag-run <dag_id> <run_id>
    # To view raw task logs (replace placeholders):
    docker compose exec airflow-scheduler cat /opt/airflow/logs/dag_id=<dag_id>/run_id=<run_id>/task_id=<task_id>/attempt=<attempt_number>.log
    ```

### MLflow
-   **Check MLflow UI:** For experiments, runs, registered models, and artifacts.
-   **Service logs:** `docker compose logs -f mlflow`
-   **MLflow CLI (inside container):**
    ```bash
    docker compose exec mlflow mlflow experiments list
    docker compose exec mlflow mlflow runs list --experiment-id <id>
    docker compose exec mlflow mlflow models list
    ```
-   **Tracking URI:** Ensure clients (DAGs, scripts) are using the correct URI (`http://mlflow:5000` for inter-service communication.
    *   **Key Fix Reminder**: A schemeless URI (e.g., `mlflow`) is treated as a local path, not a hostname. Always use `http://mlflow:5000` for services within the Docker network.

### Kubernetes API Deployment (Minikube on EC2)
-   **Check pod status:** `kubectl get pods -l app=health-predict-api -n default`
-   **Pod logs:** `kubectl logs <pod_name> -n default -c <container_name>` (e.g., `health-predict-api-container`)
-   **Describe pod (for events/errors):** `kubectl describe pod <pod_name> -n default`
-   **Deployment status:** `kubectl describe deployment health-predict-api-deployment -n default`
-   **Service URL:** `minikube service health-predict-api-service --url -n default` (run on the EC2 instance)
-   **API Test Connectivity:** If testing the API from the Airflow worker, network configuration is crucial. The `run_api_tests` task currently hardcodes `MINIKUBE_IP` to `127.0.0.1` which is suitable if the Airflow worker can resolve Minikube services as localhost. This step is currently skipped in the DAG due to CI environment limitations.

## Key Environment Variables (Expected in `../.env`)
-   `AWS_ACCESS_KEY_ID`: Your AWS Access Key ID.
-   `AWS_SECRET_ACCESS_KEY`: Your AWS Secret Access Key.
-   `AWS_DEFAULT_REGION`: Default AWS region (e.g., `us-east-1`).
-   `ECR_REPOSITORY`: Full ECR repository URI for the API image (e.g., `account_id.dkr.ecr.region.amazonaws.com/health-predict-api`).
-   `AWS_ACCOUNT_ID`: Your AWS Account ID.
-   `MLFLOW_TRACKING_URI`: Should be `http://mlflow:5000` for inter-service communication.
-   `EC2_PRIVATE_IP`: The private IP of the EC2 instance, used by K8s pods to reach MLflow on the host network.
-   `K8S_DEPLOYMENT_NAME`: Name of the K8s deployment (e.g., `health-predict-api-deployment`).
-   `K8S_CONTAINER_NAME`: Name of the container within the K8s deployment (e.g., `health-predict-api-container`).
-   `K8S_SERVICE_NAME`: Name of the K8s service (e.g., `health-predict-api-service`).
-   Airflow specific variables like `FERNET_KEY`, `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` are also present. 