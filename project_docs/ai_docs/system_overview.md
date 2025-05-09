# MLOps System Overview for Health-Predict Project

## 1. Introduction

This document provides an overview of the MLOps system built for the "Health Predict" capstone project. The system is designed to orchestrate machine learning model training, hyperparameter optimization (HPO), and experiment tracking using a suite of MLOps tools running in Docker containers on an AWS EC2 instance. Its primary goal is to predict patient readmission.

## 2. Core Infrastructure

*   **AWS EC2 Instance**: A virtual server in AWS cloud hosting all Dockerized services. The specific instance type and OS may vary but should have Docker and Docker Compose installed.
*   **Docker & Docker Compose**:
    *   Docker is used for containerizing individual services.
    *   Docker Compose (via `mlops-services/docker-compose.yml`) is used to define and manage the multi-container application.

## 3. MLOps Services (via `mlops-services/docker-compose.yml`)

All services are defined in `~/health-predict/mlops-services/docker-compose.yml`.

*   **`postgres`**:
    *   **Purpose**: Provides a PostgreSQL database backend for both Airflow and MLflow.
    *   **Configuration**:
        *   Uses a standard PostgreSQL Docker image.
        *   Persistent data is stored in a Docker volume (`postgres_data`).
        *   Includes an initialization script `mlops-services/init-mlflow-db.sh` (mounted to `/docker-entrypoint-initdb.d/`) that automatically creates the `mlflowdb` database required by MLflow upon first startup.
*   **`mlflow`**:
    *   **Purpose**: MLflow Tracking Server for logging experiments, parameters, metrics, and artifacts.
    *   **Configuration**:
        *   Uses the official MLflow Docker image.
        *   Command configured to start the MLflow server with:
            *   Backend store URI pointing to the `postgres` service (`postgresql://airflow:airflow@postgres:5432/mlflowdb`).
            *   Default artifact root pointing to an S3 bucket (e.g., `s3://<your-mlflow-s3-bucket>/mlflow-artifacts/`).
        *   Depends on the `postgres` service.
        *   Accessible on host port `5000`.
*   **`airflow-init`**:
    *   **Purpose**: A one-time service to initialize the Airflow database and create an initial admin user.
    *   **Configuration**:
        *   Uses the Apache Airflow Docker image.
        *   `entrypoint: /bin/bash`
        *   `command: -c "airflow db init && airflow users create --username admin --password admin --firstname Anonymous --lastname User --role Admin --email admin@example.com"`
        *   Depends on the `postgres` service.
*   **`airflow-scheduler`**:
    *   **Purpose**: Monitors Airflow DAGs and triggers their tasks as per schedule or manual triggers.
    *   **Configuration**:
        *   Uses the Apache Airflow Docker image.
        *   Mounts the Docker socket (`/var/run/docker.sock:/var/run/docker.sock`) to allow tasks to execute `docker-compose exec` commands on the host.
        *   `group_add: ["<host_docker_gid>"]` (e.g., "998") is crucial to grant the containerized Airflow user permission to use the host's Docker socket.
        *   Installs specific versions of `docker`, `docker-compose`, and `jsonschema` via pip in its startup command to ensure compatibility for executing `docker-compose` commands.
        *   Depends on `postgres` and `airflow-init`.
        *   Mounts local `mlops-services/dags` to `/opt/airflow/dags` in the container.
*   **`airflow-webserver`**:
    *   **Purpose**: Provides the web-based UI for Airflow.
    *   **Configuration**:
        *   Similar Docker socket mount, `group_add` configuration, and pip installs as `airflow-scheduler`.
        *   Depends on `postgres` and `airflow-init`.
        *   Accessible on host port `8080`.
*   **`jupyterlab`**:
    *   **Purpose**: Provides a JupyterLab environment for interactive development, data exploration, and execution of Python scripts (including the main model training script).
    *   **Configuration**:
        *   Uses a Jupyter Docker image (e.g., `jupyter/scipy-notebook`).
        *   Mounts the entire project root `~/health-predict` to `/home/jovyan/work` within the container, allowing access to all project files.
        *   Installs Python dependencies from `/home/jovyan/work/scripts/requirements-training.txt` upon startup.
        *   Accessible on host port `8888`.
        *   This container is the execution environment for `scripts/train_model.py` when triggered by Airflow.

## 4. Key Files and Directories (relative to `~/health-predict/`)

*   `mlops-services/docker-compose.yml`: The heart of the MLOps stack, defining all services.
*   `mlops-services/dags/`: Contains Airflow DAG definitions.
    *   `training_pipeline_dag.py`: Orchestrates the model training and HPO pipeline.
*   `scripts/train_model.py`: The main Python script for training models, performing HPO with Ray Tune, and logging to MLflow.
*   `src/feature_engineering.py`: Contains functions for data cleaning, feature engineering, and preprocessing.
*   `scripts/requirements-training.txt`: Lists Python dependencies for the `jupyterlab` (training) environment.
*   `mlops-services/init-mlflow-db.sh`: Script to create the `mlflowdb` in Postgres on initial startup.
*   `project_docs/`: For project-related documentation (like this file, `project_plan.md`, etc.).
*   `data/`: Intended for local data storage, though primary data is usually sourced from S3 for pipeline runs.
*   `config_variables.md`: Stores common configuration variables like S3 bucket names, AWS region.
*   S3 Bucket (e.g., `s3://health-predict-mlops-XXXXXXXX`): Used for raw data, processed data, and MLflow artifacts.

## 5. Workflow: Training Pipeline Execution

1.  **Trigger**: The `health_predict_training_hpo` DAG is typically triggered manually from the Airflow UI.
2.  **Airflow Task**: The DAG contains a `BashOperator` task that executes the model training script.
3.  **Execution Command**: The `BashOperator` runs a command similar to:
    ```bash
    cd /home/ubuntu/health-predict/mlops-services && \
    docker-compose exec jupyterlab python3 /home/jovyan/work/scripts/train_model.py \
        --s3-bucket <your-s3-bucket> \
        --s3-processed-data-key processed_data/ \
        --mlflow-tracking-uri http://mlflow:5000 \
        --mlflow-experiment-name HealthPredict_Training_HPO_Airflow \
        --ray-tune-local-dir /home/jovyan/work/ray_results_airflow
    ```
    *   This command is executed from within the `airflow-scheduler` (or `airflow-worker`) container, using the mounted Docker socket to call `docker-compose exec` on the host.
    *   It targets the `jupyterlab` service to run the Python script.
4.  **`train_model.py` Execution (inside `jupyterlab` container)**:
    *   Downloads data from S3 using paths constructed from arguments.
    *   Performs feature engineering using functions from `src/feature_engineering.py`.
    *   Conducts Hyperparameter Optimization (HPO) using Ray Tune for specified models (e.g., Logistic Regression, Random Forest, XGBoost).
    *   Logs all relevant information (parameters, metrics, tags, artifacts like the trained model and preprocessor) to the MLflow server specified by `--mlflow-tracking-uri`.

## 6. Operating the System

*   **Prerequisites**:
    *   EC2 instance with Docker and Docker Compose installed.
    *   AWS credentials configured (for S3 access).
    *   Project repository cloned to `~/health-predict`.
    *   Host Docker group GID identified (e.g., `getent group docker | cut -d: -f3`) and updated in `docker-compose.yml` for Airflow services.
*   **Starting/Stopping Services**:
    *   All `docker-compose` commands should be run from the `~/health-predict/mlops-services/` directory.
    *   **Full Restart & Rebuild**: `docker-compose down --volumes && docker-compose up -d --build` (Use `--volumes` to clear persistent data, e.g., for a fresh DB setup).
    *   **Standard Start**: `docker-compose up -d`.
    *   **Stop Services**: `docker-compose down`.
    *   **Restart Specific Service(s)**: `docker-compose restart <service_name_1> <service_name_2>`.
    *   **View Service Status**: `docker-compose ps`.
    *   **View Logs**: `docker-compose logs <service_name>` (e.g., `docker-compose logs airflow-scheduler`). Add `-f` to follow logs.
*   **Accessing UIs**:
    *   Airflow: `http://<EC2_PUBLIC_IP>:8080` (Login: admin/admin or as created)
    *   MLflow: `http://<EC2_PUBLIC_IP>:5000`
    *   JupyterLab: `http://<EC2_PUBLIC_IP>:8888` (Token might be needed if default Jupyter config is used, but current setup aims for no token)
*   **Code & Configuration Changes**:
    *   **Python Scripts (`scripts/`, `src/`) or DAGs (`mlops-services/dags/`)**: Changes are reflected immediately in the `jupyterlab` and Airflow containers due to volume mounts. Airflow might need a DAG refresh in the UI.
    *   **`scripts/requirements-training.txt`**: After changes, rebuild the `jupyterlab` service: `docker-compose up -d --build jupyterlab`.
    *   **`docker-compose.yml` (e.g., environment variables, service commands, new ports)**: Rebuild affected services: `docker-compose up -d --build <service_name_1> ...`. For substantial changes or issues, a full `docker-compose down --volumes && docker-compose up -d --build` is recommended.
    *   **`mlops-services/init-mlflow-db.sh`**: Changes apply only when the `postgres` volume is cleared and the service is recreated (e.g., via `docker-compose down --volumes`).

## 7. Troubleshooting Common Issues

*   **`KeyError: 'ContainerConfig'` during `docker-compose up`**:
    *   This often indicates stale or corrupted Docker internal state.
    *   **Fix**: Run `docker-compose down --volumes` before `docker-compose up -d --build`.
*   **Permission Denied for Docker Socket (`/var/run/docker.sock`) in Airflow**:
    *   Symptom: Airflow tasks fail when trying to run `docker-compose exec ...`.
    *   **Fix**: Ensure the `group_add: ["<host_docker_gid>"]` line in `docker-compose.yml` for `airflow-scheduler` and `airflow-webserver` correctly reflects the GID of the `docker` group on the EC2 host. Rebuild the services after updating.
*   **MLflow UI Inaccessible / "Database `mlflowdb` does not exist" errors in MLflow logs**:
    *   **Fix**:
        1.  Verify `mlops-services/init-mlflow-db.sh` is present and correct.
        2.  Ensure the `postgres` service configuration correctly mounts this script to `/docker-entrypoint-initdb.d/`.
        3.  Run `docker-compose down --volumes` (to clear old Postgres data) and then `docker-compose up -d --build postgres mlflow`.
*   **Airflow UI Inaccessible / "Database not initialized" errors in Airflow Webserver logs**:
    *   **Fix**:
        1.  Check logs of the `airflow-init` service: `docker-compose logs airflow-init`. It should show successful DB initialization and user creation, then exit.
        2.  If `airflow-init` failed or didn't run properly, ensure its `command` in `docker-compose.yml` is correct and it depends on `postgres`.
        3.  May require `docker-compose down --volumes` followed by `docker-compose up -d --build airflow-init airflow-scheduler airflow-webserver`.
*   **Python Package Issues (e.g., `TypeError: kwargs_from_env() got an unexpected keyword argument 'ssl_version'` in Airflow)**:
    *   Symptom: Caused by incompatible versions of `docker-compose` (Python package) and its `docker` library dependency within Airflow containers.
    *   **Fix**: Airflow services (`airflow-scheduler`, `airflow-webserver`, and sometimes `airflow-init` if it uses `docker-compose`) have their startup commands in `docker-compose.yml` modified to `pip install docker==5.0.3 docker-compose==1.29.2 jsonschema>=4.18.0` (or similar pinned versions). Ensure these are correctly applied and services rebuilt.
*   **Long startup times for `jupyterlab`**:
    *   This is expected if `scripts/requirements-training.txt` is large, as dependencies are installed on each container start if the image hasn't been rebuilt with them baked in. For stability, consider building a custom Docker image for `jupyterlab` with dependencies pre-installed if startup time is an issue. 