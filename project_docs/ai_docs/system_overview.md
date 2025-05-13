# Health Predict MLOps System Overview

## 1. Project Goal & MLOps Philosophy

The primary goal of the "Health Predict" project is to develop a machine learning system capable of predicting patient readmission risk. This project emphasizes robust MLOps practices, including automated pipelines, experiment tracking, model versioning, deployment, monitoring, and automated retraining. The aim is to build a scalable and maintainable ML system, not just a one-off model, with a focus on cost-effective solutions.

## 2. Core Technologies & Services

The system leverages a combination of cloud services, containerization, and specialized MLOps tools:

*   **AWS (Amazon Web Services):**
    *   **EC2:** Hosts the Dockerized MLOps services (Airflow, MLflow, PostgreSQL, JupyterLab) and the local Kubernetes cluster (Minikube).
    *   **S3:** Used for persistent storage of raw and processed data, MLflow artifacts (models, preprocessors, metrics, reports). The primary bucket is `health-predict-mlops-f9ac6509`.
    *   **ECR (Elastic Container Registry):** Stores the Docker image for the model serving API.
*   **Docker & Docker Compose:**
    *   Containerizes each core MLOps service (PostgreSQL, Airflow components, MLflow, JupyterLab) for consistent environments and simplified dependency management.
    *   `docker-compose.yml` in the `mlops-services/` directory defines and manages these services.
*   **Kubernetes (Minikube):**
    *   Runs locally on the EC2 instance (using Minikube with the Docker driver) to orchestrate the deployed model serving API. This approach is chosen for demonstrating Kubernetes deployment principles in a cost-effective manner.
*   **Apache Airflow:**
    *   Orchestrates MLOps workflows: model training/HPO, model registration, (planned) API deployment, and (planned) drift monitoring/retraining.
    *   Uses a PostgreSQL backend for metadata.
    *   DAGs are located in `mlops-services/dags/`.
*   **MLflow:**
    *   Manages the ML lifecycle:
        *   **Tracking:** Logs experiments, parameters, metrics, code versions, and artifacts.
        *   **Models:** Stores and versions trained models and associated preprocessors.
        *   **Registry:** Manages model stages (e.g., "Staging", "Production"), with automated promotion based on criteria like performance and artifact presence (e.g., preprocessor).
    *   Uses a PostgreSQL backend for metadata and S3 for artifact storage.
    *   Accessible via UI at `http://<EC2_IP>:5000`.
*   **PostgreSQL:**
    *   Serves as the backend metadata store for both Airflow and MLflow, running as a Docker container.
*   **FastAPI:**
    *   Python framework used for building the robust and efficient model serving API (`/src/api/main.py`).
*   **Evidently AI:**
    *   (Planned) Python library for implementing data and concept drift detection in the monitoring pipeline.
*   **Python:**
    *   The primary language for data processing, feature engineering, model training, API development, and utility scripts.
    *   Key libraries: `pandas`, `scikit-learn`, `xgboost`, `ray[tune]`, `boto3`, `mlflow`, `fastapi`, `uvicorn`.

## 3. Directory Structure Highlights

*   `iac/`: Infrastructure as Code (Terraform) for initial setup of core AWS resources (VPC, EC2, S3, ECR).
*   `mlops-services/`: Contains Docker Compose configuration, Airflow DAGs, logs, and plugins.
    *   `dags/`: Airflow DAG Python files for training, deployment, and monitoring.
    *   `docker-compose.yml`: Defines all core MLOps services.
*   `project_docs/`: Project documentation, including this overview, guides, and plans.
*   `scripts/`: Utility scripts for tasks like data splitting (`split_data.py`), model training (`train_model.py`), and (planned) drift monitoring (`monitor_drift.py`).
*   `src/`: Source code for core ML logic and the API.
    *   `feature_engineering.py`: Defines data cleaning, feature engineering, and preprocessing steps.
    *   `api/main.py`: FastAPI application for model serving.
    *   `api/requirements.txt`: Python dependencies for the API.
*   `k8s/`: Contains Kubernetes manifest files (e.g., `deployment.yaml`) for deploying the API to the local K8s cluster.
*   `Dockerfile`: Located at the project root, defines the container image for the FastAPI application.
*   `data/`: (Primarily for local interaction if needed) S3 is the source of truth for datasets.
*   `notebooks/`: Contains the initial EDA and baseline model script (`01_eda_baseline.py`).

## 4. Data Pipeline & Management

1.  **Raw Data:** The project uses the `diabetic_data.csv` dataset.
2.  **S3 Storage:**
    *   Raw data is uploaded to `s3://health-predict-mlops-f9ac6509/raw_data/`.
    *   Processed data (splits for initial training and future simulation) is stored in `s3://health-predict-mlops-f9ac6509/processed_data/`.
3.  **Data Splitting (`scripts/split_data.py`):**
    *   The script downloads raw data from S3.
    *   Partitions data into: Initial Data (20% for train/validation/test) and Future Data (80% for simulated batches).
    *   Uploads resulting CSVs to `processed_data/` in S3.
4.  **Target Variable:** `readmitted_binary` (0 if readmitted == 'NO', 1 if readmitted == '<30' or '>30').

## 5. Feature Engineering (`src/feature_engineering.py`)

This module encapsulates the logic for transforming raw data into a model-ready format. It includes:
1.  **`clean_data()`:** Handles missing values and filters irrelevant discharge dispositions.
2.  **`engineer_features()`:** Creates the binary target `readmitted_binary` and ordinal feature `age_ordinal`.
3.  **`get_preprocessor()`:** Defines a `ColumnTransformer` for `StandardScaler` on numerical and `OneHotEncoder` on categorical features.
The fitted preprocessor (`preprocessor.joblib`) is logged as an artifact to MLflow alongside the trained models it's associated with. The API loads this preprocessor dynamically with the model from MLflow.

## 6. Model Training, HPO & Tracking (`scripts/train_model.py`)

This script handles the end-to-end training process for multiple model types (Logistic Regression, Random Forest, XGBoost), incorporating HPO and comprehensive MLflow logging.
1.  **Data Handling:** Loads data from S3, applies cleaning/feature engineering, fits/logs the preprocessor.
2.  **HPO with Ray Tune:** Uses `tune.Tuner` and `ASHAScheduler` for each model type. Each trial is a nested MLflow run.
3.  **Final Model Training & Logging:** Trains a final model using best HPO params, logs it to a new MLflow run ("Best\_<ModelType>\_Model") with metrics, parameters, the model artifact, and a co-located copy of `preprocessor.joblib`.
4.  **MLflow Experiment:** `HealthPredict_Training_HPO_Airflow` (when run via DAG).

## 7. API Serving (`src/api/main.py` & Kubernetes)

1.  **FastAPI Application:**
    *   Provides `/predict` and `/health` endpoints.
    *   On startup, loads the specified model (e.g., "HealthPredict_RandomForest") and its co-located preprocessor from the "Production" stage in the MLflow Model Registry.
    *   The `/predict` endpoint takes patient data, applies cleaning, feature engineering, and the loaded preprocessor, then returns the prediction and probability score.
    *   Handles Pydantic model validation for input/output.
2.  **Containerization & ECR:**
    *   The API is packaged into a Docker image using the root `Dockerfile`.
    *   This image is pushed to AWS ECR.
3.  **Kubernetes Deployment:**
    *   Manifests in `k8s/deployment.yaml` define a `Deployment` and a `Service` (NodePort type).
    *   The API is deployed to the local Minikube cluster running on the EC2 instance, pulling the image from ECR.
    *   The `MLFLOW_TRACKING_URI` for the API pods is configured to point to the MLflow service on the EC2 host network (e.g., using the EC2 private IP or `host.minikube.internal`).

## 8. Orchestration with Airflow

*   **DAG Files:** Located in `mlops-services/dags/`.
*   **`health_predict_training_hpo` (Implemented):**
    *   Orchestrates `scripts/train_model.py` execution.
    *   Includes a `find_and_register_best_model` task that queries MLflow for the best model from the HPO runs, checks for a co-located `preprocessor.joblib` artifact, and if found, registers the model and promotes its version to the "Production" stage in the MLflow Model Registry.
*   **`deployment_pipeline_dag.py` (Planned):**
    *   Automates the API deployment process:
        *   Fetches the latest "Production" model information from MLflow.
        *   Builds the API Docker image (using the root `Dockerfile`).
        *   Authenticates with and pushes the image to ECR.
        *   Updates the Kubernetes deployment on Minikube to use the new image (`kubectl set image`).
        *   Verifies the deployment rollout.
*   **`monitoring_retraining_dag.py` (Planned):**
    *   Simulates arrival of new data batches from `future_data.csv`.
    *   Executes `scripts/monitor_drift.py` (using Evidently AI) to detect data and concept drift against a reference dataset.
    *   Logs drift reports and metrics to MLflow.
    *   If significant drift is detected, triggers the `health_predict_training_hpo` DAG for model retraining using the new data combined with reference data.
    *   (Optionally) Triggers the `deployment_pipeline_dag.py` if retraining leads to a new "Production" model.

## 9. MLOps Services Infrastructure (`mlops-services/docker-compose.yml`)

Defines and manages the core MLOps services:
*   `postgres`: Backend for Airflow and MLflow.
*   `airflow-scheduler`, `airflow-webserver`, `airflow-init`: Airflow components.
*   `mlflow`: MLflow tracking server, using Postgres backend and S3 for artifacts.
*   `jupyterlab`: JupyterLab service with necessary DS/ML libraries and project root mounted.
Services communicate over a shared Docker network.

## 10. Current Status & Next Steps (Summary)

*   **Completed & Implemented:**
    *   Core AWS infrastructure setup (EC2, S3, ECR via Terraform).
    *   Dockerized MLOps services (Airflow, MLflow, Postgres, JupyterLab) on EC2.
    *   Data splitting and S3 storage.
    *   Feature engineering pipeline (`src/feature_engineering.py`).
    *   Comprehensive training script (`scripts/train_model.py`) with HPO (Ray Tune) and MLflow integration, including co-location of preprocessor.
    *   Airflow DAG (`health_predict_training_hpo`) orchestrating training and model promotion to "Production" in MLflow Model Registry.
    *   FastAPI model serving API (`src/api/main.py`) that loads model/preprocessor from MLflow.
    *   Containerization of the API (`Dockerfile`) and push to ECR.
    *   Deployment of the API to a local Kubernetes (Minikube) cluster on EC2, with successful testing.
*   **Immediate Next Steps (High-Level from `project_steps.md`):**
    *   Develop and test the Airflow CI/CD DAG (`deployment_pipeline_dag.py`) for automated API deployment.
    *   Implement the drift monitoring script (`scripts/monitor_drift.py`) using Evidently AI.
    *   Develop and test the Airflow DAG (`monitoring_retraining_dag.py`) for the drift detection and automated retraining loop.
    *   Complete comprehensive project documentation, finalization, and showcase materials (Phase 6).

## 11. Key Configuration Points & Access URLs

*   **S3 Bucket:** `health-predict-mlops-f9ac6509`
*   **MLflow Tracking URI (internal, for services in Docker network):** `http://mlflow:5000`
*   **MLflow UI (external):** `http://<EC2_PUBLIC_IP>:5000`
*   **Airflow UI (external):** `http://<EC2_PUBLIC_IP>:8080`
*   **JupyterLab UI (external):** `http://<EC2_PUBLIC_IP>:8888`
*   **API Endpoint (Local K8s on EC2):** `http://<EC2_PUBLIC_IP>:<NodePort>` (NodePort obtained via `minikube service health-predict-api-service --url` or `kubectl get svc health-predict-api-service`).
*   **ECR Repository URI:** e.g., `536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api`

This document aims to provide a comprehensive overview of the Health Predict MLOps system architecture, components, and workflows. 