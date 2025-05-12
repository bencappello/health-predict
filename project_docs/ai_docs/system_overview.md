# Health Predict MLOps System Overview

## 1. Project Goal & MLOps Philosophy

The primary goal of the "Health Predict" project is to develop a machine learning system capable of predicting patient readmission risk. This project emphasizes robust MLOps practices, including automated pipelines, experiment tracking, model versioning, and preparations for future deployment and monitoring. The aim is to build a scalable and maintainable ML system, not just a one-off model.

## 2. Core Technologies & Services

The system leverages a combination of cloud services, containerization, and specialized MLOps tools:

*   **AWS (Amazon Web Services):**
    *   **EC2:** Hosts the Dockerized services (Airflow, MLflow, PostgreSQL, JupyterLab).
    *   **S3:** Used for persistent storage of raw and processed data, MLflow artifacts (models, preprocessors, metrics), and potentially reports. The primary bucket is `health-predict-mlops-f9ac6509`.
*   **Docker & Docker Compose:**
    *   Containerizes each service (PostgreSQL, Airflow components, MLflow, JupyterLab) for consistent environments and simplified dependency management.
    *   `docker-compose.yml` in the `mlops-services/` directory defines and manages these services.
*   **Apache Airflow:**
    *   Orchestrates workflows, primarily the model training and HPO pipeline.
    *   Uses a PostgreSQL backend for metadata.
    *   DAGs are located in `mlops-services/dags/`.
*   **MLflow:**
    *   Manages the ML lifecycle:
        *   **Tracking:** Logs experiments, parameters, metrics, code versions, and artifacts.
        *   **Models:** Stores and versions trained models and associated preprocessors.
        *   **Registry (Planned):** Will be used to manage model stages (e.g., Staging, Production).
    *   Uses a PostgreSQL backend for metadata and S3 for artifact storage.
    *   Accessible via UI at `http://<EC2_IP>:5000`.
*   **PostgreSQL:**
    *   Serves as the backend metadata store for both Airflow and MLflow, running as a Docker container.
*   **Python:**
    *   The primary language for data processing, feature engineering, model training, and utility scripts.
    *   Key libraries:
        *   `pandas`: Data manipulation.
        *   `scikit-learn`: Machine learning (models, metrics, preprocessing).
        *   `xgboost`: Gradient boosting model.
        *   `ray[tune]`: Distributed hyperparameter optimization (HPO).
        *   `boto3`: AWS SDK for Python (interacting with S3).
        *   `mlflow`: Python client for MLflow.
        *   `fastapi` (Planned): For building the model serving API.
*   **JupyterLab:**
    *   Provided as a Docker service for interactive EDA and experimentation.
    *   The environment includes necessary libraries for data science and AWS connectivity.

## 3. Directory Structure Highlights

*   `iac/`: Infrastructure as Code (Terraform) for setting up AWS resources (currently less emphasized as focus shifted to Dockerized EC2 setup).
*   `mlops-services/`: Contains Docker Compose configuration, Airflow DAGs, logs, and plugins.
    *   `dags/`: Airflow DAG Python files.
    *   `docker-compose.yml`: Defines all core MLOps services.
*   `project_docs/`: Project documentation, including this overview, guides, and plans.
*   `scripts/`: Utility scripts for tasks like data splitting (`split_data.py`) and model training (`train_model.py`).
*   `src/`: Source code for core ML logic.
    *   `feature_engineering.py`: Defines data cleaning, feature engineering, and preprocessing steps.
    *   `api/` (Planned): Will contain the FastAPI application for model serving.
*   `data/`: (Primarily for local interaction if needed) S3 is the source of truth for datasets.
*   `notebooks/`: (Primarily for local interaction if needed) JupyterLab service provides a workspace.

## 4. Data Pipeline & Management

1.  **Raw Data:** The project uses the `diabetic_data.csv` dataset.
2.  **S3 Storage:**
    *   Raw data is uploaded to `s3://health-predict-mlops-f9ac6509/raw_data/`.
    *   Processed data (splits) is stored in `s3://health-predict-mlops-f9ac6509/processed_data/`.
3.  **Data Splitting (`scripts/split_data.py`):**
    *   **Objective:** To prepare data for initial model development and simulate future data for drift monitoring.
    *   The script downloads the raw data from S3.
    *   It partitions the data:
        *   **Initial Data (20%):** Used for the first round of model training. This is further split into `initial_train.csv`, `initial_validation.csv`, and `initial_test.csv`.
        *   **Future Data (80%):** Reserved to simulate incoming data batches for later model monitoring and retraining exercises. Saved as `future_data.csv`.
    *   All resulting CSV files are uploaded to the `processed_data/` prefix in S3.
4.  **Target Variable:** `readmitted_binary` (0 if readmitted == 'NO', 1 if readmitted == '<30' or '>30').

## 5. Feature Engineering (`src/feature_engineering.py`)

This module encapsulates the logic for transforming raw data into a model-ready format.

1.  **`clean_data()`:**
    *   Replaces '?' with NaN.
    *   Drops columns with high missing percentages (`weight`, `payer_code`, `medical_specialty`).
    *   Fills remaining NaNs in specified categorical columns (`race`, `diag_1`, `diag_2`, `diag_3`) with 'Missing'.
    *   Filters out rows where `discharge_disposition_id` indicates hospice or expired, as these are not relevant to readmission prediction.
2.  **`engineer_features()`:**
    *   Creates the binary target `readmitted_binary` from the original `readmitted` column.
    *   Converts `age` to an ordinal feature (`age_ordinal`).
    *   Drops original columns that have been transformed or are not needed for modeling (e.g., `encounter_id`, `patient_nbr`).
3.  **`get_preprocessor()`:**
    *   Defines a `ColumnTransformer` from `scikit-learn`.
    *   Applies `StandardScaler` to numerical features.
    *   Applies `OneHotEncoder` (with `handle_unknown='ignore'`) to categorical features.
4.  **`save_preprocessor()` & `load_preprocessor()`:** Utilities for saving/loading the fitted preprocessor using `joblib`.
5.  **`preprocess_data()`:** Applies a fitted preprocessor to new data.

The fitted preprocessor is saved as `preprocessor.joblib` and logged as an artifact to MLflow alongside the models.

## 6. Model Training, HPO & Tracking (`scripts/train_model.py`)

This script handles the end-to-end training process for multiple model types, including HPO and comprehensive MLflow logging.

1.  **Models:**
    *   Logistic Regression
    *   Random Forest
    *   XGBoost
2.  **Data Handling:**
    *   Loads initial train and validation sets from S3 (using paths defined in the Airflow DAG).
    *   Applies cleaning and feature engineering steps from `src/feature_engineering.py`.
    *   Fits the preprocessor on the training data and saves/logs it via MLflow ("Preprocessing_Run").
    *   Transforms training and validation sets using the fitted preprocessor.
3.  **Hyperparameter Optimization (HPO) with Ray Tune:**
    *   For each model type, a predefined search space is configured.
    *   Ray Tune (`tune.Tuner`) is used to perform HPO, driven by `train_model_hpo` trainable function.
    *   `ASHAScheduler` is used for early stopping of unpromising trials.
    *   Each HPO trial is logged as a nested MLflow run, including its parameters and validation metrics.
    *   The number of HPO samples, max epochs, and grace period are configurable (e.g., via Airflow DAG params).
4.  **Final Model Training & Logging:**
    *   After HPO completes for a model type, the best hyperparameters are retrieved.
    *   A final model of that type is trained on the full (processed) training dataset using these best hyperparameters.
    *   This "Best <ModelType> Model" is logged to MLflow as a separate run, including:
        *   Best hyperparameters.
        *   Validation metrics.
        *   The model artifact itself (e.g., `sklearn.log_model`).
        *   A copy of the `preprocessor.joblib` used for this model.
5.  **MLflow Experiment:**
    *   When run via the Airflow DAG, experiments are logged under `HealthPredict_Training_HPO_Airflow`.
    *   Tracking URI: `http://mlflow:5000` (accessed by components within the Docker network).

## 7. Orchestration with Airflow

*   **DAG File:** `mlops-services/dags/training_pipeline_dag.py`
*   **DAG ID:** `health_predict_training_hpo`
*   **Purpose:** To automate the execution of the model training and HPO pipeline.
*   **Schedule:** Manual trigger (`schedule_interval=None`).
*   **Key Task:**
    *   `run_training_and_hpo`: A `BashOperator` that executes `scripts/train_model.py`. It passes necessary parameters (S3 paths, MLflow URI, Ray Tune settings, etc.) to the script using Airflow's templating mechanism with `params` derived from `env_vars` defined in the DAG.
*   **Planned Task (Currently Commented Out):**
    *   `find_and_register_best_model`: A `PythonOperator` intended to query MLflow for the overall best performing model across all types from a given experiment run and then register it to the MLflow Model Registry, potentially promoting it to a stage like "Production".

## 8. MLOps Services Infrastructure (`mlops-services/docker-compose.yml`)

This file is central to running the MLOps environment.

*   **Services Defined:**
    *   `postgres`: PostgreSQL database for Airflow and MLflow metadata. Data is persisted in a Docker volume.
    *   `airflow-scheduler`, `airflow-webserver`, `airflow-init`: Standard Airflow components using the official `apache/airflow` image. Configured to use the `postgres` service for their backend. Host directories (`./dags`, `./logs`, `./plugins`) are mounted into the containers.
    *   `mlflow`: MLflow tracking server. Configured to use the `postgres` service for backend storage and S3 (bucket `health-predict-mlops-f9ac6509`) for artifact storage.
    *   `jupyterlab`: JupyterLab service using `jupyter/scipy-notebook`. Installs additional Python dependencies from `scripts/requirements-training.txt` on startup (including `boto3`, `sklearn`, `mlflow`, etc.). The project root directory is mounted into the container at `/home/jovyan/work` for easy access to all project files.
*   **Networking:** Services are on a shared Docker network, allowing them to communicate (e.g., Airflow/MLflow talking to Postgres, training script talking to MLflow server).
*   **Environment Configuration:** Critical environment variables (like database connections, MLflow artifact root) are set within the `docker-compose.yml`.

## 9. Current Status & Next Steps (Summary from Project Plan)

*   **Completed:**
    *   Infrastructure setup on EC2 with Docker Compose for core services.
    *   Data splitting and S3 storage.
    *   Feature engineering pipeline.
    *   Comprehensive training script with HPO (Ray Tune) for multiple models and MLflow integration.
    *   Airflow DAG for orchestrating the training script.
    *   Successful execution and logging of HPO trials and best models to MLflow.
*   **Immediate Next Steps (High-Level):**
    *   Implement and test the `find_and_register_best_model` task in the Airflow DAG to register models to the MLflow Model Registry.
    *   Develop a model serving API using FastAPI.
*   **Future Phases (as per `project_steps.md`):**
    *   Containerize the API and push to ECR.
    *   Deploy API to a local Kubernetes cluster (Minikube/Kind) on EC2.
    *   Develop CI/CD pipelines using Airflow for model deployment.
    *   Implement data/concept drift monitoring and a retraining loop.

## 10. Key Configuration Points

*   **S3 Bucket:** `health-predict-mlops-f9ac6509`
*   **MLflow Tracking URI (internal):** `http://mlflow:5000`
*   **MLflow UI (external):** `http://<EC2_PUBLIC_IP>:5000`
*   **Airflow UI (external):** `http://<EC2_PUBLIC_IP>:8080`
*   **JupyterLab UI (external):** `http://<EC2_PUBLIC_IP>:8888`
*   **Python environment for training:** Managed within the `jupyterlab` Docker service, which installs packages from `scripts/requirements-training.txt`.
*   **Airflow DAG parameters:** Configurable within `mlops-services/dags/training_pipeline_dag.py`, particularly under `env_vars`.

This document should provide a solid foundation for understanding the current state and intended architecture of the Health Predict MLOps system. 