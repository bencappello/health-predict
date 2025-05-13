## Summary of Phase 1: Foundation, Cloud Setup & Exploration

- **Initial Project & AWS Setup:**
    - GitHub repository `bencappello/health-predict` created and cloned.
    - Local Python 3.11.3 environment (`.venv`) and initial project structure (`/src`, `/notebooks`, etc.) established.
    - `.gitignore` created with standard exclusions.
    - AWS credentials configured locally.
- **Infrastructure as Code (Terraform):**
    - Initial Terraform scripts (`main.tf`, `variables.tf`, etc.) created in `iac/` for VPC, EC2, S3, ECR, IAM roles, and security groups.
    - `project_docs/terraform_guide.md` created with instructions for user deployment.
- **MLOps Tools on EC2 (Docker Compose):**
    - `docker-compose.yml` created in `~/health-predict/mlops-services/` to manage Postgres, Airflow (webserver, scheduler, init), and MLflow.
    - Configured MLflow with Postgres backend and S3 artifact storage.
    - Troubleshooting for MLflow (`psycopg2` missing) and Airflow webserver (Gunicorn timeouts, worker count) was performed.
    - UI access for Airflow and MLflow confirmed.
- **Kubernetes Setup on EC2:**
    - `kubectl` installed.
    - Minikube installed and local cluster started (after resolving EC2 disk space issues).
- **Data Management:**
    - Dataset (`diabetic_data.csv`) uploaded to S3 (`raw_data/`).
    - `scripts/split_data.py` created and run to split data into initial train/validation/test (20%) and future data (80%) for drift simulation, all uploaded to S3 (`processed_data/`).
    - JupyterLab service added to `docker-compose.yml` for EDA, mounting the project root.
- **Initial EDA & Baseline Model:**
    - `notebooks/01_eda_baseline.py` developed for EDA, cleaning, feature engineering (binary target `readmitted_binary`, ordinal `age_ordinal`), and baseline Logistic Regression model training using the initial 20% dataset.
    - Preprocessing included StandardScaler and OneHotEncoder.

## Summary of Phase 2: Scalable Training & Tracking on AWS

- **Feature Engineering & Training Scripts:**
    - `src/feature_engineering.py` created with data cleaning, feature engineering, and preprocessing pipeline functions.
    - `scripts/train_model.py` developed for HPO with Ray Tune and MLflow integration, supporting Logistic Regression, Random Forest, and XGBoost.
- **Training Environment & Execution:**
    - Training environment dockerized by updating `jupyterlab` service in `docker-compose.yml` to use `scripts/requirements-training.txt`.
    - `project_docs/run_training_guide.md` created for user execution.
    - Training script executed, involving iterative debugging (Ray Tune errors, `ObjectStoreFullError`, `ValueError` in `get_preprocessor`).
- **Model Iteration & Performance:**
    - Initial perfect scores indicated data leakage; fixed by ensuring target `readmitted` was excluded from features.
    - Training rerun with realistic scores (LR F1 ~0.27, RF F1 ~0.28, XGB F1 ~0.07).
    - Target variable `readmitted_binary` redefined (any readmission vs. no readmission), leading to significantly improved F1 scores (RF ~0.63, XGB ~0.63, LR ~0.61).
- **Airflow DAG for Training & HPO:**
    - `mlops-services/dags/training_pipeline_dag.py` created with BashOperator for `train_model.py` execution.
    - `system_overview.md` created for project documentation.
    - MLflow model registration task (PythonOperator calling `find_and_register_best_model`) implemented in the DAG, correctly identifying best HPO models and transitioning them to "Production".
    - DAG successfully triggered and verified via CLI and `scripts/verify_mlflow_registration.py`, confirming models registered in MLflow Model Registry.

## Summary of Phase 3: API Development & Deployment to Local K8s

- **API Development:**
    - Created a robust FastAPI application in `src/api/main.py` with `/predict` and `/health` endpoints.
    - Implemented model and preprocessor loading from MLflow at application startup, correctly handling MLflow model URIs and stage filtering.
    - Defined Pydantic models for request/response validation with alias handling for hyphenated field names.
    - Added comprehensive error handling and logging throughout the API.
    - Addressed cross-module imports by creating a fallback mechanism for loading feature engineering functions.
    - Created `src/api/requirements.txt` with appropriate dependencies and versions.
- **Containerization with Docker:**
    - Created a `Dockerfile` at the project root to package the FastAPI application.
    - Used Python 3.10 slim base image for efficiency.
    - Implemented proper environment variable handling and port exposure.
    - Created `.dockerignore` to exclude unnecessary files from the build context.
    - Successfully built the image and authenticated with AWS ECR.
    - Pushed the image to the ECR repository (`536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`).
- **Kubernetes Deployment on EC2 (Minikube):**
    - Created `k8s/deployment.yaml` with `Deployment` and `Service` (NodePort) resources.
    - Configured the deployment with the ECR image, environment variables, and appropriate resource requests.
    - **Troubleshooting ECR Authentication in Minikube:**
        - Initially faced `ErrImagePull` errors due to Minikube's Kubelet not authenticating to ECR.
        - Attempted `minikube image load <image_uri>`, but encountered out-of-memory error (exit code 137).
        - Resolved by logging into ECR directly within Minikube's Docker environment using `aws ecr get-login-password | (eval $(minikube -p minikube docker-env) && docker login --username AWS --password-stdin <ecr_uri> && eval $(minikube docker-env -u))`.
- **API Testing and Debugging:**
    - **Network Connectivity:** Fixed MLflow connectivity by using EC2 private IP instead of `host.minikube.internal:5000`.
    - **MLflow Stage vs. Alias:** Corrected the API to use `get_latest_versions` with stage filtering instead of `get_model_version_by_alias`.
    - **Column Name Mismatch:** Resolved critical issue where preprocessor expected hyphenated column names (e.g., `glyburide-metformin`) while API used underscores (e.g., `glyburide_metformin`).
        - Added logic to dynamically identify and create columns with both formats before applying the preprocessor.
    - **Preprocessor Co-Location:** Ensured the training script (`train_model.py`) correctly logs the preprocessor artifact alongside each "Best\_<ModelName>\_Model" run.
    - **Model Promotion Criteria:** Modified the Airflow DAG to only promote models to "Production" stage if they have the co-located preprocessor artifact.
    - Created automated tests in `tests/api/test_api_endpoints.py` for both endpoints.
    - Performed thorough manual testing with multiple test payloads (`test_payload1.json`, `test_payload2.json`, etc.).
    - Verified proper error handling for malformed payloads (`malformed_payload1.json`, `malformed_payload2.json`, `invalid_json.json`).
- **User Documentation:**
    - Updated `system_overview.md` to accurately reflect the API deployment architecture.
    - Enhanced `project_steps.md` with detailed API development and deployment instructions.
    - Reviewed and enhanced `project_docs/ai_docs/project_steps.md` to provide comprehensive details for future phases.
    - Added detailed sub-tasks and explanations for the remaining phases in the project plan.
    - Introduced a new Phase 6: Documentation, Finalization & AWS Showcase, with comprehensive steps for final deliverables.
    - Updated system overview to include FastAPI, containerization (Dockerfile, ECR), and deployment details.

## Summary of Phase 4: CI/CD Automation using AWS Resources

- **API Deployment Automation:**
    - Created `mlops-services/dags/deployment_pipeline_dag.py` with sequential tasks to automate the deployment process:
        - `get_production_model_info`: Python function that fetches the latest "Production" model version from MLflow.
        - `define_image_details`: Python function that generates a unique Docker image tag based on model version and timestamp.
        - `authenticate_docker_to_ecr`: BashOperator for ECR authentication.
        - `build_api_docker_image`: BashOperator to build the Docker image from the project root.
        - `push_image_to_ecr`: BashOperator to push the tagged image to ECR.
        - `update_kubernetes_deployment`: BashOperator to update the K8s deployment and ensure correct MLflow tracking URI.
        - `verify_deployment_rollout`: BashOperator to verify successful deployment and print service URL.
    - Defined appropriate environment variables and task dependencies for the DAG.
    - Updated `project_steps.md` to mark Phase 4, Step 1 and all its sub-tasks as complete.
    - Next step is to verify the CI/CD DAG by triggering it manually from the Airflow UI.
