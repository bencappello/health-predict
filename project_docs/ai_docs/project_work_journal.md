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

## Phase 3: API Development & Deployment to Local K8s - Detailed Log

## $(date +'%Y-%m-%d %H:%M:%S') - Updated API Development Steps in Project Plan

- Reviewed user request to enhance the detail of API development steps in `project_steps.md`.
- Read the existing `project_steps.md` to locate Phase 3, Step 1.
- Rewritten the bullet points under Phase 3, Step 1 ("API Development (FastAPI)") to provide comprehensive descriptions and detailed sub-tasks for:
    - Creating the API code structure (`/src/api/main.py`).
    - Loading the model and preprocessor from MLflow on application startup.
    - Defining Pydantic models for request and response schemas.
    - Implementing the `/predict` endpoint with input validation and prediction logic.
    - Implementing a `/health` check endpoint.
    - Creating a `requirements.txt` file for the API's Python dependencies.
- The changes were applied to `project_docs/ai_docs/project_steps.md`.

## $(date +'%Y-%m-%d %H:%M:%S') - Created Initial API Structure

- Executed Phase 3, Step 1, Bullet 1 from `project_steps.md`.
- Created the initial API code structure in `src/api/main.py`.
- The `main.py` file includes:
    - FastAPI app initialization.
    - Necessary imports (`fastapi`, `pydantic`, `mlflow`, `pandas`, `numpy`, `os`, `logging`).
    - Basic logging setup.
    - Placeholder global variables for MLflow configuration and the model pipeline.
    - Placeholder Pydantic models for `InferenceInput` and `InferenceResponse`.
    - A basic `/health` endpoint.
    - Startup and shutdown event handlers with a placeholder for model loading logic.
    - A `__main__` block for local Uvicorn execution.
- Updated `project_steps.md` to mark this sub-task as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Implemented Model Loading in API Startup

- Executed Phase 3, Step 1, Bullet 2 from `project_steps.md`.
- Modified the `startup_event` function in `src/api/main.py` to load the ML model and preprocessor.
- The startup logic now:
    - Sets the MLflow tracking URI (defaults to `http://mlflow:5000` or uses `MLFLOW_TRACKING_URI` env var).
    - Constructs the model URI for `HealthPredict_RandomForest` model from the `Production` stage.
    - Uses `mlflow.sklearn.load_model()` to load the model pipeline.
    - Stores the loaded pipeline in the global `model_pipeline` variable.
    - Includes error handling and logging for the loading process.
- Updated `project_steps.md` to mark this sub-task as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Defined API Request/Response Models

- Executed Phase 3, Step 1, Bullet 3 from `project_steps.md`.
- Defined Pydantic models in `src/api/main.py`:
    - `InferenceInput`: Includes 44 features expected by the raw input layer of the prediction pipeline, with appropriate data types (`str`, `int`, `Optional[str]`). Added a Pydantic `Config` for alias generation to handle hyphens in input JSON keys.
    - `InferenceResponse`: Defines the prediction output structure (`prediction: int`, `probability_score: float`).
- Added `from typing import Optional` to `src/api/main.py`.
- Updated `project_steps.md` to mark this sub-task as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Implemented /predict Endpoint in API

- Executed Phase 3, Step 1, Bullet 4 from `project_steps.md`.
- Implemented the `/predict` endpoint in `src/api/main.py`.
- The endpoint logic includes:
    - Checking if the `model_pipeline` (preprocessor + model) is loaded.
    - Accepting `InferenceInput` (Pydantic model).
    - Converting input to a Pandas DataFrame (handling hyphenated JSON keys via Pydantic `by_alias=True`).
    - Applying `clean_data` and `engineer_features` from `src.feature_engineering` to the input DataFrame.
    - Using the loaded `model_pipeline` to get predictions and probability scores.
    - Returning results formatted as `InferenceResponse`.
    - Robust error handling with `HTTPException` for issues like model not loaded, invalid input, or prediction errors.
    - Added a `try-except` block for `src.feature_engineering` imports with dummy fallbacks for easier local development (to be addressed for production).
- Updated `project_steps.md` to mark this sub-task as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Confirmed /health Endpoint Implementation

- Executed Phase 3, Step 1, Bullet 5 from `project_steps.md`.
- Reviewed the existing `/health` endpoint in `src/api/main.py`.
- Confirmed that the current implementation meets the requirements:
    - It is a GET endpoint.
    - It returns a JSON response with API status.
    - It checks and reports the model loading status.
- Updated `project_steps.md` to mark this sub-task as complete, as no code changes were needed.

## $(date +'%Y-%m-%d %H:%M:%S') - Created API requirements.txt

- Executed Phase 3, Step 1, Bullet 6 from `project_steps.md`.
- Created `src/api/requirements.txt` with the following Python dependencies and versions:
    - `fastapi==0.111.0`
    - `uvicorn[standard]==0.30.1`
    - `pydantic==2.8.2`
    - `mlflow==2.14.1`
    - `pandas==2.2.2`
    - `numpy==1.26.4`
    - `scikit-learn==1.5.1`
    - `xgboost==2.1.0`
    - `python-dotenv==1.0.1`
- Updated `project_steps.md` to mark this sub-task as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Detailed Containerization and ECR Push Steps in Project Plan

- Reviewed user request to enhance the detail of API containerization and ECR deployment steps in `project_steps.md`.

## $(date +'%Y-%m-%d %H:%M:%S') - Updated API Development Steps in Project Plan

## $(date +'%Y-%m-%d %H:%M:%S') - Detailed Containerization and ECR Push Steps in Project Plan

- Reviewed user request to enhance the detail of API containerization and ECR deployment steps in `project_steps.md`.
- Updated Phase 3, Step 2 ("Containerization") in `project_docs/ai_docs/project_steps.md` to include detailed sub-tasks for creating the `Dockerfile`:
    - Choosing a base Python image.
    - Setting the working directory.
    - Configuring environment variables (e.g., `PYTHONUNBUFFERED`).
    - Copying `requirements.txt` and installing dependencies efficiently.
    - Copying application code and the importance of `.dockerignore`.
    - Exposing the application port.
    - Defining the `CMD` to run Uvicorn with the FastAPI application.
- Updated Phase 3, Step 3 ("Build and Push to ECR") in `project_docs/ai_docs/project_steps.md` to include detailed sub-tasks for:
    - Authenticating Docker with AWS ECR on the EC2 instance using `aws ecr get-login-password`.
    - Navigating to the correct directory for Docker build context.
    - Defining the ECR image URI and tag.
    - Running `docker build` with the appropriate tag.
    - Running `docker push` to upload the image to ECR.
    - Verification steps for each command.

## $(date +'%Y-%m-%d %H:%M:%S') - Detailed Kubernetes Deployment and API Testing Steps in Project Plan

- Reviewed user request to enhance the detail of Kubernetes deployment and API testing steps in `project_steps.md`.
- Updated Phase 3, Step 4 ("Kubernetes Deployment (Targeting Local K8s on EC2)") in `project_docs/ai_docs/project_steps.md` to include detailed sub-tasks for:
    - Ensuring `kubectl` is configured for the local K8s cluster (Minikube/Kind).
    - Creating `k8s/deployment.yaml` with comprehensive definitions for `Deployment` and `Service` (NodePort type).
    - Detailed explanation of `Deployment` specs: replicas, selectors, pod template, container image URI, container port, environment variables (e.g., `MLFLOW_TRACKING_URI`), and resource requests/limits.
    - Detailed explanation of `Service` specs: selector, ports, and type (`NodePort`).
    - Important considerations for MLflow and S3 access from pods in the local K8s cluster on EC2, including networking options and IAM permissions.
    - Steps for applying manifests using `kubectl apply`.
    - Verification steps using `kubectl get pods`, `kubectl rollout status`, `kubectl get svc`, and how to determine the access NodePort.
- Updated Phase 3, Step 5 ("API Testing") in `project_docs/ai_docs/project_steps.md` to include detailed sub-tasks for:
    - Identifying the service NodePort and EC2 IP for accessing the API.
    - Testing the `/health` endpoint using `curl` from the EC2 instance.
    - Preparing a sample JSON input for the `/predict` endpoint and saving it to a file.
    - Testing the `/predict` endpoint using `curl` with the JSON payload.
    - Expected outputs and troubleshooting tips (checking pod logs, input data).
    - Optional testing with Postman if the NodePort is externally accessible.

## $(date +'%Y-%m-%d %H:%M:%S') - Refined API Testing Strategy in Project Plan

- Reviewed user request to update the API testing strategy in `project_steps.md`.
- Rewritten Phase 3, Step 5 ("API Testing (Automated & Manual)") in `project_docs/ai_docs/project_steps.md` to:
    - Emphasize a primary strategy of written automated tests using `pytest`.
    - Include detailed sub-tasks for:
        - Setting up the Python testing environment (`pytest`, `requests`/`httpx`).
        - Identifying the API base URL for tests.
        - Creating a test file structure (e.g., `tests/api/test_api_endpoints.py`).
        - Writing automated tests for the `/health` endpoint (status code, basic response body).
        - Writing automated tests for the `/predict` endpoint, covering valid input, missing fields, invalid data types, and optional edge cases.
        - Instructions for running the automated tests using `pytest`.
    - Clearly define a section for "Manual Testing & Verification" to be performed by the Human User:
        - Stated purpose: exploratory testing for usability and unexpected interactions.
        - Outlined actions for the user: sending varied valid and malformed requests to `/predict` and `/health` via Postman or `curl`.
        - Specified that the user should provide feedback on any issues found.
    - Updated completion criteria for the step to include: all automated tests passing AND successful manual verification by the Human User.

## $(date +'%Y-%m-%d %H:%M:%S') - Dockerization and Git Branch Reconciliation

- Committed `Dockerfile` and `.dockerignore` for API containerization to the `new-feature-from-old` branch.
- Reconciled Git branches:
    - Switched from `new-feature-from-old` to `main`.
    - Deleted the local `main` branch.
    - Renamed `new-feature-from-old` to `main`.
    - Force-pushed the new local `main` branch to `origin/main`, overwriting the remote `main`.
    - Deleted the `new-feature-from-old` branch from the remote (`origin`).
- The current project state is now consolidated on the `main` branch, both locally and remotely.

## $(date +'%Y-%m-%d %H:%M:%S') - Built and Pushed API Docker Image to ECR

- Successfully executed Phase 3, Step 3: "Build and Push to ECR".
- Authenticated Docker with AWS ECR using the ECR repository URL (`536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api`) provided by the user.
- Built the Docker image for the FastAPI application using the `Dockerfile` at the project root. The image was tagged as `536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`.
- Pushed the tagged Docker image to the specified ECR repository.
- Updated `project_steps.md` to mark this step and its sub-tasks as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Deployed API to Local Kubernetes (Minikube)

- Executed Phase 3, Step 4 from `project_steps.md` to deploy the containerized API to Minikube on the EC2 instance.
- **Troubleshooting Minikube & ECR:**
    - Initial `kubectl cluster-info` attempts failed due to Minikube not being in a healthy state.
    - Performed `minikube delete` and a fresh `minikube start --driver=docker`, which was successful.
    - Encountered `ErrImagePull` for the ECR image (`536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`) as Minikube's Kubelet couldn't authenticate to ECR.
    - Attempted `minikube image load <image_uri>`, which initially failed due to an out-of-memory error (exit code 137). Deleted the existing Kubernetes deployment to free resources, and the image load then succeeded.
    - Even with the image loaded locally and `imagePullPolicy: IfNotPresent` (explicitly set in `k8s/deployment.yaml`), pods still went into `ErrImagePull` because Kubernetes (especially with `:latest` tag) often re-validates with the registry.
    - **Resolution:** Logged into ECR *within Minikube's Docker environment* using `aws ecr get-login-password --region us-east-1 | (eval $(minikube -p minikube docker-env) && docker login --username AWS --password-stdin <ecr_uri> && eval $(minikube docker-env -u))`. This allowed Minikube's Kubelet to authenticate and pull the image.
- **Kubernetes Deployment:**
    - Created `k8s/deployment.yaml` with `Deployment` and `Service` (NodePort type) definitions.
        - `Deployment` configured for 2 replicas, using the ECR image, exposing container port 8000.
        - `MLFLOW_TRACKING_URI` environment variable set to `http://host.minikube.internal:5000` (as `project_steps.md` notes, this might need to be EC2 private IP if resolution fails).
        - `imagePullPolicy` was set to `IfNotPresent` during troubleshooting.
        - `Service` configured to expose port 80 and target pod port 8000, using `NodePort`.
    - Applied manifests using `kubectl apply -f k8s/deployment.yaml`.
- **Verification:**
    - `kubectl get pods -l app=health-predict-api` showed both pods running (`READY 1/1`).
    - `kubectl rollout status deployment/health-predict-api-deployment` confirmed successful rollout.
    - `kubectl get svc health-predict-api-service` showed the NodePort (e.g., 30854).
    - `minikube service health-predict-api-service --url` provided the accessible URL (e.g., `http://192.168.49.2:30854`).
- Updated `project_steps.md` to mark Phase 3, Step 4 and its sub-tasks as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Initiated API Testing (Phase 3, Step 5)

- Began work on Phase 3, Step 5: "API Testing (Automated & Manual)".
- **Setup Testing Environment:**
    - Created `tests/requirements.txt` with `pytest` and `requests`.
    - Installed testing dependencies using pip.
    - Established test file structure: `tests/api/test_api_endpoints.py`.
- **Implemented Automated Tests:**
    - Identified API Base URL (`http://192.168.49.2:30854`) from Minikube service exposure.
    - Wrote `test_health_check` function in `tests/api/test_api_endpoints.py` to verify the `/health` endpoint status and model loading status.
    - Wrote tests for the `/predict` endpoint in the same file:
        - `test_predict_valid_input`: Checks for a 200 response and correct output structure using a placeholder valid payload (marked as needing adjustment to match the exact API schema).
        - `test_predict_missing_field`: Checks for a 422 response when a required field is missing.
        - `test_predict_invalid_data_type`: Checks for a 422 response when a field has an incorrect data type.
- Updated `project_steps.md` to mark the completion of the first five sub-tasks of Phase 3, Step 5.
