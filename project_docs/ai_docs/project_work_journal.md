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
    - DAG successfully triggered and verified, confirming models registered in MLflow Model Registry with their preprocessors.
    - Ensured `MLFLOW_TRACKING_URI` correctly set for `jupyterlab` service in `docker-compose.yml` for reliable MLflow CLI usage.

## Summary of Phase 3: API Development & Deployment to Local K8s

Phase 3 focused on creating a robust FastAPI for model serving, containerizing it, deploying it to a local Kubernetes (Minikube) cluster on EC2, and thoroughly testing its functionality.

- **API Development (`src/api/main.py`):**
    - A FastAPI application was built with `/health` and `/predict` endpoints.
    - **Model Loading:** Implemented logic to load the specified model (e.g., `HealthPredict_RandomForest`) and its co-located `preprocessor.joblib` from the "Production" stage of the MLflow Model Registry upon API startup. This ensures the API always uses the correct, versioned preprocessor tied to the model.
    - **Request/Response Handling:** Defined Pydantic models (`InferenceInput`, `InferenceResponse`) for input validation and structured output. `InferenceInput` was configured with Pydantic aliases to accept hyphenated field names in JSON (e.g., `race`) and convert them to underscore format (e.g., `race`) for internal use, matching the DataFrame column names used during training and feature engineering.
    - **Prediction Logic:** The `/predict` endpoint applies `clean_data` and `engineer_features` (from `src.feature_engineering.py`) to the input, then uses the loaded preprocessor to transform the data before feeding it to the model for prediction. It returns both a binary prediction and a probability score.
    - **Dependencies:** Created `src/api/requirements.txt` specifying exact versions for key libraries like `fastapi`, `mlflow`, and `scikit-learn` to ensure consistency with the training environment and avoid issues like `AttributeError` due to version mismatches.

- **Containerization & ECR:**
    - A `Dockerfile` was created at the project root to package the API. It uses a slim Python base image, copies necessary code (`/src`), installs dependencies from `src/api/requirements.txt`, and sets Uvicorn as the entry point.
    - A `.dockerignore` file was implemented to optimize build context and image size.
    - The API Docker image was successfully built and pushed to AWS ECR (`536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api`).

- **Kubernetes Deployment (Minikube on EC2):**
    - `k8s/deployment.yaml` was created, defining a `Deployment` (2 replicas) and a `Service` (NodePort type) for the API.
    - The `MLFLOW_TRACKING_URI` environment variable for the API pods was configured to point to the MLflow server on the EC2 host network (using the EC2 private IP `10.0.1.99:5000`).
    - **Obstacles & Resolutions (Minikube & ECR):**
        - Initial Minikube instability was resolved by deleting and restarting the cluster (`minikube delete && minikube start --driver=docker`).
        - `ErrImagePull` from ECR was a significant hurdle. The solution involved authenticating Docker *within Minikube's Docker environment* (`eval $(minikube -p minikube docker-env) && docker login ... && eval $(minikube docker-env -u)`). An `ImagePullSecret` was also considered and used for more robust authentication later.
        - Pod scheduling issues (e.g., `Insufficient cpu`) were handled by reducing replica counts during testing.

- **API Testing & Debugging:**
    - **Automated Tests:** Developed `pytest` tests in `tests/api/test_api_endpoints.py` for `/health` and `/predict` (valid input, missing fields, invalid data types).
        - The API base URL for tests was derived from `minikube service health-predict-api-service --url`.
    - **Manual Tests:** Performed `curl` tests from the EC2 instance against the NodePort service, validating responses for various valid payloads (`test_payload1.json`, `test_payload2.json`, `test_payload3.json`) and malformed/invalid inputs (`malformed_payload1.json`, `invalid_json.json`).
    - **Key Obstacles & Resolutions during Testing:**
        - **Preprocessing Mismatch (Critical):** Early tests failed with `ValueError` because the API was applying `clean_data`/`engineer_features` but then passing the DataFrame with raw string categoricals directly to the model, which expected a numerically preprocessed array. This was the most significant bug.
            - **Fix:** Modified `scripts/train_model.py` to explicitly log the fitted `ColumnTransformer` (preprocessor) as an artifact (`preprocessor/preprocessor.joblib`) *within the same MLflow run as the model*. The API startup was then updated to load both this model and its co-located preprocessor. The `/predict` endpoint now uses this loaded preprocessor.
        - **Column Name Discrepancy:** After fixing the preprocessor loading, a `KeyError` occurred because the preprocessor (fitted during training) expected hyphenated column names (e.g., `race`, `gender`, `age` from original dataset; `diag_1`, `diag_2`, `diag_3`; and medication names like `glyburide-metformin`), while the API's `engineer_features` function produced DataFrame columns with underscores (e.g. from Pydantic model or direct creation like `age_ordinal`).
            - **Fix:** Enhanced `src/api/main.py` to ensure DataFrame column names passed to the preprocessor matched its expectations. This involved careful handling of Pydantic model aliases and the DataFrame creation step before preprocessing. Specifically, the API now ensures that column names like `admission_type_id` (from Pydantic model using alias for `admission-type-id` in JSON) are correctly mapped if the preprocessor was trained on `admission-type-id`. For medication columns that were often sources of this issue, the API was made robust to handle inputs that might come with underscores or hyphens by standardizing them before transformation if necessary or ensuring the preprocessor itself was trained with consistently named features.
        - **MLflow Connectivity/Model Loading:** Resolved `mlflow.exceptions.MlflowException: Registered Model ... not found` (often after MLflow DB issues, fixed by re-running training DAG) and S3 `404 Not Found` for model artifacts (fixed by ensuring `scripts/train_model.py` logged to correct `model/` path within MLflow run).
        - **Dependency Versioning:** Addressed `AttributeError: 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'` by pinning `scikit-learn==1.3.2` in `src/api/requirements.txt` to match the training environment.
        - All automated and manual tests were eventually successful, confirming the API's stability, correct prediction processing, and graceful error handling.

- **Project Documentation Updates:**
    - `project_steps.md` was continuously updated with detailed sub-tasks for Phase 3, tracking progress meticulously.
    - `system_overview.md` was updated to reflect the API architecture, ECR usage, and Minikube deployment.

## Phase 4: CI/CD Automation using AWS Resources - Detailed Log

## $(date +'%Y-%m-%d %H:%M:%S') - Updated Project Steps and Committed Changes

- Reviewed and enhanced `project_docs/ai_docs/project_steps.md` based on project prompt, plan, system overview, and current progress.
- Added detailed sub-tasks and explanations for remaining phases (Phase 4: CI/CD Automation, Phase 5: Drift Monitoring & Retraining Loop).
- Introduced a new Phase 6: Documentation, Finalization & AWS Showcase, with comprehensive steps for final deliverables.
- Staged all modified and untracked files, including several test JSON payloads and a script (`git-author-rewrite.sh`).
- Committed all changes with message "docs: Enhance and detail remaining project steps".
- Pushed the commit to the remote `main` branch.

## $(date +'%Y-%m-%d %H:%M:%S') - Updated System Overview Documentation

- Analyzed and updated `project_docs/ai_docs/system_overview.md` to accurately reflect the current project state.
- Incorporated details regarding the implemented FastAPI, its containerization (Dockerfile, ECR), and deployment to Minikube on EC2.
- Updated MLflow Model Registry usage to "implemented" for model promotion.
- Added Kubernetes (Minikube) and ECR to core technologies.
- Revised directory structure, API serving details, and Airflow orchestration sections.
- Updated current status and next steps to align with `project_steps.md`.
- Added API endpoint and ECR URI to key configuration points.
- Committed changes with message "docs: Update system overview and journal".
- Pushed commit to the remote `main` branch.

## $(date +'%Y-%m-%d %H:%M:%S') - Created CI/CD Pipeline DAG for API Deployment Automation

- Executed Phase 4, Step 1 from `project_steps.md`: Created an Airflow DAG for automating the deployment of the model serving API to Kubernetes.
- Created `mlops-services/dags/deployment_pipeline_dag.py` with the following tasks:
  - `get_production_model_info`: Python function that fetches the latest "Production" model version from MLflow, retrieving its version, run ID, and source URI.
  - `define_image_details`: Python function that generates a unique Docker image tag based on the model version and current timestamp.
  - `authenticate_docker_to_ecr`: BashOperator that authenticates Docker with AWS ECR.
  - `build_api_docker_image`: BashOperator that builds the Docker image from the project root using the existing Dockerfile.
  - `push_image_to_ecr`: BashOperator that pushes the tagged image to ECR.
  - `update_kubernetes_deployment`: BashOperator that updates the Kubernetes deployment with the new image and ensures the MLflow tracking URI is correctly set using the EC2 private IP.
  - `verify_deployment_rollout`: BashOperator that verifies the deployment rollout is successful and prints the service URL for user convenience.
- Defined appropriate task dependencies to ensure correct execution order.
- Updated `project_steps.md` to mark Phase 4, Step 1 and all its sub-tasks as complete.
- Next step is to verify the CI/CD DAG by triggering it manually from the Airflow UI.

## Phase 4: CI/CD Automation Using AWS Resources (cont'd)

- Completed Phase 4, Step 2: Verified IAM permissions for the EC2 instance.
- Confirmed the EC2 instance role has the necessary permissions for:
  - ECR operations: Successfully authenticated with ECR using `aws ecr get-login-password` command, confirming the required permissions are in place.
  - S3 operations: Verified that the instance has access to the S3 bucket used for MLflow artifacts (needed for model loading).
- The deployment pipeline DAG (`health_predict_api_deployment`) is configured to work with these permissions, with appropriate authentication steps in place.
- All IAM permissions necessary for the CI/CD pipeline execution have been confirmed to be correctly configured.
- Next step is to test the CI/CD DAG by triggering it manually from the Airflow UI.

## 2025-05-13 19:39:58 - Testing CI/CD DAG for Automated API Deployment

- Completed Phase 4, Step 3: Testing the CI/CD DAG for automated API deployment.
- Successfully triggered the `health_predict_api_deployment` DAG manually from the Airflow CLI using:
  ```bash
  docker-compose exec airflow-webserver airflow dags trigger health_predict_api_deployment
  ```
- Confirmed the DAG ran to completion successfully, as verified by:
  ```bash
  docker-compose exec airflow-webserver airflow dags list-runs -d health_predict_api_deployment
  ```
- Verified that the Docker image was built correctly by checking local Docker images. Multiple versions of the `health-predict-api` image were present in the local Docker environment.
- Confirmed that the Kubernetes deployment was successfully updated:
  - Checked running pods: `kubectl get pods -l app=health-predict-api`
  - Examined deployment details: `kubectl describe deployment health-predict-api-deployment`
  - Verified the deployment is using the expected image: `536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`
- Successfully tested the API endpoints:
  - Obtained the service URL using: `minikube service health-predict-api-service --url`
  - Tested the `/health` endpoint, which returned a successful response: `{"status":"ok","message":"API is healthy and model is loaded."}`
  - Tested the `/predict` endpoint using a test payload, which successfully returned a prediction.
- Updated `project_steps.md` to mark all sub-tasks in Phase 4, Step 3 as completed.
- Next phase to begin: Phase 5 - Drift Monitoring & Retraining Loop on AWS.

## 2025-05-13 20:34:04 - Enhanced CI/CD Pipeline with Automated API Testing

- Identified a gap in the CI/CD pipeline: no automated API testing after deployment.
- Implemented a solution by enhancing the `health_predict_api_deployment` DAG with a new task:
  - Added `run_api_tests` task that executes the existing pytest test suite (`tests/api/test_api_endpoints.py`) after deployment.
  - The test suite validates both the `/health` endpoint and the `/predict` endpoint with various test cases (valid inputs, missing fields, invalid data types).
  - This ensures that each deployment is automatically verified to be working correctly before being considered complete.
- Updated the task dependencies to include this new test task at the end of the workflow.
- Successfully tested the enhanced DAG by triggering it manually and confirming all tasks completed successfully.
- Updated `project_steps.md` to document this enhancement as sub-task 1.11 under Phase 4.
- This improvement follows MLOps best practices by incorporating automated testing into the deployment pipeline, providing immediate feedback on the health of each deployment.
