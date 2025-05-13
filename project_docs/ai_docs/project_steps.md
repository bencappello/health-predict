## Health Predict MLOps Project - Detailed Step-by-Step Plan (Cost-Optimized AWS & Drift Simulation)

**Phase 1: Foundation, Cloud Setup & Exploration (Completed)**

*   **Project & AWS Setup:** Initialized GitHub repo (`health-predict-mlops`), local Python environment, project structure (`/src`, `/notebooks`, `/iac`, etc.), and configured AWS credentials.
*   **Infrastructure as Code (Terraform in `/iac`):** Developed scripts to provision core AWS resources: VPC, Public Subnet, Security Groups, IAM Roles (EC2 instance profile for S3/ECR access), EC2 instance (t2.micro/t3.micro), S3 bucket (for data & MLflow artifacts), and ECR repository. User Data on EC2 installed Docker, Docker Compose, and Git. Cost optimization was a key consideration, leading to running Kubernetes and PostgreSQL directly on the EC2 instance rather than using managed EKS/RDS.
*   **MLOps Tools on EC2 (`~/mlops-services/docker-compose.yml`):** Deployed PostgreSQL (for Airflow/MLflow metadata), Airflow (Webserver, Scheduler, Init - using `LocalExecutor`), and MLflow server (configured with Postgres backend and S3 for `--default-artifact-root`). Ensured services were accessible via EC2 public IP.
*   **Local Kubernetes on EC2:** Installed `kubectl` and Minikube (using Docker driver) on the EC2 instance to host the API deployment locally.
*   **Data Preparation & S3 Storage (`/scripts/split_data.py`):** Downloaded the dataset, uploaded raw data (`diabetic_data.csv`) to `s3://<bucket>/raw_data/`. The script then partitioned this into `initial_data` (20% for train/validation/test, saved as `initial_train.csv`, etc.) and `future_data.csv` (remaining 80% for drift simulation), both uploaded to `s3://<bucket>/processed_data/`.
*   **Initial EDA & Baseline Model (`/notebooks/01_eda_baseline.py`):** Using JupyterLab (via Docker Compose service), performed EDA on `initial_train.csv`. This involved data cleaning, basic feature engineering (e.g., creating `readmitted_binary` target, `age_ordinal`), training a baseline LogisticRegression model, and documenting initial findings.

**Phase 2: Scalable Training & Tracking on AWS (Completed)**

*   **Feature Engineering Pipeline (`/src/feature_engineering.py`):** Centralized data cleaning, feature creation, and preprocessing (Scikit-learn `ColumnTransformer` with `StandardScaler` for numerical, `OneHotEncoder` for categorical) into reusable functions.
*   **Training Script (`/scripts/train_model.py`):**
    *   Developed an `argparse`-driven script to load data from S3, apply the feature engineering pipeline from `/src/feature_engineering.py`.
    *   Supported experimentation with Logistic Regression, Random Forest, and XGBoost.
    *   **MLflow Integration:** Connected to the MLflow server on EC2. For each HPO trial and final model, logged parameters, the *fitted preprocessor object itself* (`preprocessor.joblib`), metrics (F1-score, etc.), and model artifacts (using `mlflow.sklearn.log_model`) to MLflow, with artifacts stored on S3.
    *   **Hyperparameter Optimization (HPO):** Implemented HPO using RayTune with `ASHAScheduler` for selected model architectures.
    *   **Key Architectural Choice:** A crucial step was logging the `preprocessor.joblib` artifact within the *same MLflow run* as the trained model it was fitted with (e.g., in a `preprocessor/` subfolder within the model's artifact URI). This ensures the API can deterministically load the exact preprocessor version tied to a specific model version.
*   **Airflow DAG for Training & HPO (`/dags/training_pipeline_dag.py`):**
    *   Created an Airflow DAG to orchestrate the model training and HPO process.
    *   A `BashOperator` executed `scripts/train_model.py`, passing necessary configurations (S3 paths, MLflow URI, HPO settings).
    *   A `PythonOperator` (`find_and_register_best_model`) was implemented to:
        *   Query MLflow (using `mlflow.tracking.MlflowClient`) for the best performing model from the HPO runs (prioritizing RandomForest, based on `val_f1_score`).
        *   **Critical Preprocessor Check:** Before registration, this task verifies that the chosen model's MLflow run contains the `preprocessor/preprocessor.joblib` artifact. This is a key control point for ensuring model-preprocessor consistency.
        *   If the preprocessor exists, the task archives any existing "Production" versions of the target model, registers the new model version, and promotes it to the "Production" stage in the MLflow Model Registry.

**Phase 3: API Development & Deployment to Local K8s (Completed)**

This phase involved creating a FastAPI application to serve the best model, containerizing it, deploying it to the Minikube cluster on EC2, and performing thorough testing.

*   **API Development (`/src/api/main.py`):**
    *   Developed a FastAPI application with `/health` and `/predict` (POST) endpoints.
    *   **Model Loading on Startup:** Implemented an `@app.on_event("startup")` handler to load the ML model (e.g., `HealthPredict_RandomForest`) and its co-located `preprocessor.joblib` from the "Production" stage of the MLflow Model Registry. The `MLFLOW_TRACKING_URI` was made configurable via an environment variable (defaulting to `http://mlflow:5000` for Docker Compose context, overridden for K8s).
    *   **Request/Response Schemas:** Defined Pydantic models (`InferenceInput`, `InferenceResponse`) for robust input validation and structured JSON responses. `InferenceInput` used Pydantic field aliases (e.g., `alias='race'`) to allow hyphenated keys in incoming JSON (e.g. `"race"`) while using underscored attribute names internally (e.g. `self.race`), aligning with DataFrame column naming conventions from training.
    *   **Prediction Logic:** The `/predict` endpoint:
        1.  Converts the `InferenceInput` (Pydantic model) to a Pandas DataFrame.
        2.  Applies `clean_data()` and `engineer_features()` from `src.feature_engineering.py`.
        3.  Uses the loaded `preprocessor.transform()` method on the engineered DataFrame.
        4.  Feeds the transformed data to the loaded `model.predict()` and `model.predict_proba()`.
        5.  Returns a JSON response with `prediction` and `probability_score`.
    *   **Dependencies (`/src/api/requirements.txt`):** Created with pinned versions for key libraries (e.g., `fastapi`, `mlflow`, `scikit-learn`) to ensure consistency between training and serving environments, mitigating issues like Scikit-learn version mismatches.
*   **Containerization (`/Dockerfile` at project root) & ECR:**
    *   Created a `Dockerfile` using a `python:3.10-slim` base image, set `WORKDIR /app`, copied `src/api/requirements.txt` and installed dependencies, then copied the `/src` directory (containing `api` and `feature_engineering` modules). Used `.dockerignore` to exclude unnecessary files.
    *   The API image was built and pushed to AWS ECR (e.g., `536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api`).
*   **Kubernetes Deployment (`/k8s/deployment.yaml` on Minikube on EC2):**
    *   Defined a K8s `Deployment` (initially 2 replicas) and a `Service` of `type: NodePort`.
    *   The `Deployment` spec configured the container image from ECR and set the `MLFLOW_TRACKING_URI` environment variable for the pods to `http://<EC2-Private-IP>:5000` to enable access to the MLflow server running on the EC2 host.
    *   **Key K8s Challenge & Solution (ECR Authentication with Minikube):** `ErrImagePull` was a common issue. Resolved by ensuring Minikube's internal Docker daemon could authenticate to ECR. This was done by executing `aws ecr get-login-password ... | docker login ...` *within the Minikube Docker environment* (accessed via `eval $(minikube -p minikube docker-env)`).
*   **API Testing (`/tests/api/test_api_endpoints.py`) & Debugging:**
    *   Developed automated `pytest` tests for `/health` and `/predict` (valid inputs, missing fields, invalid data types). The API base URL for tests was dynamically determined using `minikube service health-predict-api-service --url`.
    *   Performed manual `curl` testing using sample valid (`test_payload1.json`, etc.) and malformed JSON payloads.
    *   **Critical Preprocessing Debugging:** The most significant challenge was ensuring the API's preprocessing steps perfectly mirrored those used in training. This involved:
        *   **Initial Mismatch:** The API initially loaded only the model, not the preprocessor, leading to errors when string categoricals were passed to the model. **Fix:** Ensured `scripts/train_model.py` logged the `ColumnTransformer` (preprocessor) as `preprocessor.joblib` in the same MLflow run as the model. The API was updated to load both.
        *   **Column Name Consistency:** Post-fix, `KeyError`s occurred due to discrepancies in column naming (e.g., hyphenated `diag-1` vs. underscored `diag_1`) between what the preprocessor expected (from training data) and what the API generated. **Fix:** Standardized column naming in the API before applying the preprocessor, often by ensuring Pydantic model field names or aliases correctly mapped to the feature names the preprocessor was trained with.
    *   Successful completion of all automated and manual tests verified the API's robustness, correct data handling, and error responses.

**Phase 4: CI/CD Automation using AWS Resources (Weeks 7-8)**

1.  **Airflow DAG for Deployment (`/dags/deployment_pipeline_dag.py`):** This DAG will automate the deployment of the model serving API to the local Kubernetes cluster on EC2.
    *   [x] **Sub-task 1.1: Define DAG Structure and Parameters:**
        *   Create a new Python file `deployment_pipeline_dag.py` in the `mlops-services/dags/` directory.
        *   Define `dag_id` (e.g., `health_predict_api_deployment`), `schedule_interval=None` (for manual trigger or trigger from another DAG), `catchup=False`, and appropriate `default_args` (start_date, owner, retries).
        *   Consider Airflow `Params` for the DAG if you want to manually specify things like `MODEL_NAME` or `MODEL_STAGE` at trigger time, though fetching the latest "Production" is usually preferred.
    *   [x] **Sub-task 1.2: Task 1 - Get Latest Production Model Information (`PythonOperator`):**
        *   Name: `get_production_model_info`.
        *   Python Callable: A function that uses the MLflow client (`mlflow.tracking.MlflowClient()`).
            *   Set tracking URI: `mlflow.set_tracking_uri("http://mlflow:5000")`.
            *   Input: `MODEL_NAME` (e.g., "HealthPredict_RandomForest").
            *   Logic: Fetch the latest version of the model in the "Production" stage using `client.get_latest_versions(name=MODEL_NAME, stages=["Production"])`.
            *   Output: Push the `model_uri` (e.g., `versions[0].source`) and `run_id` (e.g., `versions[0].run_id`) to XComs. The `run_id` is crucial for fetching co-located artifacts like the preprocessor if needed by the API build process (though the current API loads at runtime).
    *   [x] **Sub-task 1.3: Task 2 - Define Image URI and Tag (`PythonOperator` or `BashOperator`):**
        *   Name: `define_image_details`.
        *   Logic: Construct the full ECR image URI and a new unique tag.
            *   ECR Base URI: `<your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com/<your-ecr-repo-name>` (e.g., `536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api`).
            *   Tag Strategy: Use the Airflow DAG run ID (`{{ run_id }}`), a timestamp (`{{ ts_nodash }}`), or the model version being deployed. Example: `latest` or `model_v{{ ti.xcom_pull(task_ids="get_production_model_info", key="model_version") }}`.
            *   Output: Push `FULL_IMAGE_URI` (e.g., `<ECR_BASE_URI>:<TAG>`) to XComs.
    *   [x] **Sub-task 1.4: Task 3 - Build Docker Image (`BashOperator`):**
        *   Name: `build_api_docker_image`.
        *   Bash Command:
            ```bash
            docker build -t {{ ti.xcom_pull(task_ids="define_image_details", key="FULL_IMAGE_URI") }} .
            ```
        *   Ensure the command is executed from the project root directory (`/home/ubuntu/health-predict/`) which is the Docker build context.
        *   The existing `Dockerfile` at the project root will be used.
        *   The Airflow worker (EC2 instance) must have Docker installed.
    *   [x] **Sub-task 1.5: Task 4 - Authenticate Docker with ECR (`BashOperator`):**
        *   Name: `authenticate_docker_to_ecr`.
        *   Bash Command:
            ```bash
            aws ecr get-login-password --region <your-aws-region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com
            ```
            *   Replace `<your-aws-region>` and `<your-aws-account-id>` (e.g., `us-east-1`, `536474293413`).
        *   The Airflow worker needs AWS CLI installed and configured with necessary IAM permissions (via EC2 instance profile).
    *   [x] **Sub-task 1.6: Task 5 - Push Docker Image to ECR (`BashOperator`):**
        *   Name: `push_image_to_ecr`.
        *   Bash Command:
            ```bash
            docker push {{ ti.xcom_pull(task_ids="define_image_details", key="FULL_IMAGE_URI") }}
            ```
    *   [x] **Sub-task 1.7: Task 6 - Update Kubernetes Deployment (`BashOperator`):**
        *   Name: `update_kubernetes_deployment`.
        *   Bash Command:
            ```bash
            kubectl set image deployment/health-predict-api-deployment \
              health-predict-api-container={{ ti.xcom_pull(task_ids="define_image_details", key="FULL_IMAGE_URI") }} \
              --record
            # Optionally, if MLFLOW_TRACKING_URI needs to be updated or ensured for the new deployment:
            # kubectl set env deployment/health-predict-api-deployment MLFLOW_TRACKING_URI="http://<EC2-Private-IP>:5000"
            ```
            *   `health-predict-api-deployment` is the name of your Kubernetes Deployment object.
            *   `health-predict-api-container` is the name of the container within your pod spec.
            *   `--record` is deprecated but was used to record the command in rollout history; consider using alternative audit methods if needed.
        *   The Airflow worker needs `kubectl` installed and configured with the context for the local Minikube/Kind cluster.
    *   [x] **Sub-task 1.8: Task 7 - Verify Deployment Rollout (`BashOperator`):**
        *   Name: `verify_deployment_rollout`.
        *   Bash Command:
            ```bash
            kubectl rollout status deployment/health-predict-api-deployment --timeout=5m
            ```
    *   [x] **Sub-task 1.9: Define Task Dependencies:** Set the correct order of execution for these tasks.
    *   [x] **Sub-task 1.10: Upload DAG to Airflow:** Copy the completed `deployment_pipeline_dag.py` to the `mlops-services/dags/` directory on the EC2 instance and ensure Airflow picks it up.
    *   [x] **Sub-task 1.11: Enhance DAG with Automated API Testing:**
        *   Add a new task after `verify_deployment_rollout` to run the API test suite.
        *   Name: `run_api_tests`.
        *   Bash Command:
            ```bash
            cd /home/ubuntu/health-predict && \
            python -m pytest tests/api/test_api_endpoints.py -v
            ```
        *   This ensures that the newly deployed API version passes all tests before considering the deployment complete.
        *   Update task dependencies to include this new test task at the end of the workflow.

2.  **IAM Permissions (Review & Confirm):**
    *   [x] **EC2 Instance Profile Role:** Verify the IAM role attached to your EC2 instance has sufficient permissions for:
        *   **ECR:** `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage`. (The `AmazonEC2ContainerRegistryPowerUser` or `FullAccess` managed policies usually cover these).
        *   **S3:** Full access to the MLflow artifact bucket and data buckets (already in place for training DAGs).
        *   (No explicit K8s permissions needed here as `kubectl` commands are local to EC2, assuming `kubectl` is configured correctly).

3.  **Testing CI/CD DAG:**
    *   [x] **Trigger DAG Manually:** From the Airflow UI, trigger the `health_predict_api_deployment` DAG.
    *   [x] **Monitor Task Logs:** Observe the logs for each task in the Airflow UI to ensure successful execution and troubleshoot any errors.
    *   [x] **Verify ECR Image:** Check your ECR repository in the AWS console to confirm the new Docker image was pushed with the correct tag.
    *   [x] **Verify Kubernetes Update:**
        *   Use `kubectl get pods -l app=health-predict-api` to see new pods being created and old ones terminating.
        *   Check `kubectl describe deployment health-predict-api-deployment` to see the updated image.
        *   Confirm the `verify_deployment_rollout` task in Airflow completes successfully.
    *   [x] **Test Deployed API:** Briefly test the `/health` and `/predict` endpoints of the newly deployed API version to ensure it's operational.

**Phase 5: Drift Monitoring & Retraining Loop on AWS (Next)**