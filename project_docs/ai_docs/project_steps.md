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

2.  **IAM Permissions (Review & Confirm):**
    *   [x] **EC2 Instance Profile Role:** Verify the IAM role attached to your EC2 instance has sufficient permissions for:
        *   **ECR:** `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage`. (The `AmazonEC2ContainerRegistryPowerUser` or `FullAccess` managed policies usually cover these).
        *   **S3:** Full access to the MLflow artifact bucket and data buckets (already in place for training DAGs).
        *   (No explicit K8s permissions needed here as `kubectl` commands are local to EC2, assuming `kubectl` is configured correctly).

3.  **Testing CI/CD DAG:**
    *   [ ] **Trigger DAG Manually:** From the Airflow UI, trigger the `health_predict_api_deployment` DAG.
    *   [ ] **Monitor Task Logs:** Observe the logs for each task in the Airflow UI to ensure successful execution and troubleshoot any errors.
    *   [ ] **Verify ECR Image:** Check your ECR repository in the AWS console to confirm the new Docker image was pushed with the correct tag.
    *   [ ] **Verify Kubernetes Update:**
        *   Use `kubectl get pods -l app=health-predict-api` to see new pods being created and old ones terminating.
        *   Check `kubectl describe deployment health-predict-api-deployment` to see the updated image.
        *   Confirm the `verify_deployment_rollout` task in Airflow completes successfully.
    *   [ ] **Test Deployed API:** Briefly test the `/health` and `/predict` endpoints of the newly deployed API version to ensure it's operational.

**Phase 5: Drift Monitoring & Retraining Loop on AWS (Weeks 9-11)**

1.  **Monitoring Script (`/scripts/monitor_drift.py`):** This script will use Evidently AI to detect data and concept drift by comparing new data batches against a reference dataset.
    *   [ ] **Sub-task 1.1: Script Setup and Argument Parsing:**
        *   Create `scripts/monitor_drift.py`.
        *   Use `argparse` to accept parameters:
            *   `--s3_new_data_path`: S3 URI for the new data batch to analyze (e.g., `s3://<bucket>/current_batch_for_monitoring/batch_data.csv`).
            *   `--s3_reference_data_path`: S3 URI for the reference dataset (e.g., `s3://<bucket>/processed_data/initial_train.csv`).
            *   `--mlflow_model_uri`: MLflow model URI of the current production model (e.g., `models:/HealthPredict_RandomForest/Production`).
            *   `--s3_evidently_reports_path`: S3 prefix to save Evidently HTML reports (e.g., `s3://<bucket>/drift_reports/{{execution_date}}`).
            *   `--mlflow_experiment_name`: Name of the MLflow experiment to log monitoring results (e.g., "HealthPredict_Monitoring").
    *   [ ] **Sub-task 1.2: Initialize Clients and Load Data:**
        *   Initialize `boto3.client('s3')` and `mlflow.tracking.MlflowClient()`.
        *   Set MLflow tracking URI: `mlflow.set_tracking_uri("http://mlflow:5000")`.
        *   Load reference data and new data batch from S3 into Pandas DataFrames.
    *   [ ] **Sub-task 1.3: Load Production Model and Preprocessor:**
        *   Load the production model using `mlflow.pyfunc.load_model(model_uri=args.mlflow_model_uri)`. This should give you the pipeline (preprocessor + model).
        *   Extract the preprocessor. If it's the first step of a scikit-learn pipeline: `preprocessor = model.steps[0][1]`. If logged separately, load it using its own artifact URI (requires knowing the production model's run_id).
    *   [ ] **Sub-task 1.4: Data Preprocessing:**
        *   Apply the *same* data cleaning (`clean_data`) and feature engineering (`engineer_features`) functions from `src.feature_engineering.py` to both reference and new data DataFrames. **Important:** Ensure these functions can handle data without the target variable if it's not present in the new batch for some drift checks.
        *   Apply the *loaded production preprocessor* (`preprocessor.transform()`) to both the engineered reference data and the engineered new data.
    *   [ ] **Sub-task 1.5: Data Drift Detection with Evidently AI:**
        *   Import `DatasetDriftPreset` from `evidently.pipeline.preset_types`.
        *   Create the drift report: `drift_report = Pipeline(stages=[DatasetDriftPreset()])`.
        *   Run the report: `drift_report.run(current_data=preprocessed_new_df, reference_data=preprocessed_reference_df, column_mapping=None)`. (Define `column_mapping` if necessary, especially for target/prediction columns if used here).
        *   Extract drift summary: `drift_results = drift_report.as_dict()`.
        *   Determine if drift is detected: `is_drifted = drift_results['data_drift']['data']['metrics']['dataset_drift']`.
        *   Save HTML report: `drift_report.save_html(os.path.join(local_report_dir, "data_drift_report.html"))`. Upload this to the S3 path specified in args.
    *   [ ] **Sub-task 1.6: Concept Drift Detection (Prediction/Target Drift) with Evidently AI:**
        *   **If new data has target variable:**
            *   Add `DataQualityPreset` and `ClassificationPreset` (if classification) to the Evidently pipeline.
            *   Make predictions with the production model on both preprocessed reference and new data. Add these predictions as columns to the DataFrames.
            *   Run the report including these presets. Evaluate changes in performance metrics.
        *   **If new data does not have target (more common for immediate drift):**
            *   Focus on prediction distribution drift. Make predictions on both preprocessed reference and new data.
            *   Use `DatasetDriftPreset` on the *prediction columns* themselves to see if their distributions have changed.
        *   Save relevant HTML reports and extract metrics.
    *   [ ] **Sub-task 1.7: Log Metrics and Reports to MLflow:**
        *   Start an MLflow run: `with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(args.mlflow_experiment_name).experiment_id, run_name="monitoring_run_{{ts_nodash}}") as run:`.
        *   Log parameters: `s3_new_data_path`, `s3_reference_data_path`, `mlflow_model_uri`.
        *   Log metrics: `mlflow.log_metric("data_drift_detected", int(is_drifted))`, number of drifting features, any key performance metrics if concept drift was calculated.
        *   Log artifacts: Upload the saved Evidently HTML reports from local disk using `mlflow.log_artifact(local_report_dir)`.
    *   [ ] **Sub-task 1.8: Output Drift Status for Airflow:**
        *   The script should print a simple string like "DRIFT_DETECTED" or "NO_DRIFT" to stdout. This can be captured by Airflow's `BashOperator` (if used) or returned by a `PythonOperator` for XComs.

2.  **Airflow DAG for Simulation & Retraining Loop (`/dags/monitoring_retraining_dag.py`):**
    *   [ ] **Sub-task 2.1: Define DAG Structure and Parameters:**
        *   Create `monitoring_retraining_dag.py` in `mlops-services/dags/`.
        *   `dag_id` (e.g., `health_predict_monitoring_retraining`), `schedule_interval` (e.g., `@daily`, `@weekly`, or `None` for manual/simulated runs).
        *   Airflow `Params`:
            *   `s3_full_future_data_path`: S3 URI for `future_data.csv`.
            *   `s3_reference_data_path`: S3 URI for `initial_train.csv`.
            *   `batch_size`: Number of rows per simulated batch (e.g., 1000).
            *   `mlflow_model_name`: Name of the production model (e.g., "HealthPredict_RandomForest").
            *   `mlflow_experiment_name`: For logging monitoring runs.
    *   [ ] **Sub-task 2.2: Task 1 - Get Iteration State / Initialize (`PythonOperator`):**
        *   Name: `get_iteration_state`.
        *   Logic: Try to pull `next_batch_start_index` from the previous DAG run's XComs. If not found (first run), initialize to 0.
        *   Output: `current_batch_start_index` via XCom.
    *   [ ] **Sub-task 2.3: Task 2 - Simulate Data Batch Arrival (`PythonOperator`):**
        *   Name: `simulate_data_batch`.
        *   Inputs: `s3_full_future_data_path`, `batch_size`, `current_batch_start_index` (from XCom).
        *   Logic:
            *   Load `future_data.csv` (or a chunk of it) from S3.
            *   Select the next `batch_size` rows starting from `current_batch_start_index`.
            *   If no more data, set a flag `no_more_data=True` and push to XComs.
            *   Save this current batch to a temporary S3 location (e.g., `s3://<bucket>/current_batch_for_monitoring/batch_{{ts_nodash}}.csv`).
            *   Output: `s3_current_batch_path`, `next_batch_start_index` (current + batch_size), `no_more_data` flag via XComs.
    *   [ ] **Sub-task 2.4: Task 3 - Check if More Data Exists (`BranchPythonOperator`):**
        *   Name: `check_for_more_data`.
        *   Input: `no_more_data` flag from XCom.
        *   Logic: If `no_more_data` is true, branch to an `end_pipeline` task. Else, branch to `get_production_model_uri`.
    *   [ ] **Sub-task 2.5: Task 4 - Get Current Production Model URI (`PythonOperator`):**
        *   Name: `get_production_model_uri`.
        *   Logic: Similar to the deployment DAG task, fetch the MLflow URI of the current "Production" model for `{{ dag_run.conf.get('mlflow_model_name', 'HealthPredict_RandomForest') }}`.
        *   Output: `production_model_uri` via XCom.
    *   [ ] **Sub-task 2.6: Task 5 - Run Drift Monitoring Script (`BashOperator`):**
        *   Name: `run_drift_monitoring`.
        *   Bash Command:
            ```bash
            python /home/ubuntu/health-predict/scripts/monitor_drift.py \
              --s3_new_data_path {{ ti.xcom_pull(task_ids="simulate_data_batch", key="s3_current_batch_path") }} \
              --s3_reference_data_path {{ dag_run.conf.get('s3_reference_data_path') }} \
              --mlflow_model_uri {{ ti.xcom_pull(task_ids="get_production_model_uri", key="production_model_uri") }} \
              --s3_evidently_reports_path s3://<your-bucket>/drift_reports/{{ dag_run.id }} \
              --mlflow_experiment_name {{ dag_run.conf.get('mlflow_experiment_name') }}
            ```
        *   Enable `do_xcom_push=True` for the `BashOperator` to capture the script's stdout (drift status).
    *   [ ] **Sub-task 2.7: Task 6 - Decide on Retraining (`BranchPythonOperator`):**
        *   Name: `decide_on_retraining`.
        *   Input: Drift status from `run_drift_monitoring` task's XCom (e.g., `{{ ti.xcom_pull(task_ids="run_drift_monitoring") }}`).
        *   Logic: If status is "DRIFT_DETECTED", branch to `trigger_retraining_dag`. Else, branch to `log_no_drift_and_loop` (or directly to a task that continues the loop if no explicit logging is needed).
    *   [ ] **Sub-task 2.8: Task 7 - Trigger Retraining DAG (`TriggerDagRunOperator`):**
        *   Name: `trigger_retraining_dag`.
        *   `trigger_dag_id="health_predict_training_hpo"` (the ID of your main training DAG).
        *   `conf`: Pass configuration to the training DAG.
            *   `"retraining_data_s3_path": "{{ ti.xcom_pull(task_ids='simulate_data_batch', key='s3_current_batch_path') }}"` (if training DAG is modified to use it).
            *   `"is_retraining_run": True`.
        *   `wait_for_completion=True` (Optional, if the monitoring DAG should wait before proceeding).
        *   **Crucial Modification Needed for Training DAG:** The `health_predict_training_hpo` DAG and its underlying `scripts/train_model.py` must be adapted to:
            *   Accept an optional S3 path for new data.
            *   If provided, combine this new data with the original `initial_train.csv` for retraining.
            *   Ensure the preprocessor is refit on the combined dataset.
    *   [ ] **Sub-task 2.9: Task 8 - (If Retraining Triggered) Optionally Trigger Deployment DAG (`TriggerDagRunOperator`):**
        *   Name: `trigger_deployment_after_retraining`.
        *   This task would run if the retraining was successful and a new model was promoted. This requires a mechanism for the training DAG to signal success and the new model version, or this monitoring DAG would need to check MLflow again. Simpler for now might be manual trigger of deployment after successful retraining.
        *   If implemented: `trigger_dag_id="health_predict_api_deployment"`.
    *   [ ] **Sub-task 2.10: Task 9 - Log No Drift and Loop / End Pipeline (`PythonOperator` / `EmptyOperator`):**
        *   `log_no_drift_and_loop`: If no drift, log this. This task would then be followed by a task that ensures the loop continues by setting up for the next iteration (implicitly handled if tasks are structured to pick up `next_batch_start_index`).
        *   `end_pipeline`: An `EmptyOperator` if `no_more_data` was true.
    *   [ ] **Sub-task 2.11: Define Task Dependencies and Control Flow for Looping/Branching.**
    *   [ ] **Sub-task 2.12: Upload DAG to Airflow.**

**Phase 6: Documentation, Finalization & AWS Showcase (Weeks 12-13)** (New Phase)

1.  **Comprehensive Code & Project Documentation:**
    *   [ ] **Sub-task 1.1: Update/Finalize `README.md` (Project Root):**
        *   Ensure a clear project overview, goals, and MLOps focus.
        *   **Detailed Setup Instructions:**
            *   Prerequisites (AWS account, CLI, Terraform, Docker, Python, etc.).
            *   IaC: `terraform init, plan, apply` for AWS resources (`iac/`).
            *   EC2 Setup: SSH access, cloning repo, Docker Compose for services (`mlops-services/docker-compose.yml up -d --build`).
            *   Local K8s (Minikube/Kind) setup on EC2.
            *   Verification steps for each service (Airflow UI, MLflow UI, Postgres, K8s cluster).
        *   **Pipeline Execution Guides:**
            *   How to run the training pipeline (Airflow DAG: `health_predict_training_hpo`).
            *   How to run the API deployment pipeline (Airflow DAG: `health_predict_api_deployment`).
            *   How to run the monitoring & retraining simulation (Airflow DAG: `health_predict_monitoring_retraining`).
            *   Include example parameters and how to check outputs/status in MLflow, K8s, etc.
        *   Link to `system_overview.md` for architecture details.
        *   **Crucial: Explicit Teardown Instructions:**
            *   `docker-compose down -v` (in `mlops-services/`).
            *   `minikube delete` (or `kind delete cluster`).
            *   `terraform destroy -auto-approve` (in `iac/`).
    *   [ ] **Sub-task 1.2: Code Comments and Docstrings:**
        *   Review all Python scripts (`.py` files in `src/`, `scripts/`, `dags/`) for clear, concise docstrings for modules, classes, and functions, explaining purpose, arguments, and returns.
        *   Add inline comments for complex or non-obvious logic.
    *   [ ] **Sub-task 1.3: Finalize `system_overview.md`:**
        *   Update with any changes to architecture, data flow, or component interactions that occurred during Phases 4 & 5.
        *   Ensure the system diagram (if any, or create a textual one) is accurate.
    *   [ ] **Sub-task 1.4: Update `project_plan.md` and `project_steps.md`:**
        *   Mark all steps as complete.
        *   Briefly summarize key achievements or deviations in the plan if necessary.
    *   [ ] **Sub-task 1.5: (Optional) User Manual for Pipelines:**
        *   If README sections become too long, consolidate detailed pipeline execution steps into a separate `USER_MANUAL.md`.
    *   [ ] **Sub-task 1.6: Future Enhancements Document (`FUTURE_ENHANCEMENTS.md`):**
        *   Create a new markdown file.
        *   Outline potential future improvements as per `project_prompt.md` requirements (e.g., more advanced models, different drift detection methods, scaling to managed K8s like EKS, full CI with Git triggers, security hardening, advanced monitoring dashboards).

2.  **Finalize Project Deliverables:**
    *   [ ] **Sub-task 2.1: GitHub Repository:**
        *   Ensure all code (Python scripts, notebooks, API, IaC), Dockerfiles, K8s manifests, Airflow DAGs, and all documentation (`.md` files) are committed and pushed to the `main` branch.
        *   Clean up any temporary files, commented-out old code (unless for illustration), or unused assets.
        *   Verify `.gitignore` is comprehensive and correctly excludes state files, virtual environments, IDE files, etc.
    *   [ ] **Sub-task 2.2: Experimentation Notebook (`/notebooks/01_eda_baseline.py`):**
        *   Ensure it's well-commented, cells are logically ordered, and presents EDA, preprocessing, and baseline model exploration clearly.
        *   Include outputs/visualizations directly in the notebook if submitting `.ipynb`, or ensure it renders correctly if exported to PDF/HTML.
    *   [ ] **Sub-task 2.3: Drift Visualizations / Reports:**
        *   Generate a few sample Evidently drift reports (HTML) using a simulated data batch from `future_data.csv`.
        *   Include these sample HTML files in a designated project directory (e.g., `project_docs/sample_drift_reports/`) and commit them.
        *   Reference/link to these sample reports in the main README or documentation to showcase the monitoring output.
    *   [ ] **Sub-task 2.4: IaC Scripts (`/iac`):**
        *   Ensure Terraform scripts are clean, well-commented, and variables are clearly defined.
        *   Verify the `project_docs/terraform_guide.md` is accurate and guides a user through `init/plan/apply/destroy`.
    *   [ ] **Sub-task 2.5: Deployed API (For Demonstration):**
        *   The API will be transient (running on local K8s on EC2). Ensure the deployment process is smooth and documented for demonstration purposes.

3.  **(Recommended) Video Walkthrough:**
    *   [ ] **Sub-task 3.1: Plan Video Content & Script:**
        *   Introduction to the project and its MLOps goals.
        *   IaC deployment (`terraform apply` - can be pre-recorded or sped up).
        *   Tool setup on EC2 (show Docker Compose services running: Airflow, MLflow, Postgres, JupyterLab).
        *   Execution of the training pipeline (trigger `health_predict_training_hpo` DAG, show MLflow experiment tracking with parameters, metrics, artifacts including model and preprocessor).
        *   Model registration in MLflow (show the `find_and_register_best_model` task promoting a model to "Production").
        *   Execution of the API deployment pipeline (trigger `health_predict_api_deployment` DAG, show image build, ECR push, K8s deployment update).
        *   API interaction (use `curl` or Postman to hit `/health` and `/predict` endpoints of the deployed API on Minikube).
        *   Demonstration of drift detection (trigger `health_predict_monitoring_retraining` DAG, show Evidently reports being generated and drift status).
        *   Demonstration of retraining trigger (show training DAG being triggered if drift was detected).
        *   **Crucial: Demonstrate full resource teardown** (`docker-compose down -v`, `minikube delete`, `terraform destroy`).
    *   [ ] **Sub-task 3.2: Record and Edit Video:**
        *   Use screen recording software. Ensure clear audio narration.
        *   Keep it concise (e.g., 10-20 minutes if possible), highlighting key MLOps practices.
        *   Edit for clarity and flow.

4.  **Final Review and Submission Packaging:**
    *   [ ] **Sub-task 4.1: Review Against `project_prompt.md` Evaluation Criteria:**
        *   Go through each criterion and ensure your project addresses it adequately:
            *   EDA and Data Preparation (5%)
            *   Large Scale HPO and Experiment Tracking (15%)
            *   Model Packaging and Deployment (15%)
            *   Model Monitoring (15%)
            *   Continuous Integration and Deployment (CI/CD) (15%)
            *   Orchestration (20%)
            *   Solution Documentation (15%)
    *   [ ] **Sub-task 4.2: Prepare Submission Package:**
        *   Ensure the GitHub repository link is correct and accessible.
        *   Include the video link (if hosted) or file.
        *   Organize any other required files as per the course's submission guidelines.
        *   Double-check all instructions in `project_prompt.md` for submission requirements.
