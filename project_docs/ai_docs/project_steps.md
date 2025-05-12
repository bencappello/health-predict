## Health Predict MLOps Project - Detailed Step-by-Step Plan (Cost-Optimized AWS & Drift Simulation)

**Phase 1: Foundation, Cloud Setup & Exploration (Weeks 1-2)**

1.  **Project Setup:**
    * [x] Create a new private GitHub repository named `health-predict-mlops`.
    * [x] Clone the repository locally.
    * [x] Set up a local Python environment (e.g., using Conda or venv) with basic libraries (`python>=3.10`, `pip`, `git`).
    * [x] Create initial project structure (e.g., `/src`, `/notebooks`, `/iac`, `/scripts`, `/config`, `/docs`, `/docker-compose`).
    * [x] Create `.gitignore` file (add common Python/IDE/OS files, secrets, terraform state).

2.  **AWS Account & Credentials:**
    * [x] Ensure you have an AWS account with sufficient permissions.
    * [x] Configure AWS credentials locally (e.g., via `aws configure` or environment variables).

3.  **Infrastructure as Code (IaC - Terraform/CDK):**
    * [x] Initialize Terraform/CDK project in the `/iac` directory. *(Terraform files created, user to init/plan/apply. See `project_docs/terraform_guide.md`)*
    * [x] **Write IaC Scripts:** Define resources for: *(Initial scripts created for VPC, Subnet, IGW, SG, IAM, EC2, S3, ECR. See `project_docs/terraform_guide.md` for customization and deployment)*
        * [x] VPC, **Public Subnet(s)**, Internet Gateway. *(Initial script created)*
        * [x] Security Groups (for EC2 - allow SSH, Airflow UI, MLflow UI, API port from your IP; allow necessary egress). *(Initial script created)*
        * [x] IAM Roles & Policies (for EC2 Instance Profile allowing S3, ECR access). *(Initial script created)*
        * [x] EC2 Instance (Choose **Free Tier eligible** type like `t2.micro` or `t3.micro` if possible, ensure sufficient RAM/CPU for Docker, Airflow, MLflow, DB, K8s). Configure User Data to install Docker, Docker Compose, Git. *(Initial script created)*
        * [x] S3 Bucket (for data, MLflow artifacts, reports - enable versioning). *(Initial script created)*
        * [x] ECR Repository (for the model API Docker image). *(Initial script created)*
        * [x] *(Remove RDS Instance definition)*.
        * [x] *(Remove EKS Cluster & Node Group definitions)*.
    * [x] **Deploy Infrastructure:** Run `terraform init`, `terraform plan`, `terraform apply`. Verify resource creation. **Remember to run `terraform destroy` when finished working.** *(User to perform. See `project_docs/terraform_guide.md`)*
    * [x] **Output Configuration:** Output EC2 Public IP/DNS, S3 bucket name, ECR repo URI. *(User to get from `terraform output` after apply. See `project_docs/terraform_guide.md`)*

4.  **Tool Installation & Configuration on EC2:**
    * [x] SSH into the provisioned EC2 instance. Verify Docker & Docker Compose are installed (via User Data or manual install).
    * [x] Create a directory for docker compose files (e.g., `~/mlops-services`).
    * [x] **Create `docker-compose.yml` in `~/mlops-services`:**
        * Define services for:
            * `postgres`: Use official `postgres` image. Set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB env vars. Mount a Docker volume for persistence (`pgdata:/var/lib/postgresql/data`).
            * `airflow-webserver`, `airflow-scheduler`, `airflow-init`: Use official `apache/airflow` image.
                * Configure `AIRFLOW__CORE__SQL_ALCHEMY_CONN` to point to the `postgres` service (e.g., `postgresql+psycopg2://user:password@postgres:5432/airflowdb`).
                * Configure `AIRFLOW__CORE__EXECUTOR=LocalExecutor`.
                * Mount host directories for `/dags`, `/logs`, `/plugins`.
                * Set `AIRFLOW_UID=$(id -u)` env var.
                * Ensure `postgres` is listed under `depends_on`.
            * `mlflow`: Use official `ghcr.io/mlflow/mlflow` image.
                * Command: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://user:password@postgres:5432/mlflowdb --default-artifact-root s3://your-mlflow-s3-bucket/` (Ensure MLflow DB user/pass are set and DB exists in Postgres, or use separate DB).
                * Ensure `postgres` is listed under `depends_on`.
    * [x] Start services using `docker-compose up -d` in `~/mlops-services`. Check logs (`docker-compose logs -f`).
    * [x] Verify access to Airflow UI (port 8080) and MLflow UI (port 5000) via browser using EC2 public IP.
    * [x] **Install Local Kubernetes (Minikube/Kind):**
        * [x] Install `kubectl` on EC2.
        * [x] Install Minikube or Kind on EC2 following their official documentation. Start the cluster (e.g., `minikube start --driver=docker` or `kind create cluster`).

5.  **Data Preparation & Storage:** (Using first 20% for initial training, rest for simulation)
    * [x] Download dataset.
    * [x] Upload raw data to S3 (`raw_data/diabetic_data.csv`).
    * [x] Modify and run data splitting script (`/scripts/split_data.py`):
        * [x] Separate first 20% of data as `initial_data`.
        * [x] Split `initial_data` into train/validation/test sets (e.g., 70/15/15 of the 20%).
        * [x] Save `initial_train.csv`, `initial_validation.csv`, `initial_test.csv` to `processed_data/` prefix in S3.
        * [x] Save the remaining 80% of data as `future_data.csv` to `processed_data/` prefix in S3.

6.  **Initial EDA & Baseline Model:** (Using Python Script with Notebook Cells in Cursor)
    * [ ] Ensure the JupyterLab service is running via `docker-compose ps` (provides kernel).
    * [x] Create Python script (`/notebooks/01_eda_baseline.py`) with notebook-style cells (e.g., using `# %%` delimiter).
    * [ ] Connect Cursor's Jupyter extension to the running Jupyter kernel (`http://localhost:8888`).
    * [x] **In Script Cells:**
        * [x] Configure `boto3` (should automatically use EC2 instance role credentials).
        * [x] Load initial training data from S3 (`processed_data/initial_train.csv`).
        * [x] Perform EDA and visualization on the initial training data.
        * [x] Perform cleaning and basic feature engineering (based on initial data).
        * [x] Train baseline model (e.g., `LogisticRegression`) on the initial training data.
        * [x] Evaluate model on the initial test set (`processed_data/initial_test.csv`).
        * [x] Document observations (in markdown cells: `# %% [markdown]`).
    * [x] Commit script changes to Git.

**Phase 2: Scalable Training & Tracking on AWS (Weeks 3-4)**

1.  **Feature Engineering Pipeline:**
    * [x] Create script (`/src/feature_engineering.py`) with Scikit-learn pipelines and save/load functions.

2.  **Training Script:**
    * [x] Create script (`/scripts/train_model.py`) using `argparse`.
    * [x] Load data from S3.
    * [x] Use `feature_engineering.py`.
    * [x] **Experiment with Model Architectures:** Implement and evaluate various models (e.g., Logistic Regression, Random Forest, XGBoost, and potentially others like LightGBM or CatBoost).
    * [x] **Integrate MLflow:** Connect to MLflow server running on EC2. Log parameters, transformer artifacts (to S3 via MLflow), metrics, and tags for each model type and experiment run.
    * [x] **Implement HPO (RayTune):** Utilize RayTune for comprehensive hyperparameter optimization across the selected model architectures. Ensure search spaces are well-defined for each model type.
    * [x] Log the best version of each model type using `mlflow.sklearn.log_model()` or equivalent for other frameworks (artifacts go to S3).
    * [x] Execute `scripts/train_model.py` on the EC2 instance with appropriate parameters (S3 paths, MLflow URI, Ray Tune settings) to perform HPO and log all experiments and final models to MLflow (artifacts stored on S3 via MLflow).
    * [x] Verify successful script execution by checking logs and MLflow UI for logged parameters, metrics, preprocessor, and the best model for each algorithm type.

3.  **Airflow DAG for Training & HPO:**
    * [x] Create DAG file (`/dags/training_pipeline_dag.py`).
    * [x] Define DAG schedule/args.
    * [x] Task 1: Optional data split.
    * [x] **Task 2 (`BashOperator`/`PythonOperator`):** Execute `train_model.py`. Pass MLflow tracking URI pointing to the MLflow container on EC2 (e.g., `http://<mlflow-service-name>:5000` or `http://localhost:5000` depending on network mode/execution context). Ensure EC2 role allows S3 access.
    * [x] **Task 3: Use MLflow client API to find and register the best model in MLflow Model Registry.**
      **Revised Detailed Implementation Guide for Task 3 (AI Agent Focused):**

      This task involves enabling, verifying, and troubleshooting the `find_and_register_best_model_task` within the `mlops-services/dags/training_pipeline_dag.py` Airflow DAG. This task uses the MLflow client API to identify the best performing models and register them.

      **Prerequisites (Assumed already completed by AI Agent in previous steps):**

      *   The file `mlops-services/dags/training_pipeline_dag.py` has been edited to:
          *   Uncomment the `find_and_register_best_model` Python function.
          *   Modify the function's `mlflow.search_runs` call to use `filter_string="tags.best_hpo_model = 'True'"`.
          *   Ensure model registration uses `model_uri=f"runs:/{best_run_id}/model"` and transitions to "Production".
          *   Uncomment the `find_and_register_best_model_task` PythonOperator.
          *   Update DAG task dependencies to `run_training_and_hpo >> find_and_register_best_model_task`.

      **Execution & Verification Steps (To be performed by AI Agent):**

      1.  **Ensure DAG File is Updated on EC2:**
          *   Confirm that the latest version of `mlops-services/dags/training_pipeline_dag.py` (with the model registration logic enabled) is present on the EC2 instance in the correct location for Airflow to pick up. *(This is typically handled when the AI uses an `edit_file` tool targeting the EC2 path, but good to state).*

      2.  **Trigger the Airflow DAG via CLI:**
          *   **Instruction:** Use a terminal command to trigger the `health_predict_training_hpo` DAG.
          *   **Command:**
              ```bash
              docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec airflow-scheduler airflow dags trigger health_predict_training_hpo
              ```
          *   **Expected Output:** Confirmation of DAG trigger (e.g., "DAG 'health_predict_training_hpo' triggered"). Note the `run_id` if provided, or you'll need to fetch it in the next step.

      3.  **Monitor DAG Run Status and Task Completion via CLI:**
          *   **Instruction:** Periodically check the status of the DAG run and its tasks until completion or failure.
          *   **a. Get Latest DAG Run ID (if not noted from trigger step):**
              *   **Command:**
                  ```bash
                  docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec airflow-scheduler airflow dags list-runs -d health_predict_training_hpo -o plain --no-header | grep -E 'manual__|scheduled__' | tail -n 1 | awk '{print $1}'
                  ```
              *   Store the output as `DAG_RUN_ID`.
          *   **b. Check Overall DAG Run Status:**
              *   **Command (replace `YOUR_DAG_RUN_ID` with the stored `DAG_RUN_ID`):**
                  ```bash
                  # First, get the execution date string for the specific DAG_RUN_ID
                  EXECUTION_DATE_STR=$(docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec airflow-scheduler airflow dags list-runs -d health_predict_training_hpo -o plain --no-header | grep "YOUR_DAG_RUN_ID" | awk '{print $3" "$4}')
                  # Then, get the report
                  docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec airflow-scheduler airflow dags report health_predict_training_hpo --start-date "$EXECUTION_DATE_STR" | grep -A 1 "YOUR_DAG_RUN_ID"
                  ```
              *   **Expected Output Analysis:** Look for `state=success` for the DAG run. If `state=failed` or `state=running` after a reasonable time, proceed to check task logs.
          *   **c. Check Task Instance Status (if DAG is successful or for debugging):**
              *   **Command (replace `YOUR_DAG_RUN_ID` with the stored `DAG_RUN_ID`):**
                  ```bash
                  docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec airflow-scheduler airflow tasks list health_predict_training_hpo YOUR_DAG_RUN_ID
                  ```
              *   **Expected Output Analysis:** Verify `run_training_and_hpo` and `find_and_register_best_model` tasks are listed and show `success`.

      4.  **Retrieve and Analyze Airflow Task Logs for `find_and_register_best_model_task`:**
          *   **Instruction:** Fetch the logs for the `find_and_register_best_model_task` and parse them for specific success indicators.
          *   **Command (replace `YOUR_DAG_RUN_ID` with the stored `DAG_RUN_ID`):**
              ```bash
              docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec airflow-scheduler airflow tasks logs YOUR_DAG_RUN_ID find_and_register_best_model health_predict_training_hpo
              ```
          *   **Log Analysis - Look for:**
              *   `Setting MLflow tracking URI to: http://mlflow:5000`
              *   `Using experiment ID: <experiment_id>`
              *   `Processing Best LogisticRegression model: Run ID <run_id>, F1 Score: <score>`
              *   `Registered LogisticRegression model as 'HealthPredict_LogisticRegression' version <X> and transitioned to Production stage.`
              *   Similar messages for `RandomForest` and `XGBoost`.
              *   Absence of error messages like "No runs found tagged as 'best_hpo_model = True'" or "Error registering or transitioning...".

      5.  **Verify Model Registration and Staging in MLflow Programmatically:**
          *   **Instruction:** Create and execute a Python script on the EC2 instance (within a suitable environment like the `jupyterlab` container) to query the MLflow API.
          *   **a. Create Verification Script (`verify_mlflow_registration.py` in `~/health-predict/scripts/` on the EC2 instance):**
              *   **Action:** Use file editing capabilities to create `/home/jovyan/work/scripts/verify_mlflow_registration.py` inside the `jupyterlab` container (which maps to `~/health-predict/scripts/` on the EC2 host) with the following content:
              ```python
              import mlflow
              import os
              import sys # Import sys for exit codes

              MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
              # EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'HealthPredict_Training_HPO_Airflow') # Not directly used in this script but good for context
              MODEL_NAMES = ["HealthPredict_LogisticRegression", "HealthPredict_RandomForest", "HealthPredict_XGBoost"]

              print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
              mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
              client = mlflow.tracking.MlflowClient()

              all_checks_passed = True

              for model_name in MODEL_NAMES:
                  print(f"--- Verifying Model: {model_name} ---")
                  try:
                      registered_model = client.get_registered_model(model_name)
                      # get_registered_model raises mlflow.exceptions.RestException if not found, 
                      # so an explicit check for None might not be necessary if we let the exception be caught.
                      # However, for clarity if the API changes or for other clients, explicit checks are safer.
                      if not registered_model: # Defensive check
                          print(f"ERROR: Registered model '{model_name}' not found (get_registered_model returned None/False).")
                          all_checks_passed = False
                          continue
                      
                      print(f"Found registered model: '{model_name}'.")
                      
                      latest_versions = registered_model.latest_versions
                      if not latest_versions:
                          print(f"ERROR: No versions found for model '{model_name}'.")
                          all_checks_passed = False
                          continue

                      production_version_found = False
                      for version_info in latest_versions:
                          if version_info.current_stage == 'Production':
                              production_version_found = True
                              print(f"  Version {version_info.version} is in 'Production' stage.")
                              print(f"    Source run_id: {version_info.run_id}")
                              
                              # Verify source run details
                              try:
                                  source_run = mlflow.get_run(version_info.run_id)
                                  expected_model_type_tag = model_name.replace('HealthPredict_', '')
                                  
                                  if source_run.data.tags.get('best_hpo_model') == 'True' and \
                                     source_run.data.tags.get('model_name') == expected_model_type_tag:
                                      print(f"    Source run ID {version_info.run_id} tags verified (best_hpo_model='True', model_name='{expected_model_type_tag}').")
                                  else:
                                      print(f"ERROR: Source run ID {version_info.run_id} tags not as expected for {model_name}.")
                                      print(f"      Expected: best_hpo_model='True', model_name='{expected_model_type_tag}'")
                                      print(f"      Found Tags: {source_run.data.tags}")
                                      all_checks_passed = False
                              except Exception as e_run:
                                  print(f"ERROR: Could not fetch or verify source run {version_info.run_id} for {model_name}: {e_run}")
                                  all_checks_passed = False
                              break # Found a production version, no need to check older latest_versions for this model_name
                      
                      if not production_version_found:
                          print(f"ERROR: No version in 'Production' stage found for model '{model_name}'. Latest versions inspected: {[(v.version, v.current_stage) for v in latest_versions]}")
                          all_checks_passed = False

                  except mlflow.exceptions.RestException as e_rest:
                      if "RESOURCE_DOES_NOT_EXIST" in str(e_rest) or e_rest.get_http_status_code() == 404:
                          print(f"ERROR: Registered model '{model_name}' not found (MLflow API 404/ResourceDoesNotExist).")
                      else:
                          print(f"ERROR: MLflow API RestException while verifying model '{model_name}': {e_rest}")
                      all_checks_passed = False
                  except Exception as e_general:
                      print(f"ERROR: A general exception occurred while verifying model '{model_name}': {e_general}")
                      all_checks_passed = False
                  print("-" * 40)

              if all_checks_passed:
                  print("\\nSUCCESS: All MLflow registration and staging checks passed.")
                  sys.exit(0)
              else:
                  print("\\nFAILURE: Some MLflow registration or staging checks failed.")
                  sys.exit(1)
              ```
          *   **b. Execute the Verification Script:**
              *   **Command:**
                  ```bash
                  docker-compose -f ~/health-predict/mlops-services/docker-compose.yml exec jupyterlab python3 /home/jovyan/work/scripts/verify_mlflow_registration.py
                  ```
              *   **Expected Output Analysis:** Look for "SUCCESS: All MLflow registration and staging checks passed." The script will exit with code 0 for success, 1 for failure. If failures occur, the script will print detailed errors for each failed check.

      **Troubleshooting Considerations (AI Agent Focused):**

      *   **MLflow Connection Issues:**
          *   If task logs or verification script show connection errors:
              *   Verify `MLFLOW_TRACKING_URI` (`http://mlflow:5000`) is correctly defined in `env_vars` in `training_pipeline_dag.py` (AI: use `read_file`).
              *   Verify the `mlflow` service is running: `docker-compose -f ~/health-predict/mlops-services/docker-compose.yml ps mlflow` (AI: use `run_terminal_cmd`).
      *   **Experiment Not Found:**
          *   If logs indicate experiment not found:
              *   Compare `EXPERIMENT_NAME` in `training_pipeline_dag.py` with the experiment name actually used/created by `scripts/train_model.py` (AI: use `read_file` on both).
      *   **No Runs Found (Tagged `best_hpo_model='True'`):**
          *   If the `find_and_register_best_model_task` log shows "No runs found tagged as 'best_hpo_model='True'":
              *   Read `scripts/train_model.py` to confirm it *is* setting the `best_hpo_model='True'` tag correctly on the final "Best Model" runs (AI: use `read_file`).
              *   Check the `run_training_and_hpo` task logs (fetched via CLI as above) to ensure it completed successfully and logged these tags.
      *   **Permissions (S3/MLflow):**
          *   If logs show permission errors (e.g., to S3 for artifacts or MLflow DB):
              *   Recall that the Airflow worker runs within a Docker container. The EC2 instance role should have S3 permissions. Docker Compose handles inter-container networking for Postgres. Focus on error messages in logs.
      *   **Model URI Issues (e.g., `runs:/{run_id}/model` not found):**
          *   If model registration fails because the artifact path is wrong:
              *   Read `scripts/train_model.py`. Confirm how `mlflow.sklearn.log_model` (or equivalent) is called for the final "Best Model" runs, specifically checking the `artifact_path` argument (AI: use `read_file`). If it's not the default "model", the `model_uri` in `find_and_register_best_model` function needs adjustment.

      By following these detailed, CLI- and script-oriented steps, an AI agent should be able to execute and verify this task.

    * [x] Upload DAG file to the mounted `/dags` directory on EC2.
    * [x] Test DAG execution. Verify results in MLflow UI (artifacts on S3, metadata in local Postgres).

**Phase 3: API Development & Deployment to Local K8s (Weeks 5-6)**

1.  **API Development (FastAPI):** This step focuses on creating a robust and production-ready API for serving the trained patient readmission prediction model. It involves setting up the FastAPI application, loading the model and preprocessor from MLflow, defining request/response schemas, and implementing the core prediction logic.
    * [x] **Create API Code Structure (`/src/api/main.py`):**
        *   Initialize a FastAPI application instance.
        *   Import necessary libraries: `fastapi`, `pydantic`, `mlflow`, `pandas`, `numpy`, `os`, `logging`, and any specific model libraries (e.g., `sklearn`, `xgboost`).
        *   Set up basic logging configuration (e.g., logging level, format).
        *   Define global variables for MLflow tracking URI, model name, and model stage (e.g., "Production"). These should ideally be configurable via environment variables.
    * [x] **Load Model and Preprocessor on Startup:**
        *   Implement a startup event handler (e.g., using `@app.on_event("startup")`) in FastAPI.
        *   Inside the startup handler:
            *   Set the MLflow tracking URI using `mlflow.set_tracking_uri()`. Consider fetching from `os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")`.
            *   Construct the model URI for the desired model stage (e.g., `f"models:/{MODEL_NAME}/{MODEL_STAGE}"`).
            *   Load the Scikit-learn flavor model using `mlflow.sklearn.load_model(model_uri=model_uri)`. This will load the model pipeline which should include the preprocessor if logged correctly.
            *   Store the loaded model/pipeline in a global variable or application state for access by endpoint functions.
            *   Log successful model loading or any errors encountered.
        *   **Note on Preprocessor:** If the preprocessor was logged separately from the model pipeline in MLflow (e.g., as a distinct artifact), load it similarly using its MLflow run ID and artifact path. Ensure the API uses the *exact same* preprocessor version that the model was trained with.
    * [x] **Define Pydantic Models for Request and Response:**
        *   Create a Pydantic model (e.g., `InferenceInput`) to define the expected input features and their data types for the `/predict` endpoint. This should mirror the raw input features before preprocessing.
        *   Create a Pydantic model (e.g., `InferenceResponse`) to define the structure of the prediction response (e.g., including `prediction` (0 or 1) and optionally `probability_score`).
    * [x] **Implement `/predict` Endpoint:**
        *   Define a POST endpoint (e.g., `@app.post("/predict", response_model=InferenceResponse)`).
        *   The endpoint function should accept an argument of type `InferenceInput`.
        *   Convert the input Pydantic model to a Pandas DataFrame suitable for the preprocessor/model.
        *   Apply `clean_data` and `engineer_features` (from `src.feature_engineering`) to the DataFrame. Ensure these functions are adapted or called correctly for API inference (e.g., handling absence of target variable, ensuring correct feature set for the preprocessor in `model_pipeline`).
        *   Perform prediction using the loaded `model_pipeline` (which includes the preprocessor and model) (e.g., `model_pipeline.predict(input_df)` and `model_pipeline.predict_proba(input_df)`).
        *   Return the prediction result formatted according to `InferenceResponse`.
        *   Implement robust error handling (e.g., for invalid input data, model prediction errors, model not loaded).
    * [x] **Implement `/health` Endpoint:**
        *   Define a GET endpoint (e.g., `@app.get("/health")`).
        *   This endpoint should return a simple JSON response indicating the API status (e.g., `{"status": "ok"}`).
        *   Optionally, it can check the status of critical components like model loading.
    * [x] **Create API `requirements.txt`:**
        *   Create a file named `requirements.txt` in the `/src/api/` directory.
        *   List all Python dependencies required to run the FastAPI application, including:
            *   `fastapi`
            *   `uvicorn[standard]` (for running the server)
            *   `pydantic`
            *   `mlflow`
            *   `pandas`
            *   `numpy`
            *   `scikit-learn`
            *   `xgboost` (if XGBoost model is used)
            *   `python-dotenv` (if using .env files for configuration)
            *   Any other specific libraries used by the model or feature engineering steps if they are re-executed or part of the model object.
        *   Specify versions for key packages to ensure reproducibility (e.g., `fastapi==0.100.0`, `mlflow==2.3.0`).

2.  **Containerization:** (No changes needed here)
    * [ ] Create Dockerfile (`/src/api/Dockerfile`).

3.  **Build and Push to ECR:** (No changes needed here)
    * [ ] Authenticate Docker with ECR on EC2.
    * [ ] Build Docker Image on EC2.
    * [ ] Push Docker Image to ECR.

4.  **Kubernetes Deployment (Targeting Local K8s on EC2):**
    * [ ] Ensure `kubectl` on EC2 is configured to talk to the local Minikube/Kind cluster.
    * [ ] **Create/Modify Kubernetes Manifests (`/k8s/deployment.yaml`):**
        * Define `Deployment`: Use ECR image URI.
        * Define `Service`: Type `NodePort` or `LoadBalancer` (if Minikube/Kind supports it via metallb or similar) to expose the service outside the K8s cluster but within the EC2 instance network.
        * **Permissions:** The pods need access to MLflow/S3. Since they run on the EC2 host's Docker daemon (Minikube/Kind), they *might* inherit the EC2 instance profile permissions, but explicitly mounting AWS credentials or using other secure methods might be needed depending on the local K8s setup. *This requires investigation.* Alternatively, bake credentials into the image (less secure) or pass via K8s secrets.
    * [ ] **Apply Manifests:** Run `kubectl apply -f k8s/deployment.yaml` on the EC2 instance.
    * [ ] **Verify Deployment:** Check pods (`kubectl get pods`), service (`kubectl get svc`). Determine the NodePort or IP/Port to access the service.

5.  **API Testing:**
    * [ ] Use `curl` or Postman *from the EC2 instance* or *via SSH tunnel* to send requests to the API service using the appropriate NodePort or ClusterIP/Port. Test `/health` and `/predict`.

**Phase 4: CI/CD Automation using AWS Resources (Weeks 7-8)**

1.  **Airflow DAG for Deployment:**
    * [ ] Create DAG file (`/dags/deployment_pipeline_dag.py`).
    * [ ] Define DAG schedule/params.
    * [ ] Task 1: Sync Git repo code.
    * [ ] Task 2: Get model URI from MLflow.
    * [ ] Task 3: Build Docker image (on EC2).
    * [ ] Task 4: Authenticate Docker with ECR (on EC2).
    * [ ] Task 5: Push image to ECR.
    * [ ] **Task 6 (`BashOperator`):** Apply the Kubernetes manifests using `kubectl apply -f k8s/deployment.yaml`. Ensure `kubectl` context points to the local Minikube/Kind cluster. *Airflow worker needs `kubectl` installed and configured.*
    * [ ] Upload DAG to Airflow.

2.  **IAM Permissions:** (Mainly EC2 instance role needs ECR push access).

3.  **Testing CI/CD DAG:**
    * [ ] Trigger deployment DAG. Monitor logs.
    * [ ] Verify image push to ECR and `kubectl apply` success. Check K8s deployment rollout status on the local cluster.

**Phase 5: Drift Monitoring & Retraining Loop on AWS (Weeks 9-11)**

1.  **Monitoring Script:** (No significant changes, ensure it reads/writes from/to S3 and connects to MLflow server)
    * [ ] Create script (`/scripts/monitor_drift.py`) using `argparse`, `evidently`, `mlflow`.
    * [ ] Load data/model/transformer from S3/MLflow.
    * [ ] Calculate Data Drift & Concept Drift.
    * [ ] Save reports/metrics to S3.
    * [ ] Determine drift status and output for XComs.

2.  **Airflow DAG for Simulation & Retraining Loop:**
    * [ ] Create DAG file (`/dags/monitoring_retraining_dag.py`).
    * [ ] Define schedule/logic to iterate through S3 batches.
    * [ ] Get current production model URI from MLflow.
    * [ ] **Loop:**
        * Run `monitor_drift.py`.
        * Use `BranchPythonOperator` based on drift status.
        * If drift, trigger `training_pipeline_dag` using `TriggerDagRunOperator`.
        * Update model reference for next loop iteration (promote in MLflow). Optionally trigger `deployment_pipeline_dag`.
    * [ ] Upload DAG to Airflow.

3.  **Visualization Script:** (No changes needed here)
    * [ ] Create script (`/scripts/visualize_drift.py`) to read metrics from S3 and generate plots using Matplotlib/Seaborn.

4.  **Testing:**
    * [ ] Trigger monitoring DAG. Verify S3 outputs, logging, and conditional triggering of training/deployment DAGs. Run visualization script.

**Phase 6: Documentation, Finalization & AWS Showcase (Weeks 12-13)**

1.  **Code Documentation:** (No changes needed here)
    * [ ] Add docstrings and comments.

2.  **README.md:**
    * [ ] Write Overview/Goals for "Health Predict".
    * [ ] **Update System Architecture Diagram:** Show EC2 hosting Airflow, MLflow, Postgres, Minikube/Kind. Show S3 and ECR.
    * [ ] **Update AWS Setup Section:** Detail IaC for EC2/S3/ECR/VPC/IAM.
    * [ ] **Add MLOps Tools Setup on EC2:** Explain `docker-compose.yml` for Airflow/MLflow/Postgres. Explain Minikube/Kind installation and startup.
    * [ ] Update Data Setup & Drift Simulation Section. Embed plots.
    * [ ] **Update How to Run Section:** Instructions for triggering DAGs. How to interact with the API via EC2 IP/NodePort.
    * [ ] **Emphasize How to Teardown Section:** Clear instructions for `terraform destroy` and **stopping the EC2 instance** to save costs.
    * [ ] List Project Structure.

3.  **IaC Code Finalization:** (No changes needed here)
    * [ ] Ensure Terraform/CDK code is clean and commented.

4.  **Video Walkthrough (Recommended):**
    * [ ] Record video demonstrating: IaC apply, tool setup on EC2 (Docker Compose, Minikube/Kind start), pipeline execution, API test (via SSH/curl), drift simulation, plots, **and crucial `terraform destroy`/EC2 stop**.

5.  **Final Review & Submission:**
    * [ ] Review all deliverables against requirements.
    * [ ] Clean repository. Commit and push.
    * [ ] Prepare submission package.
