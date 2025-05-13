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
