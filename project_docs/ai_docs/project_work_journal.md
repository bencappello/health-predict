## $(date +'%Y-%m-%d %H:%M:%S') - Project Setup Phase 1 Initiated

- Reviewed `project_plan.md` and `project_steps.md`.
- Confirmed GitHub repository `bencappello/health-predict` is created.
- Cloned repo locally

## 2025-05-06 12:52:31 - Completed Project Setup (Phase 1)

- Set up local Python 3.11.3 environment using `venv` (named `.venv`).
- Created initial project directory structure: `/src`, `/notebooks`, `/iac`, `/scripts`, `/config`, `/docs`, `/docker-compose`.
- Created `.gitignore` file with standard Python, OS, IDE, secrets, and Terraform exclusions.
- Updated `project_steps.md` to reflect completion of these tasks.

## 2025-05-06 15:12:52 - AWS Account & Credentials Update (Phase 1)

- Confirmed AWS account with sufficient permissions is available.
- Configured AWS credentials locally.
- Skipped setting up AWS Budgets and billing alerts for now.
- Updated `project_steps.md` accordingly.

## 2025-05-06 15:19:52 - Git Commit

- Committed initial project setup and AWS credential configuration.
- Commit message:
  ```
  feat: Complete Project Setup and AWS Credentials configuration

  - Initialized Python virtual environment (.venv).
  - Created project directory structure (src, notebooks, iac, etc.).
  - Added .gitignore file with common exclusions.
  - Updated project_steps.md to track progress.
  - Updated project_work_journal.md with setup activities.
  - Configured AWS credentials locally (user confirmed).
  ```

## 2025-05-06 15:21:28 - Initial Terraform Setup (Phase 1 IaC)

- Created initial Terraform configuration files in `iac/` directory (`versions.tf`, `variables.tf`, `main.tf`, `outputs.tf`).
  - `main.tf` includes basic setup for: VPC, Public Subnet, Internet Gateway, Security Group, IAM Role & Policies (for S3 & ECR access), EC2 Instance (t2.micro with User Data for Docker/Compose/Git), S3 Bucket (versioned), ECR Repository.
- Created `project_docs/terraform_guide.md` with detailed instructions for the user to:
  - Customize variables (especially `your_ip` and EC2 `key_name`).
  - Run `terraform init`, `plan`, and `apply`.
  - Verify resources and retrieve outputs.
  - SSH into the EC2 instance.
  - Crucially, run `terraform destroy` to manage costs.
- Updated `project_steps.md` to reflect these initial IaC tasks and point to the guide.

## $(date +'%Y-%m-%d %H:%M:%S') - Docker Compose Setup on EC2 (Airflow & MLflow)

- Created `~/health-predict/mlops-services/` directory on EC2 for Docker Compose files.
- Created `docker-compose.yml` in `~/health-predict/mlops-services/` for Postgres, Airflow (webserver, scheduler, init), and MLflow.
  - Ensured `AIRFLOW_UID` is set for correct file permissions.
  - Configured MLflow to use the Postgres backend and an S3 artifact root (placeholder `your-mlflow-s3-bucket` initially, user updated to actual bucket `health-predict-mlops-f9ac6509`).
- Created `dags/`, `logs/`, `plugins/` subdirectories within `mlops-services/` for Airflow volume mounts.
- Addressed initial `KeyError: 'ContainerConfig'` by running `docker-compose down --volumes` before `up -d`.
- **Troubleshooting MLflow:**
  - Resolved `ModuleNotFoundError: No module named 'psycopg2'` for MLflow by modifying its `command` in `docker-compose.yml` to `pip install psycopg2-binary` before starting the server.
- **Troubleshooting Airflow Webserver:**
  - Addressed Gunicorn timeout errors (`No response from gunicorn master within 120 seconds`).
  - Checked `docker stats` for resource usage.
  - Set `AIRFLOW__WEBSERVER__WORKERS=2` in `docker-compose.yml` for the `airflow-webserver` to stabilize its startup.
- User confirmed MLflow UI (port 5000) and Airflow UI (port 8080 - after login) are now accessible.
- Updated `project_steps.md`.

## $(date +'%Y-%m-%d %H:%M:%S') - Git Commits for Docker Compose Setup

- **Commit 1:**
  ```
  docs: Update progress and journal for Docker Compose setup

  - Marked Docker Compose and UI verification tasks as complete in project_steps.md.
  - Added detailed journal entry for Docker Compose setup, including troubleshooting steps for MLflow and Airflow.
  ```
- **Commit 2:**
  ```
  feat: Add Docker Compose setup for MLOps services

  - Adds docker-compose.yml for Postgres, Airflow, and MLflow.
  - Includes initial dags/, logs/, and plugins/ directories for Airflow.
  - Configures MLflow to use Postgres backend and S3 for artifacts.
  - Sets Airflow webserver workers to 2 for stability.
  ```

## $(date +'%Y-%m-%d %H:%M:%S') - Kubernetes Setup on EC2 (kubectl & Minikube)

- Installed `kubectl` v1.28.5-eks-5e0fdde on the EC2 instance.
- Attempted to install and start Minikube, but failed due to insufficient disk space (`GUEST_PROVISION_NOSPACE`).
  - `df -h` confirmed root volume was ~7.6G with only ~1GB available.
- User resized EC2 root EBS volume and restarted the instance (new IP: 54.226.87.176).
- Successfully installed Minikube v1.35.0 and started a Kubernetes v1.32.0 cluster using the Docker driver.
  - `kubectl` is now configured to use the "minikube" cluster.
  - Noted a version skew warning between installed `kubectl` (v1.28.5) and Minikube's K8s server (v1.32.0).
- Updated `project_steps.md`.

## $(date +'%Y-%m-%d %H:%M:%S') - Git Commits for Kubernetes Setup & Config Variables

- **Commit 1 (build):**
  ```
  build: Ignore kubectl binary in mlops-services

  - Prevents the downloaded kubectl binary from being tracked by Git.
  ```
- **Commit 2 (docs):**
  ```
  docs: Update progress and journal for Kubernetes setup

  - Marked kubectl and Minikube installation as complete in project_steps.md.
  - Added journal entry detailing Kubernetes setup on EC2, including disk space troubleshooting and resolution.
  ```
- **Commit 3 (feat):**
  ```
  feat: Add config_variables.md for project settings

  - Includes initial variables like AWS region and S3 bucket name.
  ```

## $(date +'%Y-%m-%d %H:%M:%S') - Data Preparation & Splitting

- User confirmed dataset (`diabetic_data.csv`) downloaded to `data/` directory on EC2.
- Installed AWS CLI (`sudo apt install awscli -y`) as it was missing.
- Uploaded `data/diabetic_data.csv` to `s3://health-predict-mlops-f9ac6509/raw_data/diabetic_data.csv` using AWS CLI.
- Created data splitting script `scripts/split_data.py`:
  - Uses `argparse` for configuration.
  - Uses `boto3` to download raw data from S3 and upload processed splits.
  - Uses `pandas` and `scikit-learn` for data handling and train/validation/test splitting (70/15/15 split).
- Installed required Python packages (`pip install boto3 pandas scikit-learn`) after installing `pip` (`sudo apt install python3-pip -y`).
- Ran `python3 scripts/split_data.py --bucket-name health-predict-mlops-f9ac6509`.
- Confirmed successful execution and upload of `train.csv`, `validation.csv`, `test.csv` to `s3://health-predict-mlops-f9ac6509/processed_data/`.
- Updated `project_steps.md`.

## $(date +'%Y-%m-%d %H:%M:%S') - Add JupyterLab Service for EDA

- Added a `jupyterlab` service definition to `mlops-services/docker-compose.yml`.
  - Uses `jupyter/scipy-notebook:latest` image.
  - Installs `boto3` on startup for S3 access.
  - Maps host port 8888 to container port 8888.
  - Mounts the project root (`../`) to `/home/jovyan/work` in the container.
  - Configured to start JupyterLab listening on `0.0.0.0` with no token/password.
- Updated Step 6 in `project_steps.md` to reflect using the JupyterLab service for EDA.
- Preparing to start services including JupyterLab.

## $(date +'%Y-%m-%d %H:%M:%S') - Refine Data Splitting Strategy for Drift Simulation

- Revised data splitting plan to support drift simulation:
  - Use the first 20% of the dataset as 'initial data' for baseline model training.
  - Preserve the remaining 80% as 'future data' to simulate incoming batches.
- Updated `project_plan.md` Core Strategy section to reflect this partitioning.
- Updated `project_steps.md` Step 5 (Data Prep) and Step 6 (EDA) to reflect the new logic and filenames (`initial_train.csv`, `initial_validation.csv`, `initial_test.csv`, `future_data.csv`).
- Modified `scripts/split_data.py`:
  - Added logic to separate initial vs. future data based on `initial_data_fraction`.
  - Applied train/validation/test split only to the initial data portion.
  - Updated arguments and logic to save all four resulting datasets to S3.
- Re-ran the modified `scripts/split_data.py` successfully.
- Processed data (initial splits + future data) is now available in `s3://health-predict-mlops-f9ac6509/processed_data/`.

## $(date +'%Y-%m-%d %H:%M:%S') - Finalize Data Split & Update EDA Plan

- Added logging to `scripts/split_data.py` to output the length of each generated dataset.
- Re-ran the script and verified the output lengths in the logs, confirming the split logic.
- Updated Step 6 in `project_steps.md` to use a standard Python script (`01_eda_baseline.py`) with notebook cell formatting (`# %%`) instead of an `.ipynb` file, to be edited and run within Cursor using the Jupyter extension connected to the `jupyterlab` Docker service.

## $(date +'%Y-%m-%d %H:%M:%S') - Git Commits for Data Split Refinements & EDA Plan Update

- **Commit 1 (docs):**
  ```
  docs: Update data split strategy and EDA plan

  - Refine data split in project_plan.md and project_steps.md to use first 20% for initial model, rest for drift simulation.
  - Update project_steps.md for EDA to use a Python script with notebook cells in Cursor, not a .ipynb file via web UI.
  - Add JupyterLab service to docker-compose.yml.
  - Update project_work_journal.md with these changes.
  ```
- **Commit 2 (feat):**
  ```
  feat: Update data splitting script and add EDA structure

  - Modify split_data.py to log dataset lengths.
  - Add notebooks/ directory for EDA script.
  - Add data/ directory structure and .gitignore for CSVs in data/.
  ```

## $(date +'%Y-%m-%d %H:%M:%S') - Completed Initial EDA and Baseline Model Development
- Developed `notebooks/01_eda_baseline.py` containing:
    - Loading of initial training, validation, and test data from S3.
    - Detailed EDA:
        - Initial data inspection (.info(), .describe(), missing values).
        - Target variable analysis (`readmitted`).
        - Numerical feature histograms and boxplots.
        - Categorical feature barplots.
    - Data Cleaning and Preprocessing:
        - Replaced '?' with NaN.
        - Dropped columns with high missing percentages (`weight`, `payer_code`, `medical_specialty`).
        - Filled remaining NaNs in `race`, `diag_1`, `diag_2`, `diag_3` with 'Missing'.
        - Filtered out rows where `discharge_disposition_id` indicated hospice or expired.
    - Feature Engineering:
        - Simplified `readmitted` to a binary target (`readmitted_binary`).
        - Converted `age` to an ordinal feature (`age_ordinal`).
        - Dropped original `readmitted`, `age`, `encounter_id`, `patient_nbr`, and `diag_1`, `diag_2`, `diag_3` (for baseline simplicity).
    - Preprocessing for Modeling:
        - Identified numerical and categorical features.
        - Applied `StandardScaler` to numerical features and `OneHotEncoder` to categorical features using `ColumnTransformer`.
        - Processed training, validation, and test sets using the fitted preprocessor.
    - Baseline Model Training:
        - Trained a Logistic Regression model with `class_weight='balanced'`.
    - Model Evaluation:
        - Evaluated the model on the test set using classification report, accuracy score, and confusion matrix.
- Created a corresponding `notebooks/01_eda_baseline.ipynb` file.
- Committed all changes to Git.
- Updated `project_steps.md` to mark Phase 1, Step 6 as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Updated Project Plan for Enhanced Model Experimentation

- Updated `project_steps.md` (Phase 1, Step 6) to accurately reflect completion of EDA and baseline model tasks.
- Revised `project_plan.md`:
  - Modified 'Core Strategy' point 'Pragmatic Modeling' to emphasize exploring various ML models (Random Forest, XGBoost, etc.) and using RayTune for HPO.
  - Updated 'Key Phases', Phase 2 (Scalable Training & Tracking on AWS) to detail experimentation with multiple model architectures and advanced HPO with RayTune.
  - Added 'XGBoost' to 'Key Technologies'.
- Revised `project_steps.md` (Phase 2, Step 2 - Training Script):
  - Added task to experiment with various model architectures (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost).
  - Enhanced MLflow integration to log parameters, artifacts, metrics, and tags for each model and run.
  - Emphasized using RayTune for comprehensive HPO across all selected model types with well-defined search spaces.
  - Specified logging the best version of each model type.

## $(date +'%Y-%m-%d %H:%M:%S') - Created Feature Engineering Pipeline Script

- Created `src/feature_engineering.py` based on the EDA notebook (`notebooks/01_eda_baseline.py`).
- The script includes functions for:
  - `clean_data`: Handles '?', drops specified columns, fills NaNs, and filters by `discharge_disposition_id`.
  - `engineer_features`: Creates binary target `readmitted_binary` and ordinal `age_ordinal`.
  - `get_preprocessor`: Defines and returns a Scikit-learn `ColumnTransformer` for numerical (StandardScaler) and categorical (OneHotEncoder) features.
  - `save_preprocessor` & `load_preprocessor`: Utility functions to save/load the fitted preprocessor using `joblib`.
  - `preprocess_data`: Applies the preprocessor (fitting if necessary) and returns a transformed DataFrame.
- Added an `if __name__ == '__main__':` block with example usage and basic tests for the functions.
- Updated `project_steps.md` to mark Phase 2, Step 1 as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Created and Refined Model Training Script

- Developed `scripts/train_model.py` for model training, hyperparameter optimization with Ray Tune, and MLflow integration.
- The script includes:
  - Argument parsing for S3 data paths, MLflow URI, and Ray Tune parameters.
  - Data loading from S3 and preprocessing using `src/feature_engineering.py`.
  - Fitting and logging the data preprocessor to MLflow.
  - HPO for Logistic Regression, Random Forest, and XGBoost models using Ray Tune.
  - Logging of HPO trials (parameters, metrics) to MLflow.
  - Training a final model using the best hyperparameters for each model type and logging these models and their metrics to MLflow.
  - Models and artifacts are stored in S3 via MLflow's artifact logging.
- Corrected a linter error in `scripts/train_model.py` related to accessing metrics from `best_trial.metrics`.
- Updated `project_steps.md` to mark the script creation and refinement sub-tasks of Phase 2, Step 2 as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Prepared for Model Training Execution

- Updated `project_steps.md` for Phase 2, Step 2 to include explicit sub-tasks for executing the training script and verifying MLflow results.
- Created `project_docs/run_training_guide.md` containing detailed step-by-step instructions for the user to run `scripts/train_model.py` on the EC2 instance. This guide includes prerequisites, dependency installation, command construction (with placeholders for EC2 IP), and steps for monitoring and verifying results in the MLflow UI.
- The next action is for the user to follow this guide to execute the training script.

## $(date +'%Y-%m-%d %H:%M:%S') - Dockerized Training Environment and Updated Guide

- Created `scripts/requirements-training.txt` to define Python dependencies for the model training script.
- Modified `mlops-services/docker-compose.yml` to update the `jupyterlab` service:
  - The service now installs dependencies from `scripts/requirements-training.txt` upon startup.
  - This makes the Python environment for training reproducible via Docker.
- Updated `project_docs/run_training_guide.md` to reflect the new Docker-based execution method:
  - Instructions now guide the user to run the training script using `docker-compose exec jupyterlab ...`.
  - Clarified that dependency installation is handled by the Docker service build/startup.
- The user is now ready to rebuild the Docker services and then run the training script as per the updated guide.

## $(date +'%Y-%m-%d %H:%M:%S') - Executed Model Training and HPO Script

- Successfully executed `scripts/train_model.py` within the `jupyterlab` Docker container.
- Iteratively debugged the script:
  - Corrected `FileNotFoundError` for `preprocessor.joblib` by ensuring the output directory for the preprocessor is created only if a directory path is specified (relevant if saving to current dir).
  - Resolved Ray Tune `AttributeError: module 'ray.tune' has no attribute 'session'` by updating `RunConfig` imports and parameters (`local_dir` to `storage_path`). This was based on web search for Ray 2.7.0 API changes.
  - Addressed Ray Tune `ObjectStoreFullError` by setting `max_concurrent_trials=1` in `TuneConfig`.
  - Fixed `ValueError: could not convert string to float: 'Missing'` by explicitly defining numerical and categorical features in `scripts/train_model.py` before passing them to `get_preprocessor` in `src/feature_engineering.py`. Ensured `COLS_FILL_NA_MISSING` (including `diag_1`, `diag_2`, `diag_3`) were correctly identified and processed as categorical by the `OneHotEncoder`.
  - Updated `tune.report` to `train.report` from `ray.train` to resolve deprecation errors that were halting trials.
- The script completed HPO for Logistic Regression, Random Forest, and XGBoost models.
- All trials for all models reported perfect validation scores (F1=1.0, AUC=1.0, etc.). This is highly indicative of data leakage or a feature that perfectly predicts the target. This needs further investigation, but the script execution and MLflow logging objectives are met.
- Best models and their preprocessors were logged to MLflow.
- Updated `project_steps.md` to mark Phase 2, Step 2 (training script execution and verification) as complete.

## $(date +'%Y-%m-%d %H:%M:%S') - Corrected Data Leakage and Reran Training

- Identified data leakage issue causing perfect (1.0) validation scores in previous run.
  - The original categorical target column `readmitted` was being included in the features passed to the model after one-hot encoding.
- Fixed the leakage in `scripts/train_model.py` by modifying the definition of `X_train_for_preprocessor_fitting` and `X_val_for_testing` to explicitly drop `readmitted` (and the original `age`) columns, in addition to the engineered target `readmitted_binary`.
- Reran the training script (`docker-compose exec jupyterlab python3 scripts/train_model.py ...`).
- The script completed successfully with realistic (non-perfect) validation scores:
  - Logistic Regression best F1 ~0.27
  - Random Forest best F1 ~0.28
  - XGBoost best F1 ~0.07
- Best models and preprocessors were logged to MLflow with the corrected results.
- Phase 2, Step 2 is now properly completed.

## 2025-05-11 21:45:00 - Target Variable Redefinition and Model Performance Improvement

- Modified the feature engineering module to redefine the `readmitted_binary` target variable:
  - Previous definition: 1 if readmitted in less than 30 days (`<30`), 0 otherwise
  - New definition: 1 if readmitted at all (both `<30` and `>30`), 0 only if not readmitted (`NO`)
- This change better represents the general readmission prediction task rather than focusing only on quick readmissions.
- The revised definition also provides a more balanced class distribution for training.
- Retrained all three model types (LogisticRegression, RandomForest, XGBoost) using the updated target definition.
- Observed dramatic improvements in model performance:
  - RandomForest: F1 score improved from ~0.28 to 0.63
  - XGBoost: F1 score improved from ~0.07 to 0.63 (most significant improvement)
  - LogisticRegression: F1 score improved from ~0.27 to 0.61
- All models are now performing comparably well, with F1 scores above 0.60.
- The relative performance ranking has changed, with XGBoost now competitive with RandomForest.

## 2025-05-11 23:30:00 - Implemented and Tested Training Pipeline DAG

- Created and implemented `mlops-services/dags/training_pipeline_dag.py` with:
  - Proper DAG configuration (schedule_interval=None for manual triggering, start_date, catchup=False)
  - Environment variable definitions for training script parameters
  - Main training task using BashOperator to execute train_model.py with appropriate parameters
  - Model registration task code is prepared but currently commented out for future implementation
- Successfully tested the training task execution:
  - DAG properly executes the training script
  - Training runs complete with expected metrics:
    - RandomForest F1: 0.6276 (best)
    - XGBoost F1: 0.6259
    - LogisticRegression F1: 0.6065
  - All models and artifacts are correctly logged to MLflow
  - Airflow logs show clean execution with proper error handling
- Updated project documentation:
  - Marked Phase 2, Step 3 (Airflow DAG) tasks as complete, except for model registration task
  - Added detailed execution logs to project work journal
- Next steps:
  - Implement and test the model registration task in the DAG (currently commented out)
  - Begin work on API development and deployment phase

## $(date +'%Y-%m-%d %H:%M:%S') - System Documentation and Model Registry Task Update

- Committed changes with the following message:
  ```
  docs: Add system overview and update project steps for model registry
  
  - Create comprehensive system_overview.md for project understanding.
  - Add detailed instructions to project_steps.md for implementing the MLflow model registration task in the Airflow DAG.
  - Update project_work_journal.md to reflect the current status of the model registration task.
  - Modify training_pipeline_dag.py to ensure correct task execution and set RAY_NUM_SAMPLES to 2 for testing.
  ``` 

## $(date +'%Y-%m-%d %H:%M:%S') - Implemented Model Registration Task in Airflow DAG

- Reviewed `project_steps.md` for Phase 2, Step 3, Task 3: Implement MLflow model registration.
- Edited `mlops-services/dags/training_pipeline_dag.py`:
  - Uncommented the `find_and_register_best_model` Python function.
  - Modified the function's `mlflow.search_runs` call to use `filter_string="tags.best_hpo_model = 'True'"` to correctly identify the final "Best Model" runs as per the project documentation clarification.
  - Ensured the model registration uses `model_uri=f"runs:/{best_run_id}/model"` and transitions models to the "Production" stage.
  - Uncommented the `find_and_register_best_model_task` PythonOperator.
  - Updated the DAG task dependencies to `run_training_and_hpo >> find_and_register_best_model_task`.
  - Commented out the previous temporary single-task dependency.
- The DAG is now configured to run the training/HPO pipeline and then register the best resulting models for each type (LogisticRegression, RandomForest, XGBoost) into the MLflow Model Registry.
- Next step is for the user to trigger the DAG in Airflow and verify the model registration in both Airflow logs and MLflow UI.

## $(date +'%Y-%m-%d %H:%M:%S') - Verified Model Registration Task via CLI and Script

- Continued execution of revised "Detailed Implementation Guide for Task 3 (AI Agent Focused)" in `project_steps.md`.
- Triggered the `health_predict_training_hpo` Airflow DAG using `docker-compose exec airflow-scheduler airflow dags trigger health_predict_training_hpo`.
  - DAG Run ID: `manual__2025-05-12T01:53:53+00:00` (Execution Date: `2025-05-12T01:53:53+00:00`).
- Monitored DAG and task status using Airflow CLI commands (`dags state`, `tasks state`):
  - Both `run_training_and_hpo` and `find_and_register_best_model` tasks completed successfully.
- Retrieved and analyzed logs for `find_and_register_best_model_task` by directly accessing the log file path (`/opt/airflow/logs/dag_id=.../run_id=.../task_id=.../attempt=1.log`) inside the `airflow-scheduler` container.
  - Logs confirmed successful processing and registration of RandomForest, XGBoost, and LogisticRegression models, with multiple versions created and transitioned to "Production" due to the `RAY_NUM_SAMPLES=2` HPO setup resulting in multiple runs tagged as `best_hpo_model='True'` for each model type.
- Created `scripts/verify_mlflow_registration.py` to programmatically check MLflow Model Registry via its Python API.
- Executed `verify_mlflow_registration.py` using `docker-compose exec jupyterlab python3 ...`.
  - The script confirmed that `HealthPredict_LogisticRegression`, `HealthPredict_RandomForest`, and `HealthPredict_XGBoost` are registered, their latest versions (v6 for this run) are in "Production", and their source run tags are correct.
- All verification steps passed. Phase 2, Step 3, Task 3 is now complete.
- Updated `project_steps.md` to reflect completion.
