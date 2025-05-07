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
