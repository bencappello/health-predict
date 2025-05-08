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
    * [ ] **Action:** Set up AWS Budgets and billing alerts for cost control (e.g., target < $30). *(Skipped for now)*

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
    * [ ] Create DAG file (`/dags/training_pipeline_dag.py`).
    * [ ] Define DAG schedule/args.
    * [ ] Task 1: Optional data split.
    * [ ] **Task 2 (`BashOperator`/`PythonOperator`):** Execute `train_model.py`. Pass MLflow tracking URI pointing to the MLflow container on EC2 (e.g., `http://<mlflow-service-name>:5000` or `http://localhost:5000` depending on network mode/execution context). Ensure EC2 role allows S3 access.
    * [ ] Task 3: Use MLflow client API to find and register the best model in MLflow Model Registry.
    * [ ] Upload DAG file to the mounted `/dags` directory on EC2.
    * [ ] Test DAG execution. Verify results in MLflow UI (artifacts on S3, metadata in local Postgres).

**Phase 3: API Development & Deployment to Local K8s (Weeks 5-6)**

1.  **API Development (FastAPI):** (No significant changes, ensure it uses MLflow client to load model/transformer from MLflow server/S3)
    * [ ] Create API code (`/src/api/main.py`) using FastAPI, Pydantic.
    * [ ] Load model/transformer on startup from MLflow (needs MLflow tracking URI and S3 access).
    * [ ] Define `/predict` and `/health` endpoints.
    * [ ] Create `requirements.txt`.

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
