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
    * [x] **Implement HPO (RayTune):** Utilize RayTune for comprehensive hyperparameter optimization across the selected model architectures. Ensure search spaces are well-defined for each model type. *(Default number of trials temporarily reduced to 2 for faster debugging iterations)*
    * [x] Log the best version of each model type using `mlflow.sklearn.log_model()` or equivalent for other frameworks (artifacts go to S3).
    * [x] Execute `scripts/train_model.py` on the EC2 instance with appropriate parameters (S3 paths, MLflow URI, Ray Tune settings) to perform HPO and log all experiments and final models to MLflow (artifacts stored on S3 via MLflow).
    * [x] Verify successful script execution by checking logs and MLflow UI for logged parameters, metrics, preprocessor, and the best model for each algorithm type.

3.  **Airflow DAG for Training & HPO:**
    * [x] Create DAG file (`/dags/training_pipeline_dag.py`).
    * [x] Define DAG schedule/args.
    * [x] Task 1: Optional data split.
    * [x] **Task 2 (`BashOperator`/`PythonOperator`):** Execute `train_model.py`. Pass MLflow tracking URI pointing to the MLflow container on EC2 (e.g., `http://<mlflow-service-name>:5000` or `http://localhost:5000` depending on network mode/execution context). Ensure EC2 role allows S3 access.
    * [x] **Task 3:** Use MLflow client API to find and register the best model in MLflow Model Registry.
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

2.  **Containerization:** This step involves creating a Dockerfile to package the FastAPI application, its dependencies, and necessary configurations into a portable Docker image.
    * [x] **Create Dockerfile (`/Dockerfile` at project root):** *(Note: Path adjusted to project root for robust build context)*
        *   **Base Image:** Start with an official Python base image that matches the version used for development (e.g., `FROM python:3.10-slim`). Choose a slim variant for a smaller image size. *(Implemented)*
        *   **Set Working Directory:** Define a working directory inside the container (e.g., `WORKDIR /app`). *(Implemented)*
        *   **Environment Variables (Optional but Recommended):**
            *   Set `PYTHONUNBUFFERED=1` to ensure Python output (like logs) is sent straight to the terminal without being buffered, which is helpful for Docker logging. *(Implemented)*
            *   Consider setting `MLFLOW_TRACKING_URI` if it's fixed or as a default, though it's often better to configure this at runtime via Kubernetes environment variables. *(Commented out in Dockerfile, to be set at runtime)*
        *   **Copy Requirements File:** Copy the `requirements.txt` file into the working directory (e.g., `COPY src/api/requirements.txt .`). *(Implemented)*
        *   **Install Dependencies:** Install the Python dependencies using pip (e.g., `RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt`).
            *   `--no-cache-dir` prevents pip from storing downloaded packages in a cache, reducing image size. *(Implemented)*
            *   Upgrading pip first is a good practice. *(Implemented)*
        *   **Copy Application Code:** Copy the entire `/src` directory into `/app/src` in the image to ensure all modules, including `feature_engineering.py`, are available (e.g., `COPY src ./src`). *(Implemented, ensures `src.api.main` and `src.feature_engineering` are accessible)*
            *   **Important:** Create a `.dockerignore` file in the project root to exclude unnecessary files and directories from being copied into the image (e.g., `__pycache__`, `.git`, `.venv`, `tests/`, etc.). This is crucial for security, build speed, and image size. *(Implemented at project root)*
        *   **Expose Port:** Inform Docker that the container listens on a specific network port at runtime (e.g., `EXPOSE 8000`, assuming Uvicorn runs on port 8000). This is documentation; the actual port mapping happens during `docker run` or in Kubernetes service definition. *(Implemented)*
        *   **Command to Run Application:** Specify the default command to run when the container starts (e.g., `CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`).
            *   Ensure the path to `main:app` is correct. With `WORKDIR /app` and `COPY src ./src`, `src.api.main:app` correctly points to `/app/src/api/main.py`. *(Implemented)*
            *   Using `0.0.0.0` as the host makes the application accessible from outside the container. *(Implemented)*

3.  **Build and Push to ECR:** This step involves building the Docker image defined by the Dockerfile and pushing it to Amazon Elastic Container Registry (ECR), making it available for deployment in Kubernetes. This should be done on the EC2 instance where Docker is installed and configured with AWS credentials.
    * [x] **Authenticate Docker with ECR on EC2:** *(Implemented)*
        *   **Retrieve ECR Login Command:** Use the AWS CLI to get a temporary Docker login command for your ECR registry. *(Implemented)*
            *   Command: `aws ecr get-login-password --region <your-aws-region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com`
            *   Replace `<your-aws-region>` (e.g., `us-east-1`) and `<your-aws-account-id>` with your actual AWS account ID.
            *   This command retrieves a password and pipes it to `docker login`.
        *   **Verification:** A "Login Succeeded" message should appear. *(Implemented)*
    * [x] **Build Docker Image on EC2:** *(Implemented)*
        *   **Navigate to Dockerfile Directory:** Change to the directory containing your `Dockerfile` (e.g., `cd /home/ubuntu/health-predict/src/api` if Dockerfile is there, or `/home/ubuntu/health-predict` if Dockerfile is at root and uses appropriate COPY paths). The path specified in `docker build` for the context (`.` in the example below) should contain the Dockerfile and all files it needs to `COPY`. *(Project root `/home/ubuntu/health-predict` used as build context)*
        *   **Define Image Name and Tag:** Choose a meaningful name and tag for your image. *(Implemented as `536474293413.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`)*
            *   ECR image URI format: `<your-aws-account-id>.dkr.ecr.<your-aws-region>.amazonaws.com/<your-ecr-repo-name>:<tag>`
            *   Example: `123456789012.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest` or `123456789012.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:v0.1.0`.
            *   Ensure `<your-ecr-repo-name>` matches the ECR repository created by Terraform.
        *   **Run Docker Build Command:** *(Implemented)*
            *   Command: `docker build -t <your-full-ecr-image-uri> .`
            *   Example: `docker build -t 123456789012.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest .`
            *   The `-t` flag tags the image. The `.` at the end specifies the build context (current directory).
        *   **Verification:** Monitor the build process. It should complete without errors. You can list local images using `docker images` to see your newly built image. *(Implemented, build successful)*
    * [x] **Push Docker Image to ECR:** *(Implemented)*
        *   **Run Docker Push Command:** *(Implemented)*
            *   Command: `docker push <your-full-ecr-image-uri>`
            *   Example: `docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`
        *   **Verification:** Monitor the push process. It will upload the image layers to ECR. *(Implemented, push successful)*
        *   You can verify the image in the AWS ECR console for your repository. *(User to verify in AWS console)*

4.  **Kubernetes Deployment (Targeting Local K8s on EC2):** This step focuses on deploying the containerized API to the local Kubernetes cluster (Minikube/Kind) running on the EC2 instance. This involves creating Kubernetes manifest files for a Deployment and a Service.
    * [x] **Ensure `kubectl` on EC2 is configured to talk to the local Minikube/Kind cluster:**
        *   [x] **Verify Current Context:** Run `kubectl config current-context` on the EC2 instance. It should show the context for your Minikube or Kind cluster (e.g., `minikube` or `kind-kind`).
        *   [x] **Check Cluster Info:** Run `kubectl cluster-info` to confirm connectivity to the local cluster master.
        *   [x] **(If needed) Set Context:** If `kubectl` is not pointing to the correct local cluster, use `kubectl config use-context <your-local-cluster-context-name>`.
    * [x] **Create/Modify Kubernetes Manifests (`/k8s/deployment.yaml`):**
        *   [x] **Create `k8s` directory:** If it doesn't exist, create a `k8s` directory in your project root (`mkdir k8s`).
        *   [x] **Create `deployment.yaml`:** Inside the `k8s` directory, create a file named `deployment.yaml`. This file will contain definitions for both the Deployment and the Service.
        *   [x] **Define `Deployment`:**
            *   `apiVersion: apps/v1`
            *   `kind: Deployment`
            *   `metadata:`
                *   `name: health-predict-api-deployment` (or a similar descriptive name)
                *   `labels:`
                    *   `app: health-predict-api`
            *   `spec:`
                *   `replicas: 2` (or your desired number of pod replicas for availability/load handling)
                *   `selector:`
                    *   `matchLabels:`
                        *   `app: health-predict-api` (must match the pod template labels)
                *   `template:` (Pod template)
                    *   `metadata:`
                        *   `labels:`
                            *   `app: health-predict-api` (pods will have this label)
                    *   `spec:`
                        *   `containers:`
                            *   `- name: health-predict-api-container`
                            *   `image: <your-full-ecr-image-uri>` (Replace with the ECR image URI pushed in the previous step, e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:latest`)
                            *   `ports:`
                                *   `- containerPort: 8000` (the port your FastAPI app listens on inside the container)
                            *   `env:` (Optional: Define environment variables for the container)
                                *   `- name: MLFLOW_TRACKING_URI`
                                *   `value: "http://<EC2-Private-IP>:5000"` (or `http://mlflow:5000` if using Docker Compose network and K8s can resolve it. For Minikube/Kind on EC2, accessing host services directly might require specific network configuration or using the EC2's private IP. This needs careful consideration based on K8s networking setup on EC2. Alternatively, the MLflow server might be exposed differently to K8s pods.)
                                *   *(Add other necessary environment variables like model name, stage if not hardcoded in API)*
                            *   `resources:` (Optional but highly recommended for production)
                                *   `limits:`
                                    *   `cpu: "1"` (1 CPU core)
                                    *   `memory: "512Mi"` (512 Megabytes)
                                *   `requests:`
                                    *   `cpu: "0.5"` (0.5 CPU core)
                                    *   `memory: "256Mi"`
        *   **Define `Service` (in the same `deployment.yaml` file, separated by `---`):**
            *   `apiVersion: v1`
            *   `kind: Service`
            *   `metadata:`
                *   `name: health-predict-api-service`
            *   `spec:`
                *   `selector:`
                    *   `app: health-predict-api` (must match the labels of the pods created by the Deployment)
                *   `ports:`
                    *   `- protocol: TCP`
                    *   `port: 80` (the port the service will be available on *within* the K8s cluster)
                    *   `targetPort: 8000` (the port on the pods that the service will forward traffic to)
                *   `type: NodePort` (Exposes the service on each Node's IP at a static port (the `NodePort`). Makes the service accessible from outside the cluster using `<NodeIP>:<NodePort>`.)
                    *   Alternatively, for Minikube, `type: LoadBalancer` might work if a load balancer addon like MetalLB is enabled (`minikube tunnel` can also expose LoadBalancer services).
        *   [x] **Permissions/Networking Notes for Local K8s on EC2:**
            *   **MLflow Access:** The primary challenge is how pods in the local K8s cluster (Minikube/Kind) access the MLflow service running via Docker Compose on the same EC2 host.
                *   **Option 1 (EC2 Private IP):** If MLflow port (5000) is exposed on the EC2 instance's network interface (0.0.0.0:5000), pods might be able to reach it via the EC2's private IP address. This requires the EC2 security group to allow traffic on port 5000 from the EC2 instance itself (or its internal Docker/K8s network ranges).
                *   **Option 2 (Host IP via Minikube/Kind):** Minikube/Kind might provide a way to access the host's network (e.g., `host.minikube.internal` for Minikube). This needs to be verified for the specific local K8s setup.
                *   **Option 3 (Shared Docker Network - Advanced):** If Kind/Minikube can be configured to use the same Docker network as the Docker Compose services, direct service name resolution (e.g., `http://mlflow:5000`) might work. This is less straightforward.
            *   **S3 Access:** If the EC2 instance has an IAM role with S3 permissions, pods running on that instance within Minikube/Kind *might* inherit these permissions if the local K8s setup allows it (often the case with Docker-based drivers). No explicit K8s secret for AWS credentials might be needed for S3 in this specific local setup, but this should be tested.
    * [x] **Apply Manifests:**
        *   [x] **Navigate to Manifests Directory:** `cd /home/ubuntu/health-predict/k8s` (or wherever `deployment.yaml` is).
        *   [x] **Run `kubectl apply`:** `kubectl apply -f deployment.yaml` on the EC2 instance.
        *   [x] **Verification:** Look for output like `deployment.apps/health-predict-api-deployment created` and `service/health-predict-api-service created`.
    * [x] **Verify Deployment:**
        *   [x] **Check Pods:** `kubectl get pods -l app=health-predict-api`. Wait for pods to be in `Running` state. Check logs of a pod if there are issues: `kubectl logs <pod-name> -c health-predict-api-container`.
        *   [x] **Check Deployment Status:** `kubectl rollout status deployment/health-predict-api-deployment`. It should report successful rollout.
        *   [x] **Check Service:** `kubectl get svc health-predict-api-service`.
            *   Note the `TYPE` (should be `NodePort`).
            *   Note the `PORT(S)`. It will show something like `80:<NodePort>/TCP`. The `<NodePort>` is a high-numbered port (e.g., 30000-32767) assigned by Kubernetes.
        *   [x] **Determine Access Point:** To access the service from the EC2 instance itself (or externally if EC2 security group allows the NodePort), you'll use the EC2 instance's IP address and the assigned `NodePort`.
            *   Minikube users can also try `minikube service health-predict-api-service --url` to get an accessible URL, which might use a tunnel.

5.  **API Testing (Automated & Manual):** This step ensures the deployed API is functioning correctly through a combination of automated tests and targeted manual verification. The primary focus is on written, repeatable tests.
    * [x] **Setup Python Testing Environment (if not already done for other tests):**
        *   [x] Ensure `pytest` and `requests` (or `httpx` for async tests) are added to your development dependencies (e.g., a `requirements-dev.txt` or equivalent).
            *   Example `requirements-dev.txt` line: `pytest==8.2.2`, `requests==2.32.3`, `httpx==0.27.0`
        *   [x] Install these dependencies in your local/EC2 development environment where you will run the tests from.
    * [x] **Identify API Base URL for Testing:**
        *   [x] Determine the base URL of the deployed API service. This will be `http://<EC2-IP>:<NodePort>` as identified in the previous Kubernetes deployment verification step (Step 4).
        *   [x] It's recommended to make this configurable, perhaps via an environment variable for the test suite (e.g., `API_BASE_URL`).
    * [x] **Create API Test File Structure:**
        *   [x] Create a directory for tests if it doesn't exist, e.g., `tests/` at the project root.
        *   [x] Inside `tests/`, create a subdirectory for API tests, e.g., `tests/api/`.
        *   [x] Create a test file, e.g., `tests/api/test_api_endpoints.py`.
    * [x] **Write Automated Tests for `/health` Endpoint (`tests/api/test_api_endpoints.py`):**
        *   [x] Import necessary libraries (`pytest`, `requests` or `httpx`).
        *   [x] Define a test function for the `/health` endpoint (e.g., `test_health_check`):
            *   Send a GET request to `/health` (e.g., `API_BASE_URL + "/health"`).
            *   Assert that the HTTP status code is 200.
            *   Assert that the response body (JSON) contains `{"status": "ok"}`.
            *   Optionally, assert that the message indicates the model is loaded (this might require the API to be fully up and model loaded before tests run, or have a separate test for post-model-load state).
    * [x] **Write Automated Tests for `/predict` Endpoint (`tests/api/test_api_endpoints.py`):**
        *   [x] Define multiple test functions for the `/predict` endpoint to cover different scenarios:
            *   **Valid Input Test (e.g., `test_predict_valid_input`):**
                *   Prepare a valid sample JSON payload (similar to the one used for manual `curl` testing, but defined within the test function or loaded from a test data file).
                *   Send a POST request to `/predict` with the valid payload.
                *   Assert that the HTTP status code is 200.
                *   Assert that the response body (JSON) contains the expected keys (`prediction`, `probability_score`).
                *   Assert that `prediction` is an integer (0 or 1).
                *   Assert that `probability_score` is a float between 0.0 and 1.0.
            *   **Invalid Input Test - Missing Required Field (e.g., `test_predict_missing_field`):**
                *   Prepare a JSON payload that is missing one or more required fields defined in `InferenceInput`.
                *   Send a POST request to `/predict`.
                *   Assert that the HTTP status code is 422 (Unprocessable Entity), which FastAPI typically returns for Pydantic validation errors.
            *   **Invalid Input Test - Incorrect Data Type (e.g., `test_predict_invalid_data_type`):**
                *   Prepare a JSON payload where a field has an incorrect data type (e.g., a string where an integer is expected for `time_in_hospital`).
                *   Send a POST request to `/predict`.
                *   Assert that the HTTP status code is 422.
            *   **(Optional) Edge Case Tests:** Consider tests for edge cases specific to your data or model (e.g., all zero values for numerical inputs if relevant, specific `diag_1` codes if they have special handling).
    * [x] **Run Automated API Tests:**
        *   [x] Navigate to project root.
        *   [x] Run `pytest tests/api/`.
        *   [x] Analyze output. **Result: 3 Passed, 1 Failed (`test_predict_valid_input` - 422 Error).** Requires debugging the 422 error (payload vs Pydantic model mismatch?) and the persistent `boto3` ModuleNotFoundError preventing model load.
         * [x] **Iterate on Fixes:** If tests fail, analyze API logs (`kubectl logs <pod-name>`), debug code in `src/api/main.py`, `src/feature_engineering.py`, check model/preprocessor loading, rebuild/redeploy API image, and re-test.
        *   *Current Status:* Automated tests are failing due to preprocessing issues in the API. The training script (`train_model.py`) and API code (`main.py`) have been modified to correctly log, load, and apply the preprocessor artifact. The next step is to re-run the training DAG to ensure the artifact is logged, then rebuild/deploy/test the API.
        
        ### V1: Fix `train_model.py` MLFlow HPO child runs & final model preprocessor logging
        - Status: Mostly Complete
        - Details:
        - [X] HPO trials (`train_model_hpo`) correctly create nested MLflow runs.
        - [X] The main `train_model.py` script correctly identifies the best HPO trial for each model type.
        - [X] The script trains a final model of each type using the best HPO parameters.
        - [X] The script logs this final model to a new, non-nested MLflow run (e.g., "Best_RandomForest_Model").
        - [X] The `preprocessor.joblib` (the one fitted on the full training data before HPO) is correctly logged as an artifact to the MLflow run associated with *each* "Best\_<ModelName>\_Model" (verified for runs created by the updated script).
        - [X] ~~Ensure the `preprocessor.joblib` is logged as an artifact *with the final chosen model's MLflow run* (the one that gets registered and promoted).~~
            - **Note / Issue**: The script changes achieve this for *newly trained and selected* models. However, the *currently registered Production model* (RandomForest v17, run `33e29674486942f6a7c80e2a8322e05b` from an older DAG run) was promoted by `find_and_register_best_model` due to its F1 score, but it *does not* have the preprocessor co-located as it predates this script change. **UPDATE:** This is now resolved. The `find_and_register_best_model` function now checks for the preprocessor artifact, and a new RandomForest model (v1, run `04df414476cb4dccbf8eee97f26e7cf4`) with its co-located preprocessor has been successfully promoted to Production.
        - Next Steps: **ALL COMPLETE FOR V1**
        - ~~Decide how to handle the current Production model lacking its co-located preprocessor:~~
            - ~~Option 1: Re-run training pipeline (possibly with more HPO trials) until a new model (with co-located preprocessor) is selected for Production.~~ (DONE - New model promoted)
            - ~~Option 2: Modify `find_and_register_best_model` to *only* promote models that have the `preprocessor/preprocessor.joblib` artifact.~~ (DONE - Logic updated in DAG)
            - ~~Option 3: Accept current state for V1 and address in V2.~~
        - [X] Ensure `MLFLOW_TRACKING_URI` is consistently available/set in the `jupyterlab` container environment for reliable `mlflow` CLI usage (e.g., by setting it in its Dockerfile or as an environment variable in `docker-compose.yml`). (DONE - Added to `docker-compose.yml`)
    * [x] **Manual Testing & Verification (Human User):**
        *   User to perform exploratory testing via Postman/curl using the identified NodePort URL (e.g., `http://<EC2_Public_IP>:<NodePort>`).
        *   **Purpose:** While automated tests cover expected behavior and common errors, manual exploratory testing can uncover usability issues or unexpected interactions, especially after initial deployment or major changes.
        *   **Action (User):**
            *   Using Postman or `curl` (as detailed in the original Step 5 description), send a few varied valid requests to the `/predict` endpoint using the `http://<EC2-IP>:<NodePort>/predict` URL.
            *   Try at least 2-3 different valid patient profiles that you construct.
            *   Verify that the predictions and probability scores returned seem reasonable based on your understanding of the model and data (this is a sanity check, not a rigorous model validation).
            *   Attempt to send a deliberately malformed JSON or an unexpected input type to the `/predict` endpoint and observe if the API handles it gracefully with an appropriate error message (e.g., 400 or 422 error).
            *   Confirm the `/health` endpoint returns the expected status via browser or `curl`.
        *   **Feedback (User):** Note any unexpected behavior, confusing error messages, or usability concerns. If critical issues are found, they may warrant new automated tests or API code changes.
    * [x] **Completion Criteria:**
        *   All automated tests in `tests/api/test_api_endpoints.py` must pass when run against the deployed API.
        *   Manual testing and verification (performed by the Human User) must confirm the API is responding correctly and gracefully to valid and basic invalid inputs.

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
    *   [ ] **EC2 Instance Profile Role:** Verify the IAM role attached to your EC2 instance has sufficient permissions for:
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
