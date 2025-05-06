## Health Predict MLOps Project - Overall Plan (Cost-Optimized AWS & Drift Simulation)

**1. Project Goal:**
Develop and deploy a robust, end-to-end Machine Learning Operations (MLOps) system on AWS infrastructure to predict patient readmission, incorporating automated drift detection and retraining. This project serves as a portfolio piece demonstrating proficiency in the full ML lifecycle, cloud infrastructure management (cost-consciously), and production-readiness practices for an ML Engineering role, while adhering to a strict budget.

**2. Core Strategy:**
* **MLOps Focus:** Prioritize demonstrating the complete MLOps workflow: data management, scalable training (HPO), experiment tracking, model packaging, automated deployment (CI/CD), API serving, model monitoring, and automated retraining.
* **Drift Simulation:** Leverage the dataset's time dimension to simulate incoming production data batches, implement monitoring for both data and concept drift, and automate model retraining based on drift detection. Visualize drift and retraining events over time.
* **Cost-Optimized AWS Integration:** Utilize core AWS services (EC2, S3, ECR) for essential infrastructure. **Run Kubernetes (Minikube/Kind) and PostgreSQL (Docker container) directly on the EC2 instance** to minimize costs associated with managed services (EKS, RDS).
* **Infrastructure as Code (IaC):** Employ Terraform (or AWS CDK) to define, provision, and manage the core AWS infrastructure (VPC, EC2, S3, ECR, IAM), showcasing modern cloud management.
* **Aggressive Resource Management:** Emphasize the critical importance of **stopping EC2 instances and destroying infrastructure (`terraform destroy`)** when not actively working to stay within budget.
* **Pragmatic Modeling:** Use standard ML models suitable for tabular data; the focus is on the surrounding MLOps system.
* **Demonstrability:** Ensure the project is well-documented, reproducible (via IaC and scripts), and effectively showcased through documentation, visualizations, and potentially a video walkthrough, including clear setup/teardown instructions.

**3. Key Phases:**
* **Phase 1: Foundation, Cloud Setup & Exploration:** Provision core AWS infrastructure (VPC, EC2, S3, ECR) via IaC, set up MLOps tools (Airflow, MLflow, PostgreSQL) and local Kubernetes (Minikube/Kind) via Docker Compose on EC2, prepare data on S3, perform initial EDA, and establish baseline model.
* **Phase 2: Scalable Training & Tracking on AWS:** Develop reusable feature engineering, implement scalable training with HPO (RayTune) on EC2, integrate comprehensive experiment tracking with MLflow (using S3 backend), and orchestrate initial training via Airflow (using local PostgreSQL backend).
* **Phase 3: API Development & Deployment to Local K8s:** Develop a REST API (FastAPI/Flask) to serve the model, containerize the API, push the image to ECR, and deploy it to the local Kubernetes cluster (Minikube/Kind) running on the EC2 instance.
* **Phase 4: CI/CD Automation using AWS Resources:** Create an Airflow DAG to automate the build (Docker image) and deployment (to local K8s on EC2) process, triggered manually or after model retraining.
* **Phase 5: Drift Monitoring & Retraining Loop on AWS:** Implement data and concept drift detection using Evidently, orchestrate a simulation loop in Airflow to process data batches, trigger retraining based on drift, and log metrics for visualization.
* **Phase 6: Documentation, Finalization & AWS Showcase:** Create comprehensive documentation (README, architecture diagram, setup/teardown guides), generate drift visualizations, record a video walkthrough, ensure IaC code is clean, and package the project for submission.

**4. Key Technologies:**
* **Languages/Libraries:** Python, Pandas, Scikit-learn, Matplotlib/Seaborn
* **MLOps Tools:** Airflow, MLflow, RayTune, Evidently AI
* **API/Containerization:** FastAPI/Flask, Docker, Docker Compose
* **Cloud/Deployment:** AWS (EC2, S3, ECR, VPC, IAM), Kubernetes (Minikube/Kind running on EC2), Terraform/AWS CDK
* **Database:** PostgreSQL (running in Docker on EC2)
* **CI/CD:** Airflow (triggering build/deploy steps), Git/GitHub

**5. Deliverables:**
* **GitHub Repository:** Containing all code (Python scripts, notebooks, API code, IaC scripts, Dockerfile, K8s manifests, Docker Compose files).
* **Infrastructure as Code:** Terraform/CDK scripts to provision core AWS resources (EC2, S3, ECR, VPC, IAM).
* **Deployed API (Transient):** A temporarily running API endpoint on the local K8s cluster on EC2 for demonstration (documented, shown in video).
* **Comprehensive Documentation:** README.md with architecture, setup (including tool installation on EC2), usage instructions, **explicit teardown steps**; code comments.
* **Drift Visualizations:** Plots showing data drift, concept drift, and retraining events over simulated time.
* **Experimentation Notebook:** Initial EDA, preprocessing, and baseline model exploration.
* **(Recommended) Video Walkthrough:** Demonstrating IaC deployment, tool setup on EC2, pipeline execution, drift detection/retraining, API interaction, and **resource teardown**.
