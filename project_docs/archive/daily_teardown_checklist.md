# 🛡️ Daily Teardown Readiness Checklist for Health Predict MLOps

> Use this checklist **every night before logging off** to ensure you can fully recover your project environment if AWS infrastructure is accidentally deleted.

## 1. 🗂️ Code & Config Backup

- [ ] ✅ All Python scripts and Jupyter notebooks (`/src`, `/scripts`, `/notebooks`) committed and pushed to GitHub.
- [ ] ✅ `docker-compose.yml` file and any `.env` files saved in `~/mlops-services` are versioned in GitHub or copied to a safe location (e.g., S3 or local).
- [ ] ✅ Airflow DAGs in `/dags` are committed and pushed to GitHub.
- [ ] ✅ Kubernetes manifests in `/k8s` are saved and versioned.
- [ ] ✅ Any modified configuration files (e.g., Airflow, MLflow, Postgres) are saved or documented.

## 2. 🧠 Model & Data Artifact Backup

- [ ] ✅ All model artifacts and logs from MLflow are saved to S3 (ensure `--default-artifact-root` is set to an S3 URI).
- [ ] ✅ All data (raw/split/prepared) is uploaded to S3 and accessible by your EC2 instance's IAM role.
- [ ] ✅ Any `.ipynb` files with recent EDA/modeling work are committed to GitHub.

## 3. 🐳 Docker & ECR

- [ ] ✅ If a new API Docker image was built, it was pushed to ECR (`docker push <ecr_repo_url>:tag`).
- [ ] ✅ Docker volumes (especially for Postgres) are persistent in the `docker-compose.yml` (e.g., `pgdata` volume).
- [ ] ✅ Docker service configuration is fully written in `docker-compose.yml` so it can be re-run with `docker-compose up -d`.

## 4. ☁️ Terraform & Infra

- [ ] ✅ Current state of Terraform is saved (do **not** destroy unless necessary).
- [ ] ✅ If destroying infra (`terraform destroy`), ensure all above steps are done and `terraform apply` can cleanly recreate it.
- [ ] ✅ Outputs from `terraform output` are saved or copied (e.g., EC2 public IP, S3 bucket name, ECR repo URI).

## 5. 🚪 EC2 Session Safety

- [ ] ✅ EC2 instance is stopped if you're not actively working (to avoid cost).
- [ ] ✅ SSH keys are backed up and accessible (e.g., `ik-keys.pem`).
- [ ] ✅ Commands to restart services (`docker-compose up -d`) are documented.

## 6. 📝 Optional: Snapshot State
