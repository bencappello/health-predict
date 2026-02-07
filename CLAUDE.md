# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Health Predict is a production-grade MLOps system for predicting patient hospital readmission risk using the UCI Diabetes dataset. It implements a complete ML lifecycle: data ingestion, drift detection, hyperparameter-optimized training, quality-gated deployment, and continuous monitoring.

## Architecture

### Pipeline Flow

The core pipeline is a single Airflow DAG (`health_predict_continuous_improvement`) that runs:
1. **Drift detection** (Evidently AI, KS-test + chi-squared) — gating, controls whether retraining proceeds
2. **Drift gate** (BranchPythonOperator) — if drift detected or `force_retrain=True`, proceed; else skip to end
3. **Data preparation** — cumulative learning across all previous batches
4. **HPO training** (Ray Tune + ASHA scheduler) — trains XGBoost/RF/LR, logs to MLflow
5. **Model evaluation** — AUC, precision, recall, F1 on validation set
6. **Regression guardrail** — compares new model vs production on held-out test; branches to deploy or skip
7. **Deployment** — register in MLflow → Docker build → ECR push → Kubernetes rolling update → version verification

### Key Source Locations

- `src/feature_engineering.py` — Data cleaning, feature engineering, preprocessing pipeline (StandardScaler + OneHotEncoder)
- `src/api/main.py` — FastAPI prediction service (`/health`, `/model-info`, `/predict` endpoints)
- `src/drift/utils.py` — Batch splitting, cumulative learning, quality gate helpers
- `scripts/train_model.py` — Ray Tune HPO training orchestrator
- `scripts/split_data.py` — Partitions raw data into 20% initial + 80% future batches
- `scripts/create_drift_aware_batches.py` — Creates 5 batches with intentional drift profiles from future_data
- `mlops-services/dags/health_predict_continuous_improvement.py` — Production DAG (the other DAGs in that directory are legacy)
- `k8s/deployment.yaml` — Kubernetes Deployment + NodePort Service
- `iac/main.tf` — Terraform for AWS (EC2, S3, ECR, VPC)

### Data Flow

Raw data on S3 → `split_data.py` creates initial train/val/test + numbered batches → each DAG run loads all batches up to current (cumulative) → feature engineering → training → model artifacts stored in MLflow → deployed API loads model from MLflow at startup.

### Service Stack (Docker Compose in `mlops-services/`)

| Service    | Port | Notes                          |
|------------|------|--------------------------------|
| Airflow UI | 8080 | Credentials: admin/admin       |
| MLflow     | 5000 | Experiment tracking & registry |
| Dashboard  | 8501 | Streamlit monitoring dashboard |
| JupyterLab | 8888 | Token in container logs        |
| PostgreSQL | 5432 | Backing store for Airflow + MLflow |
| API (K8s)  | 31780| After deployment via Minikube  |

## Common Commands

### Start/Stop MLOps Services
```bash
./scripts/start-mlops-services.sh              # Start all services
./scripts/start-mlops-services.sh --rebuild     # Rebuild containers
./scripts/start-mlops-services.sh --reset       # Fresh start
./scripts/stop-mlops-services.sh
```

### Run Tests
```bash
pytest tests/api/test_api_endpoints.py -v       # API endpoint tests
pytest tests/drift/test_batch_split.py -v       # Drift utility tests
pytest tests/integration/test_drift_retraining_flow.py -v  # Integration tests
```

Test dependencies: `pytest`, `requests` (see `tests/requirements.txt`).

### Trigger Training Pipeline
```bash
# Via Airflow CLI (inside scheduler container):
docker exec mlops-services-airflow-scheduler-1 \
  airflow dags trigger health_predict_continuous_improvement \
  --conf '{"batch_number": 1}'
```

### Data Preparation
```bash
python scripts/split_data.py --bucket-name <S3_BUCKET> \
  --raw-data-key raw_data/diabetic_data.csv
```

### Infrastructure
```bash
cd iac && terraform init && terraform plan && terraform apply
```

## Key Configuration

- `.env` — AWS credentials, S3 bucket name, MLflow URI, Airflow DB connection, ECR registry, drift thresholds
- S3 bucket pattern: `health-predict-mlops-{suffix}`
- MLflow experiment name: `HealthPredict_Training_HPO_Airflow`
- Model name in registry: `HealthPredictModel`
- Regression threshold: `-0.02` (max 2% AUC regression before blocking deploy)
- Drift threshold: `0.30` (30% drift share to trigger retraining)
- HPO config: `RAY_NUM_SAMPLES=4`, `RAY_MAX_EPOCHS=3`
- `force_retrain` DAG config parameter overrides drift gate

## Dependencies

There is no unified `requirements.txt` or `pyproject.toml`. Dependencies are split across:
- `src/api/requirements.txt` — API service (FastAPI, XGBoost, MLflow, scikit-learn)
- `scripts/requirements-training.txt` — Training (Ray Tune, Evidently, XGBoost, MLflow)
- `tests/requirements.txt` — Testing (pytest, requests)
- Dockerfiles in `mlops-services/` define container-specific dependencies

No linter, formatter, or pre-commit hooks are configured.

## Troubleshooting

### Service Startup Order
**Minikube must start BEFORE docker-compose.** The `docker-compose.yml` declares the `minikube` network as `external: true`. If docker-compose starts first, it will fail because the network doesn't exist. If minikube restarts while docker-compose is running, it recreates the Docker network, killing all containers on it. The start script (`scripts/start-mlops-services.sh`) handles the correct order.

### ECR Image Pull Secret
The K8s deployment requires an `ecr-registry-key` secret to pull images from ECR. This secret:
- Is created by `scripts/start-mlops-services.sh` during the K8s setup phase
- **Expires after 12 hours** (ECR token lifetime) — must be refreshed for long-running clusters
- Must be manually created if minikube is started outside the start script:
  ```bash
  ECR_TOKEN=$(aws ecr get-login-password --region us-east-1)
  kubectl create secret docker-registry ecr-registry-key \
    --docker-server=692133751630.dkr.ecr.us-east-1.amazonaws.com \
    --docker-username=AWS --docker-password="${ECR_TOKEN}"
  ```
- Missing this secret causes `verify_deployment` to fail with `ImagePullBackOff`

### Recovering Failed DAG Runs
If only late-stage tasks fail (e.g., `verify_deployment`), you can clear those tasks instead of re-running the full pipeline:
```bash
docker exec mlops-services-airflow-scheduler-1 \
  airflow tasks clear health_predict_continuous_improvement \
  -t "task_regex" -s START_DATE -e END_DATE --yes
```
**Warning**: The `-s`/`-e` date range applies across ALL DAG runs in that window. If multiple runs exist, tasks will be cleared from all of them. The DAG has `max_active_runs=1`, so only one run executes at a time — cleared runs queue behind the active one.

### Airflow CLI Notes
- Airflow 2.8.1 does not have `airflow tasks logs` — use filesystem logs at `/opt/airflow/logs/dag_id=.../run_id=.../task_id=.../attempt=N.log`
- Filter noisy output: append `2>&1 | grep -v -E "DeprecationWarning|FutureWarning|RemovedInAirflow"` to CLI commands

### Docker-Compose v1 ContainerConfig Bug (CRITICAL)
**NEVER run `docker-compose up -d <service>` without `--no-deps` when adding or rebuilding a single service.** Docker-compose v1 on this machine has a `KeyError: 'ContainerConfig'` bug that corrupts containers when it tries to recreate dependencies. This kills postgres, which takes down MLflow and Airflow.

**Safe pattern for single-service operations:**
```bash
# Rebuild and restart ONE service without touching others:
docker-compose up -d --no-deps --build dashboard

# If you hit ContainerConfig anyway, force-remove the broken container first:
docker rm -f <container_name>
docker-compose up -d --no-deps <service>
```

**If postgres or multiple containers are corrupted (ContainerConfig errors on several services):**
```bash
docker-compose down          # Clean remove all containers
docker-compose up -d         # Fresh start everything
```
This is safe — the `pgdata` volume persists across `down`/`up` so no data is lost.

**Root cause:** docker-compose v1's `get_container_data_volumes` fails to read metadata from containers that were partially recreated. The `--no-deps` flag prevents compose from touching dependency containers, avoiding the bug entirely.

### Minikube Instability
Minikube on this EC2 instance can be flaky. If the container dies during startup:
```bash
minikube delete --all --purge
minikube start --driver=docker --cpus=2 --memory=2048MB --force
```
This may need multiple attempts. Always stop docker-compose services first.

### verify_deployment Task
This task (DAG line ~1028) does more than check rollout status — it also queries the deployed API's `/model-info` endpoint and verifies the model version matches what was promoted in MLflow. If the API is slow to start (readiness probe has 45s initial delay), the 3 retry attempts with 5s delays should cover it.

## Git Commits

- **Author**: All commits must be authored as `Ben Cappello <bencappello@gmail.com>` — never use any other name or email
- **No Co-Authored-By**: Do NOT add `Co-Authored-By` trailers to commit messages. Every commit should appear as solely authored by Ben Cappello
- Git config is already set: `user.name = "Ben Cappello"`, `user.email = "bencappello@gmail.com"`

## Conventions

- The target variable is `readmitted_binary` (binary 0/1 derived from the original multi-class `readmitted` column)
- Feature engineering always follows the sequence: `clean_data` → `engineer_features` → `preprocess_data`
- The API loads its model from MLflow using `MODEL_NAME` and `MODEL_STAGE` environment variables at startup
- Kubernetes readiness/liveness probes hit the `/health` endpoint
- Drift reports are HTML files saved to `s3://<bucket>/drift_monitoring/reports/`
- Drift detection is gating: if drift_share < 0.30, retraining is skipped (use `force_retrain: true` to override)
- Batches 1-2 have no intentional drift; batches 3-5 have progressively stronger drift profiles
- High-cardinality columns (`diag_1`, `diag_2`, `diag_3`) are excluded from drift analysis to reduce noise

## CI/CD with GitHub Actions

### Workflow

The `.github/workflows/ci-cd.yml` workflow runs on a self-hosted runner on the EC2 instance. Three jobs: `test` → `build-and-push` → `deploy`.

### Trigger Workflow

```bash
# Full pipeline: build + deploy
gh workflow run ci-cd.yml -f deploy_to_k8s=true

# Build only (no deploy)
gh workflow run ci-cd.yml

# With integration tests
gh workflow run ci-cd.yml -f run_integration_tests=true -f deploy_to_k8s=true

# Monitor run
gh run watch

# View failed logs
gh run view --log-failed
```

### Self-Hosted Runner Management

```bash
# Setup
./scripts/setup-github-runner.sh

# Check status
sudo ~/actions-runner/svc.sh status

# Restart
sudo ~/actions-runner/svc.sh stop && sudo ~/actions-runner/svc.sh start

# Remove
./scripts/remove-github-runner.sh
```

### Dual-Pipeline Architecture

- **GitHub Actions**: Software CI/CD — triggered by code changes (push/PR) or `workflow_dispatch`
- **Airflow DAG**: ML CI/CD — triggered by data events (new batch) or manual DAG trigger
- Both pipelines can deploy to the same Kubernetes cluster independently
- GitHub Actions uses image tag format `gh-{sha:7}-{timestamp}`
- Airflow DAG uses image tag format `v{model_version}-{timestamp}`
