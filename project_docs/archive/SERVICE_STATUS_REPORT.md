# Service Status Report — 2026-02-06

## Summary

All MLOps services were brought up and the main DAG (`health_predict_continuous_improvement`) was triggered with `batch_number=2`. The entire ML pipeline succeeded (drift detection, data prep, HPO training, evaluation, model registration, Docker build, test, ECR push, K8s deploy). **The DAG failed at the `verify_deployment` task** because Kubernetes couldn't pull the Docker image from ECR due to a missing image pull secret.

---

## Current Service State

| Service | Status | Port | Notes |
|---------|--------|------|-------|
| PostgreSQL | Running (healthy) | 5432 | Backing store for Airflow + MLflow |
| MLflow | Running | 5000 | HTTP 200 |
| Airflow Webserver | Running | 8080 | HTTP 200, credentials: admin/admin |
| Airflow Scheduler | Running | — | Processing DAGs |
| JupyterLab | Running (healthy) | 8888 | |
| Minikube | Running | — | Node Ready, K8s v1.34.0 |

All Docker Compose services are up. Minikube is running with a single-node cluster.

---

## DAG Run Details

**Run ID**: `manual__2026-02-06T22:33:13+00:00`
**Overall State**: `failed`
**Duration**: ~12 minutes (22:35:05 to 22:46:54)

### Task Results (in execution order)

| Task | State | Duration | Notes |
|------|-------|----------|-------|
| `run_drift_detection` | success | ~10s | Evidently AI drift analysis on batch 2 |
| `prepare_drift_aware_data` | success | ~2s | Cumulative learning dataset prepared |
| `run_training_and_hpo` | success | ~2 min | Ray Tune HPO (1 sample, 1 epoch, fast mode) |
| `evaluate_model_performance` | success | ~1s | Best XGBoost model selected from MLflow |
| `compare_against_production` | success | ~11s | Decision: DEPLOY (new model passed quality gate) |
| `deployment_decision_branch` | success | instant | Routed to DEPLOY path |
| `check_kubernetes_readiness` | success | ~1s | Minikube cluster verified accessible |
| `register_and_promote_model` | success | ~1s | Model registered as HealthPredictModel v10, promoted to Production |
| `build_api_image` | success | ~2 min | Docker image built: `health-predict-api:v10-1770417468` |
| `test_api_locally` | success | ~20s | API pytest passed inside Docker container |
| `push_to_ecr` | success | ~1.5 min | Image pushed to `692133751630.dkr.ecr.us-east-1.amazonaws.com/health-predict-api` |
| `deploy_to_kubernetes` | success | ~1s | `kubectl set image` updated the deployment |
| **`verify_deployment`** | **FAILED** | ~5 min | **Rollout timed out — pod stuck in ImagePullBackOff** |
| `post_deployment_health_check` | upstream_failed | — | Skipped due to verify_deployment failure |
| `notify_deployment_success` | upstream_failed | — | Skipped |
| `end` | upstream_failed | — | Skipped |

---

## Root Cause: `verify_deployment` Failure

### Error Message

```
AirflowFailException: Deployment verification failed: Rollout status failed. Exit code: 1, stderr: error: timed out waiting for the condition
```

### What Happened

1. The `deploy_to_kubernetes` task ran `kubectl set image` to update the K8s deployment with the new ECR image tag (`v10-1770417468`). This succeeded — it just updates the deployment spec.

2. The `verify_deployment` task then ran `kubectl rollout status deployment/health-predict-api-deployment --timeout=300s`. This waits for the new pod to become ready.

3. The new pod was created but could **not pull the image from ECR**. It entered `ImagePullBackOff`:

   ```
   Warning  FailedToRetrieveImagePullSecret  kubelet  Unable to retrieve some image pull secrets (ecr-registry-key)
   Warning  Failed  kubelet  Failed to pull image "692133751630.dkr.ecr.us-east-1.amazonaws.com/health-predict-api:v10-1770417468": no basic auth credentials
   ```

4. After 300 seconds the rollout status timed out and the task failed.

### Why the ECR Secret Is Missing

The K8s deployment spec (`k8s/deployment.yaml`) references `imagePullSecrets: [name: ecr-registry-key]`. This secret must be created in the `default` namespace with valid ECR credentials. It is normally created by the start script (`scripts/start-mlops-services.sh`) during the K8s setup phase.

**In this session**, the start script failed because minikube had corrupted state and wouldn't start. Minikube was manually recovered using:

```bash
minikube delete --all --purge
minikube start --driver=docker --cpus=2 --memory=2048MB --force
```

This gave us a clean minikube cluster, but the ECR secret creation step from the start script was never executed.

### How to Fix

Create the ECR image pull secret in K8s:

```bash
# Get ECR login token
ECR_TOKEN=$(aws ecr get-login-password --region us-east-1)

# Create K8s secret
kubectl create secret docker-registry ecr-registry-key \
  --docker-server=692133751630.dkr.ecr.us-east-1.amazonaws.com \
  --docker-username=AWS \
  --docker-password="${ECR_TOKEN}" \
  --namespace=default
```

Then either re-trigger the DAG or manually restart the deployment:

```bash
kubectl rollout restart deployment/health-predict-api-deployment
```

**Note**: ECR tokens expire after 12 hours, so this secret needs to be refreshed periodically.

---

## Minikube Instability Issue

Minikube was extremely flaky during this session. The Docker container would consistently die ~10 seconds after startup during the certificate setup phase. The pattern:

1. Minikube creates the Docker container
2. Provisions SSH keys and Docker TLS certs inside the container
3. Restarts the Docker daemon inside the container with new TLS config
4. Continues setting up kubelet, containerd, certificates...
5. **~10 seconds in, the container completely disappears** (not just stops — removed from Docker)

The verbose logs (`minikube start -v=5`) show the exact failure point: the SSH session gets an EOF at `scp /home/ubuntu/.minikube/ca.crt --> /var/lib/minikube/certs/ca.crt`, and then the container is gone.

**What eventually worked**: Full purge + `--force` flag:

```bash
minikube delete --all --purge
minikube start --driver=docker --cpus=2 --memory=2048MB --force
```

This was non-deterministic — it failed multiple times before succeeding.

**Important**: When minikube starts, it recreates the `minikube` Docker network. The docker-compose.yml declares this network as `external: true`, so Docker Compose services connected to it will lose connectivity and die. **Never run `minikube start` while Docker Compose services are running.** Always start minikube first, then docker-compose.

---

## Key Configuration Reference

- **S3 Bucket**: `health-predict-mlops-f9ac6509`
- **ECR Registry**: `692133751630.dkr.ecr.us-east-1.amazonaws.com/health-predict-api`
- **MLflow Experiment**: `HealthPredict_Training_HPO_Airflow`
- **Model Name**: `HealthPredictModel` (currently at v10, Production stage)
- **K8s Deployment**: `health-predict-api-deployment` in `default` namespace
- **K8s Service**: `health-predict-api-service` (NodePort 31780)
- **Docker image tag format**: `v{model_version}-{timestamp}`
- **Latest image**: `health-predict-api:v10-1770417468` (successfully pushed to ECR)

## Files Involved

- `scripts/start-mlops-services.sh` — Start script that handles minikube, K8s setup (including ECR secret), and docker-compose
- `mlops-services/docker-compose.yml` — Docker Compose service definitions
- `mlops-services/dags/health_predict_continuous_improvement.py` — The main DAG (line ~1220 is where `verify_deployment` fails)
- `k8s/deployment.yaml` — K8s Deployment + Service (references `imagePullSecrets: ecr-registry-key`)
- `.env` — Environment variables (AWS creds, S3 bucket, ECR registry, etc.)
