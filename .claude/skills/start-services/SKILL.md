---
name: start-services
description: Start all MLOps services on EC2. Use when the user wants to bring up services, start the stack, spin things up, or check if services need to be started.
---

# Start All MLOps Services

## Primary Action

Run the startup script, which handles the full startup sequence (Minikube, ECR secret, K8s manifests, Docker Compose) with retries and health checks:

```
./scripts/start-mlops-services.sh
```

Monitor the output for any `[ERROR]` lines. The script will exit non-zero if something fails.

### Script Flags

- `--rebuild` — Rebuild all Docker images before starting (use after code changes to services)
- `--reset` — Full reset: removes Docker volumes (destroys Postgres data, MLflow experiments, Airflow history) and rebuilds everything from scratch

## Verification

After the script completes successfully, it prints a service info table. Confirm:

1. All health checks passed (look for `[SUCCESS]` lines)
2. Service URLs are accessible:
   - Airflow UI: http://localhost:8080 (admin/admin)
   - MLflow UI: http://localhost:5000
   - Dashboard: http://localhost:8501
   - JupyterLab: http://localhost:8888
3. K8s pod status: `kubectl get pods` — pod may show `ErrImagePull` if no model image has been deployed yet (this is expected and resolves after the first DAG run or CI/CD push)

## Troubleshooting

If the script fails, diagnose based on the error output:

### Minikube won't start / network errors
```
minikube delete --all --purge
minikube start --driver=docker --cpus=2 --memory=3900MB --force
```
Then rerun `./scripts/start-mlops-services.sh`.

### ContainerConfig errors (docker-compose v1 bug)
If you see `KeyError: 'ContainerConfig'`, compose tried to recreate a dependency container and corrupted it. Recovery:
```
cd /home/ubuntu/health-predict/mlops-services
docker compose --env-file /home/ubuntu/health-predict/.env down
docker compose --env-file /home/ubuntu/health-predict/.env up -d
```
The `pgdata` volume survives `down`/`up`, so no data is lost.

**Prevention:** Never run `docker-compose up -d <service>` without `--no-deps` when targeting a single service.

### ECR secret creation failed
Check that AWS credentials are valid:
```
aws sts get-caller-identity
```
If credentials are fine, create the secret manually:
```
ECR_TOKEN=$(aws ecr get-login-password --region us-east-1)
kubectl create secret docker-registry ecr-registry-key \
  --docker-server=692133751630.dkr.ecr.us-east-1.amazonaws.com \
  --docker-username=AWS --docker-password="${ECR_TOKEN}"
```
Note: ECR tokens expire after 12 hours. If the cluster has been running for a while, delete and recreate the secret:
```
kubectl delete secret ecr-registry-key
```
Then recreate with the commands above.

### Airflow not healthy after startup
The scheduler can take 30-60 seconds after container start. Wait and retry:
```
curl -s http://localhost:8080/health
```

### Partial failure — only one service is down
Restart just that service without touching dependencies:
```
cd /home/ubuntu/health-predict/mlops-services
docker compose --env-file /home/ubuntu/health-predict/.env up -d --no-deps <service>
```
If the container is corrupted, force-remove it first:
```
docker rm -f <container_name>
docker compose --env-file /home/ubuntu/health-predict/.env up -d --no-deps <service>
```
