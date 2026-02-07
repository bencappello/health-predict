---
name: stop-services
description: Stop all MLOps services on EC2. Shuts down Docker Compose and Minikube cleanly.
---

# Stop All MLOps Services

## Primary Action

Run the stop script, which handles the full shutdown sequence (Docker Compose first, then Minikube) in the correct order:

```
./scripts/stop-mlops-services.sh
```

Monitor the output for any `[ERROR]` lines. The script prints a final status check confirming everything is down.

### Script Flags

- `--keep-minikube` — Stop Docker Compose services but leave Minikube running (useful if you plan to restart compose services soon without the overhead of recreating the K8s cluster)
- `--remove-volumes` — Remove Docker volumes during shutdown (**DESTRUCTIVE**: deletes Postgres data, MLflow experiments, Airflow history). The script prompts for confirmation before proceeding.

## Verification

After the script completes, confirm everything is down:

```
minikube status 2>&1 || true
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

- Minikube should report "not found" or show no profile
- No mlops-related containers should be running

## Troubleshooting

### Script hangs during docker compose down
A container may be stuck. Cancel the script (Ctrl+C) and force-remove the hung container:
```
docker rm -f <hung_container_name>
```
Then rerun `./scripts/stop-mlops-services.sh`.

### Minikube delete fails
If `minikube delete` errors out:
```
minikube delete --all --purge
```
If that also fails, force-remove the minikube Docker container:
```
docker rm -f minikube
```

### Some containers still running after script completes
Force-stop and remove them:
```
docker rm -f $(docker ps -q --filter "name=mlops-services")
```

## Partial Shutdown

To stop just one service without bringing down the whole stack:

```
cd /home/ubuntu/health-predict/mlops-services
docker compose --env-file /home/ubuntu/health-predict/.env stop <service>
```

To restart it later:
```
docker compose --env-file /home/ubuntu/health-predict/.env up -d --no-deps <service>
```

**Always use `--no-deps`** when targeting a single service to avoid the docker-compose v1 ContainerConfig bug.
