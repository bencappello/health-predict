## 2025-05-15

- **Verified `mlops-services/docker-compose.yml`:**
    - Confirmed that the `airflow-scheduler` service has the correct volume mount for Kubeconfig: `- /home/ubuntu/.kube:/home/airflow/.kube:ro`.
    - Confirmed environment variables `HOME=/home/airflow` and `KUBECONFIG=/home/airflow/.kube/config`.
    - This setup should allow the Python Kubernetes client (`kubernetes.config.load_kube_config()`) to locate and use the Kubeconfig.
- **Next Steps:**
    - User will ensure host Kubeconfig is correct and accessible.
    - User will perform a clean restart of Docker Compose services.
    - User will unpause and trigger the `health_predict_api_deployment` DAG to test the refactored Kubernetes tasks.

## 2025-05-15 (Continued)

- **Git Workflow**: 
    - Added `K8S_SERVICE_NAME` to `env_vars` in `mlops-services/dags/deployment_pipeline_dag.py` to prevent KeyError during service URL retrieval.
    - Committed and pushed this fix along with other pending changes (related to previous refactoring of K8s tasks to use Python client and journal updates).
    - Commit message: `fix: Add K8S_SERVICE_NAME to DAG env_vars`
- **Addressed DAG Execution Problem**:
    - Confirmed `kubernetes` Python package is present in `mlops-services/Dockerfile.airflow`.
    - Confirmed `deployment_pipeline_dag.py` already incorporates the Python Kubernetes client for `update_kubernetes_deployment` and `verify_deployment_rollout` tasks, aligning with the previous strategy.
    - Performed a clean restart of Docker Compose services (`docker-compose down -v --remove-orphans && docker system prune -af && docker-compose up -d --build airflow-scheduler airflow-webserver postgres mlflow`) to ensure a fresh environment.
- **Next Steps (User)**:
    - User to verify host Kubeconfig (`/home/ubuntu/.kube/config`) is correct and accessible.
    - User to unpause and trigger the `health_predict_api_deployment` DAG in the Airflow UI.
    - User to monitor the DAG run and report back with results/logs.

## 2025-05-15: API Deployment DAG - ECR Auth & Pytest

*   Successfully resolved `ImagePullBackOff` for the API deployment in Kubernetes.
    *   Ensured host Docker was logged into ECR.
    *   Created a Kubernetes secret `ecr-registry-key` from the host's Docker `config.json`.
    *   Updated `k8s/deployment.yaml` to use `imagePullSecrets` with `ecr-registry-key`.
    *   Applied the updated deployment, leading to successful image pull and pod readiness.
*   The `verify_deployment_rollout` task in the `health_predict_api_deployment` DAG succeeded after the ECR auth fix.
*   Addressed `pytest: No module named pytest` error in the `run_api_tests` task.
    *   Added `pytest` to `mlops-services/Dockerfile.airflow`.
    *   Initiated a rebuild of Airflow services to include `pytest`.
*   The `health_predict_api_deployment` DAG is now progressing, with the `run_api_tests` task being the current focus after the image rebuild. 