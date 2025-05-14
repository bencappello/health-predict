## 2025-05-15

- **Verified `mlops-services/docker-compose.yml`:**
    - Confirmed that the `airflow-scheduler` service has the correct volume mount for Kubeconfig: `- /home/ubuntu/.kube:/home/airflow/.kube:ro`.
    - Confirmed environment variables `HOME=/home/airflow` and `KUBECONFIG=/home/airflow/.kube/config`.
    - This setup should allow the Python Kubernetes client (`kubernetes.config.load_kube_config()`) to locate and use the Kubeconfig.
- **Next Steps:**
    - User will ensure host Kubeconfig is correct and accessible.
    - User will perform a clean restart of Docker Compose services.
    - User will unpause and trigger the `health_predict_api_deployment` DAG to test the refactored Kubernetes tasks. 