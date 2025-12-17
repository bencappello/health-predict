# Screenshots Directory

This directory contains screenshots documenting the Health Predict MLOps system.

## Available Screenshots

### Airflow
- `airflow_dag_graph.png` - DAG graph view showing all tasks and dependencies
- `airflow_batch5_success.png` - Successful Batch 5 run completion

### MLflow
- `mlflow_experiments.png` - List of experiments (Training HPO, Drift Monitoring)
- `mlflow_model_registry.png` - Model registry showing versioning
- `mlflow_drift_metrics.png` - Drift monitoring metrics

### API
- `api_swagger_ui.png` - Swagger documentation at /docs
- `api_model_info.png` - /model-info endpoint response showing version
- `api_health_check.png` - /health endpoint response

### Kubernetes
- `k8s_deployment_status.png` - kubectl get deployments,pods,svc
- `k8s_pod_logs.png` - API pod logs showing model loading

### Drift Reports
- `drift_report_batch5.png` - Evidently HTML report screenshot
- `drift_metrics_comparison.png` - Drift trends across batches

### Model Verification
- `model_verification_logs.png` - verify_deployment task logs showing version match

## How to Capture

To capture your own screenshots:

1. **Airflow**: Navigate to http://localhost:8080, screenshot DAG graph view
2. **MLflow**: Navigate to http://localhost:5000, screenshot experiments page
3. **API**: Navigate to http://$(minikube ip):31780/docs
4. **Kubernetes**: Run `kubectl get all` and screenshot terminal
5. **Drift**: Download HTML from S3, open in browser

## Embedding in Documentation

Use markdown image syntax with absolute paths:
```markdown
![Description](screenshots/filename.png)
```

Or for better control:
```markdown
<img src="screenshots/filename.png" alt="Description" width="800"/>
```
