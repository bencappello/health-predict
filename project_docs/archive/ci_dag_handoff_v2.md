# CI DAG Debugging Handoff Report

## 🚀 Current Objective
Implement a **Periodic Retraining Strategy** for the `health_predict_continuous_improvement` DAG. The goal is to move from drift-triggered retraining to always retraining on new batches, while using a **Regression Guard Rail** (-0.02 AUC) to gate deployment.

## 🔄 Strategic Shifts
1.  **Periodic Retraining**: We now retrain on *every* new batch (Batch 2, 3, 4, 5).
2.  **Integrated Drift Detection**: Drift detection (`run_drift_detection`) allows monitoring but **does not gate** the pipeline. It runs on the FULL batch before splitting.
3.  **Regression Guard Rail**: New model is deployed if `new_auc >= prod_auc - 0.02`.
4.  **Cumulative Data**: Training uses `initial_train + batch_2 + batch_3 ...`.

## ✅ Progress & Status
### Verified Components
*   **Drift Detection**: Working perfectly. Runs on full batch, logs to MLflow, saves HTML report to S3.
*   **Cumulative Data Loading**: Validated. Batch 3 run correctly loaded 20,700 rows (initial + batch 2 + batch 3).
*   **Regression & Decision Logic**: Working.
    *   **Batch 2**: Decided `SKIP` (Regression > -0.02). Correct.
    *   **Batch 3**: Decided `DEPLOY` (Regression < -0.02). Correct.

### Current Blockers
*   **DAG Execution Stability**: We encountered repeated issues with tasks hanging or failing during HPO execution, likely due to resource contention or scheduler issues with `max_active_runs=1`.
    *   Batch 4 `run_training_and_hpo` failed with `SIGTERM` (externally set to failed).
*   **Kubernetes Deployment**: The `check_kubernetes_readiness` task fails due to permission issues accessing Minikube from the Airflow container. **This is expected and low priority** for now, but creates "upstream_failed" noise.

## 📉 New Strategy: "Super Lightweight" Debugging
The user has mandated a shift to **maximize feedback speed**:
1.  **HPO Settings**: Reduce to **1 trial, 1 epoch**.
    *   *Goal*: Finish training in seconds to test the DAG flow itself.
    *   *Risk*: Poor model performance might trigger the regression guard rail (SKIP).
2.  **Force Deployment Path**: If the poor models cause constant SKIP decisions, **relax the regression threshold** (e.g., to `-0.5` or similar absurdly low value) to force the `DEPLOY` branch to execute. We need to verify the *deployment path* logic works end-to-end.

## 📝 Next Steps for New Agent
1.  **Modify DAG Config**:
    *   Update `health_predict_continuous_improvement.py` with:
        ```python
        'RAY_NUM_SAMPLES': '1',
        'RAY_MAX_EPOCHS': '1',
        'REGRESSION_THRESHOLD': '-0.5', # Optional: Relax if needed to verify Deploy branch
        ```
2.  **Clean Slate**:
    *   Clear invalid/stuck DAG runs for Batch 4 and 5.
    *   Restart the scheduler to ensure a clean state.
3.  **Execute Batch 4 & 5**:
    *   Trigger runs with new lightweight settings.
    *   Monitor for end-to-end success (ignoring K8s failure).
4.  **Verify Deploy Branch**: Ensure the DAG attempts to reach the deployment tasks.

## 📂 Relevant Files
*   `mlops-services/dags/health_predict_continuous_improvement.py`: Main DAG file.
*   `task.md`: Current progress tracker.
