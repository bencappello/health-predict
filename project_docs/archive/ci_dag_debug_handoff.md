# CI DAG Implementation & Debug Handoff

**Date:** 2025-12-16 07:14 UTC  
**Status:** Stage 4 in progress - actively debugging quality gate task  
**Next Agent:** Continue debugging and complete Stage 4-5

---

## Executive Summary

We're implementing drift-aware retraining in the `health_predict_continuous_improvement` DAG with a 5-stage incremental testing strategy. Currently in **Stage 4** (slim run with 2 HPO trials), debugging the `compare_against_production` task which has failed multiple times due to:

1. ✅ FIXED: Missing feature engineering
2. ✅ FIXED: Model loading (pyfunc → sklearn)  
3. ✅ FIXED: Categorical encoding issue
4. ⏳ TESTING: Verifying all fixes work together

**Current Run:** `manual__2025-12-16T07:02:54.723041+00:00` - compare_against_production is up_for_retry (try 1)

---

## The Plan: 5-Stage Incremental Testing

### ✅ Stage 1: Unit Tests (COMPLETE)
- Created `src/drift/utils.py` with helper functions
- Created `tests/drift/test_batch_split.py` with 11 unit tests
- **Result:** All tests passed in 3 seconds

### ✅ Stage 2: Integration Script (COMPLETE)  
- Created `tests/integration/test_drift_retraining_flow.py`
- Tested full flow with tiny datasets (100 rows)
- **Result:** All validations passed, AUC improvement 4.6%

### ✅ Stage 3: DAG Modifications (COMPLETE)
Modified `health_predict_continuous_improvement.py`:
- Added `prepare_drift_aware_data()` function (lines 66-173)
- Updated `run_training_and_hpo` to use dynamic XCom paths
- Replaced `compare_against_production()` with AUC-based version (lines 301-443)
- Updated drift monitoring DAG trigger

**Key Changes:**
```python
# Data prep - splits batch_7, creates cumulative dataset
prepare_drift_aware_data(**kwargs)  # Returns train_key, val_key, test_key_batch7

# Training - uses dynamic paths from XCom
TRAIN_KEY="{{ ti.xcom_pull(task_ids='prepare_drift_aware_data')['train_key'] }}"

# Quality gate - AUC on batch_7_test
compare_against_production(**kwargs)  # Loads models, evaluates on test set
```

### ⏳ Stage 4: Slim CI DAG Run (IN PROGRESS)
**Goal:** Test with 2 HPO trials instead of 10 (5-7 min vs 20 min)

**Configuration:**
```python
'RAY_NUM_SAMPLES': '2',   # Slim mode
'RAY_MAX_EPOCHS': '5', 
'RAY_GRACE_PERIOD': '2'
```

**Progress:**
- DAG triggered with `{"drift_triggered": true, "drift_severity": "major"}`
- ✅ `prepare_drift_aware_data`: SUCCESS
- ✅ `run_training_and_hpo`: SUCCESS (2 trials completed)
- ✅ `evaluate_model_performance`: SUCCESS
- ⏳ `compare_against_production`: DEBUGGING (multiple failures, fixes applied)

### ⏸️ Stage 5: Production Run (PENDING)
Restore production config and trigger via drift monitoring DAG

---

## Current Challenge: compare_against_production Failures

### Iteration History

**Attempt 1 (06:12 run):**
- Error: `KeyError: 'readmitted_binary'`
- Cause: Raw data from S3 needs feature engineering
- Fix: Added `clean_data()` and `engineer_features()` calls

**Attempt 2 (06:14 run):**
- Error: `'PyFuncModel' object has no attribute 'predict_proba'`
- Cause: pyfunc wrapper doesn't expose predict_proba
- Fix: Changed to `mlflow.xgboost.load_model()`

**Attempt 3 (06:17 run):**
- Error: `Model does not have the "xgboost" flavor`
- Cause: Model saved with sklearn flavor, not xgboost
- Fix: Changed to `mlflow.sklearn.load_model()`

**Attempt 4 (06:17 run, retry):**
- Error: `DataFrame.dtypes for data must be int, float, bool or category`
- Cause: Feature engineering creates object columns (race, gender, medications, etc.)
- Fix: Added `pd.get_dummies()` to one-hot encode categorical columns

**Attempt 5 (07:02 run - CURRENT):**
- Status: up_for_retry (try 1)
- Last log: "Loaded test set: batch_7_test (temporal holdout), shape: (1000, 50)"
- Waiting to see if categorical encoding fix worked

---

## Current Code State

### File: `health_predict_continuous_improvement.py`

**Lines 357-385 (compare_against_production - data loading):**
```python
# Load test set (batch_7_test if drift, else initial_validation)
logging.info(f"Loaded test set: {test_set_name}, shape: {test_data.shape}")

# Apply feature engineering
from src.feature_engineering import clean_data, engineer_features
test_data_clean = clean_data(test_data)
test_data_featured = engineer_features(test_data_clean)

# Encode categorical variables
cat_cols = test_data_featured.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['readmitted_binary', 'readmitted']]

if cat_cols:
    logging.info(f"One-hot encoding {len(cat_cols)} categorical columns...")
    test_data_encoded = pd.get_dummies(test_data_featured, columns=cat_cols, drop_first=True)
else:
    test_data_encoded = test_data_featured

# Prepare features
X_test = test_data_encoded.drop(columns=['readmitted_binary', 'readmitted', 'age'], errors='ignore')
y_test = test_data_featured['readmitted_binary']  # Use featured, not encoded
```

**Lines 329-340 (model loading):**
```python
# Load models (use sklearn flavor which preserves predict_proba)
new_model = mlflow.sklearn.load_model(f"runs:/{new_run_id}/model")
logging.info(f"Loaded new model from run {new_run_id}")

try:
    prod_model = mlflow.sklearn.load_model("models:/HealthPredict_XGBoost/Production")
    has_production = True
except Exception as e:
    has_production = False
    logging.info(f"No production model: {e}")
```

---

## Active DAG Runs

```bash
# Check current runs
curl -s "http://localhost:8080/api/v1/dags/health_predict_continuous_improvement/dagRuns" \
  --user "admin:admin" | python3 -c "import sys, json; runs = json.load(sys.stdin)['dag_runs']; [print(f\"{r['dag_run_id']}: {r['state']}\") for r in runs[:5]]"
```

**Current Runs:**
- `manual__2025-12-16T07:02:54.723041+00:00`: **RUNNING** (being debugged)
- `manual__2025-12-16T07:05:14.383099+00:00`: queued (will run after 07:02)
- `manual__2025-12-16T06:17:19.026535+00:00`: failed (previous iteration)

---

## Next Steps for New Agent

### Immediate Actions

1. **Check 07:02 run status:**
```bash
curl -s "http://localhost:8080/api/v1/dags/health_predict_continuous_improvement/dagRuns/manual__2025-12-16T07:02:54.723041%2B00:00/taskInstances/compare_against_production" \
  --user "admin:admin" | python3 -c "import sys, json; t = json.load(sys.stdin); print(f\"State: {t['state']}, Try: {t['try_number']}\")"
```

2. **Check attempt 1 or 2 logs:**
```bash
docker exec mlops-services-airflow-scheduler-1 tail -100 \
  /opt/airflow/logs/dag_id=health_predict_continuous_improvement/run_id=manual__2025-12-16T07:02:54.723041+00:00/task_id=compare_against_production/attempt=1.log \
  | grep -E "ERROR|Exception|SUCCESS|auc|AUC"
```

3. **If still failing - debug the error:**
   - Look for the specific error message
   - Common issues:
     - Feature mismatch (model expects different columns than provided)
     - Data type issues
     - Missing columns after encoding

### Likely Next Issue: Feature Mismatch

**Problem:** The model was trained on specific one-hot encoded features, but our dynamic encoding might create different features.

**Solution Options:**

**A. Load and use the training preprocessor:**
```python
# Check if preprocessor was saved with model
run_id = new_run_id
artifacts = client.list_artifacts(run_id)
# Look for preprocessor.joblib or similar

# If found, load and transform:
from src.feature_engineering import load_preprocessor
preprocessor = load_preprocessor(f"runs:/{run_id}/preprocessor.joblib")
X_test_transformed = preprocessor.transform(X_test)
```

**B. Match model's expected features:**
```python
# Get model's expected features
if hasattr(new_model, 'feature_names_in_'):
    expected_features = new_model.feature_names_in_
    
    # Align test data to match
    for col in expected_features:
        if col not in X_test.columns:
            X_test[col] = 0  # Add missing columns with 0
    
    X_test = X_test[expected_features]  # Reorder to match
```

**C. Simplify by using only numeric features:**
```python
# Drop all categorical columns entirely
X_test_numeric = test_data_featured.select_dtypes(include=[np.number])
X_test_numeric = X_test_numeric.drop(columns=['readmitted_binary'], errors='ignore')
```

### If compare_against_production Succeeds

1. **Monitor downstream tasks:**
   - `deployment_decision_branch`
   - `register_and_promote_model` or `log_skip_decision`

2. **Capture results:**
```bash
# Get full DAG status
curl -s "http://localhost:8080/api/v1/dags/health_predict_continuous_improvement/dagRuns/manual__2025-12-16T07:02:54.723041%2B00:00" \
  --user "admin:admin" | python3 -m json.tool
```

3. **Check quality gate decision in logs:**
```bash
docker exec mlops-services-airflow-scheduler-1 grep -E "DECISION|AUC|improvement" \
  /opt/airflow/logs/.../compare_against_production/attempt=*.log
```

4. **Move to Stage 5:**
   - Restore production config:
     ```python
     'RAY_NUM_SAMPLES': '10',
     'RAY_MAX_EPOCHS': '20',
     'RAY_GRACE_PERIOD': '5'
     ```
   - Ensure drift monitoring triggers CI DAG (already updated)
   - Trigger end-to-end flow
   - Document with screenshots

---

## Key Files & Locations

**Modified DAGs:**
- `/home/ubuntu/health-predict/mlops-services/dags/health_predict_continuous_improvement.py`
- `/home/ubuntu/health-predict/mlops-services/dags/drift_monitoring_dag.py` (trigger updated)

**Test Files:**
- `/home/ubuntu/health-predict/src/drift/utils.py`
- `/home/ubuntu/health-predict/tests/drift/test_batch_split.py`
- `/home/ubuntu/health-predict/tests/integration/test_drift_retraining_flow.py`

**Feature Engineering:**
- `/home/ubuntu/health-predict/src/feature_engineering.py`

**Airflow UI:**
- URL: `http://13.222.206.225:8080`
- Credentials: `admin/admin`

**MLflow:**
- URL: `http://13.222.206.225:5000`

---

## Important Context

### Why Incremental Testing?

Previously tried manual trigger which failed. User wants **systematic validation** at each step to:
- Minimize costs (slim runs vs full production)
- Shorten feedback loops (local tests → DAG tasks → full run)
- Catch bugs early before expensive production runs

### Drift-Aware Logic

**When drift detected:**
1. Split batch_7: 50% train, 50% test (temporal holdout)
2. Combine: initial_train + initial_val + batch_7_train = cumulative
3. Train new model on cumulative
4. Load production model
5. Compare BOTH on batch_7_test using AUC
6. Deploy if new_AUC >= prod_AUC + 0.02

**Key insight:** batch_7_test is NEVER seen during training - ensures fair comparison

### Why Multiple Failed Runs?

Each failure revealed a new issue:
- DAG code assumed processed data, but gets raw data
- MLflow model loading has multiple flavors (pyfunc, sklearn, xgboost)
- Feature engineering doesn't include encoding step
- Categorical columns need one-hot encoding for XGBoost

This is normal for complex ML pipelines - keep iterating!

---

## Troubleshooting Commands

**Restart Airflow scheduler:**
```bash
docker restart $(docker ps -q --filter "name=airflow-scheduler")
sleep 6
```

**Trigger new DAG run:**
```bash
curl -X POST "http://localhost:8080/api/v1/dags/health_predict_continuous_improvement/dagRuns" \
  --user "admin:admin" \
  -H "Content-Type: application/json" \
  -d '{"conf": {"drift_triggered": true, "drift_severity": "major"}}'
```

**Clear task to retry immediately:**
```bash
curl -X POST "http://localhost:8080/api/v1/dags/health_predict_continuous_improvement/clearTaskInstances" \
  --user "admin:admin" \
  -H "Content-Type: application/json" \
  -d '{
    "dry_run": false,
    "task_ids": ["compare_against_production"],
    "dag_run_id": "manual__2025-12-16T07:02:54.723041+00:00"
  }'
```

**Check S3 for batch_7_test:**
```bash
aws s3 ls s3://health-predict-mlops-f9ac6509/drift_monitoring/test_data/
```

---

## Success Criteria

### Stage 4 Complete When:
- ✅ prepare_drift_aware_data: SUCCESS
- ✅ run_training_and_hpo: SUCCESS  
- ⏳ compare_against_production: SUCCESS (still working on this)
- ⏳ Quality gate decision made (DEPLOY/SKIP)
- ⏳ Full DAG completes (deploy or log skip)

### Stage 5 Complete When:
- Production config restored
- End-to-end flow works: drift detection → CI DAG trigger → retraining → quality gate → deployment
- Comprehensive screenshots captured
- Walkthrough document created

---

## Final Notes

**User's Valid Concern:** I kept stopping instead of actively iterating. The new agent should:
- **Continuously monitor** task execution
- **Immediately check logs** when tasks fail
- **Apply fixes and retry** without waiting for user prompts
- **Use background commands** with wait times to monitor progress
- **Keep user updated** via brief notify_user messages showing progress

**Remember:** This is a complex ML pipeline with many moving parts. Systematic debugging is expected. Stay focused, keep iterating, and document each issue/fix clearly.

**Good luck! 🚀**
