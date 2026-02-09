---
title: "feat: Redesign dashboard as experiment report"
type: feat
date: 2026-02-08
---

# feat: Redesign Dashboard as Experiment Report

## Overview

Rewrite the Streamlit dashboard from a live monitoring tool into an **experiment report** — a vertical scrollable story showing one clean run-through of batches 0-5. Add a baseline capture step (Batch 0) so the report starts with initial model metrics.

## Problem Statement / Motivation

The current dashboard (`scripts/drift_dashboard.py`) shows every historical MLflow run — dozens of debug runs, repeated batch triggers, development noise. It's built as a live ops console (4 tabs, all runs) when it should be a **portfolio demonstration** that tells the story: "I built this system, fed 5 batches through it, here's what happened."

**Specific issues:**
- Dashboard queries ALL runs with no dedup — shows batch 1 run 7 times, batch 3 run 18 times
- No baseline (Batch 0) showing initial model performance
- Tab-based layout fragments the narrative instead of telling a linear story
- Metrics are scattered across tabs instead of in one clear table

## Proposed Solution

### Phase 1: Baseline Capture Script

Create `scripts/create_baseline.py` that establishes "Batch 0" — the starting point of the experiment.

**What it does:**
1. Load the current Production model from MLflow registry (`HealthPredictModel/Production`)
2. Load the preprocessor from the model's training run artifacts
3. Load `initial_test.csv` from S3 (`processed_data/initial_test.csv`)
4. Apply feature engineering: `clean_data()` → `engineer_features()` → `preprocess_data()`
5. Predict and compute metrics: AUC, F1, precision, recall
6. Compute ROC curve
7. Log a quality gate run to MLflow matching the existing format:
   - Experiment: `HealthPredict_Training_HPO_Airflow`
   - Run name: `quality_gate_batch_0`
   - Params: `batch_number=0`, `test_set_used=initial_test`, `model_type=XGBoost`
   - Metrics: `new_auc_test`, `new_f1_test`, `new_precision_test`, `new_recall_test` (all set to prod model values)
   - Also: `prod_auc_test`, `prod_f1_test`, `prod_precision_test`, `prod_recall_test` (same values)
   - `auc_improvement=0.0`, `f1_improvement=0.0`
   - Tag: `quality_gate='True'`, `decision='BASELINE'`
   - Artifact: `roc_curves/roc_curve_production.json`
8. Also log a drift monitoring run for Batch 0:
   - Experiment: `HealthPredict_Drift_Monitoring`
   - Run name: `drift_batch_0`
   - Params: `batch_number=0`
   - Metrics: `drift_share=0.0`, `n_drifted_columns=0`, `dataset_drift=0`
   - (Baseline has no drift by definition)

**Key constraint:** Must match the exact MLflow logging format used by the DAG's `compare_against_production` and `evaluate_production_only` tasks so the dashboard can query it identically.

**File:** `scripts/create_baseline.py`

**Dependencies:** Same as DAG (mlflow, pandas, scikit-learn, xgboost, boto3) — already in `scripts/requirements-training.txt`

**Run command:**
```bash
python scripts/create_baseline.py \
  --s3-bucket-name health-predict-mlops-f9ac6509 \
  --mlflow-tracking-uri http://localhost:5000
```

Or from inside a container:
```bash
docker exec mlops-services-airflow-scheduler-1 \
  python /opt/airflow/scripts/create_baseline.py \
  --s3-bucket-name health-predict-mlops-f9ac6509 \
  --mlflow-tracking-uri http://mlflow:5000
```

### Phase 2: Dashboard Rewrite

Complete rewrite of `scripts/drift_dashboard.py` — same file, new content.

#### Data Loading Changes

**Current:** Queries all runs, no dedup
```python
mlflow.search_runs(experiment_ids=[...], order_by=["start_time DESC"], max_results=100)
```

**New:** Query all runs, then deduplicate to latest per batch
```python
runs = mlflow.search_runs(...)
# Group by batch_number param, keep most recent run per batch
runs['batch'] = pd.to_numeric(runs['params.batch_number'], errors='coerce')
latest_runs = runs.sort_values('start_time', ascending=False).drop_duplicates(subset=['batch'], keep='first')
latest_runs = latest_runs.sort_values('batch')
```

This applies to BOTH experiments:
- `HealthPredict_Drift_Monitoring` → latest drift run per batch
- `HealthPredict_Training_HPO_Airflow` (quality gate filter) → latest quality gate per batch

#### Page Layout: Vertical Scroll Story

Replace the 4-tab structure with a single scrollable page:

```
┌─────────────────────────────────────────────┐
│  HERO: Project Title + Summary Stats        │
│  (Production model, total batches, AUC)     │
├─────────────────────────────────────────────┤
│  EXPERIMENT SUMMARY                         │
│  ┌─────────────────────────────────────┐    │
│  │ Drift Progression Chart             │    │
│  │ (bar chart, 0-5, threshold line)    │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │ Metrics Summary Table               │    │
│  │ Batch | Drift | Retrain? | AUC | F1 │    │
│  │   0   |  0.00 |    --    | .82 | .74│    │
│  │   1   |  .27  |    No    | .82 | .74│    │
│  │   2   |  .27  |    No    | .82 | .74│    │
│  │   3   |  .32  |   Yes    | .84 | .76│    │
│  │   4   |  .34  |   Yes    | .85 | .77│    │
│  │   5   |  .35  |   Yes    | .83 | .75│    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│  BATCH 0: BASELINE                          │
│  Initial model trained on initial dataset   │
│  Metrics: AUC .82 | F1 .74 | Prec .71 ...  │
├─────────────────────────────────────────────┤
│  BATCH 1: No Drift Detected                │
│  Drift: 0.27 (below 0.30 threshold)        │
│  Decision: SKIP — no retraining             │
│  Prod model on batch 1: AUC .82 | F1 .74   │
├─────────────────────────────────────────────┤
│  BATCH 2: No Drift Detected                │
│  ...                                        │
├─────────────────────────────────────────────┤
│  BATCH 3: DRIFT DETECTED → RETRAINED       │
│  Drift: 0.32 (above 0.30 threshold)        │
│  Decision: DEPLOY                           │
│  Prod model:  AUC .82 | F1 .74             │
│  New model:   AUC .84 | F1 .76  (+0.02)    │
│  [expandable ROC curve]                     │
├─────────────────────────────────────────────┤
│  BATCH 4: DRIFT DETECTED → RETRAINED       │
│  ...                                        │
├─────────────────────────────────────────────┤
│  BATCH 5: DRIFT DETECTED → RETRAINED       │
│  ...                                        │
├─────────────────────────────────────────────┤
│  FINAL SUMMARY                              │
│  Starting AUC → Ending AUC                  │
│  X of 5 batches triggered retraining        │
│  Key takeaway sentence                      │
└─────────────────────────────────────────────┘
```

#### Section Details

**1. Hero Section**
- Title: "Health Predict — Continuous Improvement Experiment"
- 3 metric cards: Current Production Model (version + AUC), Batches Processed, Retraining Events
- One-line description of the system

**2. Experiment Summary**
- **Drift Progression Chart** (Plotly bar chart):
  - X-axis: Batch number (0-5)
  - Y-axis: Drift share (0.0-1.0)
  - Horizontal line at 0.30 threshold
  - Color: green if < 0.30, red if >= 0.30
  - Batch 0 bar = 0.0 (baseline, no drift)
- **Metrics Summary Table** (Streamlit dataframe or custom HTML):
  - Columns: Batch, Drift Share, Retrained?, Decision, Prod AUC, Prod F1, Prod Precision, Prod Recall, New AUC, New F1, New Precision, New Recall
  - Retrained batches show both prod and new model metrics
  - Non-retrained batches show prod metrics only, new columns blank or "—"
  - Batch 0 shows baseline metrics, Retrained = "—"

**3. Per-Batch Journey Sections**
Each batch gets a `st.container()` with:
- **Header**: "Batch N" + badge (color-coded: green "No Drift", red "Drift Detected → Retrained", blue "Baseline")
- **Drift info**: drift_share value, n_drifted_columns, threshold comparison
- **Decision**: What happened (BASELINE / SKIP_NO_DRIFT / DEPLOY / DEPLOY_REFRESH / SKIP)
- **Metrics cards**: Production model metrics (always), New model metrics + deltas (if retrained)
- **ROC Curve** (expandable `st.expander`): Load from MLflow artifacts if available

**Batch 0 special case:**
- No drift section (baseline by definition)
- Show only production model metrics
- Label as "Initial Training — Baseline"

**4. Final Summary**
- Starting AUC (Batch 0) → Final AUC (latest production model)
- Count of retraining events
- One-sentence takeaway

#### Sidebar (Simplified)

Keep minimal:
- MLflow URI (auto from env var, rarely changed)
- API URL (auto from env var)
- "Refresh Data" button (clears cache)
- Link to MLflow UI, Airflow UI

### Phase 3: Rebuild & Verify

1. Rebuild dashboard container: `docker-compose up -d --no-deps --build dashboard`
2. Run baseline script to create Batch 0
3. Run all 5 batches through the pipeline (or verify existing latest runs are clean)
4. Verify dashboard shows the complete story

## Technical Considerations

### MLflow Query: "Latest Per Batch" Logic

```python
def get_latest_per_batch(experiment_name, filter_string=None):
    """Query MLflow and return only the most recent run per batch number."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=100,
    )

    if runs.empty:
        return runs

    # Extract batch number, deduplicate
    runs['batch'] = pd.to_numeric(runs.get('params.batch_number', pd.Series()), errors='coerce')
    runs = runs.dropna(subset=['batch'])
    runs = runs.sort_values('start_time', ascending=False).drop_duplicates(subset=['batch'], keep='first')
    return runs.sort_values('batch')
```

### Drift Monitoring Query
```python
drift_runs = get_latest_per_batch("HealthPredict_Drift_Monitoring")
```

### Quality Gate Query
```python
quality_runs = get_latest_per_batch(
    "HealthPredict_Training_HPO_Airflow",
    filter_string="metrics.new_auc_test > 0"
)
```

### Decision Tag Extraction
Quality gate runs have a `tags.decision` field:
- `BASELINE` — Batch 0
- `SKIP_NO_DRIFT` — No drift, prod model evaluated only
- `DEPLOY` — Drift detected, new model deployed (improvement)
- `DEPLOY_REFRESH` — Drift detected, minor regression but still deployed
- `SKIP` — Drift detected, regression too large, deployment skipped

### Docker Rebuild Safety
Per CLAUDE.md: **Always use `--no-deps`** to avoid ContainerConfig bug:
```bash
cd /home/ubuntu/health-predict/mlops-services
docker-compose up -d --no-deps --build dashboard
```

### ROC Curve Artifacts
Stored as JSON in `roc_curves/` artifact folder. Load via:
```python
client = MlflowClient()
artifact_path = client.download_artifacts(run_id, "roc_curves/roc_curve_new.json", "/tmp")
with open(artifact_path) as f:
    roc_data = json.load(f)  # {fpr: [...], tpr: [...], auc: float}
```

## Acceptance Criteria

- [x] `scripts/create_baseline.py` creates a Batch 0 quality gate run in MLflow matching the existing format
- [x] `scripts/create_baseline.py` creates a Batch 0 drift run (drift_share=0.0) in MLflow
- [x] Dashboard shows exactly ONE run per batch (latest), not historical duplicates
- [x] Dashboard is a single scrollable page (no tabs)
- [x] Hero section shows project summary and current production model
- [x] Drift progression chart shows batches 0-5 with 0.30 threshold line
- [x] Metrics summary table shows AUC, F1, precision, recall for every batch
- [x] Metrics table shows both prod and new model metrics for retrained batches
- [x] Per-batch sections show drift info, decision, and metrics
- [x] ROC curves are viewable per batch (expandable)
- [x] Final summary shows starting vs ending AUC and retraining count
- [x] Dashboard container rebuilds cleanly with `docker-compose up -d --no-deps --build dashboard`

## Success Metrics

- Dashboard tells a clear, linear story readable by someone unfamiliar with the project
- No duplicate/noisy historical runs visible
- All 6 batches (0-5) visible with complete metrics
- Batches 1-2 show no retraining; batches 3-5 show retraining (matching expected drift profiles)

## Dependencies & Risks

**Dependencies:**
- Production model must exist in MLflow registry before running baseline script
- All 5 batches must have been run at least once (or re-run after baseline)
- MLflow and S3 must be accessible

**Risks:**
- If batch drift profiles have shifted (e.g., batch 2 now drifts), the story changes — verify expected outcomes
- Historical runs with `batch_number=0` could exist — baseline script should handle gracefully (latest-per-batch dedup handles this)
- ROC curve artifacts may not exist for older runs — dashboard should handle missing artifacts gracefully

## References & Research

### Internal References
- Current dashboard: `scripts/drift_dashboard.py` (792 lines, full rewrite)
- Quality gate logging (retrain): DAG lines 698-745
- Quality gate logging (prod-only): DAG lines 965-1005
- Feature engineering: `src/feature_engineering.py` (clean_data, engineer_features, preprocess_data)
- Training script: `scripts/train_model.py` (MLflow logging patterns)
- Docker setup: `mlops-services/Dockerfile.dashboard`, `mlops-services/docker-compose.yml` lines 222-236
- Dashboard deps: `scripts/requirements-dashboard.txt`

### Brainstorm
- `docs/brainstorms/2026-02-08-dashboard-redesign-brainstorm.md`

## Implementation Order

1. **Create baseline script** (`scripts/create_baseline.py`) — ~100 lines
2. **Rewrite dashboard** (`scripts/drift_dashboard.py`) — ~400-500 lines (down from 792, simpler without 4 tabs)
3. **Rebuild container** and run baseline
4. **Run batches 1-5** (or verify existing runs) and confirm dashboard
