# Dashboard Redesign: From Monitoring Tool to Experiment Report

**Date**: 2026-02-08
**Status**: Brainstorm complete, ready for planning

## What We're Building

A complete redesign of the Streamlit dashboard from a live monitoring tool into an **experiment report** that tells the story of one clean run-through of the ML pipeline.

### The Narrative

> "I built an MLOps system that detects data drift and automatically retrains models. I fed 5 batches of data through it. Here's what happened."

The dashboard should read like a finished experiment — a vertical scrollable story showing the initial baseline, then each batch's drift analysis, retraining decision, and model performance.

## Why This Approach

**Problem**: The current dashboard shows every historical MLflow run — dozens of debug runs, repeated batch triggers, noise from development. It's built as a live ops console when it should be a portfolio demonstration.

**Solution**: Filter to the **latest run per batch number** and redesign the layout as a linear story.

- No pipeline changes needed (just add a baseline step)
- No MLflow cleanup needed (filtering handles historical noise)
- Re-running any batch automatically updates the dashboard
- Simple, minimal, YAGNI-compliant

## Key Decisions

### 1. Data Filtering Strategy
**Decision**: Latest per batch (Approach 1)
- Dashboard queries MLflow, groups by `batch_number` param, takes most recent run per batch
- Works for both drift monitoring experiment and training experiment
- No session IDs, no experiment isolation, no pipeline changes

### 2. Dashboard Layout
**Decision**: Vertical scroll story
- NOT tabs — one continuous scrollable page
- Each batch gets a full-width section
- Reads like a data science report or portfolio case study

### 3. Baseline Display
**Decision**: Show "Batch 0" as starting point
- Need to create a baseline capture step (script or DAG task)
- Evaluates initial model on initial_test data
- Logs as quality gate run with `batch_number=0`, `decision='BASELINE'`
- Shows initial AUC, F1, precision, recall as the starting point

### 4. Metrics Per Batch
**Decision**: Show production model metrics for every batch; show new model metrics where retraining occurred
- Batches without retraining: production model metrics only
- Batches with retraining: both production and new model metrics side-by-side
- Metrics: AUC, F1, precision, recall

### 5. Drift Visualization
**Decision**: Integrated into the journey view
- Each batch section shows its drift score
- Clear threshold line (0.30) marking retrain/no-retrain boundary
- Color-coded: green = below threshold (no retrain), red = above threshold (retrain triggered)
- Also a summary chart showing drift progression across all batches

### 6. Pipeline Trigger Model
**Decision**: Keep separate per-batch triggers
- Each batch is triggered individually (as today)
- Dashboard identifies "latest run-through" by taking most recent run per batch
- No session grouping needed

## Dashboard Section Outline

### Section 1: Hero / Overview
- Project title and one-line description
- Current production model version and stage
- Summary stats: total batches processed, models trained, current AUC

### Section 2: Experiment Summary
- **Drift Progression Chart**: Bar chart showing drift_share for batches 0-5, with 0.30 threshold line
- **Metrics Summary Table**: One row per batch, columns for:
  - Batch number
  - Drift share
  - Retrain triggered? (Yes/No)
  - Production model AUC, F1, precision, recall
  - New model AUC, F1, precision, recall (if retrained)
  - Decision (BASELINE / SKIP_NO_DRIFT / DEPLOY / SKIP)

### Section 3: The Journey (per-batch sections)
For each batch (0 through 5), a full-width section containing:

**Batch 0 (Baseline)**:
- "Initial model trained on initial dataset"
- Baseline metrics: AUC, F1, precision, recall
- No drift analysis (this is the starting point)

**Batches 1-5** (each):
- Batch header with drift score and retrain decision badge
- Drift details: drift_share, number of drifted features, dataset_drift flag
- Model performance:
  - Production model metrics on this batch's test data
  - If retrained: new model metrics + improvement deltas
- Optional: ROC curve visualization (expandable)

### Section 4: Final Summary
- Starting AUC (Batch 0) vs. ending AUC (latest production model)
- Total retraining events
- Key takeaway: "The system detected drift in X of 5 batches and retrained Y times, improving AUC from A to B"

## What Needs to Change

### 1. New: Baseline Capture Script
- Script that evaluates the initial production model on initial_test data
- Logs a quality gate run to MLflow with batch_number=0
- Run once before starting the batch sequence

### 2. Dashboard: Complete Rewrite
- Replace current 4-tab monitoring dashboard with vertical scroll story
- New data loading: filter to latest run per batch
- New visualizations: drift progression chart, metrics table, per-batch journey sections

### 3. Pipeline: No Changes
- DAG stays the same
- MLflow logging stays the same (already logs batch_number consistently)
- Batch creation stays the same

## Open Questions

1. **Should the dashboard have an "About" or methodology section?** (explaining how drift detection works, what the threshold means, etc.) — could be useful for portfolio context
2. **ROC curves**: Include per-batch or just in a summary? The data is already logged.
3. **Exact batch drift outcomes**: Need to verify that batches 1-2 don't trigger retraining and batches 3-5 do with the current 0.30 threshold (from memory: batch 1 ~0.27, batch 2 ~0.27, batch 3 ~0.32, batch 4 ~0.34, batch 5 ~0.35)

## Next Steps

Run `/workflows:plan` to create the implementation plan.
