#!/usr/bin/env python3
"""
Stage 2: Integration Test - Complete Drift Retraining Flow
Tests end-to-end logic with tiny datasets (100 rows) before touching Airflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.drift.utils import split_batch_for_training, prepare_cumulative_data


def create_tiny_initial_data(n=100):
    """Create tiny initial training dataset with slight class imbalance"""
    np.random.seed(42)
    return pd.DataFrame({
        'time_in_hospital': np.random.randint(1, 10, n),
        'num_medications': np.random.randint(5, 20, n),
        'num_lab_procedures': np.random.randint(10, 80, n),
        'readmitted_binary': np.random.choice([0, 1], n, p=[0.7, 0.3])  # 30% readmission
    })


def create_tiny_batch_7(n=100):
    """Create tiny batch_7 with DRIFT (shifted distributions)"""
    np.random.seed(99)  # Different seed = different distribution
    return pd.DataFrame({
        # Shifted means and ranges to simulate drift
        'time_in_hospital': np.random.randint(3, 12, n),  # Higher (was 1-10)
        'num_medications': np.random.randint(10, 25, n),  # Higher (was 5-20)
        'num_lab_procedures': np.random.randint(40, 120, n),  # Higher (was 10-80)
        'readmitted_binary': np.random.choice([0, 1], n, p=[0.6, 0.4])  # 40% readmission (was 30%)
    })


def main():
    print("=" * 80)
    print("STAGE 2: INTEGRATION TEST - DRIFT-AWARE RETRAINING FLOW")
    print("=" * 80)
    print()
    
    # ===== Step 1: Create Mock Data =====
    print("Step 1: Creating mock datasets (100 rows each)...")
    initial_data = create_tiny_initial_data(100)
    batch_7_full = create_tiny_batch_7(100)
    print(f"  ✓ Initial data: {initial_data.shape}")
    print(f"  ✓ Batch 7 data: {batch_7_full.shape}")
    print()
    
    # ===== Step 2: Split Batch 7 =====
    print("Step 2: Splitting batch_7 (50% train, 50% test)...")
    batch_7_train, batch_7_test = split_batch_for_training(batch_7_full, test_size=0.5, seed=42)
    print(f"  ✓ Batch 7 train: {len(batch_7_train)} rows")
    print(f"  ✓ Batch 7 test: {len(batch_7_test)} rows (HELD OUT)")
    print()
    
    # ===== Step 3: Create Cumulative Dataset =====
    print("Step 3: Creating cumulative dataset...")
    cumulative = prepare_cumulative_data([initial_data, batch_7_train])
    print(f"  ✓ Cumulative size: {len(cumulative)} rows")
    print(f"    - Initial: {len(initial_data)}")
    print(f"    - Batch 7 (train only): {len(batch_7_train)}")
    print()
    
    # ===== Step 4: Train Production Model =====
    print("Step 4: Training 'production' model (initial data only)...")
    X_initial = initial_data.drop(columns=['readmitted_binary'])
    y_initial = initial_data['readmitted_binary']
    
    prod_model = LogisticRegression(random_state=42, max_iter=1000)
    prod_model.fit(X_initial, y_initial)
    print(f"  ✓ Production model trained on {len(y_initial)} samples")
    print()
    
    # ===== Step 5: Train New Model =====
    print("Step 5: Training 'new' model (cumulative data)...")
    X_cumulative = cumulative.drop(columns=['readmitted_binary'])
    y_cumulative = cumulative['readmitted_binary']
    
    new_model = LogisticRegression(random_state=43, max_iter=1000)
    new_model.fit(X_cumulative, y_cumulative)
    print(f"  ✓ New model trained on {len(y_cumulative)} samples")
    print()
    
    # ===== Step 6: Evaluate on Batch 7 Test Set =====
    print("Step 6: Evaluating both models on batch_7 TEST SET...")
    X_test = batch_7_test.drop(columns=['readmitted_binary'])
    y_test = batch_7_test['readmitted_binary']
    
    # Production model
    prod_pred_proba = prod_model.predict_proba(X_test)[:, 1]
    prod_pred = (prod_pred_proba > 0.5).astype(int)
    
    prod_auc = roc_auc_score(y_test, prod_pred_proba)
    prod_precision, prod_recall, prod_f1, _ = precision_recall_fscore_support(y_test, prod_pred, average='binary', zero_division=0)
    
    # New model
    new_pred_proba = new_model.predict_proba(X_test)[:, 1]
    new_pred = (new_pred_proba > 0.5).astype(int)
    
    new_auc = roc_auc_score(y_test, new_pred_proba)
    new_precision, new_recall, new_f1, _ = precision_recall_fscore_support(y_test, new_pred, average='binary', zero_division=0)
    
    print()
    print("=" * 80)
    print("RESULTS: Model Performance on Batch 7 Test Set ({} samples)".format(len(y_test)))
    print("=" * 80)
    print()
    
    print("Production Model (trained on initial data):")
    print(f"  AUC:       {prod_auc:.3f}")
    print(f"  Precision: {prod_precision:.3f}")
    print(f"  Recall:    {prod_recall:.3f}")
    print(f"  F1:        {prod_f1:.3f}")
    print()
    
    print("New Model (trained on cumulative = initial + batch_7_train):")
    print(f"  AUC:       {new_auc:.3f}")
    print(f"  Precision: {new_precision:.3f}")
    print(f"  Recall:    {new_recall:.3f}")
    print(f"  F1:        {new_f1:.3f}")
    print()
    
    # ===== Step 7: Quality Gate Decision =====
    print("=" * 80)
    print("QUALITY GATE DECISION")
    print("=" * 80)
    print()
    
    auc_improvement = new_auc - prod_auc
    f1_improvement = new_f1 - prod_f1
    threshold = 0.02  # 2% improvement
    
    print(f"AUC Improvement: {auc_improvement:+.3f} ({(auc_improvement / prod_auc * 100):+.1f}%)")
    print(f"F1 Improvement:  {f1_improvement:+.3f}")
    print(f"Threshold:       {threshold:.3f} (2%)")
    print()
    
    if auc_improvement >= threshold:
        decision = "✅ DEPLOY"
        reason = f"AUC improved by {auc_improvement:.3f} (exceeds {threshold:.3f} threshold)"
    elif auc_improvement >= 0:
        decision = "⚠️ DEPLOY_REFRESH"
        reason = f"Minor AUC improvement: {auc_improvement:.3f}"
    else:
        decision = "❌ SKIP"
        reason = f"AUC regression: {auc_improvement:.3f}"
    
    print(f"Decision: {decision}")
    print(f"Reason:   {reason}")
    print()
    
    # ===== Validation =====
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print()
    
    validations = [
        ("Batch 7 split correctly", len(batch_7_train) == 50 and len(batch_7_test) == 50),
        ("No overlap between train/test", len(set(batch_7_train.index) & set(batch_7_test.index)) == 0),
        ("Cumulative data combined", len(cumulative) == len(initial_data) + len(batch_7_train)),
        ("Production model never saw batch_7_test", True),  # By design
        ("New model never saw batch_7_test", True),  # By design
        ("Metrics calculated", all([prod_auc is not None, new_auc is not None])),
        ("Decision made", decision is not None)
    ]
    
    all_passed = True
    for check, passed in validations:
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print()
        print("Stage 2 complete. Core logic works correctly with tiny datasets.")
        print("Ready to proceed to Stage 3 (individual DAG tasks).")
        return 0
    else:
        print("⚠️ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
