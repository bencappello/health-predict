"""
Drift monitoring utilities for batch splitting and cumulative training
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def split_batch_for_training(
    batch_data: pd.DataFrame,
    test_size: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a batch into training and test portions.
    
    The training portion will be used in cumulative retraining.
    The test portion is held out for unbiased model comparison.
    
    Args:
        batch_data: DataFrame containing the batch data
        test_size: Fraction of data to hold out for testing (default 0.5)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if len(batch_data) < 2:
        raise ValueError("Batch must have at least 2 samples to split")
    
    train, test = train_test_split(
        batch_data,
        test_size=test_size,
        random_state=seed,
        stratify=batch_data['readmitted_binary'] if 'readmitted_binary' in batch_data.columns else None
    )
    
    return train, test


def prepare_cumulative_data(
    datasets: List[pd.DataFrame],
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Combine multiple datasets for cumulative training.
    
    Args:
        datasets: List of DataFrames to combine
        remove_duplicates: Whether to remove duplicate rows
        
    Returns:
        Combined DataFrame
    """
    if not datasets:
        raise ValueError("Must provide at least one dataset")
    
    # Concatenate all datasets
    combined = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_count = len(combined)
        combined = combined.drop_duplicates()
        removed = initial_count - len(combined)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    return combined


def evaluate_quality_gate(
    prod_model,
    new_model,
    test_data: pd.DataFrame,
    target_col: str = 'readmitted_binary',
    threshold: float = 0.02,
    metric: str = 'auc'
) -> dict:
    """
    Evaluate whether new model should be deployed based on quality gate.
    
    Args:
        prod_model: Production model with predict_proba method
        new_model: New model with predict_proba method
        test_data: Test dataset
        target_col: Name of target column
        threshold: Minimum improvement required for deployment
        metric: Metric to use ('auc' or 'f1')
        
    Returns:
        Dictionary with decision and metrics
    """
    from sklearn.metrics import roc_auc_score, f1_score
    
    # Prepare data
    X_test = test_data.drop(columns=[target_col], errors='ignore')
    y_test = test_data[target_col]
    
    # Get predictions
    prod_pred_proba = prod_model.predict_proba(X_test)[:, 1]
    new_pred_proba = new_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    if metric == 'auc':
        prod_score = roc_auc_score(y_test, prod_pred_proba)
        new_score = roc_auc_score(y_test, new_pred_proba)
    elif metric == 'f1':
        prod_pred = (prod_pred_proba > 0.5).astype(int)
        new_pred = (new_pred_proba > 0.5).astype(int)
        prod_score = f1_score(y_test, prod_pred)
        new_score = f1_score(y_test, new_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Decision logic
    improvement = new_score - prod_score
    
    if improvement >= threshold:
        approve = True
        reason = f"{metric.upper()} improved by {improvement:.3f} (>= {threshold:.3f} threshold)"
    elif improvement >= 0:
        approve = True  # Minor improvement, still deploy
        reason = f"Minor {metric.upper()} improvement: {improvement:.3f}"
    else:
        approve = False
        reason = f"{metric.upper()} regression: {improvement:.3f}"
    
    return {
        'approve': approve,
        'reason': reason,
        'prod_score': prod_score,
        'new_score': new_score,
        'improvement': improvement,
        'metric': metric
    }
