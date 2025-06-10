#!/usr/bin/env python3
"""
Synthetic Drift Injection for Health Predict MLOps Pipeline
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def inject_covariate_drift(data, features, intensity=0.3, drift_type="shift", seed=None):
    if seed is not None:
        np.random.seed(seed)
    logger.info(f"Injecting {drift_type} covariate drift with intensity {intensity} on features: {features}")
    drifted_data = data.copy()
    for feature in features:
        if feature not in drifted_data.columns:
            logger.warning(f"Feature {feature} not found in data, skipping")
            continue
        original_values = drifted_data[feature].copy()
        if not pd.api.types.is_numeric_dtype(original_values):
            continue
        if drift_type == "shift":
            feature_std = original_values.std()
            shift_amount = intensity * feature_std * np.random.normal(0, 1)
            drifted_data[feature] = original_values + shift_amount
            logger.info(f"Applied mean shift of {shift_amount:.3f} to {feature}")
    return drifted_data

def create_drift_scenario(data, scenario_name, target_col="readmitted_binary", seed=None):
    if seed is not None:
        np.random.seed(seed)
    logger.info(f"Creating drift scenario: {scenario_name}")
    scenario_metadata = {"scenario_name": scenario_name, "drift_types": [], "affected_features": [], "intensity_levels": {}, "target_column": target_col}
    drifted_data = data.copy()
    if scenario_name == "mild_covariate":
        features = ["age_ordinal", "time_in_hospital"] if "age_ordinal" in data.columns else ["time_in_hospital"]
        drifted_data = inject_covariate_drift(drifted_data, features, intensity=0.15, drift_type="shift", seed=seed)
        scenario_metadata["drift_types"] = ["covariate_shift"]
        scenario_metadata["affected_features"] = features
        scenario_metadata["intensity_levels"] = {"covariate": 0.15}
    return drifted_data, scenario_metadata
