"""FastAPI prediction service for the Health Predict system.

Serves patient readmission risk predictions via a REST API.  On startup
the service loads the latest Production model and its fitted preprocessor
from the MLflow model registry.  Incoming prediction requests pass
through the same ``clean_data`` / ``engineer_features`` / ``preprocess_data``
pipeline used during training to ensure feature consistency.

Endpoints:
  - ``GET /health``      — Liveness/readiness probe for Kubernetes.
  - ``GET /model-info``  — Returns the loaded model version, run ID, and
    stage for deployment verification.
  - ``POST /predict``    — Accepts patient encounter data and returns a
    binary readmission prediction with probability score.
"""

import logging
import os
import sys
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
from fastapi import status
import logging

# Add current directory to Python path for proper imports
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Attempt to import feature engineering functions
try:
    from src.feature_engineering import clean_data, engineer_features, preprocess_data, load_preprocessor
    import tempfile
    import shutil
except ImportError:
    logger = logging.getLogger(__name__) # Temp logger for import error
    logger.error("Could not import from src.feature_engineering. Ensure PYTHONPATH is set correctly or src is a package.")
    # Define dummy functions if import fails, to allow API to start for other endpoints/testing
    # This is a fallback for local development if PYTHONPATH is tricky, remove for production container
    def clean_data(df):
        logger.warning("Using DUMMY clean_data due to import error.")
        return df
    def engineer_features(df):
        logger.warning("Using DUMMY engineer_features due to import error.")
        # Simulate age_ordinal creation and age drop if age column exists
        if 'age' in df.columns:
            df['age_ordinal'] = 0 # Placeholder
            df = df.drop(columns=['age'])
        return df
    def preprocess_data(df, preprocessor, fit_preprocessor=False):
        logger.warning("Using DUMMY preprocess_data due to import error.")
        return df
    def load_preprocessor(file_path):
        logger.warning("Using DUMMY load_preprocessor due to import error.")
        return None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Health Prediction API",
    description="API for predicting health outcomes using an ML model.",
    version="0.1.0"
)

# Add custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the detailed validation errors
    # logger.error(f"Validation error for request {request.url}: {exc.errors()}")
    print(f"!!! VALIDATION ERROR HANDLER TRIGGERED for {request.url} !!!") # Simple print
    print(f"Validation Errors: {exc.errors()}") # Simple print
    # Return the default 422 response body from FastAPI
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# MLflow Configuration (Ideally, use environment variables)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") # Default for local dev
# Default to the XGBoost model registered by the training DAG
MODEL_NAME = os.getenv("MODEL_NAME", "HealthPredictModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Global variables for the model and preprocessor
model = None
preprocessor = None
model_metadata = {}  # Store model version info for verification

# --- Pydantic Models for Request and Response --- 
class InferenceInput(BaseModel):
    race: Optional[str] = None
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: Optional[str] = None
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    number_diagnoses: int
    max_glu_serum: Optional[str] = None
    A1Cresult: Optional[str] = None
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    glyburide_metformin: str # Column name in dataset is glyburide-metformin
    glipizide_metformin: str # Column name in dataset is glipizide-metformin
    glimepiride_pioglitazone: str # Column name in dataset is glimepiride-pioglitazone
    metformin_rosiglitazone: str # Column name in dataset is metformin-rosiglitazone
    metformin_pioglitazone: str # Column name in dataset is metformin-pioglitazone
    change: str
    diabetesMed: str

    class Config:
        alias_generator = lambda string: string.replace("_", "-")
        allow_population_by_field_name = True

class InferenceResponse(BaseModel):
    prediction: int
    probability_score: float

# --- API Endpoints (to be defined later) ---

# Health check endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "model_loaded": model is not None, "preprocessor_loaded": preprocessor is not None}

# Model info endpoint for verification
@app.get("/model-info", status_code=status.HTTP_200_OK)
async def model_info():
    """
    Returns metadata about the currently loaded model for verification.
    """
    if not model_metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available. Model may not have loaded successfully.")
    
    return model_metadata

# --- Startup and Shutdown Events (Model loading will go here) ---
@app.on_event("startup")
async def load_model_on_startup():
    """Load the production model and preprocessor from MLflow at startup.

    Connects to the MLflow tracking server, loads the model registered
    under MODEL_NAME at MODEL_STAGE (default: Production) using pyfunc,
    then downloads and loads the fitted sklearn preprocessor artifact
    from the same run.  Populates ``model_metadata`` with version info
    used by the ``/model-info`` verification endpoint.
    """
    global model, preprocessor, model_metadata
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading model from URI: {model_uri} via MLflow server: {MLFLOW_TRACKING_URI}")
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model '{MODEL_NAME}' version from stage '{MODEL_STAGE}' loaded successfully.")
        
        # Load the preprocessor from model artifacts
        try:
            from mlflow.tracking import MlflowClient
            from datetime import datetime
            
            client = MlflowClient()
            
            # Get the model version details
            model_version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
            run_id = model_version.run_id
            
            # Store model metadata for verification endpoint
            model_metadata = {
                "model_name": MODEL_NAME,
                "model_version": model_version.version,
                "model_stage": MODEL_STAGE,
                "run_id": run_id,
                "loaded_at": datetime.utcnow().isoformat()
            }
            print(f"Model metadata: {model_metadata}")
            
            # Download the preprocessor artifact to a temporary location
            temp_dir = tempfile.mkdtemp()
            preprocessor_path = client.download_artifacts(run_id, "preprocessor/preprocessor.joblib", temp_dir)
            
            # Load the preprocessor
            preprocessor = load_preprocessor(preprocessor_path)
            print(f"Preprocessor loaded successfully from run {run_id}")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"Warning: Could not load preprocessor: {e}")
            print("Predictions may fail due to missing preprocessing step.")
            preprocessor = None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # Depending on policy, you might want to prevent startup or allow startup without model
        # For now, we\'ll let it start and log the error. Predictions will fail.
        model = None
        preprocessor = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down.")

# --- API Endpoints --- 
@app.post("/predict", response_model=InferenceResponse)
async def predict(input_data: InferenceInput):
    """Return a readmission prediction for a single patient encounter.

    Applies the full preprocessing pipeline (clean, engineer features,
    preprocess with the fitted ColumnTransformer) to the input, then
    runs the loaded model to produce a binary prediction and a
    probability score for the positive (readmitted) class.
    """
    global model, preprocessor
    if model is None:
        logger.error("Model is not loaded. API cannot make predictions.")
        raise HTTPException(status_code=503, detail="Model not loaded. API not ready for predictions.")
    
    if preprocessor is None:
        logger.error("Preprocessor is not loaded. API cannot make predictions.")
        raise HTTPException(status_code=503, detail="Preprocessor not loaded. API not ready for predictions.")

    try:
        # Convert Pydantic model to DataFrame - we need underscore column names for consistency
        # with most of the code and feature engineering
        input_dict = input_data.dict(by_alias=False)  # Use Python attribute names (underscores)
        input_df = pd.DataFrame([input_dict])
        logger.info(f"Received input for prediction (first row): \n{input_df.head(1).to_string()}")

        # Apply cleaning
        cleaned_df = clean_data(input_df.copy())
        logger.info("Applied clean_data successfully.")

        # Apply feature engineering
        engineered_df = engineer_features(cleaned_df.copy())
        logger.info("Applied engineer_features successfully.")
        
        # Prepare data for preprocessing (drop target-related columns)
        columns_to_drop = ['readmitted_binary', 'readmitted', 'age']  # Same as training script
        feature_df = engineered_df.drop(columns=columns_to_drop, errors='ignore')
        
        # Fix column name mismatch: API uses underscores but preprocessor expects hyphens
        column_rename_map = {
            'glyburide_metformin': 'glyburide-metformin',
            'glipizide_metformin': 'glipizide-metformin', 
            'glimepiride_pioglitazone': 'glimepiride-pioglitazone',
            'metformin_rosiglitazone': 'metformin-rosiglitazone',
            'metformin_pioglitazone': 'metformin-pioglitazone'
        }
        feature_df = feature_df.rename(columns=column_rename_map)
        logger.info(f"Feature columns for preprocessing: {list(feature_df.columns)}")
        
        # Apply the sklearn preprocessing pipeline (the missing step!)
        preprocessed_df = preprocess_data(feature_df, preprocessor, fit_preprocessor=False)
        logger.info(f"Applied preprocess_data successfully. Final shape: {preprocessed_df.shape}")
        logger.info(f"Final preprocessed columns: {list(preprocessed_df.columns)[:10]}...")  # Show first 10
        
        # Apply the model to preprocessed data
        prediction_result = model.predict(preprocessed_df)
        
        # Handle different prediction result formats
        if hasattr(prediction_result, 'shape') and len(prediction_result.shape) > 1:
            # If it's a 2D array (probabilities), get the class with highest probability
            prediction_val = int(prediction_result[0].argmax())
            positive_class_proba = float(prediction_result[0][1])  # Probability of class 1 (readmission)
        elif hasattr(prediction_result, '__len__') and len(prediction_result) > 1:
            # If it's a 1D array with multiple values, take the first
            prediction_val = int(prediction_result[0])
            positive_class_proba = float(prediction_result[0])
        else:
            # Single value
            prediction_val = int(prediction_result)
            positive_class_proba = float(prediction_result)
        
        logger.info(f"Prediction: {prediction_val}, Probability of readmission: {positive_class_proba:.4f}")

        return InferenceResponse(
            prediction=prediction_val,
            probability_score=positive_class_proba
        )
    except HTTPException: # Re-raise HTTPExceptions to return proper responses
        raise
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}")
        logger.exception("Full traceback for ValueError in prediction:")
        raise HTTPException(status_code=400, detail=f"Invalid input or data processing error: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        logger.exception("Full traceback for unexpected error in prediction:")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This is for local debugging. In production, Uvicorn will be run by Docker/K8s.
    uvicorn.run(app, host="0.0.0.0", port=8000) 