import logging
import os
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib

# Attempt to import feature engineering functions
try:
    from src.feature_engineering import clean_data, engineer_features
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Health Predict API", version="0.1.0")

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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# Specific model name and stage will be determined by the best model registered.
# These are placeholders and might be loaded dynamically or set more specifically.
DEFAULT_MODEL_NAME = "HealthPredict_RandomForest" # Example, will be refined
DEFAULT_MODEL_STAGE = "Production"

# Global variable to hold the loaded model and preprocessor
model_pipeline = None
preprocessor = None # Added global for preprocessor

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
@app.get("/health")
async def health():
    # Basic health check
    if model_pipeline is not None:
        return {"status": "ok", "message": "API is healthy and model is loaded."}
    else:
        # This case might indicate the startup event hasn't finished or failed
        return {"status": "ok", "message": "API is healthy, but model is not loaded yet or loading failed."}

# --- Startup and Shutdown Events (Model loading will go here) ---
@app.on_event("startup")
async def startup_event():
    global model_pipeline, preprocessor # Include preprocessor
    logger.info("Attempting to load model and preprocessor on startup...")
    try:
        logger.info(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        model_uri = f"models:/{DEFAULT_MODEL_NAME}/{DEFAULT_MODEL_STAGE}"
        logger.info(f"Attempting to load model from URI: {model_uri}")
        
        # Load the model itself
        model_pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model '{DEFAULT_MODEL_NAME}' from stage '{DEFAULT_MODEL_STAGE}'.")

        # Find the run ID associated with this model version using stage instead of alias
        model_versions = client.get_latest_versions(DEFAULT_MODEL_NAME, stages=[DEFAULT_MODEL_STAGE])
        if not model_versions:
            raise Exception(f"No model version found for {DEFAULT_MODEL_NAME} in stage {DEFAULT_MODEL_STAGE}")
            
        model_version = model_versions[0]  # Get the first version in that stage
        source_run_id = model_version.run_id
        logger.info(f"Model version {model_version.version} in stage '{DEFAULT_MODEL_STAGE}' originates from run ID: {source_run_id}")

        # Download the preprocessor artifact from that run
        local_preprocessor_dir = "./downloaded_artifacts"
        os.makedirs(local_preprocessor_dir, exist_ok=True)
        local_path = client.download_artifacts(run_id=source_run_id, path="preprocessor", dst_path=local_preprocessor_dir)
        preprocessor_path = os.path.join(local_path, "preprocessor.joblib") # Construct full path
        logger.info(f"Downloaded preprocessor artifact to {preprocessor_path}")
        
        # Load the preprocessor
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Successfully loaded preprocessor from {preprocessor_path}")
        else:
             logger.error(f"Preprocessor artifact not found at expected path: {preprocessor_path}")
             preprocessor = None

    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        logger.exception("Full traceback:") # Logs the full exception traceback
        model_pipeline = None
        preprocessor = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down.")

# --- API Endpoints --- 
@app.post("/predict", response_model=InferenceResponse)
async def predict(input_data: InferenceInput):
    global model_pipeline, preprocessor # Include preprocessor
    if model_pipeline is None or preprocessor is None:
        logger.error("Model pipeline or preprocessor is not loaded. API cannot make predictions.")
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded. API not ready for predictions.")

    try:
        # SPECIAL HANDLING FOR COLUMN NAME MISMATCH
        # The categorical transformer in the preprocessor expects columns like 'glyburide-metformin'
        # But we're receiving them as 'glyburide_metformin'
        # Get the information from the preprocessor
        categorical_columns_with_hyphens = []
        medication_cols_map = {}
        try:
            from sklearn.compose import ColumnTransformer
            if isinstance(preprocessor, ColumnTransformer):
                # Get the column names from the ColumnTransformer
                transformers = preprocessor.transformers
                for name, transformer, cols in transformers:
                    if name == 'cat' and cols:
                        for col in cols:
                            if '-' in col and col.endswith('-metformin') or col.endswith('-pioglitazone') or col.endswith('-rosiglitazone'):
                                categorical_columns_with_hyphens.append(col)
                                # Create a mapping from underscore to hyphen version
                                medication_cols_map[col.replace('-', '_')] = col
            logger.info(f"Identified column name mapping needed: {medication_cols_map}")
        except Exception as e:
            logger.warning(f"Error extracting column names from preprocessor: {e}")
            
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
        
        # Add back the columns with hyphens that the categorical transformer expects
        # We need to rename 'glyburide_metformin' to 'glyburide-metformin' etc.
        engineered_copy = engineered_df.copy()
        
        if medication_cols_map:
            # First create new columns with hyphenated names, copying values from the underscore columns
            for underscore_name, hyphen_name in medication_cols_map.items():
                if underscore_name in engineered_copy.columns:
                    engineered_copy[hyphen_name] = engineered_copy[underscore_name]
                    # We keep both versions for now
        
        # Now we have dataframe with BOTH 'glyburide_metformin' AND 'glyburide-metformin' columns
        logger.info(f"Preprocessor input data shape: {engineered_copy.shape}")
        logger.info(f"Preprocessor columns: {engineered_copy.columns.tolist()}")
        
        # Apply the preprocessor
        preprocessed_data = preprocessor.transform(engineered_copy)
        logger.info(f"Preprocessing successful. Output shape: {preprocessed_data.shape}")
        
        # Perform prediction using the model and preprocessed data
        prediction_val = model_pipeline.predict(preprocessed_data)[0]
        proba_val = model_pipeline.predict_proba(preprocessed_data)[0]
        
        # The second element in proba_val corresponds to the probability of the positive class (readmission)
        positive_class_proba = proba_val[1]
        
        logger.info(f"Prediction: {prediction_val}, Probability of readmission: {positive_class_proba:.4f}")

        return InferenceResponse(
            prediction=int(prediction_val),
            probability_score=float(positive_class_proba)
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