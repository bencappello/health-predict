import logging
import os
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
MODEL_NAME = os.getenv("MODEL_NAME", "HealthPredict_RandomForest") 
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Global variable for the model
model = None

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
    return {"status": "healthy", "model_loaded": model is not None}

# --- Startup and Shutdown Events (Model loading will go here) ---
@app.on_event("startup")
async def load_model_on_startup():
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Loading model from URI: {model_uri} via MLflow server: {MLFLOW_TRACKING_URI}")
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model '{MODEL_NAME}' version from stage '{MODEL_STAGE}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Depending on policy, you might want to prevent startup or allow startup without model
        # For now, we\'ll let it start and log the error. Predictions will fail.
        model = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down.")

# --- API Endpoints --- 
@app.post("/predict", response_model=InferenceResponse)
async def predict(input_data: InferenceInput):
    global model
    if model is None:
        logger.error("Model is not loaded. API cannot make predictions.")
        raise HTTPException(status_code=503, detail="Model not loaded. API not ready for predictions.")

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
        
        # Apply the model
        prediction_val = model.predict(engineered_df)[0]
        proba_val = model.predict_proba(engineered_df)[0]
        
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