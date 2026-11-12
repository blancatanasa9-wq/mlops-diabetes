"""
This module handles state management, champion model loading from MLflow registry,
and Pydantic schemas for data validation.
"""

import logging
import os
from typing import Any

import mlflow
from fastapi import HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# ..............................................................................
# Configuration & Constants
# ..............................................................................

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "DiabetesBestRegressor"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

logger = logging.getLogger(__name__)

# Diccionario de estado global utilizado por los endpoints de la API
_state: dict[str, Any] = {
    "model": None,
    "model_version": None,
    "run_id": None,
}

# ..............................................................................
# Model Loading & State Helpers
# ..............................................................................

def load_champion_model() -> None:
    """Load the champion model from the MLflow registry into _state.

    Sets the MLflow tracking URI, loads the pyfunc model, and fetches version metadata
    via MlflowClient. Raises RuntimeError if the alias does not exist.
    """
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info("Loading model from registry: %s", MODEL_URI)
    try:
        _state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
        client = MlflowClient()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        
        _state["model_version"] = str(mv.version)
        _state["run_id"] = mv.run_id

        logger.info(
            "Model loaded - version: %s | run_id: %s",
            mv.version,
            mv.run_id,
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise RuntimeError(
            f"Could not load '{MODEL_URI}'!\n"
            "Run evaluate_model.py first to assign the 'champion' alias."
        ) from exc


def get_model() -> Any:
    """Return the loaded model, or raise HTTP 503 if startup has not completed."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _state["model"]


def get_model_info() -> dict:
    """Return registry metadata for the loaded model, or raise HTTP 503 if not ready."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_version": _state["model_version"],
        "run_id": _state["run_id"],
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


# ..............................................................................
# Pydantic Schemas
# ..............................................................................

class DiabetesFeatures(BaseModel):
    """Features required to predict diabetes progression."""
    age: float = Field(..., description="Age of the patient")
    sex: float = Field(..., description="Sex of the patient")
    bmi: float = Field(..., description="Body Mass Index")
    bp: float = Field(..., description="Average blood pressure")
    s1: float = Field(..., description="Total serum cholesterol (tc)")
    s2: float = Field(..., description="Low-density lipoproteins (ldl)")
    s3: float = Field(..., description="High-density lipoproteins (hdl)")
    s4: float = Field(..., description="Total cholesterol / HDL (tch)")
    s5: float = Field(..., description="Log of serum triglycerides level (ltg)")
    s6: float = Field(..., description="Blood sugar level (glu)")


class PredictionResponse(BaseModel):
    """Response schema containing the model prediction."""
    prediction: float = Field(..., description="Predicted disease progression score")
    model_name: str = Field(..., description="Registered model name")
    model_version: str = Field(..., description="Model version loaded")
