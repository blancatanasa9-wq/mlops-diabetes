"""
FastAPI application containing the lifecycle manager and routing endpoints.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.model import (
    REGISTERED_MODEL_NAME,
    _state,
    get_model,
    get_model_info,
    load_champion_model,
    DiabetesFeatures,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ..............................................................................
# Lifespan
# ..............................................................................

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Call load_champion_model() before the app starts accepting requests."""
    load_champion_model()
    yield
    # Shutdown hook: release model from memory
    logger.info("Shutting down: releasing model from memory.")
    _state["model"] = None
    _state["model_version"] = None
    _state["run_id"] = None


# ..............................................................................
# FastAPI App Instance & Endpoints
# ..............................................................................

app = FastAPI(
    title="Diabetes Regression MLOps API",
    description="API for serving predictions using MLflow Model Registry champion models.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Monitoring"])
def health() -> dict:
    """Return service liveness."""
    return {"status": "ok"}


@app.get("/model-info", tags=["Monitoring"])
def model_info() -> dict:
    """Return registry metadata for the loaded model."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: DiabetesFeatures) -> PredictionResponse:
    """Receive patient features and return the diabetes progression prediction."""
    model = get_model()

    try:
        # Convertimos los datos de entrada a DataFrame de Pandas
        input_data = pd.DataFrame([features.dict()])

        # Predicción
        predictions = model.predict(input_data)
        predicted_value = float(predictions[0])

        logger.info("Prediction generated successfully: %.4f", predicted_value)

        return PredictionResponse(
            prediction=predicted_value,
            model_name=REGISTERED_MODEL_NAME,
            model_version=_state.get("model_version", "champion"),
        )

    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
