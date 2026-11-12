"""
This module connects to the MLflow tracking server, identifies the best-performing
run based on evaluation metrics (lowest RMSE), registers that model in the MLflow
Model Registry, and adds descriptive metadata to the registered version.
"""

import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# ..............................................................................
# Configuration & Constants
# ..............................................................................

BASE_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "diabetes-regression"
METRIC_FOR_BEST_MODEL = "rmse"  # Buscamos el menor error cuadrático medio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ..............................................................................
# Main Registration Logic
# ..............................................................................

def register_best_model() -> None:
    """Finds the best run in the experiment and registers its model artifact."""
    # 1. Configurar la URI de seguimiento
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    logger.info("Connecting to MLflow Tracking URI: %s", MLFLOW_TRACKING_URI)

    # 2. Obtener el experimento
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run training first.")

    experiment_id = experiment.experiment_id
    logger.info("Found experiment '%s' with ID: %s", EXPERIMENT_NAME, experiment_id)

    # 3. Buscar la mejor ejecución (ordenada por el menor RMSE)
    logger.info("Searching for the best run based on metric: %s (ASCENDING)", METRIC_FOR_BEST_MODEL)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{METRIC_FOR_BEST_MODEL} ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No runs found in this experiment. Train models before registering.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_type = best_run.data.tags.get("model_type", "UnknownModel")
    best_metric_value = best_run.data.metrics.get(METRIC_FOR_BEST_MODEL)

    logger.info(
        "Best run identified -> ID: %s | Model: %s | %s: %.4f",
        run_id,
        model_type,
        METRIC_FOR_BEST_MODEL.upper(),
        best_metric_value,
    )

    # 4. Registrar el modelo en el Model Registry
    # El artifact_path que usamos al entrenar fue "model"
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "DiabetesBestRegressor"

    logger.info("Registering model from URI: %s as '%s'...", model_uri, registered_model_name)
    
    model_version_details = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name
    )
    
    model_version = model_version_details.version
    logger.info("Model registered successfully. Version assigned: %s", model_version)

    # 5. Añadir metadatos y descripción a la versión del modelo
    description = (
        f"Automated registration for model type: {model_type}. "
        f"Selected based on best {METRIC_FOR_BEST_MODEL.upper()} of {best_metric_value:.4f}. "
        f"Associated Run ID: {run_id}"
    )
    
    client.update_model_version(
        name=registered_model_name,
        version=model_version,
        description=description
    )
    
    logger.info(
        "Model '%s' version %s is now in the registry with description added.",
        registered_model_name,
        model_version,
    )

if __name__ == "__main__":
    register_best_model()
