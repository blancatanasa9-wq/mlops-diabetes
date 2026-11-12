"""
This module evaluates the latest registered model version from the Model Registry
using test data, checks it against defined quality gates, and promotes it by assigning
the 'champion' alias if it passes.
"""

import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.entities
import pandas as pd
from mlflow.tracking import MlflowClient

# ..............................................................................
# Configuration & Constants
# ..............................................................................

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "diabetes_features.csv"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "diabetes-regression"
REGISTERED_MODEL_NAME = "DiabetesBestRegressor"
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Definimos el Quality Gate (por ejemplo, R² mínimo de 0.25 para aceptar el modelo)
MIN_R2_THRESHOLD = 0.25

# ..............................................................................
# Logging configuration
# ..............................................................................

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ..............................................................................
# Helper Functions
# ..............................................................................

def get_latest_version(client: MlflowClient, model_name: str) -> str:
    """Return the version number of the most recently registered model version.

    Searches all versions and returns the one with the highest version integer.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(
            f"No versions found for registered model '{model_name}'.\n"
            "Run register_model.py first."
        )

    latest = max(versions, key=lambda v: int(v.version))
    logger.info(
        "Latest version of '%s': %s (run id: %s)",
        model_name,
        latest.version,
        latest.run_id,
    )
    return latest.version


def load_test_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature dataset and extract the test split."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Usamos el mismo train_test_split con el mismo random_state para replicar exactamente el conjunto de test
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Test data loaded successfully. Test size: %d rows", len(X_test))
    return X_test, y_test


def evaluate_registered_model(
    client: MlflowClient,
    model_version: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Evaluate the registered model using mlflow.evaluate() on test data."""
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    logger.info("Evaluating model URI: %s", model_uri)

    # mlflow.evaluate() requiere un DataFrame que incluya la columna objetivo
    eval_data = X_test.copy()
    eval_data[TARGET] = y_test.values

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluation_v{model_version}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "stage": "evaluation",
            "model_version": model_version,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "dataset": INPUT_PATH.name,
        })

        # mlflow.evaluate calcula las métricas estándar de regresión automáticamente
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets=TARGET,
            model_type="regressor",
            evaluators="default",
        )

        logger.info("Evaluation metrics:")
        for key, value in sorted(result.metrics.items()):
            logger.info("  %-35s %.6f", key, value)

        return result


def passes_quality_gate(metrics: dict) -> bool:
    """Check if the model metrics satisfy the defined quality gate criteria."""
    r2 = metrics.get("r2_score", metrics.get("r2", float("-inf")))
    logger.info("Checking quality gate: R2 score = %.4f (Threshold: >= %.2f)", r2, MIN_R2_THRESHOLD)
    return r2 >= MIN_R2_THRESHOLD


def assign_champion_alias(
    client: MlflowClient,
    model_name: str,
    model_version: str,
) -> None:
    """Assign the 'champion' alias to a model version in MLflow 2.9+."""
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version,
    )
    logger.info(
        "Alias 'champion' assigned to '%s' version %s.",
        model_name,
        model_version,
    )
    logger.info(
        "Load the champion model with: mlflow.pyfunc.load_model('models:/%s@champion')",
        model_name,
    )


# ..............................................................................
# Main Pipeline
# ..............................................................................

def main() -> None:
    """Evaluate the latest registered model and promote it if it passes quality gates."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Registered model to evaluate: %s", REGISTERED_MODEL_NAME)

    client = MlflowClient()

    # 1. Resolver la última versión del modelo registrado
    model_version = get_latest_version(client, REGISTERED_MODEL_NAME)

    # 2. Cargar los datos de prueba
    X_test, y_test = load_test_data(INPUT_PATH)

    # 3. Evaluar el modelo mediante mlflow.evaluate()
    result = evaluate_registered_model(client, model_version, X_test, y_test)

    # 4. Comprobar el Quality Gate y promocionar si pasa el filtro
    if passes_quality_gate(result.metrics):
        logger.info("Model passed the quality gate!")
        assign_champion_alias(client, REGISTERED_MODEL_NAME, model_version)
    else:
        logger.warning(
            "Model version %s did NOT pass the quality gate. "
            "Alias 'champion' was NOT assigned.",
            model_version,
        )

    logger.info(
        "Done. Open the MLflow UI to review the evaluation run:\n"
        "mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )


if __name__ == "__main__":
    main()
