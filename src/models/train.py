"""
This module trains two regression models on the Diabetes dataset and tracks every run with MLflow Experiment Tracking.
For each model the script logs hyperparameters, evaluation metrics and the serialised model artifact so runs can be compared in the MLflow UI.
"""

import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ..............................................................................
# Paths & Constants
# ..............................................................................

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "diabetes_features.csv"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "diabetes-regression"
TARGET = "target"  # Columna objetivo para el dataset de diabetes
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ..............................................................................
# Logging configuration
# ..............................................................................

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ..............................................................................
# Data
# ..............................................................................

def load_features(path: Path) -> pd.DataFrame:
    """Load the engineered feature dataset from CSV.
    
    Args:
        path: Path to diabetes_features.csv.
        
    Returns:
        DataFrame with all features and target column.
        
    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the data pipeline first: python src/data/feature_engineering.py"
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test sets.
    
    Args:
        df: Full feature DataFrame including the target column.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        "Train size: %d | Test size: %d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test

# ..............................................................................
# Metrics
# ..............................................................................

def compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute regression evaluation metrics.
    
    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        
    Returns:
        Dictionary with rmse, mae, and r2 keys.
    """
    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

# ..............................................................................
# MLFlow run
# ..............................................................................

def run_experiment(
    model,
    model_type: str,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Train a model and log everything to a single MLflow run.
    The run name follows the convention: {model_type}_{YYYYMMDD_HHMMSS}
    so runs are identifiable at a glance in the UI.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "model_type": model_type,
            "dataset": INPUT_PATH.name,
            "stage": "training",
        })
        
        # Log hyperparameters
        mlflow.log_params(params)

        logger.info("[%s] Training model...", model_type)
        model.fit(X_train, y_train)

        # Predictions and metrics
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        logger.info(
            "[%s] Metrics -> RMSE: %.4f | MAE: %.4f | R2: %.4f",
            model_type,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )

        # Log trained model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None
        )
        logger.info("[%s] Model logged to MLflow successfully.", model_type)

# ..............................................................................
# Main
# ..............................................................................

def main() -> None:
    """Run the full experiment tracking pipeline for both models."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment name: %s", EXPERIMENT_NAME)

    # 1. Load data
    df = load_features(INPUT_PATH)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Model 1: LinearRegression (Baseline)
    run_experiment(
        model=LinearRegression(),
        model_type="LinearRegression",
        params={"fit_intercept": True},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # 4. Model 2: RandomForestRegressor
    run_experiment(
        model=RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE),
        model_type="RandomForestRegressor",
        params={"n_estimators": 100, "max_depth": 15, "random_state": RANDOM_STATE},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    logger.info(
        "Done. Open the MLflow UI to compare runs:\n"
        "mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )

if __name__ == "__main__":
    main()
