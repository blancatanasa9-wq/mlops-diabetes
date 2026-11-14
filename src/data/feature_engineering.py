"""This module applies feature engineering techniques (such as scaling) to the preprocessed dataset, preparing it for model training."""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "diabetes_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "diabetes_features.csv"

def load_data(path: Path) -> pd.DataFrame:
    """Load the preprocessed dataset."""
    return pd.read_csv(path)

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical feature columns using StandardScaler, keeping target intact."""
    # Separar características (features) y variable objetivo (target) si existe
    if "target" in df.columns:
        features = df.drop(columns=["target"])
        target = df["target"]
    else:
        features = df
        target = None

    # Aplicar StandardScaler para normalizar los datos
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features)
    
    # Reconstruir el DataFrame manteniendo los nombres de las columnas
    scaled_df = pd.DataFrame(scaled_array, columns=features.columns)
    
    # Volver a adjuntar el target si existía
    if target is not None:
        scaled_df["target"] = target.values
        
    return scaled_df

def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save feature-engineered DataFrame to CSV without index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main() -> None:
    """Run the feature engineering pipeline."""
    print(f"Loading preprocessed data from: {INPUT_PATH}")
    df = load_data(INPUT_PATH)
    
    print("Applying feature engineering (StandardScaler)...")
    df_engineered = scale_features(df)
    
    print(f"Saving engineered features to: {OUTPUT_PATH}")
    save_data(df_engineered, OUTPUT_PATH)
    print(f"Final shape after features: {df_engineered.shape}")
    print("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
