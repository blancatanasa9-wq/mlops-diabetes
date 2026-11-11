"""This module cleans the raw dataset by handling missing values, fixing data types, and removing outliers.
It outputs a processed dataset ready for machine learning."""

from pathlib import Path
import numpy as np
import pandas as pd

# Definición de rutas basadas en la estructura del profesor (subiendo 2 niveles desde src/data/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_raw.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "diabetes_clean.csv"

def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    return pd.read_csv(path)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in key features (like bmi or bp) with their median."""
    for col in ["bmi", "bp"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert specific columns to float, coercing errors to NaN."""
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the target value exceeds the 99th percentile."""
    if "target" in df.columns:
        threshold = df["target"].quantile(0.99)
        df = df[df["target"] < threshold]
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return the cleaned DataFrame."""
    df = handle_missing_values(df)
    df = fix_data_types(df)
    df = handle_outliers(df)
    return df

def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main() -> None:
    """Run the full preprocessing pipeline."""
    print("Loading raw data...")
    df = load_data(RAW_PATH)
    rows_before = len(df)
    print(f"Rows before cleaning: {rows_before}")
    
    print("Cleaning data...")
    df = clean_data(df)
    rows_after = len(df)
    print(f"Rows after cleaning: {rows_after}")
    
    print("Saving processed data...")
    save_data(df, PROCESSED_PATH)
    print(f"Final shape: {df.shape}")
    print(f"Output path: {PROCESSED_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()
