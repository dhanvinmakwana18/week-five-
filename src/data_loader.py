"""
Data loading and acquisition module for the UCI Wine Quality dataset.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import urllib.request
import pandas as pd


UCI_RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
]


def load_raw_data(
    raw_data_path: Path = Path("data/raw/winequality-red.csv"),
    download_url: str = UCI_RED_WINE_URL,
    force_download: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw red wine quality data from local cache or downloads from UCI repository.

    Parameters
    ----------
    raw_data_path : Path
        Local filesystem destination for raw CSV.
    download_url : str
        UCI repository URL for red wine dataset.
    force_download : bool
        Whether to re-download if file already exists.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Loaded raw DataFrame and validation metadata dictionary.
    """
    raw_data_path = Path(raw_data_path)
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists() or force_download:
        print(f"[INFO] Fetching raw dataset from {download_url}...")
        try:
            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                raw_data_path.write_bytes(content)
            print(f"[SUCCESS] Dataset successfully saved to {raw_data_path.resolve()}")
        except Exception as e:
            if raw_data_path.exists():
                print(f"[WARNING] Download failed ({e}), using cached local raw file.")
            else:
                raise RuntimeError(f"Failed to acquire dataset from {download_url}: {e}") from e

    df = pd.read_csv(raw_data_path, sep=";")

    # Validate dataset structure
    validation_info = validate_raw_data(df)
    print(f"[INFO] Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    return df, validation_info


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates schema, column names, missing values, and data types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation findings.
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw dataset is missing expected columns: {missing_cols}")

    null_counts = df.isnull().sum().to_dict()
    total_missing = sum(null_counts.values())

    validation_summary = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "total_missing_values": int(total_missing),
        "column_missing_breakdown": null_counts,
        "quality_range": [int(df["quality"].min()), int(df["quality"].max())],
        "quality_distribution": df["quality"].value_counts().sort_index().to_dict(),
        "is_valid": total_missing == 0 and len(df) == 1599
    }

    return validation_summary


if __name__ == "__main__":
    df_raw, val_meta = load_raw_data()
    print("Validation Metadata:", val_meta)
