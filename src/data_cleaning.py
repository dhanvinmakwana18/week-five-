"""
Data cleaning, validation, feature engineering, and train/test partition module.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_and_prepare_data(
    df: pd.DataFrame,
    processed_dir: Path = Path("data/processed"),
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw wine quality data, creates classification targets and domain features,
    and performs a stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw wine dataset.
    processed_dir : Path
        Directory where processed and partitioned datasets are saved.
    test_size : float
        Proportion of dataset reserved for the holdout test set (default: 0.20).
    random_state : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (df_cleaned, train_df, test_df, cleaning_metadata)
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_clean = df.copy()

    # Column name formatting (replace spaces with underscores for robust programmatic access)
    raw_col_names = list(df_clean.columns)
    cleaned_col_map = {col: col.strip().replace(" ", "_") for col in df_clean.columns}
    df_clean.rename(columns=cleaned_col_map, inplace=True)

    # Check for duplicate observations
    num_duplicates = int(df_clean.duplicated().sum())

    # Create target variables
    # Binary Target: 0 = Standard Quality (scores 3-5), 1 = Good Quality (scores 6-8)
    df_clean["quality_label"] = (df_clean["quality"] >= 6).astype(int)

    # 3-tier categorical target for exploratory granularity
    def categorize_quality(score: int) -> str:
        if score <= 4:
            return "Low (3-4)"
        elif score <= 6:
            return "Medium (5-6)"
        else:
            return "High (7-8)"

    df_clean["quality_category"] = df_clean["quality"].apply(categorize_quality)

    # Domain engineered features
    # Total acidity: composite sum of organic acid concentrations
    df_clean["total_acidity"] = (
        df_clean["fixed_acidity"] + df_clean["volatile_acidity"] + df_clean["citric_acid"]
    )
    # Bound sulfur dioxide: difference between total and free SO2
    df_clean["bound_sulfur_dioxide"] = (
        df_clean["total_sulfur_dioxide"] - df_clean["free_sulfur_dioxide"]
    )
    # Ensure bound sulfur dioxide has no negative values due to sensor variance
    df_clean["bound_sulfur_dioxide"] = df_clean["bound_sulfur_dioxide"].clip(lower=0.0)

    # Stratified Train/Test Split
    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=df_clean["quality_label"]
    )

    # Export processed data files
    full_path = processed_dir / "cleaned_wine_data.csv"
    train_path = processed_dir / "train_data.csv"
    test_path = processed_dir / "test_data.csv"

    df_clean.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "original_rows": len(df),
        "cleaned_rows": len(df_clean),
        "num_duplicate_rows": num_duplicates,
        "duplicate_retention_rationale": (
            "The dataset contains 240 duplicate feature/label records. Because the available dataset does not provide a "
            "unique production-batch identifier, these records cannot be independently verified as repeated measurements "
            "from the same production batch. Therefore, the duplicate records were retained rather than removed, and this "
            "decision is acknowledged as a limitation of the dataset."
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_split_ratio": test_size,
        "random_seed": random_state,
        "target_distribution_overall": df_clean["quality_label"].value_counts().to_dict(),
        "target_distribution_train": train_df["quality_label"].value_counts().to_dict(),
        "target_distribution_test": test_df["quality_label"].value_counts().to_dict(),
        "features_list": [
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol"
        ],
        "engineered_features": ["total_acidity", "bound_sulfur_dioxide"]
    }

    print(f"[SUCCESS] Processed dataset saved to {full_path.resolve()}")
    print(f"[INFO] Train set: {len(train_df)} rows | Test set: {len(test_df)} rows.")

    return df_clean, train_df, test_df, metadata


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df_raw, _ = load_raw_data()
    df_clean, train_df, test_df, meta = clean_and_prepare_data(df_raw)
    print("Cleaning Metadata:", meta)
