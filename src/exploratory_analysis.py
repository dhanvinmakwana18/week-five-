"""
Exploratory Data Analysis (EDA) module for UCI Wine Quality dataset.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


PHYSICOCHEMICAL_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


def run_exploratory_analysis(
    df: pd.DataFrame,
    tables_dir: Path = Path("outputs/tables")
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes comprehensive descriptive statistics, group comparisons,
    and correlation structures.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    tables_dir : Path
        Output directory for exported tables.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (overall_stats_df, group_stats_df, eda_summary_dict)
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall Feature Descriptive Statistics
    stats_list = []
    for col in PHYSICOCHEMICAL_FEATURES:
        series = df[col]
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        stats_list.append({
            "Feature": col,
            "Mean": round(float(series.mean()), 4),
            "Std": round(float(series.std()), 4),
            "Median": round(float(series.median()), 4),
            "IQR": round(iqr, 4),
            "Min": round(float(series.min()), 4),
            "Max": round(float(series.max()), 4),
            "Skewness": round(float(stats.skew(series)), 4),
            "Kurtosis": round(float(stats.kurtosis(series)), 4)
        })

    overall_stats_df = pd.DataFrame(stats_list)
    overall_stats_df.to_csv(tables_dir / "descriptive_statistics.csv", index=False)

    # 2. Descriptive Stats by Quality Class (Standard vs Good)
    group_records = []
    for col in PHYSICOCHEMICAL_FEATURES:
        std_vals = df[df["quality_label"] == 0][col]
        good_vals = df[df["quality_label"] == 1][col]
        group_records.append({
            "Feature": col,
            "Standard_Mean": round(float(std_vals.mean()), 4),
            "Standard_Std": round(float(std_vals.std()), 4),
            "Standard_Median": round(float(std_vals.median()), 4),
            "Good_Mean": round(float(good_vals.mean()), 4),
            "Good_Std": round(float(good_vals.std()), 4),
            "Good_Median": round(float(good_vals.median()), 4),
            "Mean_Difference": round(float(good_vals.mean() - std_vals.mean()), 4),
            "Relative_Change_%": round(
                float((good_vals.mean() - std_vals.mean()) / (std_vals.mean() + 1e-9) * 100), 2
            )
        })

    group_stats_df = pd.DataFrame(group_records)
    group_stats_df.to_csv(tables_dir / "group_descriptive_statistics.csv", index=False)

    # 3. Correlation Analysis (Pearson & Spearman)
    numeric_cols = PHYSICOCHEMICAL_FEATURES + ["quality", "quality_label"]
    corr_pearson = df[numeric_cols].corr(method="pearson").round(4)
    corr_spearman = df[numeric_cols].corr(method="spearman").round(4)

    corr_pearson.to_csv(tables_dir / "correlation_matrix_pearson.csv")
    corr_spearman.to_csv(tables_dir / "correlation_matrix_spearman.csv")

    # 4. Dataset Overview Summary Table
    dataset_summary = pd.DataFrame([
        {"Attribute": "Total Sample Size (N)", "Value": str(len(df))},
        {"Attribute": "Number of Physicochemical Features", "Value": str(len(PHYSICOCHEMICAL_FEATURES))},
        {"Attribute": "Sensory Quality Score Range", "Value": f"{df['quality'].min()} to {df['quality'].max()}"},
        {"Attribute": "Standard Quality Count (<6)", "Value": f"{(df['quality_label'] == 0).sum()} ({((df['quality_label'] == 0).mean() * 100):.2f}%)"},
        {"Attribute": "Good Quality Count (>=6)", "Value": f"{(df['quality_label'] == 1).sum()} ({((df['quality_label'] == 1).mean() * 100):.2f}%)"},
        {"Attribute": "Missing Values Count", "Value": "0 (100% complete)"},
        {"Attribute": "Duplicate Records Count", "Value": f"{df[PHYSICOCHEMICAL_FEATURES].duplicated().sum()} (retained distinct batches)"}
    ])
    dataset_summary.to_csv(tables_dir / "dataset_summary.csv", index=False)

    # Top correlates with quality
    top_pearson = corr_pearson["quality"].drop(["quality", "quality_label"]).sort_values(ascending=False).to_dict()
    top_spearman = corr_spearman["quality"].drop(["quality", "quality_label"]).sort_values(ascending=False).to_dict()

    eda_summary = {
        "dataset_summary": dataset_summary.to_dict(orient="records"),
        "top_positive_correlates_pearson": {k: v for k, v in top_pearson.items() if v > 0},
        "top_negative_correlates_pearson": {k: v for k, v in top_pearson.items() if v < 0},
        "top_spearman_correlations": top_spearman,
        "highest_positive_feature": max(top_pearson.items(), key=lambda x: x[1]),
        "highest_negative_feature": min(top_pearson.items(), key=lambda x: x[1])
    }

    print(f"[SUCCESS] EDA completed. Descriptive tables exported to {tables_dir.resolve()}")
    return overall_stats_df, group_stats_df, eda_summary


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaning import clean_and_prepare_data
    raw, _ = load_raw_data()
    clean, _, _, _ = clean_and_prepare_data(raw)
    run_exploratory_analysis(clean)
