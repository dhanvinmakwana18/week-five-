"""
Statistical hypothesis testing and inferential analysis module for wine quality characteristics.
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


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Calculates Cohen's d effect size for two independent samples."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_sd)


def calculate_rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Calculates rank-biserial correlation effect size for Mann-Whitney U."""
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


def run_statistical_analysis(
    df: pd.DataFrame,
    tables_dir: Path = Path("outputs/tables")
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes formal statistical hypothesis testing on physicochemical features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned wine dataset.
    tables_dir : Path
        Output directory for exported statistical tables.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (statistical_tests_df, normality_tests_df, statistical_summary_dict)
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    group_standard = df[df["quality_label"] == 0]
    group_good = df[df["quality_label"] == 1]
    n_standard = len(group_standard)
    n_good = len(group_good)

    # 1. Normality and Variance Homogeneity Testing
    normality_records = []
    for feat in PHYSICOCHEMICAL_FEATURES:
        std_vals = group_standard[feat]
        good_vals = group_good[feat]

        # Shapiro-Wilk test (subsampled if > 5000, here N=744 and 855)
        # Note: Scipy shapiro supports up to 5000
        sw_stat_std, sw_p_std = stats.shapiro(std_vals)
        sw_stat_good, sw_p_good = stats.shapiro(good_vals)

        # Levene's test for equality of variance
        lev_stat, lev_p = stats.levene(std_vals, good_vals)

        normality_records.append({
            "Feature": feat,
            "Shapiro_Stat_Standard": round(float(sw_stat_std), 4),
            "Shapiro_P_Standard": float(sw_p_std),
            "Normal_Standard_Alpha0.05": bool(sw_p_std > 0.05),
            "Shapiro_Stat_Good": round(float(sw_stat_good), 4),
            "Shapiro_P_Good": float(sw_p_good),
            "Normal_Good_Alpha0.05": bool(sw_p_good > 0.05),
            "Levene_Stat": round(float(lev_stat), 4),
            "Levene_P": float(lev_p),
            "Equal_Variance_Alpha0.05": bool(lev_p > 0.05)
        })

    normality_df = pd.DataFrame(normality_records)
    normality_df.to_csv(tables_dir / "normality_and_variance_tests.csv", index=False)

    # 2. Comprehensive Two-Sample Hypothesis Tests
    test_records = []
    for feat in PHYSICOCHEMICAL_FEATURES:
        std_vals = group_standard[feat]
        good_vals = group_good[feat]

        # Non-parametric Mann-Whitney U test (primary robust test due to non-normality)
        u_stat, mw_p = stats.mannwhitneyu(good_vals, std_vals, alternative="two-sided")
        rank_biserial = calculate_rank_biserial(u_stat, n_good, n_standard)

        # Parametric Welch's t-test (secondary reference)
        t_stat, t_p = stats.ttest_ind(good_vals, std_vals, equal_var=False)
        cohens_d = calculate_cohens_d(good_vals, std_vals)

        # Kruskal-Wallis H-test across all 6 discrete quality scores (3, 4, 5, 6, 7, 8)
        kw_groups = [group[feat].values for _, group in df.groupby("quality")]
        kw_stat, kw_p = stats.kruskal(*kw_groups)

        # Interpretation of direction and practical significance
        if mw_p < 0.001:
            sig_label = "p < 0.001 (Highly Significant)"
        elif mw_p < 0.05:
            sig_label = f"p = {mw_p:.4f} (Significant)"
        else:
            sig_label = f"p = {mw_p:.4f} (Not Significant)"

        if abs(cohens_d) >= 0.8:
            effect_label = "Large"
        elif abs(cohens_d) >= 0.5:
            effect_label = "Medium"
        elif abs(cohens_d) >= 0.2:
            effect_label = "Small"
        else:
            effect_label = "Negligible"

        direction = "Elevated in Good Quality" if good_vals.mean() > std_vals.mean() else "Suppressed in Good Quality"

        test_records.append({
            "Feature": feat,
            "H0": f"Distribution of {feat} is identical between Good and Standard quality wines",
            "H1": f"Distribution of {feat} differs significantly between Good and Standard quality wines",
            "Mann_Whitney_U": round(float(u_stat), 2),
            "Mann_Whitney_P": float(mw_p),
            "Rank_Biserial_Effect": round(rank_biserial, 4),
            "Welch_t_Stat": round(float(t_stat), 4),
            "Welch_t_P": float(t_p),
            "Cohens_d": round(cohens_d, 4),
            "Effect_Size_Magnitude": effect_label,
            "Kruskal_Wallis_H": round(float(kw_stat), 4),
            "Kruskal_Wallis_P": float(kw_p),
            "Significance": sig_label,
            "Observed_Trend": direction
        })

    tests_df = pd.DataFrame(test_records)

    # Benjamini-Hochberg FDR correction for multiple comparisons
    p_vals = tests_df["Mann_Whitney_P"].values
    sorted_indices = np.argsort(p_vals)
    n_tests = len(p_vals)
    fdr_adjusted_p = np.zeros(n_tests)
    for rank, idx in enumerate(sorted_indices, 1):
        fdr_adjusted_p[idx] = min(p_vals[idx] * n_tests / rank, 1.0)
    # Ensure monotonicity
    for i in range(len(sorted_indices) - 2, -1, -1):
        idx_curr = sorted_indices[i]
        idx_next = sorted_indices[i + 1]
        fdr_adjusted_p[idx_curr] = min(fdr_adjusted_p[idx_curr], fdr_adjusted_p[idx_next])

    tests_df["FDR_Adjusted_P"] = [round(float(p), 6) for p in fdr_adjusted_p]
    tests_df.to_csv(tables_dir / "statistical_tests.csv", index=False)

    # Summary key insights
    significant_features = tests_df[tests_df["Mann_Whitney_P"] < 0.05]["Feature"].tolist()
    top_effects = tests_df.sort_values(by="Cohens_d", key=abs, ascending=False).head(4)[
        ["Feature", "Cohens_d", "Effect_Size_Magnitude", "Observed_Trend", "Mann_Whitney_P"]
    ].to_dict(orient="records")

    statistical_summary = {
        "total_hypotheses_tested": len(PHYSICOCHEMICAL_FEATURES),
        "significant_features_alpha0.05": significant_features,
        "non_significant_features": [f for f in PHYSICOCHEMICAL_FEATURES if f not in significant_features],
        "top_effect_sizes": top_effects,
        "normality_summary": {
            "all_features_non_normal": bool((normality_df["Normal_Standard_Alpha0.05"] == False).all())
        }
    }

    print(f"[SUCCESS] Statistical analysis completed. Exported to {tables_dir.resolve()}")
    return tests_df, normality_df, statistical_summary


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaning import clean_and_prepare_data
    raw, _ = load_raw_data()
    clean, _, _, _ = clean_and_prepare_data(raw)
    run_statistical_analysis(clean)
