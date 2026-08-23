"""
Visualization generation module for publication-grade exploratory, statistical, and ML figures.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix


# Color Palette constants (Executive & Professional)
NAVY = "#1B365D"
SLATE = "#4A5568"
TEAL = "#0D9488"
CRIMSON = "#BE123C"
AMBER = "#D97706"
LIGHT_BG = "#F8FAFC"
ACCENT_BLUE = "#2563EB"
ACCENT_GREEN = "#16A34A"


def set_plot_style():
    """Applies a clean, modern, professional visual theme."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })


def generate_all_visualizations(
    df: pd.DataFrame,
    ml_results: Dict[str, Any],
    figures_dir: Path = Path("outputs/figures")
) -> Dict[str, Path]:
    """
    Generates all 9 required figures and saves them as high-resolution PNGs.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned wine quality dataset.
    ml_results : Dict[str, Any]
        Dictionary returned by machine_learning.train_and_evaluate_models.
    figures_dir : Path
        Directory to save generated figures.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping figure names to their saved filepaths.
    """
    set_plot_style()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_figures = {}

    # -------------------------------------------------------------
    # 1. Target Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Discrete Quality Scores
    score_counts = df["quality"].value_counts().sort_index()
    palette_scores = sns.color_palette("Blues_r", len(score_counts))
    bars1 = axes[0].bar(score_counts.index, score_counts.values, color=palette_scores, edgecolor="black", alpha=0.85)
    axes[0].set_title("Sensory Quality Score Distribution (Discrete 3–8)", fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Sensory Quality Score")
    axes[0].set_ylabel("Number of Observations")
    for bar in bars1:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval}\n({pct:.1f}%)",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Binary Quality Class
    bin_counts = df["quality_label"].value_counts().sort_index()
    bin_labels = ["Standard Quality (<6)", "Good Quality (>=6)"]
    bars2 = axes[1].bar(bin_labels, bin_counts.values, color=[CRIMSON, TEAL], edgecolor="black", alpha=0.85, width=0.55)
    axes[1].set_title("Binary Target Class Balance", fontweight="bold", color=NAVY)
    axes[1].set_ylabel("Number of Observations")
    for bar in bars2:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 15, f"{yval}\n({pct:.1f}%)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p1 = figures_dir / "01_target_distribution.png"
    plt.savefig(p1)
    plt.close()
    generated_figures["target_distribution"] = p1

    # -------------------------------------------------------------
    # 2. Physicochemical Distributions
    # -------------------------------------------------------------
    features = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        sns.histplot(data=df, x=feat, hue="quality_label", kde=True, ax=ax,
                     palette={0: CRIMSON, 1: TEAL}, alpha=0.4, element="step")
        clean_name = feat.replace("_", " ").title()
        ax.set_title(clean_name, fontweight="bold", fontsize=11, color=NAVY)
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=9)
        if idx == 0:
            ax.legend(title="Quality", labels=["Good (>=6)", "Standard (<6)"], fontsize=8)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    # Hide extra subplot in 3x4 grid
    fig.delaxes(axes[11])

    fig.suptitle("Physicochemical Feature Distributions Stratified by Wine Quality Class",
                 fontsize=15, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    p2 = figures_dir / "02_physicochemical_distributions.png"
    plt.savefig(p2)
    plt.close()
    generated_figures["physicochemical_distributions"] = p2

    # -------------------------------------------------------------
    # 3. Correlation Matrix Heatmap
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    corr_cols = features + ["quality"]
    labels = [c.replace("_", " ").title() for c in corr_cols]

    p_corr = df[corr_cols].corr(method="pearson")
    s_corr = df[corr_cols].corr(method="spearman")

    mask = np.triu(np.ones_like(p_corr, dtype=bool), k=1)

    sns.heatmap(p_corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, ax=axes[0], xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Linear Correlation Matrix", fontweight="bold", color=NAVY)
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(s_corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, ax=axes[1], xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Spearman Monotonic Correlation Matrix", fontweight="bold", color=NAVY)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    p3 = figures_dir / "03_correlation_matrix.png"
    plt.savefig(p3)
    plt.close()
    generated_figures["correlation_matrix"] = p3

    # -------------------------------------------------------------
    # 4. Key Features by Quality Boxplots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    key_drivers = [
        ("alcohol", "Alcohol (% vol.)"),
        ("volatile_acidity", "Volatile Acidity (g/dm³)"),
        ("sulphates", "Sulphates (g/dm³)"),
        ("citric_acid", "Citric Acid (g/dm³)")
    ]

    for idx, (feat, title) in enumerate(key_drivers):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        sns.boxplot(data=df, x="quality", y=feat, hue="quality", ax=ax, palette="Blues_r", legend=False, boxprops=dict(alpha=0.8))
        sns.stripplot(data=df, x="quality", y=feat, ax=ax, color="black", alpha=0.15, jitter=0.2, size=3)
        ax.set_title(f"{title} across Quality Ratings", fontweight="bold", color=NAVY)
        ax.set_xlabel("Sensory Quality Score (3 to 8)")
        ax.set_ylabel(title)

    plt.tight_layout()
    p4 = figures_dir / "04_key_features_by_quality_boxplot.png"
    plt.savefig(p4)
    plt.close()
    generated_figures["key_features_boxplots"] = p4

    # -------------------------------------------------------------
    # 5. Model Performance Comparison
    # -------------------------------------------------------------
    metrics_df = ml_results["metrics_df"]
    fig, ax = plt.subplots(figsize=(11, 5.5))

    plot_metrics = ["Test_Accuracy", "Test_Precision", "Test_Recall_Sensitivity", "Test_Macro_F1", "Test_ROC_AUC"]
    plot_labels = ["Accuracy", "Precision", "Recall", "Macro F1", "ROC-AUC"]

    x = np.arange(len(plot_labels))
    width = 0.25

    model_colors = {"Logistic Regression": NAVY, "Decision Tree": AMBER, "Random Forest": TEAL}

    for i, row in metrics_df.iterrows():
        model_name = row["Model"]
        values = [row[m] for m in plot_metrics]
        rects = ax.bar(x + (i - 1) * width, values, width, label=model_name,
                       color=model_colors.get(model_name, SLATE), edgecolor="black", alpha=0.85)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Performance Score (0.0 – 1.0)")
    ax.set_title("Comparative Model Performance on Holdout Test Set (N=320)", fontweight="bold", color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    p5 = figures_dir / "05_model_performance_comparison.png"
    plt.savefig(p5)
    plt.close()
    generated_figures["model_performance"] = p5

    # -------------------------------------------------------------
    # 6. Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    y_test = ml_results["y_test"]

    for idx, (name, preds) in enumerate(ml_results["predictions"].items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, preds["y_pred"])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        annot_text = np.array([
            [f"{count}\n({pct:.1%})" for count, pct in zip(c_row, n_row)]
            for c_row, n_row in zip(cm, cm_norm)
        ])

        sns.heatmap(cm, annot=annot_text, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred: Std (<6)", "Pred: Good (>=6)"],
                    yticklabels=["Actual: Std (<6)", "Actual: Good (>=6)"])
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold", color=NAVY)

    plt.tight_layout()
    p6 = figures_dir / "06_confusion_matrices.png"
    plt.savefig(p6)
    plt.close()
    generated_figures["confusion_matrices"] = p6

    # -------------------------------------------------------------
    # 7. ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, preds in ml_results["predictions"].items():
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {roc_auc_val:.3f})",
                color=model_colors.get(name, SLATE))

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", color=NAVY)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    p7 = figures_dir / "07_roc_curves.png"
    plt.savefig(p7)
    plt.close()
    generated_figures["roc_curves"] = p7

    # -------------------------------------------------------------
    # 8. Feature Importance Comparison
    # -------------------------------------------------------------
    feat_df = ml_results["feature_importance_df"].copy()
    feat_df.sort_values(by="RF_Permutation_Importance_Mean", ascending=True, inplace=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Permutation Importance
    y_pos = np.arange(len(feat_df))
    clean_feature_names = [f.replace("_", " ").title() for f in feat_df["Feature"]]

    axes[0].barh(y_pos, feat_df["RF_Permutation_Importance_Mean"],
                 xerr=feat_df["RF_Permutation_Importance_Std"],
                 color=TEAL, edgecolor="black", alpha=0.85, capsize=3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(clean_feature_names, fontweight="bold")
    axes[0].set_xlabel("Mean Decrease in Accuracy (Permutation)")
    axes[0].set_title("Random Forest Permutation Importance\n(Out-of-Sample Test Set)", fontweight="bold", color=NAVY)

    # Logistic Regression Standardized Coefficients
    feat_df_lr = ml_results["feature_importance_df"].sort_values(by="Logistic_Std_Coefficient", ascending=True)
    lr_colors = [CRIMSON if c < 0 else ACCENT_BLUE for c in feat_df_lr["Logistic_Std_Coefficient"]]
    axes[1].barh(np.arange(len(feat_df_lr)), feat_df_lr["Logistic_Std_Coefficient"],
                 color=lr_colors, edgecolor="black", alpha=0.85)
    axes[1].set_yticks(np.arange(len(feat_df_lr)))
    axes[1].set_yticklabels([f.replace("_", " ").title() for f in feat_df_lr["Feature"]], fontweight="bold")
    axes[1].axvline(0, color="black", linestyle="--", lw=1)
    axes[1].set_xlabel("Standardized Logistic Regression Coefficient (Log-Odds)")
    axes[1].set_title("Standardized Logistic Regression Coefficients\n(Direction & Magnitude)", fontweight="bold", color=NAVY)

    plt.tight_layout()
    p8 = figures_dir / "08_feature_importance.png"
    plt.savefig(p8)
    plt.close()
    generated_figures["feature_importance"] = p8

    # -------------------------------------------------------------
    # 9. Error Analysis Distribution Profile
    # -------------------------------------------------------------
    test_analysis = ml_results["test_analysis_df"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Scatter plot: Alcohol vs Volatile Acidity with Classification Outcome
    palette_errors = {
        "True Positive": TEAL,
        "True Negative": NAVY,
        "False Positive (Type I)": AMBER,
        "False Negative (Type II)": CRIMSON
    }

    sns.scatterplot(
        data=test_analysis,
        x="volatile_acidity",
        y="alcohol",
        hue="error_type",
        palette=palette_errors,
        style="error_type",
        s=70,
        alpha=0.85,
        ax=axes[0]
    )
    axes[0].set_title("Misclassification Distribution in Primary Feature Space\n(Alcohol vs. Volatile Acidity)",
                      fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Volatile Acidity (g/dm³)")
    axes[0].set_ylabel("Alcohol (% vol.)")
    axes[0].legend(title="Outcome", loc="upper right", frameon=True, fontsize=9)

    # Predicted Probability Distribution by True Class
    sns.histplot(
        data=test_analysis,
        x="predicted_probability",
        hue="quality_label",
        palette={0: CRIMSON, 1: TEAL},
        kde=True,
        bins=15,
        element="step",
        ax=axes[1]
    )
    axes[1].axvline(0.5, color="black", linestyle="--", lw=1.5, label="Decision Threshold (0.50)")
    axes[1].set_title("Predicted Probability Distribution by True Class\n(Overlap indicates classification uncertainty)",
                      fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Predicted Probability of Good Quality (P >= 0.50)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(title="True Label", labels=["Decision Boundary", "Good (>=6)", "Standard (<6)"], fontsize=9)

    plt.tight_layout()
    p9 = figures_dir / "09_error_analysis_residual_profiles.png"
    plt.savefig(p9)
    plt.close()
    generated_figures["error_analysis"] = p9

    print(f"[SUCCESS] All {len(generated_figures)} publication figures generated at {figures_dir.resolve()}")
    return generated_figures


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaning import clean_and_prepare_data
    from src.machine_learning import train_and_evaluate_models

    raw, _ = load_raw_data()
    clean, train, test, _ = clean_and_prepare_data(raw)
    ml_res, _, _, _ = train_and_evaluate_models(train, test)
    generate_all_visualizations(clean, ml_res)
