"""
Main execution pipeline for Week 5 Comprehensive Data Science Project.
Orchestrates end-to-end data acquisition, cleaning, exploratory analysis,
statistical hypothesis testing, machine learning modeling, visualization rendering,
and automated Word document report generation.
"""

import sys
import time
from pathlib import Path

# Add project root directory to path for robust direct and module execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import load_raw_data
from src.data_cleaning import clean_and_prepare_data
from src.exploratory_analysis import run_exploratory_analysis
from src.statistical_analysis import run_statistical_analysis
from src.machine_learning import train_and_evaluate_models
from src.visualizations import generate_all_visualizations
from src.report_generator import generate_comprehensive_report


def run_pipeline() -> int:
    """
    Executes the end-to-end Data Science analytical pipeline.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    start_time = time.time()
    print("=" * 80)
    print("  WEEK 5 DATA SCIENCE CAPSTONE: COMPREHENSIVE WINE QUALITY PIPELINE")
    print("=" * 80)

    try:
        # Step 1: Data Acquisition & Validation
        print("\n[STEP 1/7] Loading and validating raw dataset...")
        raw_df, val_meta = load_raw_data()
        print(f"  -> Successfully loaded {val_meta['num_rows']} rows, {val_meta['num_columns']} columns.")

        # Step 2: Data Cleaning, Target Engineering & Partitioning
        print("\n[STEP 2/7] Cleaning data and creating stratified train/test partitions...")
        clean_df, train_df, test_df, clean_meta = clean_and_prepare_data(raw_df)
        print(f"  -> Cleaned dataset: {clean_meta['cleaned_rows']} rows.")
        print(f"  -> Train partition: {clean_meta['train_rows']} rows | Test partition: {clean_meta['test_rows']} rows.")

        # Step 3: Exploratory Data Analysis
        print("\n[STEP 3/7] Conducting exploratory data analysis & correlation profiling...")
        overall_stats, group_stats, eda_summary = run_exploratory_analysis(clean_df)
        print(f"  -> Strongest positive correlate: {eda_summary['highest_positive_feature'][0]} ({eda_summary['highest_positive_feature'][1]:.4f})")
        print(f"  -> Strongest negative correlate: {eda_summary['highest_negative_feature'][0]} ({eda_summary['highest_negative_feature'][1]:.4f})")

        # Step 4: Inferential Statistical Hypothesis Testing
        print("\n[STEP 4/7] Performing formal hypothesis testing (Mann-Whitney U, Welch's t, FDR correction)...")
        stat_df, norm_df, stat_summary = run_statistical_analysis(clean_df)
        print(f"  -> Significant features (alpha=0.05): {len(stat_summary['significant_features_alpha0.05'])} / 11")
        print(f"  -> Top effect size: {stat_summary['top_effect_sizes'][0]['Feature']} (d = {stat_summary['top_effect_sizes'][0]['Cohens_d']:.4f})")

        # Step 5: Machine Learning Modeling & Error Analysis
        print("\n[STEP 5/7] Training classification models & evaluating out-of-sample performance...")
        ml_results, metrics_df, feat_imp_df, error_df = train_and_evaluate_models(train_df, test_df)
        print(f"  -> Best Model: {ml_results['best_model_name']}")
        print(f"  -> Test Accuracy: {ml_results['best_accuracy']*100:.2f}% | Test ROC-AUC: {ml_results['best_roc_auc']:.4f}")

        # Step 6: Publication-Grade Visualizations
        print("\n[STEP 6/7] Rendering 9 high-resolution publication figures...")
        fig_map = generate_all_visualizations(clean_df, ml_results)
        print(f"  -> Generated {len(fig_map)} figures in outputs/figures/")

        # Step 7: Comprehensive Report Generation (DOCX)
        print("\n[STEP 7/7] Generating executive-ready comprehensive Word document report...")
        report_path = generate_comprehensive_report(clean_df, ml_results, fig_map)
        print(f"  -> Report created at: {report_path.resolve()}")

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS (EXIT 0)")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline encountered an exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())
