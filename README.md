# Week 5 Data Science Capstone: Comprehensive Wine Quality Analytics & Strategic Recommendations

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Clean](https://img.shields.io/badge/Code%20Style-PEP8-teal.svg)](https://pep8.org/)
[![Pipeline: Validated](https://img.shields.io/badge/Pipeline-Validated%20(Exit%200)-success.svg)]()

An end-to-end Data Science capstone project synthesizing exploratory data analysis, non-parametric inferential hypothesis testing, leakage-free supervised machine learning classification, permutation feature interpretability, granular error diagnostics, and data-driven strategic operational recommendations for viticulture and enology stakeholders.

---

## Executive Summary & Core Research Question

### Primary Research Question
> *"Can objective physicochemical characteristics be utilized to reliably classify red wine quality, and what empirical insights can support laboratory screening workflows and future analytical enhancements?"*

### Synthesis of Core Findings
- **Data Integrity & Scale**: Evaluated on $N = 1,599$ red Vinho Verde observations (Cortez et al., 2009) comprising 11 continuous laboratory features and median sensory quality scores (ranging from 3 to 8).
- **Statistical Significance**: Non-parametric Mann-Whitney U tests with Benjamini-Hochberg False Discovery Rate (FDR) adjustments confirmed that **9 out of 11 physicochemical features** diverge significantly ($p < 0.05$) between Standard Quality ($<6$, $46.5\%$) and Good Quality ($\ge 6$, $53.5\%$) wines.
- **Key Discriminative Drivers**: **Alcohol concentration** exhibits the largest positive effect size ($\text{Cohen's } d = 0.9673$, $p < 0.001$), while **volatile acidity** (acetic acid) displays the largest negative suppression effect ($\text{Cohen's } d = -0.6801$, $p < 0.001$).
- **Non-Significant Parameters**: Univariate tests revealed **no statistically significant difference** for **pH** ($p = 0.8363$) or **residual sugar** ($p = 0.5797$) between quality classes.
- **Model Performance**: On an untouched holdout test partition ($N = 320$, stratified 80/20 split), the **Random Forest Classifier** achieved the strongest discriminatory performance:
  - **Test Accuracy**: $74.69\%$
  - **Test ROC-AUC**: $0.8571$
  - **Test Macro F1**: $0.7463$
  - **Test Sensitivity (Good Quality Recall)**: $74.27\%$
  - **Test Specificity (Standard Quality Filtering)**: $75.17\%$
  - **5-Fold Cross-Validation Accuracy (Training Set)**: $78.26\% \pm 1.81\%$
- **Operational Utility**: Physicochemical profiling provides strong automated discrimination suitable for a **two-stage quality triage pipeline**, expediting high-confidence releases while prioritizing borderline batches ($0.20 \le \hat{P} \le 0.80$) for human sommelier sensory panels.

---

## Repository Architecture

```text
week5-comprehensive-data-science/
├── data/
│   ├── raw/
│   │   └── winequality-red.csv           # Cached raw benchmark dataset (Cortez et al., 2009)
│   └── processed/
│       ├── cleaned_wine_data.csv         # Cleaned full dataset with target labels
│       ├── train_data.csv                # Stratified training split (80%, N=1279)
│       └── test_data.csv                 # Stratified holdout test split (20%, N=320)
├── outputs/
│   ├── figures/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_physicochemical_distributions.png
│   │   ├── 03_correlation_matrix.png
│   │   ├── 04_key_features_by_quality_boxplot.png
│   │   ├── 05_model_performance_comparison.png
│   │   ├── 06_confusion_matrices.png
│   │   ├── 07_roc_curves.png
│   │   ├── 08_feature_importance.png
│   │   └── 09_error_analysis_residual_profiles.png
│   └── tables/
│       ├── dataset_summary.csv
│       ├── descriptive_statistics.csv
│       ├── group_descriptive_statistics.csv
│       ├── correlation_matrix_pearson.csv
│       ├── correlation_matrix_spearman.csv
│       ├── normality_and_variance_tests.csv
│       ├── statistical_tests.csv
│       ├── cross_validation_results.csv
│       ├── model_evaluation_metrics.csv
│       ├── confusion_matrix_metrics.csv
│       ├── feature_importances.csv
│       └── error_analysis_summary.csv
├── report/
│   └── Week_5_Comprehensive_Data_Science_Project.docx  # 22-section publication Word report
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Automated dataset acquisition and schema validation
│   ├── data_cleaning.py                  # Cleaning, feature engineering, and stratified splitting
│   ├── exploratory_analysis.py          # Summary statistics, skewness, and correlation matrices
│   ├── statistical_analysis.py          # Inferential hypothesis testing, Mann-Whitney U, and FDR
│   ├── machine_learning.py              # ML pipelines, cross-validation, metrics, error analysis
│   ├── visualizations.py                # 9 high-resolution 300 DPI publication plots
│   ├── report_generator.py              # Programmatic Word report builder with embedded tables/charts
│   └── main.py                          # Pipeline orchestration entry point
├── .gitignore                            # Clean environment exclusions (.venv, __pycache__, .idea)
├── requirements.txt                      # Minimum package dependencies
└── README.md                             # Comprehensive technical documentation
```

---

## Dataset Description

The dataset corresponds to the **UCI Red Wine Quality Benchmark** collected by P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis (2009).

| Feature Name | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `fixed_acidity` | Continuous | $\text{g(tartaric acid)/dm}^3$ | Non-volatile organic acids contributing to wine freshness |
| `volatile_acidity` | Continuous | $\text{g(acetic acid)/dm}^3$ | Acetic acid concentration; excessive levels cause vinegar off-flavor |
| `citric_acid` | Continuous | $\text{g/dm}^3$ | Freshness and flavor enhancement agent |
| `residual_sugar` | Continuous | $\text{g/dm}^3$ | Unfermented sugars remaining post-fermentation |
| `chlorides` | Continuous | $\text{g(sodium chloride)/dm}^3$ | Salt concentration originating from terroir and water |
| `free_sulfur_dioxide` | Continuous | $\text{mg/dm}^3$ | Unbound active antioxidant / antimicrobial agent |
| `total_sulfur_dioxide`| Continuous | $\text{mg/dm}^3$ | Free plus bound $\text{SO}_2$; excessive levels cause sulfur pungency |
| `density` | Continuous | $\text{g/cm}^3$ | Specific gravity, dependent on ethanol and sugar concentrations |
| `pH` | Continuous | pH scale | Scale of wine acidity / alkalinity |
| `sulphates` | Continuous | $\text{g(potassium sulphate)/dm}^3$| Mineral salts enhancing $\text{SO}_2$ preservative efficacy |
| `alcohol` | Continuous | $\%$ vol. | Ethanol concentration percentage by volume |
| `quality` | Discrete | $3 \text{ to } 8$ | Median sensory rating assigned by expert tasting panels |

### Target Formulations
- **Primary Binary Classification**:
  - **Standard Quality ($0$)**: Sensory score $<6$ ($N = 744$, $46.53\%$)
  - **Good Quality ($1$)**: Sensory score $\ge 6$ ($N = 855$, $53.47\%$)
- **Stratified Partitioning**: $80\%$ Train ($N = 1,279$) and $20\%$ Test ($N = 320$), random state fixed to `42`.

---

## Methodology & Pipeline Overview

```text
[1. Data Acquisition]  --> Auto-download & schema validation (N=1599, 12 variables)
          ↓
[2. Data Cleaning]     --> Target engineering & 80/20 Stratified Partitioning
          ↓
[3. Exploratory EDA]   --> Descriptive moments, skewness, Pearson/Spearman matrices
          ↓
[4. Statistical Tests] --> Shapiro-Wilk, Mann-Whitney U, Welch's t, Cohen's d, FDR correction
          ↓
[5. Machine Learning]  --> Leakage-free Pipelines (Logistic Regression, Decision Tree, Random Forest)
                           5-Fold Cross-Validation + Holdout Test Set Evaluation
          ↓
[6. Diagnostics]       --> Permutation Importance, Confusion Matrices, Error Profiling
          ↓
[7. Artifacts & DOCX]  --> 9 High-Res Figures + 12 CSV Tables + 22-Section Word Report
```

---

## Statistical Hypothesis Testing Summary

Each feature was evaluated under the null hypothesis $H_0$: *The distribution of the chemical feature is identical between Good and Standard quality wines.*

| Feature | Mann-Whitney $U$ | Mann-Whitney $p$-value | FDR Adjusted $p$ | Cohen's $d$ | Effect Magnitude | Direction in Good Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **alcohol** | $481,313.0$ | $2.01 \times 10^{-70}$ | $< 0.001$ | $+0.9673$ | **Large** | Elevated |
| **volatile_acidity** | $197,208.0$ | $2.39 \times 10^{-39}$ | $< 0.001$ | $-0.6801$ | **Medium** | Suppressed |
| **sulphates** | $440,968.0$ | $1.18 \times 10^{-40}$ | $< 0.001$ | $+0.4477$ | **Small/Medium** | Elevated |
| **total_sulfur_dioxide** | $245,006.0$ | $2.14 \times 10^{-15}$ | $< 0.001$ | $-0.4778$ | **Small** | Suppressed |
| **citric_acid** | $376,272.5$ | $2.55 \times 10^{-10}$ | $< 0.001$ | $+0.3229$ | **Small** | Elevated |
| **density** | $257,552.0$ | $5.02 \times 10^{-11}$ | $< 0.001$ | $-0.3229$ | **Small** | Suppressed |
| **chlorides** | $254,091.0$ | $3.72 \times 10^{-12}$ | $< 0.001$ | $-0.2207$ | **Small** | Suppressed |
| **fixed_acidity** | $347,895.5$ | $0.0012$ | $0.0016$ | $+0.1914$ | **Negligible** | Elevated |
| **free_sulfur_dioxide**| $298,401.5$ | $0.0326$ | $0.0399$ | $-0.1240$ | **Negligible** | Suppressed |
| **residual_sugar** | $323,150.5$ | $0.5797$ | $0.6377$ | $-0.0043$ | **None** | *Not Significant* |
| **pH** | $316,157.5$ | $0.8363$ | $0.8363$ | $-0.0065$ | **None** | *Not Significant* |

---

## Machine Learning Benchmark Results

Models were evaluated on the held-out test partition ($N = 320$):

| Model | Test Accuracy | Precision (Good) | Recall (Sensitivity) | Specificity | Macro F1 | Test ROC-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **74.69%** | **77.44%** | **74.27%** | **75.17%** | **0.7463** | **0.8571** | **78.26% ± 1.81%** |
| **Logistic Regression** | 74.06% | 76.83% | 73.68% | 74.50% | 0.7401 | 0.8242 | 74.59% ± 1.81% |
| **Decision Tree** | 73.44% | 76.88% | 71.93% | 75.17% | 0.7341 | 0.7875 | 72.40% ± 2.56% |

### Key Diagnostic Takeaways
1. **Model Stability**: Random Forest demonstrated the highest discrimination capacity ($\text{AUC} = 0.8571$) and consistent cross-validation stability.
2. **Error Audit**: Misclassifications occur overwhelmingly in borderline wines. False Positives average higher ethanol ($10.98\%$) mimicking good wines, while False Negatives average elevated volatile acidity ($0.505\text{ g/dm}^3$) depressing sensory perception.

---

## Strategic Recommendations

Recommendations are structured strictly via the **Data Finding $\rightarrow$ Interpretation $\rightarrow$ Stakeholder Implication $\rightarrow$ Strategic Recommendation** framework:

```text
┌─────────────────┐     ┌────────────────┐     ┌────────────────────────────┐     ┌────────────────────────┐
│  DATA FINDING   │ ──> │ INTERPRETATION │ ──> │  STAKEHOLDER IMPLICATION   │ ──> │ STRATEGIC RECOMMNDATION│
└─────────────────┘     └────────────────┘     └────────────────────────────┘     └────────────────────────┘
```

1. **Two-Stage Quality Screening Architecture**:
   - *Data Finding*: Random Forest yields $74.69\%$ test accuracy and $0.8571$ ROC-AUC with balanced specificity ($75.17\%$) and sensitivity ($74.27\%$).
   - *Interpretation*: Models provide powerful screening, but a $\approx 25\%$ error rate precludes fully autonomous release.
   - *Implication*: Automated-only release introduces brand quality risk, whereas tasting every single production batch is cost-inefficient.
   - *Recommendation*: Deploy a **two-tier triage system**: batches with high model confidence ($\hat{P} > 0.80$ or $\hat{P} < 0.20$) proceed through fast-track routing; borderline batches ($0.20 \le \hat{P} \le 0.80$) are scheduled for sommelier panel review.

2. **Target Acidity Fermentation Controls**:
   - *Data Finding*: Volatile acidity exhibits strong negative effect ($\text{Cohen's } d = -0.6801$, $p < 0.001$), with Good wines averaging $0.474\text{ g/dm}^3$ vs. $0.589\text{ g/dm}^3$ in Standard wines.
   - *Interpretation*: Elevated acetic acid imparts pungent vinegar defects.
   - *Implication*: Fermentation anomalies that permit microbial oxidation compromise batch commercial viability.
   - *Recommendation*: Enforce inline fermentation alerts at $0.52\text{ g/dm}^3$ volatile acidity, prompting immediate cellar interventions (inert gas blanketing, temperature regulation).

3. **Laboratory Analytical Resource Allocation**:
   - *Data Finding*: Neither residual sugar ($p = 0.5797$) nor pH ($p = 0.8363$) separates quality classes in this corpus.
   - *Interpretation*: Within typical dry wine ranges, minor sugar and pH variances do not dictate quality ratings.
   - *Implication*: Over-investing in continuous sugar/pH optimization yields diminishing returns.
   - *Recommendation*: Shift laboratory testing budgets toward ethanol yield monitoring, volatile acid tracking, and bound/free sulfur dioxide ratios.

4. **Preservative Balance & Bound $\text{SO}_2$ Monitoring**:
   - *Data Finding*: Sulphates correlate positively ($d = +0.4477$), whereas total $\text{SO}_2$ correlates negatively ($d = -0.4778$).
   - *Interpretation*: Sulphates provide antioxidant benefits, but excessive total $\text{SO}_2$ introduces harsh, sulfurous aromatic defects.
   - *Implication*: Unbalanced preservation harms organoleptic appeal.
   - *Recommendation*: Track the bound-to-free sulfur dioxide ratio to optimize antimicrobial longevity while keeping total $\text{SO}_2$ below pungent sensory thresholds.

---

## Limitations & Threats to Validity

1. **Geographic & Varietal Bounds**: Sample is drawn exclusively from Portuguese red Vinho Verde; findings should not be extrapolated to aged reserve wines, sparkling wines, or disparate cultivars without local re-calibration.
2. **Omission of Complex Phenolics & Aromas**: The 11 measured continuous variables lack gas chromatography volatile aroma profiles, tannins, and anthocyanins.
3. **Presence of Duplicate Records**: The 240 duplicate feature/label records cannot be independently verified as repeated measurements from the same production batch due to the absence of batch identifiers, and their retention is acknowledged as a dataset limitation.
4. **Sensory Ground Truth Subjectivity**: Target scores reflect median human panel evaluations, which inherently contain taster fatigue and perceptual subjectivity.
5. **Correlational Constraint**: Feature importance metrics represent statistical associations and do not prove direct biochemical causality.

---

## Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/week5-comprehensive-data-science.git
   cd week5-comprehensive-data-science
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # On Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # On Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the Full End-to-End Pipeline**:
   ```bash
   python src/main.py
   ```

### Execution Log & Expected Output
```text
================================================================================
  WEEK 5 DATA SCIENCE CAPSTONE: COMPREHENSIVE WINE QUALITY PIPELINE
================================================================================

[STEP 1/7] Loading and validating raw dataset...
  -> Successfully loaded 1599 rows, 12 columns.

[STEP 2/7] Cleaning data and creating stratified train/test partitions...
  -> Cleaned dataset: 1599 rows.
  -> Train partition: 1279 rows | Test partition: 320 rows.

[STEP 3/7] Conducting exploratory data analysis & correlation profiling...
  -> Strongest positive correlate: alcohol (0.4762)
  -> Strongest negative correlate: volatile_acidity (-0.3906)

[STEP 4/7] Performing formal hypothesis testing (Mann-Whitney U, Welch's t, FDR correction)...
  -> Significant features (alpha=0.05): 9 / 11
  -> Top effect size: alcohol (d = 0.9673)

[STEP 5/7] Training classification models & evaluating out-of-sample performance...
  -> Best Model: Random Forest
  -> Test Accuracy: 74.69% | Test ROC-AUC: 0.8571

[STEP 6/7] Rendering 9 high-resolution publication figures...
  -> Generated 9 figures in outputs/figures/

[STEP 7/7] Generating executive-ready comprehensive Word document report...
  -> Report created at: report/Week_5_Comprehensive_Data_Science_Project.docx

================================================================================
  PIPELINE EXECUTION COMPLETED SUCCESSFULLY (EXIT 0)
================================================================================
```

---

## Technical Dependencies

```text
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.13.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
python-docx>=1.1.0
requests>=2.31.0
```

---

## References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547-553.
2. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
4. Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery rate: a practical and powerful approach to multiple testing.* Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
5. Jackson, R. S. (2020). *Wine Science: Principles and Applications (5th ed.).* Academic Press.
