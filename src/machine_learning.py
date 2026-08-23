"""
Machine learning modeling, evaluation, cross-validation, feature importance, and error analysis.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance


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


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tables_dir: Path = Path("outputs/tables"),
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trains Logistic Regression, Decision Tree, and Random Forest classifiers;
    performs 5-fold cross-validation, out-of-sample evaluation, feature importance,
    and error profiling.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Hold-out testing dataset.
    tables_dir : Path
        Output directory for exported metrics and tables.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (results_dict, metrics_df, feature_importance_df, error_analysis_df)
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[PHYSICOCHEMICAL_FEATURES]
    y_train = train_df["quality_label"]
    X_test = test_df[PHYSICOCHEMICAL_FEATURES]
    y_test = test_df["quality_label"]

    # Define models with standardized pipelines
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=15,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1
        )
    }

    # 1. 5-Fold Stratified Cross-Validation on Training Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_records = []

    for name, model in models.items():
        scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1_macro": "f1_macro"}
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_records.append({
            "Model": name,
            "CV_Accuracy_Mean": round(float(scores["test_accuracy"].mean()), 4),
            "CV_Accuracy_Std": round(float(scores["test_accuracy"].std()), 4),
            "CV_ROC_AUC_Mean": round(float(scores["test_roc_auc"].mean()), 4),
            "CV_ROC_AUC_Std": round(float(scores["test_roc_auc"].std()), 4),
            "CV_F1_Macro_Mean": round(float(scores["test_f1_macro"].mean()), 4),
            "CV_F1_Macro_Std": round(float(scores["test_f1_macro"].std()), 4),
        })

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(tables_dir / "cross_validation_results.csv", index=False)

    # 2. Fit on Full Train Set & Evaluate on Hold-Out Test Set
    metrics_records = []
    cm_records = []
    model_predictions = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        model_predictions[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob
        }

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec_bin = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_bin = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_bin = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        # Confusion Matrix breakdown
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Match with CV metrics
        cv_match = cv_df[cv_df["Model"] == name].iloc[0]

        metrics_records.append({
            "Model": name,
            "Test_Accuracy": round(float(acc), 4),
            "Test_Precision": round(float(prec_bin), 4),
            "Test_Recall_Sensitivity": round(float(rec_bin), 4),
            "Test_Specificity": round(float(specificity), 4),
            "Test_Macro_F1": round(float(f1_mac), 4),
            "Test_Weighted_F1": round(float(f1_wt), 4),
            "Test_ROC_AUC": round(float(roc_auc), 4),
            "Test_PR_AUC": round(float(pr_auc), 4),
            "CV_Accuracy_Mean": cv_match["CV_Accuracy_Mean"],
            "CV_Accuracy_Std": cv_match["CV_Accuracy_Std"],
            "CV_ROC_AUC_Mean": cv_match["CV_ROC_AUC_Mean"],
            "CV_ROC_AUC_Std": cv_match["CV_ROC_AUC_Std"]
        })

        cm_records.append({
            "Model": name,
            "True_Negative": int(tn),
            "False_Positive": int(fp),
            "False_Negative": int(fn),
            "True_Positive": int(tp),
            "Specificity_TNR": round(float(specificity), 4),
            "Sensitivity_TPR": round(float(rec_bin), 4),
            "False_Positive_Rate": round(float(fpr), 4),
            "False_Negative_Rate": round(float(fnr), 4)
        })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(tables_dir / "model_evaluation_metrics.csv", index=False)

    cm_df = pd.DataFrame(cm_records)
    cm_df.to_csv(tables_dir / "confusion_matrix_metrics.csv", index=False)

    # 3. Feature Importance Extraction
    rf_model = fitted_models["Random Forest"]
    rf_gini = rf_model.feature_importances_

    # Permutation importance on test set
    perm_result = permutation_importance(
        rf_model, X_test, y_test, n_repeats=20, random_state=random_state, n_jobs=-1
    )
    rf_perm_mean = perm_result.importances_mean
    rf_perm_std = perm_result.importances_std

    # Logistic Regression standardized coefficients
    lr_pipeline = fitted_models["Logistic Regression"]
    lr_coefs = lr_pipeline.named_steps["classifier"].coef_[0]

    # Decision Tree Gini importance
    dt_model = fitted_models["Decision Tree"]
    dt_gini = dt_model.feature_importances_

    feat_imp_records = []
    for i, feat in enumerate(PHYSICOCHEMICAL_FEATURES):
        feat_imp_records.append({
            "Feature": feat,
            "RF_Gini_Importance": round(float(rf_gini[i]), 4),
            "RF_Permutation_Importance_Mean": round(float(rf_perm_mean[i]), 4),
            "RF_Permutation_Importance_Std": round(float(rf_perm_std[i]), 4),
            "Logistic_Std_Coefficient": round(float(lr_coefs[i]), 4),
            "Odds_Ratio": round(float(np.exp(lr_coefs[i])), 4),
            "Decision_Tree_Gini": round(float(dt_gini[i]), 4)
        })

    feat_imp_df = pd.DataFrame(feat_imp_records)
    feat_imp_df.sort_values(by="RF_Permutation_Importance_Mean", ascending=False, inplace=True)
    feat_imp_df.to_csv(tables_dir / "feature_importances.csv", index=False)

    # 4. Error Analysis for the Best Performing Model (Random Forest)
    rf_preds = model_predictions["Random Forest"]["y_pred"]
    rf_probs = model_predictions["Random Forest"]["y_prob"]

    test_analysis_df = test_df.copy()
    test_analysis_df["predicted_label"] = rf_preds
    test_analysis_df["predicted_probability"] = rf_probs

    def classify_error(row: pd.Series) -> str:
        actual = row["quality_label"]
        pred = row["predicted_label"]
        if actual == 1 and pred == 1:
            return "True Positive"
        elif actual == 0 and pred == 0:
            return "True Negative"
        elif actual == 0 and pred == 1:
            return "False Positive (Type I)"
        else:
            return "False Negative (Type II)"

    test_analysis_df["error_type"] = test_analysis_df.apply(classify_error, axis=1)

    error_summary_records = []
    for etype in ["True Negative", "True Positive", "False Positive (Type I)", "False Negative (Type II)"]:
        subset = test_analysis_df[test_analysis_df["error_type"] == etype]
        record = {
            "Classification_Outcome": etype,
            "Count": len(subset),
            "Percentage_of_Test_Set_%": round(len(subset) / len(test_analysis_df) * 100, 2),
            "Mean_Alcohol": round(float(subset["alcohol"].mean()), 2) if len(subset) > 0 else 0.0,
            "Mean_Volatile_Acidity": round(float(subset["volatile_acidity"].mean()), 3) if len(subset) > 0 else 0.0,
            "Mean_Sulphates": round(float(subset["sulphates"].mean()), 3) if len(subset) > 0 else 0.0,
            "Mean_Total_SO2": round(float(subset["total_sulfur_dioxide"].mean()), 2) if len(subset) > 0 else 0.0,
            "Mean_Predicted_Prob": round(float(subset["predicted_probability"].mean()), 3) if len(subset) > 0 else 0.0
        }
        error_summary_records.append(record)

    error_df = pd.DataFrame(error_summary_records)
    error_df.to_csv(tables_dir / "error_analysis_summary.csv", index=False)

    # Structured Results Output
    best_model_name = metrics_df.sort_values(by="Test_Accuracy", ascending=False).iloc[0]["Model"]
    best_accuracy = metrics_df.sort_values(by="Test_Accuracy", ascending=False).iloc[0]["Test_Accuracy"]
    best_roc_auc = metrics_df.sort_values(by="Test_Accuracy", ascending=False).iloc[0]["Test_ROC_AUC"]

    results_dict = {
        "models": fitted_models,
        "predictions": model_predictions,
        "metrics_df": metrics_df,
        "cv_df": cv_df,
        "confusion_matrix_df": cm_df,
        "feature_importance_df": feat_imp_df,
        "error_analysis_df": error_df,
        "test_analysis_df": test_analysis_df,
        "best_model_name": best_model_name,
        "best_accuracy": best_accuracy,
        "best_roc_auc": best_roc_auc,
        "y_test": y_test.values
    }

    print(f"[SUCCESS] Model training and evaluation completed. Best Model: {best_model_name} (Acc: {best_accuracy:.4f}, AUC: {best_roc_auc:.4f})")
    return results_dict, metrics_df, feat_imp_df, error_df


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaning import clean_and_prepare_data
    raw, _ = load_raw_data()
    clean, train, test, _ = clean_and_prepare_data(raw)
    train_and_evaluate_models(train, test)
