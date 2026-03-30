"""
Heart Disease Predictor — ML Model
Dataset: Heart Disease UCI (Kaggle)
Model: Logistic Regression + Decision Tree
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, classification_report
)


# ── Feature descriptions for UI ──
FEATURE_INFO = {
    "age":      {"label": "Age", "unit": "years", "min": 20, "max": 80, "type": "number"},
    "sex":      {"label": "Sex", "unit": "", "min": 0, "max": 1, "type": "select",
                 "options": {"0": "Female", "1": "Male"}},
    "cp":       {"label": "Chest Pain Type", "unit": "", "min": 0, "max": 3, "type": "select",
                 "options": {"0": "Typical Angina", "1": "Atypical Angina",
                             "2": "Non-anginal Pain", "3": "Asymptomatic"}},
    "trestbps": {"label": "Resting Blood Pressure", "unit": "mm Hg", "min": 80, "max": 200, "type": "number"},
    "chol":     {"label": "Serum Cholesterol", "unit": "mg/dl", "min": 100, "max": 600, "type": "number"},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl", "unit": "", "min": 0, "max": 1, "type": "select",
                 "options": {"0": "No", "1": "Yes"}},
    "restecg":  {"label": "Resting ECG Results", "unit": "", "min": 0, "max": 2, "type": "select",
                 "options": {"0": "Normal", "1": "ST-T Wave Abnormality", "2": "Left Ventricular Hypertrophy"}},
    "thalach":  {"label": "Max Heart Rate Achieved", "unit": "bpm", "min": 60, "max": 220, "type": "number"},
    "exang":    {"label": "Exercise Induced Angina", "unit": "", "min": 0, "max": 1, "type": "select",
                 "options": {"0": "No", "1": "Yes"}},
    "oldpeak":  {"label": "ST Depression (Exercise vs Rest)", "unit": "", "min": 0.0, "max": 6.2, "type": "float"},
    "slope":    {"label": "Slope of Peak Exercise ST", "unit": "", "min": 0, "max": 2, "type": "select",
                 "options": {"0": "Upsloping", "1": "Flat", "2": "Downsloping"}},
    "ca":       {"label": "Major Vessels Colored by Flourosopy", "unit": "", "min": 0, "max": 4, "type": "select",
                 "options": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}},
    "thal":     {"label": "Thalassemia", "unit": "", "min": 0, "max": 3, "type": "select",
                 "options": {"0": "Normal", "1": "Fixed Defect", "2": "Reversible Defect", "3": "Unknown"}},
}

FEATURES = list(FEATURE_INFO.keys())
TARGET   = "target"


def load_data(path="../data/heart.csv"):
    """Load and clean the Heart Disease UCI dataset."""
    df = pd.read_csv(path)
    print(f"  Columns found: {df.columns.tolist()}")

    # Rename columns to match expected names
    df.rename(columns={"thalch": "thalach", "num": "target"}, inplace=True)

    # Drop unneeded columns
    for col in ["id", "dataset"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Encode text columns to numbers
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).fillna(0)
    df["cp"] = df["cp"].map({"typical angina": 0, "atypical angina": 1,
                              "non-anginal": 2, "asymptomatic": 3}).fillna(0)
    df["fbs"] = df["fbs"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)
    df["restecg"] = df["restecg"].map({"normal": 0, "st-t abnormality": 1,
                                        "lv hypertrophy": 2}).fillna(0)
    df["exang"] = df["exang"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)
    df["slope"] = df["slope"].map({"upsloping": 0, "flat": 1, "downsloping": 2}).fillna(1)
    df["thal"] = df["thal"].map({"normal": 0, "fixed defect": 1,
                                  "reversible defect": 2, "unknown": 3}).fillna(0)

    # Binarize target (>0 = disease)
    df["target"] = (df["target"] > 0).astype(int)

    # Handle remaining missing values
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Keep only the 13 expected features + target
    keep = FEATURES + ["target"]
    df = df[[c for c in keep if c in df.columns]]
    print(f"  Final shape: {df.shape}")
    return df


def eda_summary(df):
    """Return EDA statistics for the frontend."""
    summary = {
        "total_records":    int(len(df)),
        "disease_count":    int(df[TARGET].sum()),
        "healthy_count":    int((df[TARGET] == 0).sum()),
        "disease_pct":      round(df[TARGET].mean() * 100, 1),
        "avg_age":          round(df["age"].mean(), 1),
        "avg_chol":         round(df["chol"].mean(), 1),
        "avg_thalach":      round(df["thalach"].mean(), 1),
        "male_disease_pct": round(df[df["sex"] == 1][TARGET].mean() * 100, 1),
        "feature_correlations": {
            col: round(df[col].corr(df[TARGET]), 3)
            for col in FEATURES
        },
        "age_distribution": {
            "labels": ["20-30", "31-40", "41-50", "51-60", "61-70", "71+"],
            "healthy": [],
            "disease": []
        }
    }

    bins = [20, 30, 40, 50, 60, 70, 100]
    labels = ["20-30", "31-40", "41-50", "51-60", "61-70", "71+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    for label in labels:
        grp = df[df["age_group"] == label]
        summary["age_distribution"]["healthy"].append(int((grp[TARGET] == 0).sum()))
        summary["age_distribution"]["disease"].append(int((grp[TARGET] == 1).sum()))

    return summary


def train_models(df):
    """Train Logistic Regression and Decision Tree, return metrics."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred  = lr.predict(X_test_s)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    lr_auc = roc_auc_score(y_test, lr_proba)

    # ── Decision Tree ──
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred  = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]

    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)
    dt_auc = roc_auc_score(y_test, dt_proba)

    # ── Feature Importance (Decision Tree) ──
    fi = dict(zip(FEATURES, dt.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # ── Logistic Regression Coefficients ──
    lr_coef = dict(zip(FEATURES, np.abs(lr.coef_[0]).tolist()))
    lr_coef_sorted = dict(sorted(lr_coef.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "logistic_regression": {
            "accuracy":         round(accuracy_score(y_test, lr_pred) * 100, 2),
            "roc_auc":          round(lr_auc * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
            "roc_curve": {
                "fpr": lr_fpr.tolist()[::5],
                "tpr": lr_tpr.tolist()[::5]
            },
            "feature_importance": lr_coef_sorted,
            "report": classification_report(y_test, lr_pred, output_dict=True)
        },
        "decision_tree": {
            "accuracy":         round(accuracy_score(y_test, dt_pred) * 100, 2),
            "roc_auc":          round(dt_auc * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, dt_pred).tolist(),
            "roc_curve": {
                "fpr": dt_fpr.tolist()[::5],
                "tpr": dt_tpr.tolist()[::5]
            },
            "feature_importance": fi_sorted,
            "report": classification_report(y_test, dt_pred, output_dict=True)
        }
    }

    # Save models
    os.makedirs("../models", exist_ok=True)
    with open("../models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("../models/logistic_regression.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open("../models/decision_tree.pkl", "wb") as f:
        pickle.dump(dt, f)

    return metrics, scaler, lr, dt


def predict_single(patient_data: dict, model_name="logistic_regression"):
    """Predict heart disease risk for a single patient."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, "..", "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, "..", "models", f"{model_name}.pkl"), "rb") as f:
        model = pickle.load(f)

    values = [float(patient_data.get(feat, 0)) for feat in FEATURES]
    arr = np.array(values).reshape(1, -1)

    if model_name == "logistic_regression":
        arr = scaler.transform(arr)

    pred  = int(model.predict(arr)[0])
    proba = float(model.predict_proba(arr)[0][1])

    return {
        "prediction": pred,
        "probability": round(proba * 100, 1),
        "risk_level": "High Risk" if proba > 0.7 else "Medium Risk" if proba > 0.4 else "Low Risk",
        "risk_color": "#ef4444" if proba > 0.7 else "#f59e0b" if proba > 0.4 else "#10b981"
    }


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()
    print(f"  Loaded {len(df)} records")

    print("Running EDA...")
    eda = eda_summary(df)
    with open("../data/eda_summary.json", "w") as f:
        json.dump(eda, f, indent=2)
    print(f"  Disease prevalence: {eda['disease_pct']}%")

    print("Training models...")
    metrics, scaler, lr, dt = train_models(df)
    with open("../data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n── Results ──")
    print(f"  Logistic Regression: Accuracy {metrics['logistic_regression']['accuracy']}% | AUC {metrics['logistic_regression']['roc_auc']}%")
    print(f"  Decision Tree:       Accuracy {metrics['decision_tree']['accuracy']}% | AUC {metrics['decision_tree']['roc_auc']}%")
    print("\nModels saved to models/")
    print("Done ✓")
