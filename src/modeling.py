"""
src/modeling.py
Train multiple classification models and save best model.
Usage:
    python src/modeling.py --input data/processed/processed.csv --out models/best_model.joblib --figdir figures
"""
import argparse
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def prepare_X_y(df):
    X = df.drop(columns=['class'])
    y = df['class'].astype(int)
    # ensure numeric types
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, y

def eval_on_test(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist()
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help="path to save best model .joblib")
    parser.add_argument("--figdir", required=False, default="figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X, y = prepare_X_y(df)

    # train/test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Define pipelines (scaler + model for models that need scaling)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ])

    # HistGradientBoosting does not require scaling - use directly
    hgb = HistGradientBoostingClassifier(random_state=42)

    # fit models
    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    hgb.fit(X_train, y_train)

    models = {"logistic": lr_pipe, "random_forest": rf_pipe, "hist_gb": hgb}

    # Evaluate on test set
    evals = {}
    for name, m in models.items():
        try:
            evals[name] = eval_on_test(m, X_test, y_test)
        except Exception as e:
            evals[name] = {"error": str(e)}

    # Create result plot (ROC AUC)
    os.makedirs(args.figdir, exist_ok=True)
    names = []
    aucs = []
    for k, v in evals.items():
        names.append(k)
        aucs.append(v.get("roc_auc", 0.0))
    plt.figure(figsize=(6,4))
    plt.bar(names, aucs)
    plt.ylim(0,1)
    plt.title("Test ROC AUC Comparison")
    plt.ylabel("ROC AUC")
    plt.savefig(os.path.join(args.figdir, "model_compare_auc.png"), bbox_inches="tight")
    plt.close()

    # select best model by roc_auc
    best_name = max(evals.keys(), key=lambda k: evals[k].get("roc_auc", 0.0))
    best_model = models[best_name]

    # Save model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(best_model, args.out)

    # Print summary to console and write evaluation to file
    print("Model evaluations (test set):")
    for k,v in evals.items():
        print(f"--> {k}")
        for metric, val in v.items():
            print(f"    {metric}: {val}")
    print(f"Best model: {best_name}. Saved to {args.out}")
