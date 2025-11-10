"""
src/evaluate.py
Load saved model, evaluate on test split and write summary to reports/evaluation.txt
Usage:
    python src/evaluate.py --model models/best_model.joblib --input data/processed/processed.csv --out reports/evaluation.txt
"""
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = joblib.load(args.model)

    proba = model.predict_proba(X_test)[:,1]
    preds = model.predict(X_test)

    results = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("Evaluation results (test set):\n")
        for k,v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved evaluation report to {args.out}")
