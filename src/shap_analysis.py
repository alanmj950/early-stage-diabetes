"""
src/shap_analysis.py
Compute SHAP importance for the saved model and save plots:
 - figures/shap_summary_beeswarm.png
 - figures/shap_feature_importance.png (bar)
Usage:
    python src/shap_analysis.py --model models/best_model.joblib --input data/processed/processed.csv --figdir figures
"""
import argparse
import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def save_shap_plots(model, X_train, X_test, figdir):
    # If the model is a Pipeline, get the final estimator and preprocessing separately
    if hasattr(model, "named_steps"):
        # assume last step is estimator, and preprocess is earlier
        preprocessing = None
        estimator = None
        # try common pipeline names
        if "scaler" in model.named_steps:
            # remove scaler from pipeline to pass raw X_train numeric array
            try:
                # use pipeline.transform to get numeric features
                X_train_trans = model.named_steps["scaler"].transform(X_train)
                X_test_trans = model.named_steps["scaler"].transform(X_test)
            except Exception:
                X_train_trans = X_train.values
                X_test_trans = X_test.values
        else:
            # fallback
            X_train_trans = X_train.values
            X_test_trans = X_test.values
        # get estimator
        estimator = list(model.named_steps.values())[-1]
    else:
        estimator = model
        X_train_trans = X_train.values
        X_test_trans = X_test.values

    # Use TreeExplainer for tree models if possible
    try:
        expl = shap.TreeExplainer(estimator)
    except Exception:
        expl = shap.Explainer(estimator, X_train_trans)

    shap_values = expl(X_test_trans)  # new SHAP API returns Explanation object

    os.makedirs(figdir, exist_ok=True)
    # summary beeswarm
    # For binary classification, pick positive class
    shap_values_pos = shap_values[:, :, 1]  # 3D -> 2D
    plt.figure(figsize=(8,6))
    shap.plots.beeswarm(shap_values_pos, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "shap_summary_beeswarm.png"), bbox_inches="tight")
    plt.close()

    # bar plot (mean abs)
    plt.figure(figsize=(8,6))
    shap.plots.bar(shap_values_pos, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "shap_feature_importance.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--figdir", default="figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = joblib.load(args.model)
    save_shap_plots(model, X_train, X_test, args.figdir)
    print(f"SHAP plots saved to {args.figdir}")
