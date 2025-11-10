import argparse
import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def save_shap_plots(model, X_train, X_test, figdir):
    # If model is a Pipeline, get final estimator and transform
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
        X_train_trans = X_train.copy()
        X_test_trans = X_test.copy()
        # Apply scaler if exists
        if "scaler" in model.named_steps:
            X_train_trans = pd.DataFrame(model.named_steps["scaler"].transform(X_train),
                                         columns=X_train.columns)
            X_test_trans = pd.DataFrame(model.named_steps["scaler"].transform(X_test),
                                        columns=X_test.columns)
    else:
        estimator = model
        X_train_trans = X_train.copy()
        X_test_trans = X_test.copy()

    # Use SHAP Explainer
    try:
        expl = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
    except Exception:
        expl = shap.Explainer(estimator, X_train_trans)

    shap_values = expl(X_test_trans)

    os.makedirs(figdir, exist_ok=True)

    # For binary classification pick positive class
    if len(shap_values.shape) == 3:  # multi-class output
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    # Beeswarm plot
    plt.figure(figsize=(10,6))
    shap.plots.beeswarm(shap_values_pos, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "shap_summary_beeswarm.png"), bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure(figsize=(10,6))
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
