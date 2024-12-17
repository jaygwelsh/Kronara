# tutorials/feature_importance_demo.py
"""
Demonstration of using SHAP for feature importance.
Tries to load a trained model checkpoint and generate a SHAP summary plot.

Usage:
    python tutorials/feature_importance_demo.py
"""

import shap
import torch
import os
from kronara.models.mlp import MLP
import numpy as np

if __name__ == "__main__":
    # Adjust path to your best model checkpoint if different
    best_model_ckpt = "best_model.ckpt"
    if not os.path.exists(best_model_ckpt):
        print("No best_model.ckpt found. Please run training first or specify a correct checkpoint path.")
        exit(1)

    model = MLP.load_from_checkpoint(best_model_ckpt)
    model.eval()

    # Generate synthetic sample data for SHAP explanation
    X = np.random.randn(100, model.hparams.input_dim)

    explainer = shap.Explainer(model.forward, torch.from_numpy(X).float())
    shap_values = explainer(torch.from_numpy(X).float())

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(root_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    shap.summary_plot(shap_values.values, X, show=False)
    shap_plot_path = os.path.join(artifacts_dir, "shap_summary_plot.png")
    shap.plt.savefig(shap_plot_path)
    print(f"SHAP summary plot saved as {shap_plot_path}")
