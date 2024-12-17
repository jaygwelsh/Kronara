# kronara/evaluate.py
import torch
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
import hydra
from omegaconf import DictConfig
from kronara.utils.logging_utils import get_logger
import numpy as np
import glob
import os
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Evaluate ensemble predictions, compute metrics, apply calibration, and produce reliability diagrams.

    Steps:
    1. Load ensemble predictions and logits from artifacts.
    2. Compute performance metrics (Acc, F1, Precision, Recall, AUC, MCC).
    3. Perform threshold optimization based on F1 score.
    4. Apply isotonic regression for calibration.
    5. Generate and save a reliability diagram.
    """
    logger = get_logger()
    logger.info("Starting evaluation with additional metrics and optional calibration...")

    # Determine project root directory based on the location of this script
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(root_dir, "kronara", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(artifacts_dir, "fold_*_preds.pt")))
    label_files = sorted(glob.glob(os.path.join(artifacts_dir, "fold_*_labels.pt")))
    logit_files = sorted(glob.glob(os.path.join(artifacts_dir, "fold_*_logits.pt")))

    if len(pred_files) == 0:
        logger.error("No fold prediction files found in kronara/artifacts/!")
        return

    all_preds = []
    all_labels = None
    all_logits = []

    for pf, lf, lof in zip(pred_files, label_files, logit_files):
        preds = torch.load(pf)
        labels = torch.load(lf)
        logits = torch.load(lof)
        all_preds.append(preds)
        all_logits.append(logits)
        if all_labels is None:
            all_labels = labels
        else:
            if not torch.allclose(all_labels, labels):
                logger.error("Labels differ between folds! Cannot ensemble.")
                return

    ensemble_preds = torch.mean(torch.stack(all_preds), dim=0)
    ensemble_logits = torch.mean(torch.stack(all_logits), dim=0)

    # Metrics setup
    val_acc = Accuracy(task="binary")
    val_f1 = F1Score(task="binary")
    val_precision = Precision(task="binary")
    val_recall = Recall(task="binary")
    val_auc = AUROC(task="binary")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Automated threshold selection by maximizing F1 score
    best_f1 = 0.0
    best_thresh = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        pc = (ensemble_preds >= t).float()
        current_f1 = val_f1(pc, all_labels)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = float(t)

    # Use the best threshold
    pc = (ensemble_preds >= best_thresh).float()
    acc = val_acc(pc, all_labels)
    f1 = val_f1(pc, all_labels)
    precision = val_precision(pc, all_labels)
    recall = val_recall(pc, all_labels)
    auc = val_auc(ensemble_preds, all_labels)
    test_loss = loss_fn(ensemble_logits, all_labels)

    # Compute confusion matrix and MCC
    y_true = all_labels.numpy().astype(int)
    y_pred = pc.numpy().astype(int)
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    logger.info(f"Ensemble Results (Threshold={best_thresh:.2f}):")
    logger.info(f"Test Loss: {test_loss.item():.4f}")
    logger.info(f"Test Acc: {acc.item():.4f}")
    logger.info(f"Test F1: {f1.item():.4f}")
    logger.info(f"Test Precision: {precision.item():.4f}")
    logger.info(f"Test Recall: {recall.item():.4f}")
    logger.info(f"Test AUC: {auc.item():.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"MCC: {mcc:.4f}")

    # Calibration using Isotonic Regression
    logger.info("Applying isotonic regression for calibration...")
    p = torch.sigmoid(ensemble_logits).numpy()
    y = all_labels.numpy()
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(p, y)
    p_cal = ir.transform(p)

    # Reliability diagram
    bin_edges = np.linspace(0,1,11)
    bin_indices = np.digitize(p_cal, bin_edges)-1
    bin_true = [y[bin_indices==i].mean() if np.any(bin_indices==i) else np.nan for i in range(len(bin_edges)-1)]

    plt.figure()
    plt.plot([0,1],[0,1], 'k--', label="Perfect calibration")
    midpoints = (bin_edges[1:] + bin_edges[:-1])/2
    plt.plot(midpoints, bin_true, marker='o', label="Calibrated model")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Reliability Diagram")
    plt.legend()
    reliability_path = os.path.join(artifacts_dir, "reliability_diagram.png")
    plt.savefig(reliability_path)
    logger.info(f"Reliability diagram saved to {reliability_path}")

if __name__ == "__main__":
    main()
