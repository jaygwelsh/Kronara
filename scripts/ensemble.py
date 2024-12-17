# scripts/ensemble.py
import os
import glob
import torch
import numpy as np
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
from kronara.utils.logging_utils import get_logger

def main():
    """
    Ensemble predictions from multiple folds and compute evaluation metrics.
    This script:
    - Loads prediction files from the artifacts directory.
    - Averages predictions and finds an optimal threshold for F1.
    - Logs final metrics for the ensemble model.
    """
    logger = get_logger()
    logger.info("Ensembling predictions from all folds...")

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

    val_acc = Accuracy(task="binary")
    val_f1 = F1Score(task="binary")
    val_precision = Precision(task="binary")
    val_recall = Recall(task="binary")
    val_auc = AUROC(task="binary")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_thresh = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        pc = (ensemble_preds >= t).float()
        current_f1 = val_f1(pc, all_labels)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = float(t)

    pc = (ensemble_preds >= best_thresh).float()
    acc = val_acc(pc, all_labels)
    f1 = val_f1(pc, all_labels)
    precision = val_precision(pc, all_labels)
    recall = val_recall(pc, all_labels)
    auc = val_auc(ensemble_preds, all_labels)
    test_loss = loss_fn(ensemble_logits, all_labels)

    logger.info(f"Ensemble Results (Threshold={best_thresh:.2f}):")
    logger.info(f"Test Loss: {test_loss.item():.4f}")
    logger.info(f"Test Acc: {acc.item():.4f}")
    logger.info(f"Test F1: {f1.item():.4f}")
    logger.info(f"Test Precision: {precision.item():.4f}")
    logger.info(f"Test Recall: {recall.item():.4f}")
    logger.info(f"Test AUC: {auc.item():.4f}")

if __name__ == "__main__":
    main()
