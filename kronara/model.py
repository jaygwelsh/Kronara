# kronara/model.py
import os
import torch
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC

class MLP(pl.LightningModule):
    ...
    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)
        all_logits = torch.cat(self.test_logits)

        # Derive the project root directory based on this file's location:
        # model.py is located at: kronara/kronara/model.py
        # One dirname up -> kronara/kronara
        # Another dirname up -> kronara root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(root_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        fold = self.trainer.datamodule.current_fold if hasattr(self.trainer.datamodule, 'current_fold') else 0

        torch.save(all_preds, os.path.join(artifacts_dir, f"fold_{fold}_preds.pt"))
        torch.save(all_labels, os.path.join(artifacts_dir, f"fold_{fold}_labels.pt"))
        torch.save(all_logits, os.path.join(artifacts_dir, f"fold_{fold}_logits.pt"))
