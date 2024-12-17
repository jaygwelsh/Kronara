# kronara/models/mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
import os

class SimpleBlock(nn.Module):
    """
    A simple feedforward block consisting of:
    - Linear layer
    - Layer normalization
    - ReLU activation
    - Dropout

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MLP(pl.LightningModule):
    """
    A Multi-Layer Perceptron model for binary classification tasks.

    Args:
        input_dim (int): Input feature dimension.
        hidden_layers (list): List of hidden layer sizes.
        output_dim (int): Output dimension (usually 1 for binary classification).
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 regularization).
        dropout (float): Dropout probability.
    """
    def __init__(self, input_dim, hidden_layers=[4096,4096,4096,4096], output_dim=1, lr=0.0005, weight_decay=1e-5, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = input_dim
        for h in self.hparams.hidden_layers:
            block = SimpleBlock(in_dim, h, dropout=self.hparams.dropout)
            layers.append(block)
            in_dim = h
        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(in_dim, self.hparams.output_dim)
        torch.nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            self.final_linear.bias.data.fill_(0.0)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.acc = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.auc = AUROC(task="binary")

        self.val_logits = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
        self.test_logits = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        self.val_logits.append(y_hat.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        return {}

    def on_validation_epoch_end(self):
        if len(self.val_logits) == 0:
            return
        all_logits = torch.cat(self.val_logits)
        all_labels = torch.cat(self.val_labels)
        self.val_logits.clear()
        self.val_labels.clear()

        preds = torch.sigmoid(all_logits)
        acc = self.acc((preds>=0.5).float(), all_labels)
        f1 = self.f1((preds>=0.5).float(), all_labels)
        precision = self.precision((preds>=0.5).float(), all_labels)
        recall = self.recall((preds>=0.5).float(), all_labels)
        auc = self.auc(preds, all_labels)
        val_loss = self.loss_fn(all_logits, all_labels)

        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)
        self.log("val_precision", precision, prog_bar=True, logger=True)
        self.log("val_recall", recall, prog_bar=True, logger=True)
        self.log("val_auc", auc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        preds = torch.sigmoid(y_hat)
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        self.test_logits.append(y_hat.detach().cpu())
        return {}

    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)
        all_logits = torch.cat(self.test_logits)

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(root_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        fold = self.trainer.datamodule.current_fold if hasattr(self.trainer.datamodule, 'current_fold') else 0

        torch.save(all_preds, os.path.join(artifacts_dir, f"fold_{fold}_preds.pt"))
        torch.save(all_labels, os.path.join(artifacts_dir, f"fold_{fold}_labels.pt"))
        torch.save(all_logits, os.path.join(artifacts_dir, f"fold_{fold}_logits.pt"))

        acc = self.acc((all_preds>=0.5).float(), all_labels)
        f1 = self.f1((all_preds>=0.5).float(), all_labels)
        precision = self.precision((all_preds>=0.5).float(), all_labels)
        recall = self.recall((all_preds>=0.5).float(), all_labels)
        auc = self.auc(all_preds, all_labels)
        test_loss = self.loss_fn(all_logits, all_labels)

        self.log("test_loss", test_loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        self.log("test_f1", f1, prog_bar=True, logger=True)
        self.log("test_precision", precision, prog_bar=True, logger=True)
        self.log("test_recall", recall, prog_bar=True, logger=True)
        self.log("test_auc", auc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        steps_per_epoch = self.trainer.estimated_stepping_batches
        total_steps = steps_per_epoch * self.trainer.max_epochs
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer, max_lr=self.hparams.lr, total_steps=total_steps, pct_start=0.1
            ),
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]
