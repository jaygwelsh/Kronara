# tests/test_train.py
import pytest
import torch
from omegaconf import OmegaConf
from kronara.train import run_training

def test_run_training():
    cfg = OmegaConf.create({
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "enable_checkpointing": False,
            "log_every_n_steps": 1,
            "precision": 32,
            "strategy": "auto",
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "early_stopping_patience": 2
        },
        "data": {
            "path": "",
            "num_samples": 10000,
            "num_features": 20,
            "test_size": 0.2,
            "k_folds": 5,
            "current_fold": 0,
            "seed": 42,
            "batch_size": 64,
            "num_workers": 0,
            "persistent_workers": False,
            "fallback_to_synthetic": True
        },
        "model": {
            "input_dim": 20,
            "hidden_layers": [64],
            "output_dim": 1,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "dropout": 0.1
        },
        "logging": {
            "mlflow_experiment_name": "Test_Experiment",
            "mlflow_tracking_uri": "file:./mlruns"
        }
    })

    run_training(cfg)
    # If it runs without error, it's considered passing for now.
