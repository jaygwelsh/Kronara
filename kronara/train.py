# kronara/train.py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
from kronara.data import DataModule
from kronara.models.mlp import MLP
from kronara.utils.logging_utils import get_logger
import mlflow
import mlflow.pytorch
import sys

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed, workers=True)

def run_training(cfg: DictConfig):
    """
    Run a single training session (single fold) given a configuration.

    Args:
        cfg (DictConfig): Configuration parameters.

    Returns:
        float: The validation F1 score at the end of training.
    """
    set_seed(cfg.data.seed)
    torch.set_float32_matmul_precision('high')

    _logger = get_logger()
    _logger.info("Initializing DataModule...")
    
    try:
        dm = DataModule(
            data_path=cfg.data.path,
            num_samples=cfg.data.num_samples,
            num_features=cfg.data.num_features,
            test_size=cfg.data.test_size,
            k_folds=cfg.data.k_folds,
            current_fold=cfg.data.current_fold,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            seed=cfg.data.seed,
            persistent_workers=cfg.data.persistent_workers,
            fallback_to_synthetic=cfg.data.fallback_to_synthetic
        )
        dm.prepare_data()
        dm.setup()
        _logger.info("DataModule setup completed.")
    except Exception as e:
        _logger.exception(f"Error during DataModule setup: {e}")
        sys.exit(1)

    _logger.info("Initializing the model...")
    try:
        model = MLP(
            input_dim=cfg.model.input_dim,
            hidden_layers=cfg.model.hidden_layers,
            output_dim=cfg.model.output_dim,
            lr=cfg.model.lr,
            weight_decay=cfg.model.weight_decay,
            dropout=cfg.model.dropout
        )
        _logger.info("Model initialization completed.")
    except Exception as e:
        _logger.exception(f"Error during model initialization: {e}")
        sys.exit(1)

    callbacks = []
    try:
        early_stop_auc = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=cfg.trainer.early_stopping_patience,
            verbose=True
        )
        callbacks.append(early_stop_auc)

        checkpoint_callback = None
        if cfg.trainer.enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_auc",
                mode="max",
                save_top_k=1,
                verbose=True,
                filename="best_model-{epoch:02d}-{val_auc:.4f}"
            )
            callbacks.append(checkpoint_callback)
    except Exception as e:
        _logger.exception(f"Error setting up callbacks: {e}")
        sys.exit(1)

    try:
        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            enable_checkpointing=cfg.trainer.enable_checkpointing,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            precision=cfg.trainer.precision,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            callbacks=callbacks
        )
        _logger.info("Trainer initialized.")
    except Exception as e:
        _logger.exception(f"Error initializing Trainer: {e}")
        sys.exit(1)

    try:
        mlflow.set_experiment(cfg.logging.mlflow_experiment_name)
        with mlflow.start_run():
            mlflow.log_params({
                "lr": cfg.model.lr,
                "weight_decay": cfg.model.weight_decay,
                "dropout": cfg.model.dropout,
                "hidden_layers": cfg.model.hidden_layers,
                "batch_size": cfg.data.batch_size,
                "max_epochs": cfg.trainer.max_epochs
            })

            _logger.info("Starting training...")
            trainer.fit(model, datamodule=dm)
            _logger.info("Training completed. Starting testing...")

            # Retrieve val_f1 immediately after training to ensure it doesn't get overridden by test metrics.
            val_f1 = trainer.callback_metrics.get("val_f1", 0.0)
            if isinstance(val_f1, torch.Tensor):
                val_f1 = val_f1.item()

            trainer.test(model, datamodule=dm)
            _logger.info("Testing completed.")

            if cfg.trainer.enable_checkpointing and checkpoint_callback is not None:
                best_ckpt = checkpoint_callback.best_model_path
                if best_ckpt:
                    mlflow.log_artifact(best_ckpt)
                    mlflow.pytorch.log_model(model, artifact_path="models")
                    _logger.info(f"Best model checkpoint saved to MLflow: {best_ckpt}")
    except Exception as e:
        _logger.exception(f"Error during training or logging: {e}")
        sys.exit(1)

    # val_f1 has already been retrieved before testing.
    _logger.info(f"Training run completed with val_f1={val_f1:.4f}.")
    return float(val_f1)
