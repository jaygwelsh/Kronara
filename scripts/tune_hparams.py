# scripts/tune_hparams.py
import hydra
from omegaconf import DictConfig
import optuna
import mlflow
import mlflow.pytorch
from kronara.train import run_training
from kronara.utils.logging_utils import get_logger

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Hyperparameter tuning using Optuna. 
    The objective function runs training and uses validation F1 as the metric.
    """
    logger = get_logger()
    logger.info("Starting hyperparameter tuning...")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        cfg.model.lr = lr
        cfg.model.dropout = dropout

        with mlflow.start_run(nested=True):
            mlflow.log_param("lr", lr)
            mlflow.log_param("dropout", dropout)
            val_f1 = run_training(cfg)
            # We want to minimize the negative of val_f1 to maximize val_f1,
            # or simply return (1 - val_f1) if we want to minimize.
            # If direction is "minimize", we return (1 - val_f1).
            return 1.0 - val_f1

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    logger.info(f"Best hyperparameters: {study.best_params}")

if __name__ == "__main__":
    main()
