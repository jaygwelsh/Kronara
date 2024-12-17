# scripts/train.py
import hydra
from omegaconf import DictConfig
from kronara.train import run_training
from kronara.utils.logging_utils import get_logger
import sys

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Entrypoint for a single-fold training run.

    This script:
    - Loads configuration from Hydra.
    - Runs the training pipeline using `run_training`.
    """
    logger = get_logger()
    logger.info("Starting single-fold training run...")
    try:
        val_f1 = run_training(cfg)
        logger.info(f"Training completed successfully with Validation F1 Score: {val_f1:.4f}")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
