# scripts/benchmark.py
import hydra
from omegaconf import DictConfig
from kronara.train import run_training
from kronara.utils.logging_utils import get_logger

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Run training on multiple folds (cross-validation) to benchmark model performance.
    After completion, run `scripts/ensemble.py` to ensemble results.
    """
    logger = get_logger()
    logger.info("Running benchmark across folds...")
    k_folds = cfg.data.k_folds

    for fold in range(k_folds):
        cfg.data.current_fold = fold
        logger.info(f"Running fold {fold}/{k_folds-1}...")
        run_training(cfg)

    logger.info("All folds completed. Run `python scripts/ensemble.py` to ensemble predictions.")

if __name__ == "__main__":
    main()
