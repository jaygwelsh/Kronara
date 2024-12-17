# Filename: test_kronara.py
# This test suite attempts to cover a wide range of functionalities within the Kronara framework.
# It includes tests for data loading, synthetic data generation, configuration handling, model initialization,
# training steps, evaluation, logging, calibration, and interpretability steps.
# Adjust imports and paths as needed depending on the project's structure.

import os
import pytest
import torch
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss

from kronara.data import DataModule
from kronara.models.mlp import MLP
from kronara.utils.logging_utils import get_logger
from kronara.evaluate import main as eval_main
from hydra import compose, initialize_config_dir
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.classification import Accuracy
import shap

TEST_SEED = 42
seed_everything(TEST_SEED, workers=True)

@pytest.fixture(scope="module")
def synthetic_data():
    np.random.seed(TEST_SEED)
    X = np.random.randn(1000, 20)
    score = X[:, :10].sum(axis=1) + np.random.randn(1000)*0.5
    y = (score > 0).astype('float32')
    return X, y

@pytest.fixture(scope="module")
def synthetic_dmodule(tmp_path_factory, synthetic_data):
    X, y = synthetic_data
    data_path = tmp_path_factory.mktemp("data") / "synthetic_test_data.csv"
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["label"] = y
    df.to_csv(data_path, index=False)
    dm = DataModule(
        data_path=str(data_path),
        num_samples=1000,
        num_features=20,
        test_size=0.2,
        k_folds=2,
        current_fold=0,
        batch_size=32,
        num_workers=0,
        seed=TEST_SEED,
        persistent_workers=False,
        fallback_to_synthetic=False
    )
    dm.prepare_data()
    dm.setup()
    return dm

def test_data_loading(synthetic_dmodule):
    assert len(synthetic_dmodule.train_dataset) > 0
    assert len(synthetic_dmodule.val_dataset) > 0
    assert len(synthetic_dmodule.test_dataset) > 0

def test_data_shapes(synthetic_dmodule):
    train_loader = synthetic_dmodule.train_dataloader()
    X, y = next(iter(train_loader))
    assert X.shape[1] == 20
    assert y.shape[0] == X.shape[0]

def test_data_splits(synthetic_dmodule):
    total = len(synthetic_dmodule.train_dataset) + len(synthetic_dmodule.val_dataset) + len(synthetic_dmodule.test_dataset)
    assert total == 1000
    # test_size=0.2 means 200 samples in test
    assert len(synthetic_dmodule.test_dataset) == 200

def test_model_initialization():
    model = MLP(input_dim=20, hidden_layers=[64,64], output_dim=1, lr=0.001, weight_decay=1e-5, dropout=0.1)
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0

def test_model_forward_pass():
    model = MLP(input_dim=20, hidden_layers=[32], output_dim=1)
    x = torch.randn(10, 20)
    y_hat = model(x)
    assert y_hat.shape == (10, 1)

def test_loss_computation():
    model = MLP(input_dim=20)
    x = torch.randn(8, 20)
    y = torch.randint(0, 2, (8,)).float()
    y_hat = model(x).view(-1)
    loss_fn = BCEWithLogitsLoss()
    loss = loss_fn(y_hat, y)
    assert loss.item() > 0

@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_batch_sizes(synthetic_dmodule, batch_size):
    synthetic_dmodule.batch_size = batch_size
    loader = synthetic_dmodule.train_dataloader()
    X, y = next(iter(loader))
    assert X.shape[0] == batch_size

def test_logging():
    logger = get_logger()
    logger.info("Test logging output")

def test_config_loading(tmp_path):
    conf_dir = tmp_path / "configs"
    conf_dir.mkdir()
    with open(conf_dir / "config.yaml", "w") as f:
        f.write("trainer:\n  max_epochs: 1\n")
    with initialize_config_dir(config_dir=str(conf_dir)):
        cfg = compose(config_name="config")
    assert cfg.trainer.max_epochs == 1

def test_training_loop(synthetic_dmodule):
    model = MLP(input_dim=20, hidden_layers=[32,32], output_dim=1, lr=1e-3)
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=synthetic_dmodule)

def test_early_stopping(synthetic_dmodule):
    model = MLP(input_dim=20)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=1)
    trainer = Trainer(max_epochs=2, callbacks=[early_stop], logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=synthetic_dmodule)

def test_evaluation_script(tmp_path, synthetic_dmodule):
    # Mock artifacts presence
    artifacts_dir = tmp_path / "kronara" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preds = torch.rand(100)
    labels = torch.randint(0,2,(100,))
    logits = torch.randn(100)
    torch.save(preds, artifacts_dir / "fold_0_preds.pt")
    torch.save(labels, artifacts_dir / "fold_0_labels.pt")
    torch.save(logits, artifacts_dir / "fold_0_logits.pt")

    # Create a minimal config directory for Hydra with controlled logging and run dir
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    with open(config_dir / "config.yaml", "w") as f:
        f.write("""trainer:
  max_epochs: 1
hydra:
  run:
    dir: .
  job_logging:
    version: 1
    disable_existing_loggers: false
    formatters:
      simple:
        format: "%(levelname)s - %(message)s"
    handlers:
      file:
        class: logging.FileHandler
        formatter: simple
        filename: evaluate.log
        mode: w
    root:
      handlers: [file]
""")

    # Run eval_main with overridden config path and name
    with patch("os.getcwd", return_value=str(tmp_path)):
        with patch("sys.argv", ["evaluate.py", f"--config-path={config_dir}", "--config-name=config"]):
            eval_main()

def test_accuracy_metric():
    acc = Accuracy(task="binary")
    preds = torch.tensor([0.1,0.9,0.6,0.4])
    labels = torch.tensor([0,1,1,0])
    # (preds>=0.5) = [0,1,1,0], labels = [0,1,1,0] -> all match, accuracy=1.0
    metric_val = acc((preds>=0.5).float(), labels).item()
    assert metric_val == 1.0

def test_shap_integration():
    import shap
    import torch
    import numpy as np

    # Define a simple linear model that returns (N,) outputs rather than (N,1)
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=20):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, 1)
        def forward(self, x):
            return self.linear(x).squeeze(-1)  # Remove the trailing dimension

    model = SimpleModel()
    model.eval()

    # Generate more data for background and testing
    # Use a reasonably large background set to avoid regression issues
    x = torch.randn(200, 20)
    background = x[:100]

    def model_fn(arr):
        with torch.no_grad():
            t = torch.from_numpy(arr).float()
            return model(t).numpy()  # shape (N,)

    explainer = shap.KernelExplainer(model_fn, background.numpy())
    vals = explainer.shap_values(x[:10].numpy(), nsamples=100)

    # For a single output, KernelExplainer returns a single NumPy array of shape (samples, features)
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (10, 20)  # 10 samples, 20 features

def test_custom_threshold():
    preds = torch.tensor([0.2,0.8,0.5,0.9,0.4])
    labels = torch.tensor([0,1,0,1,0])
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.1,1.0,0.1):
        p = (preds>=t).float()
        tp = ((p==1)*(labels==1)).sum().item()
        fp = ((p==1)*(labels==0)).sum().item()
        fn = ((p==0)*(labels==1)).sum().item()
        precision = tp/(tp+fp) if tp+fp>0 else 0
        recall = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*(precision*recall)/(precision+recall) if precision+recall>0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    assert best_f1 >= 0

def test_persistent_workers_flag(synthetic_data):
    X, y = synthetic_data
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=32, num_workers=0, persistent_workers=False)
    batch = next(iter(loader))
    assert len(batch) == 2

def test_parameter_count():
    model = MLP(input_dim=20, hidden_layers=[64,64], output_dim=1)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 20*64

def test_no_exceptions_during_train(synthetic_dmodule):
    model = MLP(input_dim=20)
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    try:
        trainer.fit(model, datamodule=synthetic_dmodule)
    except Exception as e:
        pytest.fail(f"Training raised an exception: {e}")

def test_no_exceptions_during_test(synthetic_dmodule):
    model = MLP(input_dim=20)
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=synthetic_dmodule)
    try:
        trainer.test(model, datamodule=synthetic_dmodule)
    except Exception as e:
        pytest.fail(f"Testing raised an exception: {e}")

def test_fallback_to_synthetic():
    dm = DataModule(
        data_path="",
        num_samples=100,
        num_features=20,
        test_size=0.2,
        k_folds=2,
        current_fold=0,
        batch_size=32,
        num_workers=0,
        seed=TEST_SEED,
        persistent_workers=False,
        fallback_to_synthetic=True
    )
    dm.prepare_data()
    dm.setup()
    assert len(dm.train_dataset) > 0
