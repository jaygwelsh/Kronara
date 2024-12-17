# tests/test_data.py
import pytest
from kronara.data import DataModule

def test_data_module_synthetic():
    dm = DataModule(
        data_path="",
        num_samples=10000,
        num_features=20,
        test_size=0.2,
        k_folds=5,
        current_fold=0,
        batch_size=64,
        num_workers=0,
        seed=42,
        persistent_workers=False,
        fallback_to_synthetic=True
    )
    dm.prepare_data()
    dm.setup()
    train_dl = dm.train_dataloader()
    batch = next(iter(train_dl))
    x, y = batch
    assert x.shape[1] == 20
    assert len(x) == 64
    assert y.shape[0] == 64
