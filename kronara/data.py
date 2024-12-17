# kronara/data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Optional
from loguru import logger
import polars as plr
from sklearn.preprocessing import StandardScaler

class RealOrSyntheticDataset(Dataset):
    """
    A PyTorch Dataset for handling either real or synthetic data.

    Attributes:
        X (torch.Tensor): Feature tensor.
        y (torch.Tensor): Label tensor.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataModule(pl.LightningDataModule):
    """
    A LightningDataModule for loading, preprocessing, and splitting data
    into train, validation, and test sets. Supports both synthetic and real data.
    Utilizes stratified k-fold cross-validation for train/val splitting.

    Args:
        data_path (str): Path to data file. If None or empty, synthetic data is used if fallback_to_synthetic=True.
        num_samples (int): Number of synthetic samples.
        num_features (int): Number of features.
        test_size (float): Fraction of data used for testing.
        k_folds (int): Number of folds for cross-validation.
        current_fold (int): Index of the current fold used.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for data loading.
        seed (int): Random seed for reproducibility.
        persistent_workers (bool): Keep workers persistent.
        fallback_to_synthetic (bool): Whether to fallback to synthetic data if no real data is found.
    """
    def __init__(self, 
                 data_path: Optional[str],
                 num_samples: int,
                 num_features: int,
                 test_size: float,
                 k_folds: int,
                 current_fold: int,
                 batch_size: int,
                 num_workers: int,
                 seed: int,
                 persistent_workers: bool=True,
                 fallback_to_synthetic: bool=False):
        super().__init__()
        self.data_path = data_path
        self.num_samples = num_samples
        self.num_features = num_features
        self.test_size = test_size
        self.k_folds = k_folds
        self.current_fold = current_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.persistent_workers = persistent_workers
        self.fallback_to_synthetic = fallback_to_synthetic

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        np.random.seed(self.seed)

    def prepare_data(self):
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading dataset from {self.data_path}")
            if self.data_path.endswith(".csv"):
                df = plr.read_csv(self.data_path)
            elif self.data_path.endswith(".parquet"):
                df = plr.read_parquet(self.data_path)
            else:
                raise ValueError("Unsupported file format. Provide CSV or Parquet.")

            if "label" not in df.columns:
                raise ValueError("The dataset must contain a 'label' column.")

            y = df["label"].to_numpy()
            X = df.drop("label").to_numpy()
        else:
            if not self.fallback_to_synthetic:
                raise FileNotFoundError("Data file not found and fallback_to_synthetic=False. Cannot proceed.")
            logger.warning("No dataset provided. Falling back to synthetic data.")
            X, y = self._generate_synthetic_data(self.num_samples, self.num_features, self.seed)

        if X.shape[0] != len(y):
            raise ValueError("Number of samples in features and labels must match.")
        if X.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {X.shape[1]}.")

        if len(np.unique(y)) < 2:
            raise ValueError("Label must have at least two classes for binary classification.")

        self.X_full = X
        self.y_full = y

    def setup(self, stage=None):
        X = self.X_full
        y = self.y_full

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        folds = list(skf.split(np.arange(len(X_trainval)), y_trainval))
        train_indices, val_indices = folds[self.current_fold]

        scaler = StandardScaler()
        scaler.fit(X_trainval[train_indices])

        X_train_scaled = scaler.transform(X_trainval[train_indices])
        X_val_scaled = scaler.transform(X_trainval[val_indices])
        X_test_scaled = scaler.transform(X_test)

        self.train_dataset = RealOrSyntheticDataset(X_train_scaled, y_trainval[train_indices])
        self.val_dataset = RealOrSyntheticDataset(X_val_scaled, y_trainval[val_indices])
        self.test_dataset = RealOrSyntheticDataset(X_test_scaled, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def _generate_synthetic_data(self, n, d, seed):
        np.random.seed(seed)
        X = np.random.randn(n, d)
        score = X[:, :10].sum(axis=1) + np.sin(X[:,10:20]).sum(axis=1) + np.random.randn(n)*0.5
        y = (score > 0).astype('float32')
        flip_mask = np.random.rand(n)<0.005
        y[flip_mask] = 1 - y[flip_mask]
        return X, y