from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class LOBDataset(Dataset):
    def __init__(self, npz_path):
        arr = np.load(npz_path)
        self.X = torch.tensor(arr["X"], dtype=torch.float32)
        self.y = torch.tensor(arr["y"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LOBDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        fp = Path(self.cfg.data.processed_dir) / "btc_1sec.npz"
        full = LOBDataset(fp)
        split = int(0.8 * len(full))
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            full,
            [split, len(full) - split],
            generator=torch.Generator().manual_seed(self.cfg.train.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
        )
