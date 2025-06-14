from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _get(cfg, key, default=None):
    """Ищет cfg.train.key или cfg.key — что найдётся."""
    if "train" in cfg and key in cfg.train:
        return cfg.train[key]
    return cfg.get(key, default)


class LOBDataset(Dataset):
    def __init__(self, npz_path: Path):
        arr = np.load(npz_path)
        self.X = torch.tensor(arr["X"], dtype=torch.float32)
        self.y = torch.tensor(arr["y"], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LOBDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # ──────────────────────────────── setup ────────────────────────────
    def setup(self, stage=None):
        fp = Path(self.cfg.data.processed_dir) / "btc_1sec.npz"
        full = LOBDataset(fp)

        split = int(0.8 * len(full))
        g = torch.Generator().manual_seed(_get(self.cfg, "seed", 42))
        self.train_ds, self.val_ds = random_split(
            full, [split, len(full) - split], generator=g
        )

    # ───────────────────────────── dataloaders ─────────────────────────
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=_get(self.cfg, "batch_size", 256),
            shuffle=True,
            num_workers=_get(self.cfg, "num_workers", 4),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=_get(self.cfg, "batch_size", 256),
            shuffle=False,
            num_workers=_get(self.cfg, "num_workers", 4),
        )
