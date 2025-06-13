from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class LOBDataset(Dataset):
    def __init__(self, npz_files, horizon_ms=300):
        self.x, self.y = [], []
        for fp in npz_files:
            arr = np.load(fp)
            x = arr["x"]  # bid, ask, mid
            # target: знак сдвига mid через horizon
            delta = np.roll(x[:, 2], -horizon_ms // 100) - x[:, 2]
            y = np.sign(delta[:-1]).astype(np.int8)  # -1/0/1
            self.x.append(x[:-1, :2])  # bid, ask
            self.y.append(y)
        self.x = torch.tensor(np.concatenate(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(self.y), dtype=torch.long) + 1  # 0..2

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class LOBDataModule(pl.LightningDataModule):
    def __init__(self, cfg): super().__init__(); self.cfg = cfg

    def setup(self, stage=None):
        files = sorted(Path(self.cfg.data.processed_dir).glob("*.npz"))
        split = int(0.8 * len(files))
        self.train_ds = LOBDataset(files[:split], self.cfg.data.horizon_ms)
        self.val_ds = LOBDataset(files[split:], self.cfg.data.horizon_ms)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.train.batch_size,
            shuffle=True, num_workers=self.cfg.train.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.cfg.train.batch_size,
            shuffle=False, num_workers=self.cfg.train.num_workers
        )

