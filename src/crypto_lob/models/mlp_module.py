import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import OmegaConf


class MLPLit(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        n_in = cfg.model.n_inputs
        self.net = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.crit = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1Score(num_classes=3, average="macro")

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.crit(logits, y)
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_f1": self.f1(logits.argmax(1), y)},
            prog_bar=True,
        )
        return loss

    def training_step(self, b, _):
        return self._step(b, "train")

    def validation_step(self, b, _):
        self._step(b, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
