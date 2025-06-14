import pytorch_lightning as pl
import torch
import torch.nn as nn

# import torchmetrics
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import MulticlassF1Score


class MLPLit(pl.LightningModule):
    def __init__(self, cfg):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
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
        # --- исправлено ↓ -----------------------------------------------
        self.f1 = MulticlassF1Score(num_classes=3, average="macro")
        # ----------------------------------------------------------------

    def forward(self, x):  # (B, n_in)
        return self.net(x)  # (B, 3)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.crit(logits, y)
        preds = logits.argmax(1)
        f1 = self.f1(preds, y)
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_f1": f1},
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
