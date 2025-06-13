import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from omegaconf import OmegaConf


class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp = chomp
    def forward(self, x): return x[..., :-self.chomp]


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, p):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=d)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=d)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop1(self.relu1(self.chomp1(y)))
        y = self.conv2(y)
        y = self.drop2(self.relu2(self.chomp2(y)))
        return self.relu(y + self.down(x))


class MaskedSelfAttn(nn.Module):
    """Маскируем будущее время-шаги (causal)"""
    def __init__(self, c, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.register_buffer(
            "mask", torch.triu(torch.ones(1024, 1024), diagonal=1).bool()
        )

    def forward(self, x):
        # x: [B, C, L]  →  [B, L, C]
        x = x.transpose(1, 2)
        out, _ = self.attn(x, x, x, attn_mask=self.mask[: x.size(1), : x.size(1)])
        return out.transpose(1, 2)


class MicroMoveNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chs = cfg.model.channels
        layers, c_in = [], 2
        for i, c_out in enumerate(chs):
            layers.append(ResBlock(c_in, c_out, cfg.model.kernel_size,
                                   d=2 ** i, p=cfg.model.dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.attn = MaskedSelfAttn(chs[-1], cfg.model.attn_heads)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(chs[-1], 3)
        )

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))     # [B, C, L]
        y = self.attn(y)
        return self.head(y)


class TCNLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.net = MicroMoveNet(cfg)
        self.crit = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 2.0]))
        self.f1 = torchmetrics.F1Score(num_classes=3, average="macro")
        self.latency = torchmetrics.MeanMetric()

    def forward(self, x): return self.net(x)

    def _step(self, batch, stage):
        x, y = batch
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        logits = self(x)
        t1.record(); torch.cuda.synchronize()
        self.latency.update(t0.elapsed_time(t1))  # ms
        loss = self.crit(logits, y)
        preds = logits.argmax(1)
        f1 = self.f1(preds, y)
        self.log_dict({f"{stage}_loss": loss, f"{stage}_f1": f1}, prog_bar=True)
        return loss

    def training_step(self, b, _): return self._step(b, "train")
    def validation_step(self, b, _): self._step(b, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

