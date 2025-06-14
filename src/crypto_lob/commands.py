# src/crypto_lob/commands.py
from pathlib import Path
from typing import Sequence

import fire
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir

# from omegaconf import OmegaConf

# ───────────────────────────────── CONFIG ──────────────────────────────
CFG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _cfg(overrides: Sequence[str]):
    """
    Собираем Hydra-конфиг из defaults + списка overrides.
    overrides -- iterable of strings, например ["model.n_inputs=4590"]
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(CFG_DIR)):
        return compose(config_name="train", overrides=list(overrides))


# ──────────────────────────── Служебная обёртка ────────────────────────
def _get(cfg, path: str, default=None):
    """
    Безопасно вытаскиваем cfg.train.xxx или cfg.xxx
    """
    *maybe_train, leaf = path.split(".")
    holder = cfg
    for part in maybe_train:
        if part in holder:
            holder = holder[part]
        else:
            return cfg.get(leaf, default)
    return holder.get(leaf, default)


# ───────────────────────────────── download ────────────────────────────
def download(*ovr):
    cfg = _cfg(ovr)
    from crypto_lob.data.download import fetch

    fetch(Path(cfg.data.raw_dir), fname=cfg.data.csv_file)


# ─────────────────────────────── preprocess ────────────────────────────
def preprocess(*ovr):
    cfg = _cfg(ovr)
    from crypto_lob.data.preprocess import snapshots

    snapshots(
        raw_dir=Path(cfg.data.raw_dir),
        out_dir=Path(cfg.data.processed_dir),
        lookback=cfg.data.lookback,
    )


# ─────────────────────────────────── train ─────────────────────────────
def train(*ovr):
    cfg = _cfg(ovr)

    from pytorch_lightning.loggers import MLFlowLogger

    from crypto_lob.datamodules.lob_dm import LOBDataModule
    from crypto_lob.models.mlp_module import MLPLit

    seed = _get(cfg, "train.seed", 42)
    pl.seed_everything(seed, workers=True)

    dm = LOBDataModule(cfg)
    model = MLPLit(cfg)

    logger = MLFlowLogger(
        experiment_name="micro-move",
        tracking_uri=cfg.logging.mlflow_uri,
    )

    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 3)),
        deterministic=True,
        logger=logger,
    )
    trainer.fit(model, dm)


# ───────────────────────────────── entry ───────────────────────────────
def entry():
    """
    Точка входа для poetry script «clob».
    Пример CLI:
        clob download
        clob preprocess data.lookback=30
        clob train model.n_inputs=4590 max_epochs=5
    """
    fire.Fire(
        {
            "download": download,
            "preprocess": preprocess,
            "train": train,
        }
    )


# удобен прямой запуск:  python -m crypto_lob.commands train ...
if __name__ == "__main__":
    entry()
