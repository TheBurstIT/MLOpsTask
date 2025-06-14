from pathlib import Path

import fire
from hydra import compose, initialize_config_dir

from crypto_lob.models.mlp_module import MLPLit as Model

# from hydra.core.global_hydra import GlobalHydra
# from omegaconf import OmegaConf


CFG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _cfg(overrides):
    with initialize_config_dir(version_base="1.3", config_dir=CFG_DIR):
        return compose(config_name="train", overrides=overrides)


def download(**overrides):
    cfg = _cfg(overrides)
    from crypto_lob.data.download import fetch

    fetch(Path(cfg.data.raw_dir), fname=cfg.data.csv_file)


def preprocess(**overrides):
    cfg = _cfg(list(overrides))
    from crypto_lob.data.preprocess import snapshots

    snapshots(
        raw_dir=Path(cfg.data.raw_dir),  # ✔ правильное поле
        out_dir=Path(cfg.data.processed_dir),
        lookback=cfg.data.lookback,  # передаём окно
    )


def train(**overrides):
    cfg = _cfg(overrides)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning.loggers import MLFlowLogger

    from crypto_lob.datamodules.lob_dm import LOBDataModule

    # from crypto_lob.models.tcn_module import TCNLitModule

    pl.seed_everything(cfg.seed, workers=True)
    dm = LOBDataModule(cfg)
    model = Model(cfg)
    logger = MLFlowLogger(
        experiment_name="micro-move",
        tracking_uri=cfg.mlflow_uri,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        deterministic=True,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )
    trainer.fit(model, dm)


def entry():
    fire.Fire({"download": download, "preprocess": preprocess, "train": train})
