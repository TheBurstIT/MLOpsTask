import fire
from hydra import compose, initialize
from pathlib import Path


def _cfg(overrides):
    with initialize(version_base=None, config_path="../../configs"):
        return compose(config_name="train", overrides=overrides)


def download(**overrides):
    cfg = _cfg(overrides)
    from crypto_lob.data.download import fetch

    fetch(Path(cfg.data.raw_dir), asset=cfg.data.asset)


def preprocess(**overrides):
    cfg = _cfg(overrides)
    from crypto_lob.data.preprocess import snapshots

    snapshots(
        raw_dir=Path(cfg.data.raw_dir),
        out_dir=Path(cfg.data.processed_dir),
        window_ms=cfg.data.window_ms,
    )


def train(**overrides):
    cfg = _cfg(overrides)
    from crypto_lob.models.tcn_module import TCNLitModule
    from crypto_lob.datamodules.lob_dm import LOBDataModule
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLFlowLogger
    from pytorch_lightning.callbacks import TQDMProgressBar

    pl.seed_everything(cfg.train.seed, workers=True)
    dm = LOBDataModule(cfg)
    model = TCNLitModule(cfg)
    logger = MLFlowLogger(
        experiment_name="micro-move",
        tracking_uri=cfg.logging.mlflow_uri,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=logger,
        deterministic=True,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
    )
    trainer.fit(model, dm)


def entry():
    fire.Fire(
        {"download": download, "preprocess": preprocess, "train": train}
    )

