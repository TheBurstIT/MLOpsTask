import os
from pathlib import Path
from typing import Sequence

import fire
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from pytorch_lightning.callbacks import ModelCheckpoint

# from omegaconf import OmegaConf

# ───────────────────────────────── CONFIG ──────────────────────────────
CFG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _cfg(overrides: Sequence[str]):
    with initialize_config_dir(version_base="1.3", config_dir=str(CFG_DIR)):
        return compose(config_name="train", overrides=list(overrides))


# ──────────────────────────── Служебная обёртка ────────────────────────
def _get(cfg, path: str, default=None):
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

    ckpt_cb = ModelCheckpoint(
        dirpath="artifacts",
        filename="best",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=int(_get(cfg, "train.max_epochs", 3)),
        deterministic=True,
        logger=logger,
        callbacks=[ckpt_cb],
    )
    trainer.fit(model, dm)


# ───────────────────────────────── entry ───────────────────────────────
def entry():
    fire.Fire(
        {
            "download": download,
            "preprocess": preprocess,
            "train": train,
            "export": export_cpu,
            "serve": serve_cpu,
        }
    )


# удобен прямой запуск:  python -m crypto_lob.commands train ...
if __name__ == "__main__":
    entry()


def export_cpu(*ovr):
    cfg = _cfg(ovr)
    ckpt = Path("artifacts/best.ckpt")
    onnx = Path("artifacts/model.onnx")

    # 1) ckpt → onnx
    from crypto_lob.export.onnx_export import main as to_onnx

    to_onnx(["--ckpt", str(ckpt), "--out", str(onnx)])

    # 2) Triton repo
    from crypto_lob.serving.build_repo import main as build

    build(["--onnx", str(onnx), "--n_inputs", str(cfg.model.n_inputs)])


def serve_cpu(port: int = 9001):
    """
    Запускает ONNX-Runtime Server на CPU.
    Аргумент `port` — внешний порт хоста (по умолчанию 9001).
    """
    image = "mcr.microsoft.com/onnxruntime/server:latest"
    cmd = (
        f"docker run -p{port}:8001 "
        "-v $(pwd)/artifacts/model.onnx:/models/model.onnx "
        f"{image} "
        "--model_path /models/model.onnx --http_port 8001"
    )
    os.system(cmd)
