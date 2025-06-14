from pathlib import Path

import click
import torch

from crypto_lob.models.mlp_module import MLPLit


@click.command()
@click.option(
    "--ckpt", type=click.Path(exists=True), required=True, help="Путь к Lightning .ckpt"
)
@click.option("--out", type=click.Path(), default="artifacts/model.onnx")
def main(ckpt, out):
    ckpt = Path(ckpt)
    out = Path(out)
    model = MLPLit.load_from_checkpoint(str(ckpt))
    model.eval().cpu()

    n_in = model.hparams["model"]["n_inputs"]
    dummy = torch.randn(1, n_in)

    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"✓ ONNX saved → {out}")


if __name__ == "__main__":
    main()
