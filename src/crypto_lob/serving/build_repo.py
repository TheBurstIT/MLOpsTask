import shutil
from pathlib import Path

import click


@click.command()
@click.option("--onnx", type=click.Path(exists=True), required=True)
@click.option("--repo", type=click.Path(), default="docker/triton/models")
@click.option("--n_inputs", type=int, required=True)
def main(onnx, repo, n_inputs):
    onnx = Path(onnx)
    model_dir = Path(repo) / "micromove/1"
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx, model_dir / "model.onnx")

    cfg = f"""
name: "micromove"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "features"
    data_type: TYPE_FP32
    dims: [ {n_inputs} ]
  }}
]
output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }}
]
"""
    (model_dir.parent / "config.pbtxt").write_text(cfg.strip())
    print(f"✓ Triton repo ready → {model_dir.parent}")


if __name__ == "__main__":
    main()
