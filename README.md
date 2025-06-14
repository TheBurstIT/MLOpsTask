# Crypto-LOB Micro-Move Classifier
*Predict the next-second direction of the mid-price from limit-order-book snapshots*

---

## 1 · What’s inside

| Section | TL;DR |
|---------|-------|
| **Goal** | 3-class classification **↑ / → / ↓** of `midpoint(t + 1 s) − midpoint(t)` |
| **Data** | Public **High-Frequency Crypto LOB** (Bybit). We use **BTC _1sec.csv** — ≈ 1 M rows × 150 LOB columns |
| **Features** | 30-second rolling window → `30 × 153 = 4590` float32 features |
| **Model** | 3-layer MLP (512-256-3) · CrossEntropy · Macro F1 |
| **Pipeline** | `download → preprocess → train → export(ONNX) → Triton CPU` — reproducible with **Poetry + Hydra + DVC + MLflow** |
| **Latency** | ≈ 0.8 ms / 1 sample on Ryzen 5950X via Triton (onnxruntime backend) |

---

## 2 · Repo layout
```bash
crypto-lob-micro-move/
├─ src/cryptolob/
│ ├─ commands.py # Fire CLI
│ ├─ data/ # download / preprocess
│ ├─ datamodules/ # LightningDataModule
│ ├─ models/ # Lightning MLP
│ ├─ export/ # ckpt → ONNX
│ └─ serving/ # Triton repo builder
├─ configs/ # Hydra YAMLs
├─ artifacts/ # checkpoints, ONNX
├─ docker/triton/models/ # generated model-repo
├─ dvc.yaml
└─ README.md
```

---
# Part I · Setup 🛠️

> Requires Python 3.10+ and Docker. **No GPU needed.**

```bash
git clone https://github.com/you/crypto-lob-micro-move.git
cd crypto-lob-micro-move

# Python deps
curl -sSL https://install.python-poetry.org | python -
poetry install

# Dev helpers
pre-commit install
dvc init            # if you forked; otherwise `dvc pull`
```
Part II · Train
```bash
# 0 · data
poetry run clob download

# 1 · build 30-s windows
poetry run clob preprocess data.lookback=30

# 2 · train 5 epochs (MLflow logs in ./mlruns)
poetry run clob train model.n_inputs=4590 max_epochs=5 batch_size=256

key=value after the command are Hydra overrides; combine as you like.
Full pipeline via DVC:
dvc repro train
```
#Part III · Production preparation

Lightning ckpt → ONNX
poetry run clob export --ckpt=artifacts/best.ckpt

Build Triton CPU repo
```bash
Artifacts delivered with the model:
artifacts/model.onnx
docker/triton/models/micromove/
    ├─ 1/model.onnx
    └─ config.pbtxt
```
#Part IV · Infer
```bash
1 · launch Triton CPU-only
docker run -p8000:8000 -p8001:8001 \
       -v $(pwd)/docker/triton/models:/models \
       ghcr.io/triton-inference-server/server:24.03-py3 \
       tritonserver --model-repository=/models

2 · call REST
import requests, json, numpy as np
vec = np.random.randn(1, 4590).astype("float32").tolist()

payload = {
  "inputs": [{
      "name": "features",
      "shape": [1, 4590],
      "datatype": "FP32",
      "data": vec
  }]
}

r = requests.post("http://localhost:8000/v2/models/micromove/infer",
                  headers={"Content-Type":"application/json"},
                  data=json.dumps(payload))
print(r.json())      # logits for classes [↓, →, ↑]

Input format — flattened window of 30 LOB snapshots (4590 floats).
```
