# AMHT

Project layout:

- `model/amht.py`
- `model/ssm.py`
- `model/router.py`
- `model/memory.py`
- `train/train.py`
- `train/config.yaml`
- `train/config_amht_8k.yaml`
- `train/config_transformer_8k.yaml`
- `train/distributed.py`
- `data/dataset.py`
- `data/tokenizer.py`
- `eval/benchmark.py`
- `eval/niah.py`
- `eval/scaling.py`
- `scripts/run_train.sh`
- `scripts/run_eval.sh`

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Train at 8K context

```bash
python3 train/train.py --config train/config.yaml --seq-len 8192
```

Explicit AMHT 8K preset:

```bash
PYTHONUNBUFFERED=1 python3 train/train.py --config train/config_amht_8k.yaml --seq-len 8192 --device cuda
```

## Train on CPU

Use the smaller CPU preset to keep runtime and memory usage reasonable:

```bash
PYTHONUNBUFFERED=1 python3 train/train.py --config train/config_cpu.yaml --seq-len 4096 --device cpu
```

For a faster smoke test:

```bash
PYTHONUNBUFFERED=1 python3 train/train.py --config train/config_cpu.yaml --seq-len 2048 --steps 10 --device cpu
```

## Evaluate

Throughput:

```bash
python3 eval/benchmark.py --config train/config.yaml --task throughput --seq-len 8192
```

Needle-in-a-Haystack:

```bash
python3 eval/benchmark.py --config train/config.yaml --task niah
```

Scaling:

```bash
python3 eval/benchmark.py --config train/config.yaml --task scaling
```

All:

```bash
python3 eval/benchmark.py --config train/config.yaml --task all --seq-len 8192
```

CPU eval:

```bash
python3 eval/benchmark.py --config train/config_cpu.yaml --task all --seq-len 4096 --device cpu
```

CPU eval with a trained checkpoint:

```bash
python3 eval/benchmark.py --config train/config_cpu.yaml --checkpoint checkpoints/amht_seq4096.pt --task niah --seq-len 4096 --device cpu
```

## Colab

After you open your own Google Colab runtime with GPU enabled:

```bash
bash scripts/colab_setup.sh
PYTHONUNBUFFERED=1 python train/train.py --config train/config.yaml --seq-len 8192 --steps 3 --device cuda
python eval/benchmark.py --config train/config.yaml --task all --seq-len 8192 --device cuda
```

There is also a notebook at `AMHT_Colab.ipynb`.

For a more conservative Colab T4 preset, use:

```bash
PYTHONUNBUFFERED=1 python train/train.py --config train/config_colab_t4.yaml --seq-len 8192 --steps 3 --device cuda
python eval/benchmark.py --config train/config_colab_t4.yaml --task all --seq-len 8192 --device cuda
```

## GPU Comparison

To compare AMHT against the local-attention Transformer baseline on GPU:

```bash
bash scripts/run_compare_gpu.sh 200 cuda
```

This produces:

- `results/amht_8k_train.jsonl`
- `results/amht_8k_eval.json`
- `results/transformer_8k_train.jsonl`
- `results/transformer_8k_eval.json`

Optional Google Drive checkpoint output in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
%env AMHT_CHECKPOINT_DIR=/content/drive/MyDrive/amht_checkpoints
```

## Validated Runs

- GPU retrieval training at 8K context has been validated on Colab T4-style settings.
- Checkpoint-loaded 8K NIAH reached `1.0` on the aligned toy retrieval task.
- Harder 8K GPU curriculum reached `0.8571` mean NIAH after 500 steps.
- CPU retrieval training at 4K context has been validated end to end.
- Checkpoint-loaded CPU NIAH at 4K reached `1.0` on the aligned toy retrieval task.
