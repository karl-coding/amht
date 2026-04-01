# AMHT

Project layout:

- `model/amht.py`
- `model/ssm.py`
- `model/router.py`
- `model/memory.py`
- `train/train.py`
- `train/config.yaml`
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

Optional Google Drive checkpoint output in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
%env AMHT_CHECKPOINT_DIR=/content/drive/MyDrive/amht_checkpoints
```
