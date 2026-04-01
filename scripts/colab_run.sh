#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

python train/train.py \
  --config train/config.yaml \
  --seq-len 8192 \
  --steps 3 \
  --device cuda

python eval/benchmark.py \
  --config train/config.yaml \
  --task all \
  --seq-len 8192 \
  --device cuda
