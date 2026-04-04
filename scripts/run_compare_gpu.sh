#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-200}"
DEVICE="${2:-cuda}"

mkdir -p results

.venv/bin/python train/train.py \
  --config train/config_amht_8k.yaml \
  --seq-len 8192 \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --log-jsonl results/amht_8k_train.jsonl

.venv/bin/python eval/benchmark.py \
  --config train/config_amht_8k.yaml \
  --task all \
  --seq-len 8192 \
  --device "${DEVICE}" \
  --save-json results/amht_8k_eval.json

.venv/bin/python train/train.py \
  --config train/config_transformer_8k.yaml \
  --seq-len 8192 \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --log-jsonl results/transformer_8k_train.jsonl

.venv/bin/python eval/benchmark.py \
  --config train/config_transformer_8k.yaml \
  --task all \
  --seq-len 8192 \
  --device "${DEVICE}" \
  --save-json results/transformer_8k_eval.json

.venv/bin/python scripts/compare_results.py \
  --amht results/amht_8k_eval.json \
  --baseline results/transformer_8k_eval.json \
  --baseline-name transformer
