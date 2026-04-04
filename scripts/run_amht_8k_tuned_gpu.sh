#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-500}"
DEVICE="${2:-cuda}"
SEED="${3:-42}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p results

"${PYTHON_BIN}" train/train.py \
  --config train/config_amht_8k_tuned.yaml \
  --seq-len 8192 \
  --steps "${STEPS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --log-jsonl "results/amht_8k_tuned_seed${SEED}_train.jsonl"

"${PYTHON_BIN}" eval/benchmark.py \
  --config train/config_amht_8k_tuned.yaml \
  --task all \
  --seq-len 8192 \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --save-json "results/amht_8k_tuned_seed${SEED}_eval.json"
