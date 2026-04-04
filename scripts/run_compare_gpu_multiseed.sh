#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-200}"
DEVICE="${2:-cuda}"
SEEDS="${3:-42 43 44}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p results

for SEED in ${SEEDS}; do
  "${PYTHON_BIN}" train/train.py \
    --config train/config_amht_8k.yaml \
    --seq-len 8192 \
    --steps "${STEPS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --log-jsonl "results/amht_8k_seed${SEED}_train.jsonl"

  "${PYTHON_BIN}" eval/benchmark.py \
    --config train/config_amht_8k.yaml \
    --task all \
    --seq-len 8192 \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --save-json "results/amht_8k_seed${SEED}_eval.json"

  "${PYTHON_BIN}" train/train.py \
    --config train/config_transformer_8k.yaml \
    --seq-len 8192 \
    --steps "${STEPS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --log-jsonl "results/transformer_8k_seed${SEED}_train.jsonl"

  "${PYTHON_BIN}" eval/benchmark.py \
    --config train/config_transformer_8k.yaml \
    --task all \
    --seq-len 8192 \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --save-json "results/transformer_8k_seed${SEED}_eval.json"
done

"${PYTHON_BIN}" scripts/aggregate_results.py \
  --glob-amht "results/amht_8k_seed*_eval.json" \
  --glob-baseline "results/transformer_8k_seed*_eval.json" \
  --baseline-name transformer
