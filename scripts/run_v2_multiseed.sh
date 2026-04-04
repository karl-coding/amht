#!/usr/bin/env bash
set -euo pipefail

SEEDS="42 43 44"
CONFIGS=(
  "train/config_amht_8k_fast.yaml:200:amht_v2_8k_fast:amht_seq8192"
  "train/config_amht_8k_accurate.yaml:500:amht_v2_8k_accurate:amht_seq8192"
  "train/config_transformer_8k.yaml:200:transformer_8k:transformer_seq8192"
)

mkdir -p results checkpoints

for cfg_entry in "${CONFIGS[@]}"; do
  IFS=":" read -r cfg steps prefix checkpoint <<< "$cfg_entry"
  for seed in $SEEDS; do
    PYTHONUNBUFFERED=1 python3 train/train.py \
      --config "$cfg" \
      --seq-len 8192 \
      --steps "$steps" \
      --device cuda \
      --seed "$seed" \
      --log-jsonl "results/${prefix}_seed${seed}_train.jsonl"

    python3 eval/benchmark.py \
      --config "$cfg" \
      --checkpoint "checkpoints/${checkpoint}.pt" \
      --task all \
      --seq-len 8192 \
      --device cuda \
      --seed "$seed" \
      --save-json "results/${prefix}_seed${seed}_eval.json"
  done
done
