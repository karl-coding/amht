#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python train/train.py --config train/config.yaml --seq-len 8192 "$@"
