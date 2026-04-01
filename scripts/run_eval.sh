#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python eval/benchmark.py --config train/config.yaml --task all --seq-len 8192 "$@"
