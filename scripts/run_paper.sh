#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

PRESET="${AMHT_PRESET:-paper_v4}"
DEVICE="${AMHT_DEVICE:-auto}"
OUTDIR="${AMHT_OUTDIR:-paper_runs/${PRESET}}"

"$PYTHON_BIN" scripts/run_colab_paper.py \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --outdir "$OUTDIR" \
  --skip-train \
  --skip-eval \
  "$@"
