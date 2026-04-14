#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

PRESET="${AMHT_PRESET:-stage1_round4_long}"
DEVICE="${AMHT_DEVICE:-auto}"
OUTDIR="${AMHT_OUTDIR:-}"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --preset)
      if (( i + 1 < ${#ARGS[@]} )); then
        PRESET="${ARGS[$((i + 1))]}"
      fi
      ;;
    --preset=*)
      PRESET="${ARGS[$i]#--preset=}"
      ;;
    --device)
      if (( i + 1 < ${#ARGS[@]} )); then
        DEVICE="${ARGS[$((i + 1))]}"
      fi
      ;;
    --device=*)
      DEVICE="${ARGS[$i]#--device=}"
      ;;
    --outdir)
      if (( i + 1 < ${#ARGS[@]} )); then
        OUTDIR="${ARGS[$((i + 1))]}"
      fi
      ;;
    --outdir=*)
      OUTDIR="${ARGS[$i]#--outdir=}"
      ;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="paper_runs/${PRESET}"
fi

"$PYTHON_BIN" scripts/run_colab_paper.py \
  --preset "$PRESET" \
  --device "$DEVICE" \
  --outdir "$OUTDIR" \
  --continue-on-error \
  --skip-figures \
  "${ARGS[@]}"

SUMMARY_JSON="$OUTDIR/report/summary.json"
ADJUSTMENTS_MD="$OUTDIR/report/adjustments.md"
if [[ -f "$SUMMARY_JSON" ]]; then
  "$PYTHON_BIN" scripts/suggest_v4_adjustments.py \
    --summary "$SUMMARY_JSON" \
    --out "$ADJUSTMENTS_MD"
fi
