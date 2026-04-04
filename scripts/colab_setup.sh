#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

CHECKPOINT_DIR="${1:-${AMHT_CHECKPOINT_DIR:-}}"

# Show runtime details for quick sanity checks in Colab.
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

if [ -n "${CHECKPOINT_DIR}" ]; then
  mkdir -p "${CHECKPOINT_DIR}"
  cat <<EOF

configured_checkpoint_dir=${CHECKPOINT_DIR}

This script cannot export AMHT_CHECKPOINT_DIR into the parent shell by itself.
To keep training and evaluation consistent in later commands, run one of:

  export AMHT_CHECKPOINT_DIR=${CHECKPOINT_DIR}

or in a notebook cell:

  %env AMHT_CHECKPOINT_DIR=${CHECKPOINT_DIR}

EOF
fi

cat <<'EOF'

Optional Google Drive checkpoint path:

  from google.colab import drive
  drive.mount('/content/drive')

Notebook cell:

  %env AMHT_CHECKPOINT_DIR=/content/drive/MyDrive/amht_checkpoints

Shell cell:

  export AMHT_CHECKPOINT_DIR=/content/drive/MyDrive/amht_checkpoints
  mkdir -p "$AMHT_CHECKPOINT_DIR"

Training and evaluation now both resolve checkpoints from:

  1. $AMHT_CHECKPOINT_DIR, if set
  2. training.checkpoint_dir from the YAML config, otherwise

EOF
