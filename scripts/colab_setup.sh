#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Show runtime details for quick sanity checks in Colab.
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

cat <<'EOF'

Optional Google Drive checkpoint path:

  from google.colab import drive
  drive.mount('/content/drive')
  %env AMHT_CHECKPOINT_DIR=/content/drive/MyDrive/amht_checkpoints

EOF
