#!/usr/bin/env bash
set -euo pipefail

# ---- EDIT THESE ----
LOCAL_REPO="$HOME/Documents/smanna/iisc/work1/breaching"                 # local repo root
REMOTE_HOST="siladittyam@10.16.63.23"                      # cluster login
REMOTE_REPO="/home/anirban/siladittyam/attack/breaching"           # repo root on cluster (unused)
PYTHON_BIN="$HOME/Documents/smanna/iisc/bin/python"                # local venv python
TORCH_ARCH="8.9"                                    # Ada (semicolon-separated)
# --------------------

export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"

# Build extensions by running a tiny import that triggers compilation
cd "$LOCAL_REPO"
"$PYTHON_BIN" - <<'PY'
import os
os.environ["STYLEGANXL_DISABLE_CUSTOM_OPS"] = "0"
from breaching.attacks.auxiliaries.stylegan_xl.torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu
# trigger init
import torch
x = torch.randn(1, 3, 4, 4, device='cuda')
bias_act.bias_act(x, impl='cuda')
upfirdn2d.upfirdn2d(x, f=None, impl='cuda')
filtered_lrelu.filtered_lrelu(x, impl='cuda')
print("Built custom ops")
PY

# Find cached extensions and copy to cluster
LOCAL_CACHE_DIR="$HOME/.cache/torch_extensions"
REMOTE_CACHE_DIR="/home/anirban/siladittyam/.cache/torch_extensions"
rsync -av "$LOCAL_CACHE_DIR/" "${REMOTE_HOST}:${REMOTE_CACHE_DIR}/"

echo "Copied torch_extensions cache to cluster."
