#!/usr/bin/env bash
# Needle 3 LoRA fine-tune, run under WSL2 (Windows Smart App Control blocks
# jaxlib on the Windows side; see logs/experiments.md E5).
# Data and outputs stay in the Windows repo via /mnt/c.
set -euo pipefail

CASE=/mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot
VENV=/root/.venvs/claimpilot
cd "$CASE"

if [ ! -f train/needle_out/checkpoints/needle3.safetensors ]; then
  echo "downloading base checkpoint (needle3.safetensors) ..."
  "$VENV/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("Cactus-Compute/needle3", "checkpoints/needle3.safetensors",
                    local_dir="train/needle_out")
print("checkpoint:", p)
PY
fi

echo "starting fine-tune ..."
"$VENV/bin/needle" finetune train/needle_train.jsonl \
  --checkpoint train/needle_out/checkpoints/needle3.safetensors \
  --epochs 3 \
  --batch-size 16 \
  --max-len 2304 \
  --val-split 0.1 \
  --seed 0 \
  --out train/needle_out/adapter.safetensors \
  --checkpoint-dir train/needle_out/checkpoints

echo "fine-tune done"
