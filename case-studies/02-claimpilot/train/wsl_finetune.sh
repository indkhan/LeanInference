#!/usr/bin/env bash
# Needle 3 LoRA fine-tune, run under WSL2 (Windows Smart App Control blocks
# jaxlib on the Windows side; see logs/experiments.md E5).
# Data and outputs stay in the Windows repo via /mnt/c.
# 2 epochs: the first 3-epoch run already reached val 0.049 after epoch 1, then died at
# step 1944/4050 with nothing saved (the CLI only writes the adapter at the end).
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform

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

echo "starting fine-tune $(date -Is) ..."
"$VENV/bin/needle" finetune train/needle_train.jsonl \
  --checkpoint train/needle_out/checkpoints/needle3.safetensors \
  --epochs 2 \
  --batch-size 1 \
  --max-len 2304 \
  --val-split 0.1 \
  --seed 0 \
  --out train/needle_out/adapter.safetensors \
  --checkpoint-dir train/needle_out/checkpoints

echo "fine-tune done $(date -Is)"
"$VENV/bin/needle" build train/needle_out/checkpoints/needle3.safetensors \
  --lora train/needle_out/adapter.safetensors --out train/needle_out/tuned.cact
echo "build done $(date -Is)"
