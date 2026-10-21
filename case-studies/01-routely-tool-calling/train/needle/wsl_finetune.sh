#!/usr/bin/env bash
# Needle 3 LoRA for Routely, trained under WSL2 (jax has no CUDA build for native Windows), using
# case study 2's WSL venv (/root/.venvs/claimpilot, cactus-needle[train]). Rows carry the compact
# tool list (train/needle/make_data.py): p50 2,074 / max 2,238 Needle tokens -> --max-len 2304.
# --val-split 0: the CLI's built-in held-out score generates with a 96-token cap and only writes the
# adapter afterwards (>1 h in case 2); our own val/test eval is the measurement.
#   wsl -d Ubuntu-24.04 -u root -- bash train/needle/wsl_finetune.sh
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
CASE=/mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-ce2b84b8/case-studies/01-routely-tool-calling
C2=/mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot
VENV=/root/.venvs/claimpilot
BASE=$C2/train/needle_out/checkpoints/needle3.safetensors
cd "$CASE"
mkdir -p train/needle/out
echo "start $(date -Is)"
"$VENV/bin/needle" finetune train/needle/data/train.jsonl --checkpoint "$BASE" \
  --epochs 2 --batch-size 1 --max-len 2304 --val-split 0 --seed 0 \
  --out train/needle/out/adapter.safetensors --checkpoint-dir train/needle/out
echo "fine-tune done $(date -Is)"
# full KV window (see 02-claimpilot/train/needle_build_fullkv.py): the default 256-token window
# collapsed long prompts in case 2
"$VENV/bin/python" - <<PY
from argparse import Namespace
import needle.model.architecture as arch
from needle.model.finetune import build_main
arch.effective_kv_window = arch.kv_budget_window
build_main(Namespace(checkpoint="$BASE", lora="train/needle/out/adapter.safetensors",
                     out="train/needle/out/tuned_fullkv.cact", platform=None, layers=None, upload=False))
PY
echo "build done $(date -Is)"
