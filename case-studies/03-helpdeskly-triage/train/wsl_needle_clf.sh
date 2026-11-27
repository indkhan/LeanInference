#!/usr/bin/env bash
# Candidate D: Needle 3 LoRA classifier, trained under WSL2 (jax has no CUDA build for
# native Windows), then merged + exported to .cact. Reuses case 2's WSL venv
# (/root/.venvs/claimpilot, `cactus-needle[train]`). Data/outputs stay on /mnt/c.
# --val-split 0: the CLI's built-in held-out scoring generates one example at a time (with a
# 96-token cap) and only writes the adapter after it - over an hour in case 2. Our own
# val/test eval (eval/predict_d_needle.py) is the measurement.
#   wsl -d Ubuntu-24.04 -u root -- bash train/wsl_needle_clf.sh
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
CASE=/mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4cc4e17c/case-studies/03-helpdeskly-triage
NEEDLE=/root/.venvs/claimpilot/bin/needle
BASE=/mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot/train/needle_out/checkpoints/needle3.safetensors
cd "$CASE"
mkdir -p train/needle_out
echo "start $(date -Is)"
"$NEEDLE" finetune data/needle_train.jsonl --checkpoint "$BASE" \
  --epochs 2 --batch-size 8 --max-len 1024 --val-split 0 --seed 0 \
  --out train/needle_out/adapter.safetensors --checkpoint-dir train/needle_out
echo "fine-tune done $(date -Is)"
"$NEEDLE" build "$BASE" --lora train/needle_out/adapter.safetensors --out train/needle_out/tuned.cact
echo "build done $(date -Is)"
