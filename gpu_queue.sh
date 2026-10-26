#!/usr/bin/env bash
# Sequential GPU queue for the three case studies (one 6GB card, no lock juggling).
# Each job logs to its case's train/logs; a failure is recorded and the queue moves on.
W=/c/Users/mgsuk/.t3/worktrees/LeanInference
C1=$W/t3-ce2b84b8/case-studies/01-routely-tool-calling
C2=$W/t3-4e600564/case-studies/02-claimpilot
Q=/c/codebases/LeanInference/gpu_queue.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$Q"; }
run() { # name dir logfile cmd...
  local name=$1 dir=$2 lf=$3; shift 3
  log "START $name"
  (cd "$dir" && "$@") >"$lf" 2>&1 && log "OK $name" || log "FAIL $name (exit $?, see $lf)"
}

# 1-2. case 1 smoke failed (Unsloth fused-CE OOM); case 2 Qwen QLoRA already running as PID 10300
while tasklist //FI "PID eq 10300" | grep -q 10300; do sleep 60; done
log "c2-qwen process ended (see $C2/train/logs/qwen35-0.8b-full.log)"

# 3. case 1 Qwen3.5-0.8B QLoRA via plain PEFT, tail-only loss (no Unsloth)
mkdir -p "$C1/train/logs"
run c1-qwen "$C1" "$C1/train/logs/qwen35-0.8b-peft.log"   .venv/Scripts/python.exe train/qwen/train_peft.py

# 4. case 2 Needle LoRA in WSL (needs CUDA jaxlib there; installed once)
run c2-needle "$C2" "$C2/train/logs/needle-finetune.log" \
  wsl -d Ubuntu-24.04 -u root -- bash -c \
  '/root/.venvs/claimpilot/bin/pip install -q -U "jax[cuda12]" && bash /mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot/train/wsl_finetune.sh'

log "QUEUE DONE"
