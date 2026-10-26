#!/usr/bin/env bash
# queue 2: case 2 Needle (batch 1), then case 1 router(k=5)+Qwen3.5-0.8B
W=/c/Users/mgsuk/.t3/worktrees/LeanInference
C1=$W/t3-ce2b84b8/case-studies/01-routely-tool-calling
Q=/c/codebases/LeanInference/gpu_queue.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$Q"; }
log "START c2-needle (batch 1)"
bash /c/codebases/LeanInference/needle_c2.sh | tail -c 400 | tee -a "$Q"
log "START c1-qwen-k5"
(cd "$C1" && .venv/Scripts/python.exe train/qwen/train_peft.py --train train-chat-k5.jsonl --tag peft-k5 --epochs 2) \
  > "$C1/train/logs/qwen35-0.8b-peft-k5.log" 2>&1 && log "OK c1-qwen-k5" || log "FAIL c1-qwen-k5 (see log)"
log "QUEUE2 DONE"
