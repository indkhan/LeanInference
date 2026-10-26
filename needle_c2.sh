#!/usr/bin/env bash
W=/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot
wsl -d Ubuntu-24.04 -u root -- bash -c '/root/.venvs/claimpilot/bin/python -c "import jax;print(jax.devices())" && bash /mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot/train/wsl_finetune.sh' > $W/train/logs/needle-finetune.log 2>&1
echo "exit $?"; tail -c 1500 $W/train/logs/needle-finetune.log
