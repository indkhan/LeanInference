#!/usr/bin/env bash
# Set up the Linux-side environment used for Needle 3 fine-tuning.
# Windows Smart App Control blocks jaxlib's native lapack DLL, so training
# runs under WSL2 (Ubuntu 24.04) while data and outputs stay on /mnt/c.
set -euo pipefail

CASE=/mnt/c/Users/mgsuk/.t3/worktrees/LeanInference/t3-4e600564/case-studies/02-claimpilot
VENV=/root/.venvs/claimpilot   # Linux filesystem: avoids DrvFs symlink/permission issues

if [ ! -x /root/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

/root/.local/bin/uv venv "$VENV" --python 3.12
/root/.local/bin/uv pip install --python "$VENV/bin/python" 'cactus-needle[train]'

cd "$CASE"
"$VENV/bin/python" - <<'PY'
import jax, flax, optax
print("jax", jax.__version__, jax.devices())
import needle
print("needle", needle.__version__)
PY
echo "WSL setup OK"
