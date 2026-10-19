# Creates .venv and installs the local training stack (CUDA torch + unsloth + trl).
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File train/qwen/setup_env.ps1
$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
Write-Host "project root: $root"

uv venv .venv
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "installing torch + torchvision (CUDA 12.8 wheels) ..."
uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "installing unsloth + training stack ..."
uv pip install --python $py unsloth trl peft accelerate bitsandbytes datasets

Write-Host "verifying ..."
& $py -c "import torch; import unsloth; print('torch', torch.__version__); print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"

Write-Host "done. Next: prepare_data.py then train_qlora.py"
