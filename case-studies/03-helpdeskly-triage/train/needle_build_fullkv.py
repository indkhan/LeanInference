"""`needle build` with the full KV budget instead of the base model's 256-token window.

The base checkpoint carries kv_window=256 and `needle build` writes min(budget, 256) into the
.cact, but our LoRA trains in JAX over the whole row (record schema + ticket, ~300-400 tokens), so
the default archive can serve prompts with numerics the adapter never saw. Same fix as case study 2
(02-claimpilot/train/needle_build_fullkv.py, where the default build collapsed to 0.10 F1 on val).
The two archives are compared on val before one runs on test.

Usage: python train/needle_build_fullkv.py   ->  train/needle_out/tuned_fullkv.cact
"""
from argparse import Namespace
from pathlib import Path

import needle.model.architecture as arch
from needle.model.finetune import build_main

OUT = Path(__file__).resolve().parent / "needle_out"
BASE = (Path(__file__).resolve().parents[3] / "t3-4e600564" / "case-studies" / "02-claimpilot"
        / "train" / "needle_out" / "checkpoints" / "needle3.safetensors")

arch.effective_kv_window = arch.kv_budget_window  # build_main imports it at call time
build_main(Namespace(checkpoint=str(BASE), lora=str(OUT / "adapter.safetensors"),
                     out=str(OUT / "tuned_fullkv.cact"), platform=None, layers=None, upload=False))
