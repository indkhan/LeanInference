"""`needle build` with the full KV budget instead of the base model's 256-token window.

The base checkpoint carries kv_window=256 (its on-device post-training setting) and
`needle build` writes min(budget, 256) into the .cact. Our LoRA fine-tune never ran with that
window (JAX training attends over the whole <=2,304-token row), so the default archive serves
long claim documents with numerics the adapter never saw: on val the output collapsed after a
few tokens on the long docs and was near-exact on the short one. This build writes the engine's
KV budget (3,456 tokens for this geometry), which covers every training row.

Usage: python train/needle_build_fullkv.py   ->  train/needle_out/tuned_fullkv.cact
"""
from argparse import Namespace
from pathlib import Path

import needle.model.architecture as arch
from needle.model.finetune import build_main

CASE = Path(__file__).resolve().parent.parent
OUT = CASE / "train" / "needle_out"

arch.effective_kv_window = arch.kv_budget_window  # build_main imports it at call time
build_main(Namespace(checkpoint=str(OUT / "checkpoints" / "needle3.safetensors"),
                     lora=str(OUT / "adapter.safetensors"), out=str(OUT / "tuned_fullkv.cact"),
                     platform=None, layers=None, upload=False))
