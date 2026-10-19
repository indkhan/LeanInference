"""Environment check for the local QLoRA training stack.

Run with:  .venv\\Scripts\\python.exe train/qwen/check_env.py
Prints: python/torch/unsloth/trl versions, CUDA + bf16 availability, the SFTConfig
fields and SFTTrainer params this TRL version actually accepts, and whether the
planned Hugging Face base-model repos exist.
"""
from __future__ import annotations

import inspect
import sys


def main():
    print("python", sys.version.split()[0])
    try:
        import torch
        print("torch", torch.__version__,
              "| cuda", torch.cuda.is_available(),
              "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
              "| bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a")
    except Exception as e:  # noqa: BLE001
        print("torch import FAILED:", e)
        return 1

    try:
        import unsloth
        print("unsloth", getattr(unsloth, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print("unsloth import FAILED:", e)
        return 1

    try:
        import trl
        from trl import SFTConfig, SFTTrainer
        interesting = ["max_length", "max_seq_length", "dataset_text_field", "eval_strategy",
                       "save_strategy", "bf16", "fp16", "optim", "report_to", "logging_steps",
                       "warmup_ratio", "weight_decay", "num_train_epochs", "max_steps"]
        params = inspect.signature(SFTConfig.__init__).parameters
        print("trl", trl.__version__, "| SFTConfig has:", [k for k in interesting if k in params])
        tparams = inspect.signature(SFTTrainer.__init__).parameters
        print("SFTTrainer: processing_class", "processing_class" in tparams,
              "| tokenizer", "tokenizer" in tparams)
    except Exception as e:  # noqa: BLE001
        print("trl import FAILED:", e)
        return 1

    try:
        from huggingface_hub import model_info
    except Exception as e:  # noqa: BLE001
        print("huggingface_hub FAILED:", e)
        return 1
    for mid in ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "unsloth/Qwen3.5-0.8B"]:
        try:
            model_info(mid)
            print("HF OK  ", mid)
        except Exception as e:  # noqa: BLE001
            print("HF FAIL", mid, "|", type(e).__name__, str(e)[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
