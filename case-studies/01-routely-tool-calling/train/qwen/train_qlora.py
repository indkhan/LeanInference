"""Unsloth QLoRA fine-tune for the Routely tool-calling benchmark (local, RTX 3050 6 GB).

Usage (from the case-study dir, after train/qwen/setup_env.ps1 + prepare_data.py):
  .venv\\Scripts\\python.exe train/qwen/train_qlora.py --model Qwen/Qwen3.5-0.8B
  .venv\\Scripts\\python.exe train/qwen/train_qlora.py --model Qwen/Qwen3.5-2B --max-seq-len 6144
  .venv\\Scripts\\python.exe train/qwen/train_qlora.py --model Qwen/Qwen3.5-4B --max-seq-len 4096 --lora-r 8

Notes:
- The fine-tune is non-thinking (chat template enable_thinking=False), same as eval.
- Before training, the script tokenizes the whole set and ABORTS if any example exceeds
  --max-seq-len (a silently truncated answer would poison training; --allow-truncation
  overrides).
- Every run appends a row to train/experiments.csv - the "journey" is part of the story.
- LoRA adapters land in train/qwen/out/<model>/final (serve with vLLM --enable-lora).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "train" / "experiments.csv"
HEADER = ["run_id", "date", "stage", "base_model", "epochs", "seq_len", "lr", "lora_r",
          "batch", "grad_accum", "status", "minutes", "notes"]


def log_experiment(row: dict):
    new_file = not EXPERIMENTS.exists()
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HEADER})


def make_sft_config(**kw):
    """TRL renamed max_seq_length -> max_length across versions; support both."""
    import inspect
    from trl import SFTConfig
    params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in params and "max_seq_length" in kw:
        kw["max_length"] = kw.pop("max_seq_length")
    return SFTConfig(**kw)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--train", default=str(ROOT / "train" / "qwen" / "data" / "train-chat.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "train" / "qwen" / "data" / "val-chat.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seq-len", type=int, default=6144)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke runs (overrides epochs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-steps", type=int, default=150)
    ap.add_argument("--save-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-truncation", action="store_true")
    ap.add_argument("--merge", action="store_true", help="also save a merged 16-bit model")
    args = ap.parse_args(argv)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + args.model.split("/")[-1].lower()
    out_dir = Path(args.out) if args.out else ROOT / "train" / "qwen" / "out" / args.model.replace("/", "__")
    row = {"run_id": run_id, "date": time.strftime("%Y-%m-%d %H:%M"),
           "stage": "qlora", "base_model": args.model, "epochs": args.epochs,
           "seq_len": args.max_seq_len, "lr": args.lr, "lora_r": args.lora_r,
           "batch": args.batch, "grad_accum": args.grad_accum, "status": "RUNNING"}

    try:
        import torch
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from trl import SFTTrainer
    except ImportError as e:
        raise SystemExit(f"training stack missing - run train/qwen/setup_env.ps1 first ({e})")

    print(f"[{run_id}] loading {args.model} (4-bit) ...")
    t0 = time.time()
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_seq_len, dtype=None,
            load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=args.seed)
        # Qwen3.5 checkpoints load as a multimodal Processor (Qwen3VLProcessor). The text-only
        # task never has images, and SFTTrainer's multimodal path raised
        # "Unsupported image file" (run 20261017-164132). Train with the inner text tokenizer.
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        print(f"text-only tokenizer: {type(tokenizer).__name__}")
        tokenizer.padding_side = "right"

        def render(example):
            try:
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
            except TypeError:  # template without enable_thinking support
                text = tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False)
            return {"text": text}

        ds_train = load_dataset("json", data_files=args.train, split="train").map(render)
        ds_val = load_dataset("json", data_files=args.val, split="train").map(render)
        print(f"train={len(ds_train)}  val={len(ds_val)}")

        # length guard: no silent truncation
        max_len = 0
        # Qwen3.5 ships a multimodal Processor; use the inner text tokenizer for lengths
        tok_raw = getattr(tokenizer, "tokenizer", tokenizer)
        for ds in (ds_train, ds_val):
            lengths = []
            for start in range(0, len(ds["text"]), 256):
                lengths.extend(len(x) for x in
                               tok_raw(ds["text"][start:start + 256])["input_ids"])
            lengths.sort()
            p = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]  # noqa: E731
            print(f"  lengths: p50={p(0.5)} p95={p(0.95)} max={lengths[-1]}")
            max_len = max(max_len, lengths[-1])
        if max_len > args.max_seq_len and not args.allow_truncation:
            raise SystemExit(
                f"ABORT: longest example is {max_len} tokens > --max-seq-len {args.max_seq_len}. "
                "Raise --max-seq-len (needs more VRAM) or pass --allow-truncation (not recommended).")

        cfg = make_sft_config(
            output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
            max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_ratio=0.05, weight_decay=0.01, logging_steps=10,
            eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
            seed=args.seed, max_seq_length=args.max_seq_len, dataset_text_field="text",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit", report_to="none")

        import inspect
        t_sig = inspect.signature(SFTTrainer.__init__).parameters
        ctor = {"model": model, "train_dataset": ds_train, "eval_dataset": ds_val, "args": cfg}
        # text-only tokenizer: passing a multimodal Processor made TRL try to decode images (run 1)
        ctor["processing_class" if "processing_class" in t_sig else "tokenizer"] = tok_raw
        trainer = SFTTrainer(**ctor)
        # loss on the assistant JSON only - the 5k-token system prompt is identical every row
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                          response_part="<|im_start|>assistant\n")
        print(f"[{run_id}] training ... (this is the long part)")
        stats = trainer.train()
        minutes = round((time.time() - t0) / 60, 1)

        final_dir = out_dir / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        (out_dir / "trainer_log.json").write_text(
            json.dumps(stats.metrics, indent=2) + "\n", encoding="utf-8")
        if args.merge:
            model.save_pretrained_merged(str(out_dir / "merged"), tokenizer,
                                         save_method="merged_16bit")

        row.update(status="DONE", minutes=minutes,
                   notes=f"train_loss={stats.metrics.get('train_loss'):.4f} "
                         f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
        log_experiment(row)
        print(f"[{run_id}] DONE in {minutes} min -> {final_dir}")
        print(f"  metrics: {json.dumps(stats.metrics)}")
        return 0
    except SystemExit:
        raise
    except Exception as e:  # OOM etc. - log and surface
        minutes = round((time.time() - t0) / 60, 1)
        row.update(status="FAILED", minutes=minutes, notes=f"{type(e).__name__}: {e}"[:200])
        log_experiment(row)
        print(f"[{run_id}] FAILED after {minutes} min: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
