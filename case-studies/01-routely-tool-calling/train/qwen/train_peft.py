"""Plain transformers+PEFT QLoRA for Routely (no Unsloth).

The Unsloth run 20261019-180139 died with "No or negligible GPU memory available for fused
cross entropy" on the 6 GB card: each row carries the ~4.4k-token tool prompt. This trainer
(same recipe as case study 2's) computes logits only for the short assistant JSON tail.

Usage (from the case-study dir):
  .venv\\Scripts\\python.exe train/qwen/train_peft.py [--epochs 1] [--max-steps N]
Adapter -> train/qwen/out/<model>-peft/final ; appends a row to train/experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_qlora import ROOT, log_experiment  # noqa: E402


def build(tok, path: Path) -> Dataset:
    rows, lens = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        msgs = json.loads(line)["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        aids = tok(msgs[-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = pids + aids
        lens.append(len(ids))
        rows.append({"input_ids": ids, "labels": [-100] * len(pids) + aids})
    lens.sort()
    print(f"{path.name}: {len(rows)} rows, p50={lens[len(lens) // 2]} max={lens[-1]}", flush=True)
    return Dataset.from_list(rows)


class TailLossTrainer(Trainer):
    """micro-batch 1, no padding: the answer is the last k tokens, so only those logits."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        ids = inputs["input_ids"]
        k = int((labels[0] != -100).sum().item())
        out = model(input_ids=ids, logits_to_keep=k + 1)
        loss = torch.nn.functional.cross_entropy(out.logits[0, :-1, :].float(), ids[0, -k:])
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--epochs", type=float, default=1.0)  # ponytail: 1 epoch ~ 6-7 h on the 3050; 2 if time allows
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train", default="train-chat.jsonl", help="file in train/qwen/data/")
    ap.add_argument("--tag", default="peft")
    ap.add_argument("--batch", type=int, default=1,
                    help=">1: padded micro-batches with the standard answer-only loss (short no-schema rows)")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + args.model.split("/")[-1].lower() + "-" + args.tag
    out_dir = ROOT / "train" / "qwen" / "out" / (args.model.replace("/", "__") + "-" + args.tag)
    row = {"run_id": run_id, "date": time.strftime("%Y-%m-%d %H:%M"), "stage": "qlora-peft",
           "base_model": args.model, "epochs": args.epochs, "seq_len": "full", "lr": args.lr,
           "lora_r": args.lora_r, "batch": args.batch, "grad_accum": args.grad_accum, "status": "RUNNING"}
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        tok = getattr(tok, "tokenizer", tok)  # Qwen3.5 may load as a multimodal processor
        ds = build(tok, ROOT / "train" / "qwen" / "data" / args.train)

        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": 0}, dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=True))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()
        model.config.use_cache = False

        targs = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
            max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_steps=10, logging_steps=5, save_strategy="steps", save_steps=40,
            save_total_limit=2, fp16=True, max_grad_norm=1.0, report_to=[], seed=42,
            dataloader_num_workers=0)
        if args.batch == 1:
            collate = lambda b: {k: torch.tensor([b[0][k]]) for k in ("input_ids", "labels")}  # noqa: E731
            trainer = TailLossTrainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
        else:
            def collate(b):
                n, pad = max(len(x["input_ids"]) for x in b), tok.pad_token_id
                return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in b]),
                        "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in b]),
                        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"]))
                                                        for x in b])}
            trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
        # resume from the last saved checkpoint (every 40 steps) if a previous run was killed
        resume = any((out_dir / "checkpoints").glob("checkpoint-*"))
        stats = trainer.train(resume_from_checkpoint=resume or None)
        trainer.save_model(str(out_dir / "final"))
        tok.save_pretrained(str(out_dir / "final"))
        minutes = round((time.time() - t0) / 60, 1)
        row.update(status="DONE", minutes=minutes,
                   notes=f"train_loss={stats.metrics.get('train_loss'):.4f} "
                         f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
        log_experiment(row)
        print(f"[{run_id}] DONE in {minutes} min -> {out_dir / 'final'}", flush=True)
    except Exception as e:
        row.update(status="FAILED", minutes=round((time.time() - t0) / 60, 1),
                   notes=f"{type(e).__name__}: {e}"[:200])
        log_experiment(row)
        raise


if __name__ == "__main__":
    main()
