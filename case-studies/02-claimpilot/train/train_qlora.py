"""QLoRA fine-tune of a small Qwen model for ClaimPilot-style extraction.

Same prompt format as inference (eval/extract_local.py): the shared system
prompt (data/extract_system.md) + the document, target = canonical JSON.

Designed for a 6 GB GPU: 4-bit NF4 base + LoRA (r=16), micro-batch 1,
grad-accum 16, gradient checkpointing, fp16 (no bf16 on Ampere-laptop).

Usage:
  python train/train_qlora.py --model Qwen/Qwen3-4B-Instruct-2507 \
      --out train/artifacts/qwen3-4b-lora [--epochs 2] [--max-len 1536] [--limit N] [--max-steps N]

Logs: stdout (redirect to train/logs/...). Config is written to <out>/train_config.json.
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

sys.stdout.reconfigure(encoding="utf-8")

CASE = Path(__file__).resolve().parent.parent
TRAIN_JSONL = CASE / "data" / "splits" / "train.jsonl"
VAL_JSONL = CASE / "data" / "splits" / "val.jsonl"
SYSTEM_PROMPT = (CASE / "data" / "extract_system.md").read_text(encoding="utf-8")


def load_model_kwargs() -> dict:
    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        return {"dtype": torch.float16} if v >= (4, 56) else {"torch_dtype": torch.float16}
    except Exception:  # noqa: BLE001
        return {"dtype": torch.float16}


def build_dataset(tokenizer, docs, max_len: int) -> Dataset:
    rows = []
    lengths = []
    for doc in docs:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": doc["text"]}],
            tokenize=False, add_generation_prompt=True,
        )
        answer = json.dumps(doc["target"], ensure_ascii=False, separators=(",", ":")) + tokenizer.eos_token
        pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        aids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if len(aids) >= max_len:
            aids = aids[:max_len]
        if len(pids) + len(aids) > max_len:
            pids = pids[: max_len - len(aids)]
        ids = pids + aids
        lengths.append(len(ids))
        rows.append({
            "input_ids": ids,
            "labels": [-100] * len(pids) + aids,
            "attention_mask": [1] * len(ids),
        })
    lengths.sort()
    print(f"dataset: {len(rows)} rows, token lens p50={lengths[len(lengths)//2]} "
          f"p95={lengths[int(len(lengths)*0.95)]} max={lengths[-1]}", flush=True)
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default=str(CASE / "train" / "artifacts" / "qwen3-4b-lora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0, help="cap training docs (smoke runs)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-val", action="store_true", help="run loss on the val set each epoch")
    args = ap.parse_args()

    torch.manual_seed(42)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    docs = [json.loads(l) for l in TRAIN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        docs = docs[: args.limit]
    ds = build_dataset(tokenizer, docs, args.max_len)

    eval_ds = None
    if args.eval_val:
        val_docs = [json.loads(l) for l in VAL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_ds = build_dataset(tokenizer, val_docs, args.max_len)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("loading base model (4-bit NF4)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0}, **load_model_kwargs()
    )
    # prepare_model_for_kbit_training upcasts the unquantized embeddings + LM head to fp32
    # (~3 GB on this 151K-vocab model) and OOMs the 6 GB card; only the checkpointing hooks are needed.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    def collator(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append(b["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=1,
        # TailLossTrainer assumes one unpadded row; the default eval batch (8, right-padded)
        # scored pad tokens and reported eval_loss ~9 on a model that is exact on val.
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        report_to=[],
        seed=42,
        data_seed=42,
        dataloader_num_workers=0,
        save_total_limit=3,
    )

    class TailLossTrainer(Trainer):
        """Loss only over the answer tail. Full-vocab logits for a ~1.8K-token prompt
        (151K vocab, fp32 upcast) OOM'd the 6 GB card. With micro-batch 1 and no padding
        the answer is the last k tokens, so only those logit positions are computed."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            ids = inputs["input_ids"]
            k = int((labels[0] != -100).sum().item())
            out = model(input_ids=ids, attention_mask=inputs["attention_mask"], logits_to_keep=k + 1)
            logits = out.logits[0, :-1, :].float()
            loss = torch.nn.functional.cross_entropy(logits, ids[0, -k:])
            return (loss, out) if return_outputs else loss

    trainer = TailLossTrainer(model=model, args=targs, train_dataset=ds, eval_dataset=eval_ds, data_collator=collator)

    train_config = {
        "model": args.model,
        "train_docs": len(docs),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": lora.target_modules,
        "quant": "4-bit nf4, double quant, fp16 compute",
        "batch": "micro 1 x grad_accum %d" % args.grad_accum,
        "seed": 42,
        "limit": args.limit,
        "max_steps": args.max_steps,
    }
    (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str), encoding="utf-8")

    t0 = time.time()
    trainer.train()
    print(f"training wall time: {(time.time() - t0) / 60:.1f} min", flush=True)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("saved adapter to", out_dir, flush=True)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
