"""Candidate C - Qwen3.5-0.8B + LoRA as a label classifier (short generation).

Prompt = label lists + ticket; target = "intent|urgency|sentiment" (one short line).
  before  base model, same prompt, zero training
  after   LoRA (r=16) on the 4,000 train tickets, loss on the answer tokens only
Both are scored on the 700-ticket test set; val is only used for the epoch-end loss.
Unparseable labels fall back to other/medium/neutral and are counted (never dropped).

Usage:
  python train/qwen_lora_clf.py --stage before
  python train/qwen_lora_clf.py --stage train      # writes train/artifacts/qwen35-0.8b-clf-lora
  python train/qwen_lora_clf.py --stage after
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_lib import ARTIFACTS, CASE, INTENTS, SENTIMENT, URGENCY, load, log_experiment, metrics  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
OUT = CASE / "train" / "artifacts" / "qwen35-0.8b-clf-lora"
SYSTEM = ("Label the customer-support ticket. Reply with one line: intent|urgency|sentiment\n"
          f"intent: {', '.join(INTENTS)}\nurgency: {', '.join(URGENCY)}\nsentiment: {', '.join(SENTIMENT)}")
DEFAULT = {"intent": "other", "urgency": "medium", "sentiment": "neutral"}


def prompt(tok, text):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def answer(r):
    return f"{r['intent']}|{r['urgency']}|{r['sentiment']}"


def parse(s):
    parts = [p.strip().lower() for p in s.strip().splitlines()[0].split("|")] if s.strip() else []
    if len(parts) != 3 or parts[0] not in INTENTS or parts[1] not in URGENCY or parts[2] not in SENTIMENT:
        return None
    return dict(zip(("intent", "urgency", "sentiment"), parts))


def load_model(adapter=None):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter))
    return m.eval()


def train(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    tok = AutoTokenizer.from_pretrained(MODEL)

    def rows(split):
        out = []
        for r in load(split):
            p = tok(prompt(tok, r["text"]), add_special_tokens=False)["input_ids"][-(args.max_len - 24):]
            a = tok(answer(r) + tok.eos_token, add_special_tokens=False)["input_ids"]
            out.append({"input_ids": p + a, "labels": [-100] * len(p) + a})
        return out

    tr, va = rows("train"), rows("val")
    model = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():  # trainable LoRA weights in fp32 (fp16 AMP needs fp32 master weights)
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    def collate(b):
        n = max(len(x["input_ids"]) for x in b)
        pad = tok.pad_token_id
        return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in b]),
                "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in b]),
                "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in b])}

    # ~170-token rows: batch 16 in one step keeps the GPU busy (same effective batch as 4 x 4)
    targs = TrainingArguments(output_dir=str(OUT / "checkpoints"), per_device_train_batch_size=16,
                              per_device_eval_batch_size=16, gradient_accumulation_steps=1,
                              num_train_epochs=args.epochs, learning_rate=2e-4, lr_scheduler_type="cosine",
                              warmup_steps=20, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
                              save_total_limit=2, fp16=True, report_to=[], seed=42, dataloader_num_workers=0,
                              remove_unused_columns=False)
    t0 = time.time()
    Trainer(model=model, args=targs, train_dataset=tr, eval_dataset=va, data_collator=collate).train()
    minutes = (time.time() - t0) / 60
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    cfg = {"model": MODEL, "epochs": args.epochs, "lr": 2e-4, "lora_r": 16, "batch": "16 x accum 1",
           "train_rows": len(tr), "train_minutes": round(minutes, 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (OUT / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log_experiment("c_qwen_lora_train", config=cfg)


@torch.no_grad()
def evaluate(stage):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = load_model(OUT if stage == "after" else None)
    test = load("test")
    preds, raws, fails = [], [], 0
    t0 = time.time()
    for i in range(0, len(test), 16):
        batch = [prompt(tok, r["text"]) for r in test[i:i + 16]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        for seq in out[:, enc["input_ids"].shape[1]:]:
            s = tok.decode(seq, skip_special_tokens=True)
            p = parse(s)
            fails += p is None
            preds.append(p or dict(DEFAULT))
            raws.append(s)
    run_s = time.time() - t0
    lats = []  # batch-1 single-request latency, as served
    for r in test[:100]:
        enc = tok(prompt(tok, r["text"]), return_tensors="pt", add_special_tokens=False).to(0)
        t = time.perf_counter()
        model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        lats.append((time.perf_counter() - t) * 1000)
    lats.sort()
    lat = {"p50_ms": round(lats[50], 1), "p95_ms": round(lats[95], 1), "n": 100, "device": "RTX 3050 Laptop fp16"}
    golds = [{k: r[k] for k in ("intent", "urgency", "sentiment")} for r in test]
    res = dict(metrics(preds, golds))
    tag = f"c_qwen_{stage}"
    log_experiment(tag, intent=res["intent"], urgency=res["urgency"], sentiment=res["sentiment"],
                   unparseable=fails, latency=lat, runtime_s=round(run_s, 1), test_n=len(test),
                   config={"model": MODEL, "adapter": str(OUT.relative_to(CASE)) if stage == "after" else None})
    (ARTIFACTS / f"preds_{tag}.json").write_text(json.dumps(
        {"name": tag, "predictions": preds, "gold": golds, "raw": raws, "ids": [r["id"] for r in test],
         "metrics": res, "latency": lat, "unparseable": fails}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(tag, res, "unparseable:", fails, lat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["before", "train", "after"], required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()
    train(a) if a.stage == "train" else evaluate(a.stage)
