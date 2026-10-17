"""Local extraction runs: Qwen in 4-bit on the RTX 3050, with optional LoRA
adapter, constrained to the ClaimPilot JSON schema via lm-format-enforcer.

Same prompt format as training (data/extract_system.md as system prompt, the
document as user turn). Writes results in the same shape as eval/extract.py so
eval/metrics.py scores both transports identically.

Usage:
  python eval/extract_local.py --tag before-ft [--adapter train/artifacts/...] \
      [--split test] [--limit N] [--max-new 400]
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

sys.stdout.reconfigure(encoding="utf-8")

CASE = Path(__file__).resolve().parent.parent
SPLITS = CASE / "data" / "splits"
RESULTS = CASE / "eval" / "results"
CACHE = CASE / "eval" / "cache"
SYSTEM_PROMPT = (CASE / "data" / "extract_system.md").read_text(encoding="utf-8")

sys.path.insert(0, str(CASE / "eval"))
from extract import extract_json_object, slugify, ENCODER  # noqa: E402


def load_dtype_kwarg():
    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        return {"dtype": torch.float16} if v >= (4, 56) else {"torch_dtype": torch.float16}
    except Exception:  # noqa: BLE001
        return {"dtype": torch.float16}


LM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claim_id", "policy_number", "claimant_name", "date_of_incident", "date_of_service",
                 "provider_name", "line_items", "total_amount", "currency", "diagnosis_or_damage_code",
                 "vehicle_vin", "claim_type", "payout_iban", "notes"],
    "properties": {
        "claim_id": {"type": "string"},
        "policy_number": {"type": ["string", "null"]},
        "claimant_name": {"type": ["string", "null"]},
        "date_of_incident": {"type": ["string", "null"]},
        "date_of_service": {"type": ["string", "null"]},
        "provider_name": {"type": ["string", "null"]},
        "line_items": {
            "anyOf": [
                {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["description", "amount"],
                    "properties": {"description": {"type": "string"}, "amount": {"type": "string"}}}},
                {"type": "null"},
            ]
        },
        "total_amount": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"], "enum": ["USD", "EUR", "GBP", "CHF", None]},
        "diagnosis_or_damage_code": {"type": ["string", "null"]},
        "vehicle_vin": {"type": ["string", "null"]},
        "claim_type": {"type": "string", "enum": ["auto", "medical", "property", "travel", "other"]},
        "payout_iban": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new", type=int, default=448)
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"loading {args.model} (4-bit)... adapter={args.adapter}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map={"": 0}, **load_dtype_kwarg())
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prefix_fn = None
    if not args.no_grammar:
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            parser = JsonSchemaParser(LM_SCHEMA)
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
            print("constrained decoding: lm-format-enforcer ON", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"WARNING lm-format-enforcer unavailable ({ex}); running unconstrained", flush=True)

    docs = [json.loads(l) for l in (SPLITS / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = docs[args.offset:]
    if args.limit:
        docs = docs[: args.limit]

    label = f"local__{args.model.split('/')[-1]}" + (f"__{Path(args.adapter).name}" if args.adapter else "")
    cache_dir = CACHE / slugify(label) / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    records = []
    t_start = time.time()
    for i, doc in enumerate(docs, 1):
        cache_file = cache_dir / f"{doc['id']}.json"
        if cache_file.exists() and not args.no_cache:
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            rec["cached"] = True
            records.append(rec)
            print(f"[{i}/{len(docs)}] {doc['id']} cached", flush=True)
            continue

        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": doc["text"]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        latency = time.perf_counter() - t0
        out_ids = gen[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(out_ids, skip_special_tokens=True)
        err = None
        parsed = None
        try:
            parsed = extract_json_object(raw)
        except Exception as ex:  # noqa: BLE001
            err = f"parse: {ex}"

        rec = {
            "doc_id": doc["id"], "family": doc["family"],
            "model": label, "tag": args.tag, "transport": "local_transformers",
            "ok": parsed is not None, "error": err,
            "attempts": 1,
            "latency_s": round(latency, 3),
            "raw": raw, "parsed": parsed,
            "tokens_in_est": int(inputs["input_ids"].shape[1]),
            "tokens_out_est": int(out_ids.shape[0]),
            "ts": int(time.time()), "cached": False,
        }
        cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        records.append(rec)
        print(f"[{i}/{len(docs)}] {doc['id']} {'ok ' if parsed is not None else 'ERR'} "
              f"{latency:5.1f}s in={rec['tokens_in_est']} out={rec['tokens_out_est']}"
              + (f"  {err[:70]}" if err else ""), flush=True)

    slug = slugify(label)
    out_file = RESULTS / f"results_{slug}_{args.tag}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    oks = [r for r in records if r["ok"]]
    lats = sorted(r["latency_s"] for r in records)
    summary = {
        "model": label, "tag": args.tag, "transport": "local_transformers",
        "split": args.split, "docs": len(records), "ok": len(oks),
        "ok_rate": round(len(oks) / max(1, len(records)), 4),
        "wall_s": round(time.time() - t_start, 1),
        "latency_p50": lats[len(lats) // 2] if lats else None,
        "latency_p95": lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else None,
        "errors": [r["error"] for r in records if r["error"]][:10],
        "results_file": str(out_file.relative_to(CASE)),
        "grammar": prefix_fn is not None,
        "adapter": args.adapter,
    }
    (RESULTS / f"run_{slug}_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
