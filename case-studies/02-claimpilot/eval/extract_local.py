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
from extract import extract_json_object, slugify, ENCODER, LM_SCHEMA  # noqa: E402


def load_dtype_kwarg():
    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        return {"dtype": torch.float16} if v >= (4, 56) else {"torch_dtype": torch.float16}
    except Exception:  # noqa: BLE001
        return {"dtype": torch.float16}




def build_grammar_prefix_fn(tokenizer, schema):
    """lm-format-enforcer prefix function, built without its transformers
    integration - that module imports transformers.tokenization_utils, which
    transformers 5.x removed. Logic mirrors the integration."""
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData

    vocab_size = len(tokenizer)
    token_0 = tokenizer.encode("0")[-1]
    special_ids = set(tokenizer.all_special_ids)
    regular_tokens = []
    for token_idx in range(vocab_size):
        if token_idx in special_ids:
            continue
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))

    def decode_fn(tokens):
        return tokenizer.decode(tokens).rstrip("\ufffd")

    data = TokenEnforcerTokenizerData(regular_tokens, decode_fn, tokenizer.eos_token_id,
                                      False, vocab_size)
    def new_prefix_fn():
        # A fresh enforcer per document: TokenEnforcer caches every prefix state it has
        # seen, and one shared instance grew until the process died ("bad allocation")
        # every ~70-90 docs. The expensive vocab scan above stays shared.
        enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
        return lambda batch_id, sent: enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

    return new_prefix_fn


def load_model(model_id: str, adapter: str | None, fp16: bool = False):
    """Load the 4-bit model. The checkpoint used here is already quantized
    (unsloth bnb-4bit), so only pass BitsAndBytesConfig when it is not."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id)
    qcfg = getattr(cfg, "quantization_config", None)
    if fp16:  # unquantized weights: no 4-bit dequant per step (fits easily for 0.8B)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, **load_dtype_kwarg())
    elif qcfg is not None:
        print(f"pre-quantized checkpoint: {qcfg.get('bnb_4bit_quant_type', '?')} "
              f"compute={qcfg.get('bnb_4bit_compute_dtype', '?')}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0},
                                                    **load_dtype_kwarg())
    else:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                     device_map={"": 0}, **load_dtype_kwarg())
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ids-file", help="run only these doc ids, one per line")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new", type=int, default=448)
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="docs per generate call")
    ap.add_argument("--fp16", action="store_true", help="load fp16 weights instead of 4-bit NF4")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading {args.model} (4-bit)... adapter={args.adapter}", flush=True)
    model = load_model(args.model, args.adapter, args.fp16)

    prefix_fn = None
    if not args.no_grammar:
        try:
            prefix_fn = build_grammar_prefix_fn(tokenizer, LM_SCHEMA)
            print("constrained decoding: lm-format-enforcer ON", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"WARNING lm-format-enforcer unavailable ({ex}); running unconstrained", flush=True)

    docs = [json.loads(l) for l in (SPLITS / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids_file:
        keep = set((CASE / args.ids_file).read_text(encoding="utf-8").split())
        docs = [d for d in docs if d["id"] in keep]
    docs = docs[args.offset:]
    if args.limit:
        docs = docs[: args.limit]

    label = f"local__{args.model.split('/')[-1]}" + (f"__{Path(args.adapter).name}" if args.adapter else "")
    cache_dir = CACHE / slugify(label) / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    records = []
    t_start = time.time()
    todo = []
    for i, doc in enumerate(docs, 1):
        cache_file = cache_dir / f"{doc['id']}.json"
        if cache_file.exists() and not args.no_cache:
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            rec["cached"] = True
            records.append(rec)
            print(f"[{i}/{len(docs)}] {doc['id']} cached", flush=True)
        else:
            todo.append(doc)

    # Batched generation keeps the GPU busy (batch 1 is CPU-bound: per-token grammar checks and
    # Qwen3.5's PyTorch fallback kernels). Left padding; one grammar enforcer per row.
    # latency_s = batch wall time / rows (throughput); measure serving latency with --batch 1.
    tokenizer.padding_side = "left"
    for b in range(0, len(todo), args.batch):
        chunk = todo[b:b + args.batch]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": d["text"]}],
            tokenize=False, add_generation_prompt=True) for d in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        fns = [prefix_fn() for _ in chunk] if prefix_fn else None
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=(lambda bid, sent: fns[bid](bid, sent)) if fns else None,
            )
        latency = (time.perf_counter() - t0) / len(chunk)
        for j, doc in enumerate(chunk):
            out_ids = gen[j][inputs["input_ids"].shape[1]:]
            out_ids = out_ids[out_ids != tokenizer.pad_token_id]
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
                "attempts": 1, "batch": len(chunk),
                "latency_s": round(latency, 3),
                "raw": raw, "parsed": parsed,
                "tokens_in_est": int(inputs["attention_mask"][j].sum()),
                "tokens_out_est": int(out_ids.shape[0]),
                "ts": int(time.time()), "cached": False,
            }
            (cache_dir / f"{doc['id']}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            records.append(rec)
            print(f"[{len(records)}/{len(docs)}] {doc['id']} {'ok ' if parsed is not None else 'ERR'} "
                  f"{latency:5.1f}s/doc (batch {len(chunk)}) in={rec['tokens_in_est']} out={rec['tokens_out_est']}"
                  + (f"  {err[:70]}" if err else ""), flush=True)
    records.sort(key=lambda r: r["doc_id"])

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
