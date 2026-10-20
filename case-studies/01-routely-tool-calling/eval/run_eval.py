"""Run one candidate model over a split and write raw per-example outputs to eval/results/.

The prompt contract (same for every candidate - comparability is the point):
  system message: benchmark instructions + all 40 tool schemas (data/tools.json)
  user message:   "Today: <request_date>\\nRequest: <request>"
  model replies with ONE strict JSON object: call / clarify / none.

Examples:
  # pipeline smoke test (no model, no cost)
  uv run python eval/run_eval.py --adapter mock --model mock-pipeline --seed 42

  # local fine-tuned model served by vLLM
  uv run python eval/run_eval.py --adapter vllm --model qwen3.5-2b-ft --no-thinking

  # API model (needs a key in the env; the $0 rule means we do not run these)
  uv run python eval/run_eval.py --adapter openai --model gpt-6-luna --api-key-env OPENAI_API_KEY

Output: eval/results/<slug>.jsonl - one line per example, resumable (existing ids are
skipped, so a long run can be interrupted and continued).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `adapters` importable

from adapters import get_adapter  # noqa: E402
from prompt import SYSTEM_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data" / "splits" / "test.jsonl"
TOOLS_PATH = ROOT / "data" / "tools.json"
RESULTS_DIR = ROOT / "eval" / "results"

def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def parse_output(raw: str):
    """Raw model text -> (prediction|None, parse_status).

    status in: json_call | json_clarify | json_none | free_text | invalid_json
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text:
        return None, "invalid_json"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"type": "text", "text": raw}, "free_text"
    if isinstance(obj, list):  # tolerate a bare [call, ...]
        obj = {"type": "call", "calls": obj}
    if not isinstance(obj, dict):
        return None, "invalid_json"
    if "calls" not in obj and "name" in obj:  # tolerate a bare call object
        obj = {"type": "call", "calls": [obj]}
    otype = obj.get("type")
    if otype == "call":
        calls = obj.get("calls")
        if not isinstance(calls, list) or not calls:
            return None, "invalid_json"
        clean = []
        for c in calls:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                return None, "invalid_json"
            args = c.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return None, "invalid_json"
            clean.append({"name": c["name"], "arguments": args})
        return {"type": "call", "calls": clean}, "json_call"
    if otype == "clarify":
        reason = obj.get("reason")
        return {"type": "clarify", "reason": reason if isinstance(reason, str) else ""}, "json_clarify"
    if otype == "none":
        return {"type": "none"}, "json_none"
    return None, "invalid_json"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, choices=["mock", "openai", "opencode", "vllm", "needle", "hf"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--out", default=None, help="results JSONL (default eval/results/<slug>.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N examples (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--extra-body", default=None, help="JSON merged into the request body")
    ap.add_argument("--no-thinking", action="store_true", help="vllm: enable_thinking=false")
    ap.add_argument("--needle-cmd", default=None)
    ap.add_argument("--adapter-path", default=None, help="hf: LoRA adapter dir")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--prompt-file", default=None, help="override the built-in system prompt")
    ap.add_argument("--route-k", type=int, default=0,
                    help="prompt carries only the top-k tools from the learned router (eval/router.py)")
    args = ap.parse_args(argv)

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    records = load_jsonl(split_path)
    if args.limit:
        records = records[:args.limit]

    tools_doc = json.dumps(json.loads(TOOLS_PATH.read_text(encoding="utf-8")),
                           ensure_ascii=False, separators=(",", ":"))
    if args.prompt_file:
        system = Path(args.prompt_file).read_text(encoding="utf-8").replace("<<TOOLS>>", tools_doc)
    else:
        system = SYSTEM_PROMPT.replace("<<TOOLS>>", tools_doc)

    slug = slugify(args.model)
    stem = split_path.stem
    if stem != "test":
        slug = f"{slug}-{stem}"
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        kept = [r for r in load_jsonl(out_path) if not r.get("error")]  # retry adapter errors
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                            encoding="utf-8")
        done_ids = {r["id"] for r in kept}
        if done_ids:
            print(f"resuming: {len(done_ids)} ids already in {out_path.name}")

    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    adapter_cls = get_adapter(args.adapter)
    adapter = adapter_cls(
        model=args.model, seed=args.seed, base_url=args.base_url,
        api_key_env=args.api_key_env, extra_body=extra_body,
        no_thinking=args.no_thinking, needle_cmd=args.needle_cmd, adapter_path=args.adapter_path,
        temperature=args.temperature, max_tokens=args.max_tokens)

    todo = [r for r in records if r["id"] not in done_ids]
    print(f"model={args.model} adapter={args.adapter} split={split_path.name} "
          f"todo={len(todo)}/{len(records)} -> {out_path}")

    consecutive_errors = 0
    counts = {}
    with open(out_path, "a", encoding="utf-8", newline="\n") as fh:
        for i, rec in enumerate(todo, start=1):
            t0 = time.perf_counter()
            if args.route_k:
                from router import learned_router
                tools_doc = learned_router().tools_doc(rec["request"], args.route_k)
                system = SYSTEM_PROMPT.replace("<<TOOLS>>", tools_doc)
            result = adapter.predict(rec, system, tools_doc)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            raw = result.get("text", "")
            prediction, status = parse_output(raw)
            if result.get("error"):
                status = "adapter_error"
                consecutive_errors += 1
            else:
                consecutive_errors = 0
            latency = result.get("latency_ms")
            line = {
                "id": rec["id"], "bucket": rec["bucket"], "language": rec["language"],
                "request": rec["request"], "model": args.model, "adapter": args.adapter,
                "parse_status": status, "prediction": prediction, "raw": raw[:4000],
                "latency_ms": latency if latency is not None else elapsed_ms,
                "usage": result.get("usage"),
            }
            if result.get("error"):
                line["error"] = result["error"][:500]
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            fh.flush()
            counts[status] = counts.get(status, 0) + 1
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            if consecutive_errors >= 5:
                print(f"ABORTING: 5 consecutive adapter errors. Last: {result.get('error', '')[:300]}")
                break

    print(f"done. wrote {sum(counts.values())} lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
