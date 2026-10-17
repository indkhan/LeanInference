"""Run Needle 3 extraction over a split, writing results in eval/extract.py's shape.

Sequential by design: the Needle engine is a single global C instance per
generation, so one agent processes documents one at a time. Documents are
independent - the agent is built with stateless=True so every `complete()`
starts from a clean conversation.

Usage:
  python eval/needle_extract.py --label needle3-base --tag before --split val --limit 20
  python eval/needle_extract.py --label needle3-ft --weights train/needle_out/tuned.cact --tag after --split test

Writes eval/results/results_<label>_<tag>.jsonl, caches per doc under
eval/cache/<label>/<tag>/, then runs eval/metrics.py on the results file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import tiktoken

sys.stdout.reconfigure(encoding="utf-8")

EVAL = Path(__file__).resolve().parent
CASE = EVAL.parent
SPLITS = CASE / "data" / "splits"
CACHE = EVAL / "cache"
RESULTS = EVAL / "results"

ENCODER = tiktoken.get_encoding("cl100k_base")


def slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name).strip("_")


def run_one(agent, doc: dict, model: str, tag: str, max_new_tokens: int) -> dict:
    system = (CASE / "data" / "needle_system.txt").read_text(encoding="utf-8").strip()
    t0 = time.perf_counter()
    parsed = None
    raw = ""
    error = None
    try:
        resp = agent.complete(doc["text"], max_new_tokens=max_new_tokens)
        raw = json.dumps(resp, ensure_ascii=False)
        calls = resp.get("function_calls") or []
        if calls:
            args = calls[0].get("arguments")
            if isinstance(args, dict):
                parsed = args
            else:
                error = "arguments not an object"
        else:
            error = "no function_calls in response"
    except Exception as ex:  # noqa: BLE001
        error = f"{type(ex).__name__}: {ex}"
    latency = round(time.perf_counter() - t0, 3)
    return {
        "doc_id": doc["id"],
        "family": doc["family"],
        "model": model,
        "tag": tag,
        "transport": "local_needle",
        "ok": parsed is not None,
        "error": error,
        "attempts": 1,
        "latency_s": latency,
        "raw": raw,
        "parsed": parsed,
        "tokens_in_est": len(ENCODER.encode(system)) + len(ENCODER.encode(doc["text"])),
        "tokens_out_est": len(ENCODER.encode(raw)) if raw else 0,
        "ts": int(time.time()),
        "cached": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="run label, e.g. needle3-base / needle3-ft")
    ap.add_argument("--weights", default=None, help="path to a tuned .cact (omit for the base model)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all documents")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="before", help="run label suffix, e.g. before / after")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    docs = [json.loads(line) for line in
            (SPLITS / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = docs[args.offset:]
    if args.limit:
        docs = docs[:args.limit]

    slug = slugify(args.label)
    cache_dir = CACHE / slug / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"label={args.label} weights={args.weights or '(base)'} split={args.split} "
          f"docs={len(docs)} tag={args.tag}")

    from needle import Needle

    tools = json.loads((EVAL / "needle_record_tool.json").read_text(encoding="utf-8"))
    system = (CASE / "data" / "needle_system.txt").read_text(encoding="utf-8").strip()

    t_init = time.perf_counter()
    agent = Needle(tools=tools, system=system, weights=args.weights, stateless=True,
                   auto_date=False)
    print(f"agent ready in {time.perf_counter() - t_init:.2f}s")

    records = []
    t_all = time.perf_counter()
    for i, doc in enumerate(docs, 1):
        cache_file = cache_dir / f"{doc['id']}.json"
        if not args.no_cache and cache_file.exists():
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            rec["cached"] = True
        else:
            rec = run_one(agent, doc, args.label, args.tag, args.max_new_tokens)
            cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        records.append(rec)
        status = "ok" if rec["ok"] else "ERR"
        print(f"[{i}/{len(docs)}] {doc['id']} {status:3s} {rec['latency_s']:6.2f}s"
              + (f"  {rec['error']}" if rec["error"] else ""))

    agent.close()

    n_ok = sum(1 for r in records if r["ok"])
    lats = sorted(r["latency_s"] for r in records)

    def pct(p):
        if not lats:
            return 0.0
        idx = min(len(lats) - 1, int(round((p / 100) * (len(lats) - 1))))
        return lats[idx]

    summary = {
        "model": args.label,
        "tag": args.tag,
        "transport": "local_needle",
        "split": args.split,
        "docs": len(docs),
        "ok": n_ok,
        "ok_rate": round(n_ok / len(docs), 4) if docs else 0.0,
        "cache_hits": sum(1 for r in records if r.get("cached")),
        "wall_s": round(time.perf_counter() - t_all, 1),
        "latency_p50": round(pct(50), 3),
        "latency_p95": round(pct(95), 3),
        "errors": [r["error"] for r in records if r.get("error")][:10],
    }
    out = RESULTS / f"results_{slug}_{args.tag}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out.relative_to(CASE)}")

    if not args.no_metrics:
        cmd = [sys.executable, str(EVAL / "metrics.py"), "--run", f"eval/results/{out.name}",
               "--split", args.split]
        r = subprocess.run(cmd, cwd=CASE)
        if r.returncode != 0:
            print(f"metrics.py exited {r.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
