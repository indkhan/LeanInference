"""Extraction runner: send every document of a split to a model, store raw
outputs + parsed JSON + latency in eval/results/, cache every call on disk.

Transports:
  hosted_free   - `opencode run --agent extractor` against the free hosted
                  models on the local OpenCode provider (no API cost).
  local_llamacpp - grammar-constrained local inference (M3; implemented later).

Every model call is made at most once per (model, tag, document): results are
cached under eval/cache/<slug>/<tag>/<doc_id>.json and reused on re-runs.

Usage:
  python eval/extract.py --model opencode/deepseek-v4.1-flash-free \
      --split train --limit 12 --tag pilot --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent          # .../02-claimpilot/eval
CASE = ROOT.parent                              # .../02-claimpilot
SPLITS = CASE / "data" / "splits"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

SYSTEM_PROMPT = (CASE / "data" / "extract_system.md").read_text(encoding="utf-8")
ENCODER = tiktoken.get_encoding("cl100k_base")


def slugify(model: str) -> str:
    return model.replace("/", "__").replace("#", "_")


def find_opencode() -> list[str]:
    exe = shutil.which("opencode.exe")
    if exe:
        return [exe]
    shim = shutil.which("opencode")
    if shim and shim.lower().endswith(".ps1"):
        cand = Path(shim).parent / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
        if cand.exists():
            return [str(cand)]
    cmd = shutil.which("opencode.cmd")
    if cmd:
        return ["cmd", "/c", cmd]
    raise FileNotFoundError("opencode executable not found")


def extract_json_object(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply (fences, prose)."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON object found in reply")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("unbalanced JSON in reply")


def run_hosted(model: str, text: str, timeout: float) -> dict:
    argv = find_opencode() + [
        "run", "--agent", "extractor", "-m", model, "--format", "json",
    ]
    # The document goes in via stdin: passing a multi-line document as an argv
    # message truncates it at the first newline on this platform.
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(CASE), input=text, capture_output=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"raw": "", "parsed": None, "error": "timeout", "latency_s": timeout}
    latency = time.perf_counter() - t0
    out = proc.stdout or ""
    err_text = None
    chunks = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "text":
            chunks.append(ev.get("part", {}).get("text", ""))
        elif etype == "error":
            err = ev.get("error", {})
            err_text = err.get("message") or json.dumps(err)[:400]
    raw = "".join(chunks)
    if err_text and not raw:
        return {"raw": raw, "parsed": None, "error": f"provider error: {err_text}", "latency_s": latency}
    if not raw:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"raw": "", "parsed": None, "error": "empty response: " + " | ".join(tail), "latency_s": latency}
    try:
        parsed = extract_json_object(raw)
        return {"raw": raw, "parsed": parsed, "error": None, "latency_s": latency}
    except Exception as ex:  # noqa: BLE001
        return {"raw": raw, "parsed": None, "error": f"parse: {ex}", "latency_s": latency}


def run_local(model: str, text: str, timeout: float) -> dict:  # noqa: ARG001
    raise NotImplementedError("local_llamacpp transport lands in M3")


TRANSPORTS = {"hosted_free": run_hosted, "local_llamacpp": run_local}


def one_doc(doc: dict, model: str, tag: str, transport: str, timeout: float,
            use_cache: bool, cache_dir: Path, retries: int = 2,
            retry_sleep: float = 20.0, keep_errors: bool = False,
            rate_limiter=None) -> dict:
    cache_file = cache_dir / f"{doc['id']}.json"
    if use_cache and cache_file.exists():
        rec = json.loads(cache_file.read_text(encoding="utf-8"))
        if rec.get("ok") or keep_errors:
            rec["cached"] = True
            return rec
        # errored record (e.g. a rate-limit failure): fall through and retry

    attempts = []
    res = None
    for attempt in range(retries + 1):
        if rate_limiter is not None:
            rate_limiter()
        res = TRANSPORTS[transport](model, doc["text"], timeout)
        attempts.append(res)
        if res.get("parsed") is not None:
            break
        if "rate limit" in (res.get("error") or "").lower() and attempt < retries:
            time.sleep(retry_sleep)
    total_latency = round(sum(a.get("latency_s", 0.0) for a in attempts), 3)

    raw = res.get("raw") or ""
    rec = {
        "doc_id": doc["id"],
        "family": doc["family"],
        "model": model,
        "tag": tag,
        "transport": transport,
        "ok": res.get("parsed") is not None,
        "error": res.get("error"),
        "attempts": len(attempts),
        "latency_s": total_latency,
        "raw": raw,
        "parsed": res.get("parsed"),
        "tokens_in_est": len(ENCODER.encode(SYSTEM_PROMPT)) + len(ENCODER.encode(doc["text"])),
        "tokens_out_est": len(ENCODER.encode(raw)) if raw else 0,
        "ts": int(time.time()),
        "cached": False,
    }
    cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transport", default="hosted_free", choices=sorted(TRANSPORTS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all documents")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="before", help="run label, e.g. pilot / before / after")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--keep-errors", action="store_true",
                    help="reuse cached failed documents instead of retrying them")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="global minimum seconds between provider calls (rate-limit pacing)")
    ap.add_argument("--retry-sleep", type=float, default=20.0,
                    help="seconds to wait before retrying a rate-limited document")
    ap.add_argument("--stop-on-ratelimit", action="store_true",
                    help="abort the batch as soon as a document fails with a rate-limit error")
    args = ap.parse_args()

    docs = [json.loads(line) for line in (SPLITS / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = docs[args.offset:]
    if args.limit:
        docs = docs[:args.limit]

    slug = slugify(args.model)
    cache_dir = CACHE / slug / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model} transport={args.transport} split={args.split} docs={len(docs)} "
          f"tag={args.tag} concurrency={args.concurrency} min_interval={args.min_interval}")

    limiter_lock = threading.Lock()
    last_call = [0.0]

    def rate_limiter():
        if args.min_interval <= 0:
            return
        with limiter_lock:
            wait = last_call[0] + args.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.monotonic()

    records = []
    rate_limited = False
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(one_doc, d, args.model, args.tag, args.transport,
                        args.timeout, not args.no_cache, cache_dir,
                        retry_sleep=args.retry_sleep, keep_errors=args.keep_errors,
                        rate_limiter=rate_limiter): d
            for d in docs
        }
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done = len(records)
            status = "ok " if rec["ok"] else "ERR"
            print(f"[{done}/{len(docs)}] {rec['doc_id']} {status} {rec['latency_s']:6.1f}s"
                  + (f"  {rec['error'][:80]}" if rec["error"] else ""), flush=True)
            if (args.stop_on_ratelimit and not rec["ok"] and not rec.get("cached")
                    and "rate limit" in (rec.get("error") or "").lower()):
                rate_limited = True
                print("RATE LIMITED: aborting batch; rerun later to resume from cache", flush=True)
                for f in futs:
                    f.cancel()
                break
    wall = time.perf_counter() - t0

    records.sort(key=lambda r: r["doc_id"])
    out_file = RESULTS / f"results_{slug}_{args.tag}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    oks = [r for r in records if r["ok"]]
    lats = sorted(r["latency_s"] for r in records)
    summary = {
        "model": args.model,
        "tag": args.tag,
        "transport": args.transport,
        "split": args.split,
        "docs": len(records),
        "ok": len(oks),
        "ok_rate": round(len(oks) / max(1, len(records)), 4),
        "cache_hits": sum(1 for r in records if r.get("cached")),
        "wall_s": round(wall, 1),
        "latency_p50": lats[len(lats) // 2] if lats else None,
        "latency_p95": lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else None,
        "errors": [r["error"] for r in records if r["error"]][:10],
        "results_file": str(out_file.relative_to(CASE)),
    }
    (RESULTS / f"run_{slug}_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if rate_limited:
        print(f"partial: only {len(records)}/{len(docs)} documents processed before the rate limit",
              flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
