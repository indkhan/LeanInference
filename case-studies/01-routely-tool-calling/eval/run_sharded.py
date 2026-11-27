"""Run eval/run_eval.py over N parallel shards of a split, then merge into one results file.

Needed for slow hosted adapters (opencode CLI takes 20-40 s per call). Each shard is a normal
run_eval.py run (resumable), so an interrupted batch continues where it stopped.

  uv run python eval/run_sharded.py --adapter opencode --model opencode/deepseek-v4.1-flash-free --shards 6
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data" / "splits" / "test.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--pause", type=float, default=60.0, help="seconds between retry rounds")
    args = ap.parse_args()

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-")
    work = ROOT / "eval" / "results" / ".shards" / slug
    work.mkdir(parents=True, exist_ok=True)
    lines = [l for l in Path(args.split).read_text(encoding="utf-8").splitlines() if l.strip()]
    for i in range(args.shards):
        part = work / f"split-{i}.jsonl"
        part.write_text("\n".join(lines[i::args.shards]) + "\n", encoding="utf-8")
    # Rounds: rows that hit an adapter error (rate limit, socket) are dropped and retried.
    for attempt in range(args.rounds):
        for i in range(args.shards):  # drop error rows so run_eval's resume retries them
            out = work / f"results-{i}.jsonl"
            if out.exists():
                keep = [l for l in out.read_text(encoding="utf-8").splitlines()
                        if l.strip() and '"parse_status": "adapter_error"' not in l]
                out.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        procs = []
        for i in range(args.shards):
            cmd = [sys.executable, str(ROOT / "eval" / "run_eval.py"), "--adapter", args.adapter,
                   "--model", args.model, "--split", str(work / f"split-{i}.jsonl"),
                   "--out", str(work / f"results-{i}.jsonl")]
            procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
        rc = [p.wait() for p in procs]
        done = 0
        for i in range(args.shards):
            out = work / f"results-{i}.jsonl"
            if out.exists():
                done += sum(1 for l in out.read_text(encoding="utf-8").splitlines()
                            if l.strip() and '"parse_status": "adapter_error"' not in l)
        print(f"round {attempt + 1}: {done}/{len(lines)} rows (exit codes {rc})", flush=True)
        if done >= len(lines):
            break
        time.sleep(args.pause)
    merged = ROOT / "eval" / "results" / f"{slug}.jsonl"
    seen, rows = set(), []
    for i in range(args.shards):
        out = work / f"results-{i}.jsonl"
        if not out.exists():
            continue
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip() and '"parse_status": "adapter_error"' not in l:
                rid = l.split('"id": "', 1)[1].split('"', 1)[0]
                if rid not in seen:
                    seen.add(rid)
                    rows.append(l)
    merged.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"merged {len(rows)}/{len(lines)} rows -> {merged} (shard exit codes {rc})")


if __name__ == "__main__":
    main()
