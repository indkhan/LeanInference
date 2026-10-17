"""Batch-run extraction + metrics for several models, sequentially.

Usage:
  python eval/run_models.py --models "opencode/a,opencode/b" --split train --limit 12 --tag pilot --concurrency 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT.parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--transport", default="hosted_free")
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--stop-on-ratelimit", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    failures = []
    for i, model in enumerate(models, 1):
        print(f"\n########## [{i}/{len(models)}] {model} ##########", flush=True)
        cmd = [PY, str(ROOT / "extract.py"), "--model", model, "--split", args.split,
               "--limit", str(args.limit), "--offset", str(args.offset), "--tag", args.tag,
               "--concurrency", str(args.concurrency), "--transport", args.transport,
               "--min-interval", str(args.min_interval)]
        if args.stop_on_ratelimit:
            cmd.append("--stop-on-ratelimit")
        r = subprocess.run(cmd, cwd=str(CASE))
        if r.returncode == 2:
            print("batch stopped: rate limited - rerun later to resume from cache", flush=True)
            break
        if r.returncode != 0:
            failures.append((model, "extract", r.returncode))
            continue
        slug = model.replace("/", "__").replace("#", "_")
        r2 = subprocess.run(
            [PY, str(ROOT / "metrics.py"), "--run", f"eval/results/results_{slug}_{args.tag}.jsonl",
             "--split", args.split],
            cwd=str(CASE),
        )
        if r2.returncode != 0:
            failures.append((model, "metrics", r2.returncode))

    print("\n########## batch done ##########")
    for f in failures:
        print("FAILED:", f)
    print(f"models: {len(models)}, failures: {len(failures)}")


if __name__ == "__main__":
    main()
