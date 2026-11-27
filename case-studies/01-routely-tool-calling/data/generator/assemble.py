"""Assemble validated batches into frozen train/val/test splits.

Usage (from the case-study dir):
  uv run --with rapidfuzz --with numpy python data/generator/assemble.py

Steps:
  1. load every data/batches/*.jsonl (already validator-clean)
  2. drop exact duplicates, then near-duplicates (>=0.95 similarity on ID-masked
     text), keep first
  3. report balance vs the bucket targets in data/FORMAT.md
  4. stratified split by (bucket, language); within a stratum examples are spread
     across tools by round-robin so each split sees every tool; allocate exactly
     500 test / 200 val via largest-remainder, everything else is train
  5. write data/splits/{train,val,test}.jsonl + sha256.txt + summary.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
BATCH_GLOB = str(ROOT / "data" / "batches" / "*.jsonl")
SPLITS_DIR = ROOT / "data" / "splits"

SEED = 42
TEST_N = 500
VAL_N = 200
NEAR_DUP_THRESHOLD = 0.95  # on ID-masked normalized text

# bucket targets (fractions from FORMAT.md), for the balance report
TARGET_PCT = {
    "clear": 0.55,
    "confusable": 0.15,
    "relative_date": 0.10,
    "multilingual": 0.05,
    "no_tool": 0.07,
    "clarify": 0.08,
}

ID_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}-\d{3,5}\b")


def norm(text: str, mask_ids: bool) -> str:
    if mask_ids:
        text = ID_TOKEN_RE.sub(" ID ", text)
    out = []
    for ch in text.lower():
        # keep letters/digits in any script plus combining marks (Devanagari matras)
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def tool_key(rec: dict) -> str:
    gold = rec.get("gold") or {}
    if gold.get("type") == "call" and gold.get("calls"):
        return gold["calls"][0].get("name") or "?"
    return gold.get("type") or "?"


def load_records(files: list[Path]) -> list[dict]:
    records = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def drop_duplicates(records: list[dict]):
    """Exact dupes, then near-dupes on ID-masked text; keep first. Returns kept + drop log."""
    keep = []
    seen_exact = {}
    dropped_exact = []
    for rec in records:
        key = norm(rec["request"], mask_ids=False)
        if key in seen_exact:
            dropped_exact.append((seen_exact[key], rec["id"]))
            continue
        seen_exact[key] = rec["id"]
        keep.append(rec)

    dropped_near = []
    if len(keep) > 400:
        try:
            from rapidfuzz import fuzz as rf_fuzz
            from rapidfuzz import process as rf_process
            import numpy as np

            masked = [norm(r["request"], mask_ids=True) for r in keep]
            scores = rf_process.cdist(masked, masked, scorer=rf_fuzz.ratio, workers=-1)
            thr = NEAR_DUP_THRESHOLD * 100
            dropped_idx = set()
            n = len(keep)
            for i in range(n):
                if i in dropped_idx:
                    continue
                row = scores[i]
                for j in np.nonzero(row[i + 1:] >= thr)[0]:
                    j = int(j) + i + 1
                    if j not in dropped_idx:
                        dropped_idx.add(j)
                        dropped_near.append((keep[i]["id"], keep[j]["id"], float(row[j]) / 100.0))
            kept = [r for i, r in enumerate(keep) if i not in dropped_idx]
        except Exception as exc:  # pragma: no cover
            print(f"note: near-dup check skipped ({exc})")
            kept = keep
    else:
        kept = keep
    return kept, dropped_exact, dropped_near


def allocate(counts: dict, target: int) -> dict:
    """Largest-remainder allocation of `target` items over strata `counts`."""
    total = sum(counts.values())
    raw = {k: v * target / total for k, v in counts.items()}
    base = {k: int(math.floor(x)) for k, x in raw.items()}
    room = target - sum(base.values())
    order = sorted(counts, key=lambda k: (raw[k] - base[k], counts[k]), reverse=True)
    for k in order[:room]:
        base[k] += 1
    return base


def split(records: list[dict]):
    rng = random.Random(SEED)
    strata = defaultdict(list)
    for rec in records:
        strata[(rec["bucket"], rec["language"])].append(rec)

    # within a stratum: shuffle each tool group (seeded), then round-robin
    # interleave so the head of the list touches every tool
    ordered = {}
    for key, items in strata.items():
        groups = defaultdict(list)
        for rec in items:
            groups[tool_key(rec)].append(rec)
        for g in groups.values():
            rng.shuffle(g)
        names = sorted(groups, key=lambda t: (-len(groups[t]), t))
        merged = []
        while any(groups[t] for t in names):
            for t in names:
                if groups[t]:
                    merged.append(groups[t].pop())
        ordered[key] = merged

    counts = {k: len(v) for k, v in ordered.items()}
    test_alloc = allocate(counts, TEST_N)
    remaining = {k: counts[k] - test_alloc[k] for k in counts}
    val_alloc = allocate(remaining, VAL_N)

    splits = {"train": [], "val": [], "test": []}
    for key in sorted(ordered):
        items = ordered[key]
        t, v = test_alloc[key], val_alloc[key]
        splits["test"].extend(items[:t])
        splits["val"].extend(items[t:t + v])
        splits["train"].extend(items[t + v:])
    return splits


def table(splits) -> dict:
    out = {}
    for name, recs in splits.items():
        out[name] = {
            "total": len(recs),
            "bucket": dict(Counter(r["bucket"] for r in recs).most_common()),
            "language": dict(Counter(r["language"] for r in recs).most_common()),
            "gold_type": dict(Counter(r["gold"]["type"] for r in recs).most_common()),
            "tools": dict(Counter(tool_key(r) for r in recs).most_common()),
        }
    return out


def main():
    files = sorted(Path(p) for p in glob.glob(BATCH_GLOB))
    if not files:
        print("no batch files found at", BATCH_GLOB)
        return 2
    print(f"batch files: {len(files)}")
    records = load_records(files)
    print(f"loaded records: {len(records)}")

    kept, dropped_exact, dropped_near = drop_duplicates(records)
    print(f"dropped exact duplicates: {len(dropped_exact)}")
    for a, b in dropped_exact[:10]:
        print(f"  {a} == {b}")
    print(f"dropped near-duplicates (>= {NEAR_DUP_THRESHOLD} masked): {len(dropped_near)}")
    for a, b, s in sorted(dropped_near, key=lambda x: -x[2])[:15]:
        print(f"  {s:.3f}  {a} ~ {b}")
    print(f"kept: {len(kept)}")

    total = len(kept)
    print("\n--- balance vs FORMAT.md targets ---")
    bucket_counts = Counter(r["bucket"] for r in kept)
    for bucket, pct in TARGET_PCT.items():
        n = bucket_counts.get(bucket, 0)
        print(f"  {bucket:14s} {n:5d}  ({100 * n / total:5.1f}%  target {pct:.0%})")

    splits = split(kept)
    stats = table(splits)

    print("\n--- splits ---")
    for name in ("train", "val", "test"):
        s = stats[name]
        print(f"  {name:5s} n={s['total']:5d}  buckets={s['bucket']}")
    print(f"  val/test language: {stats['val']['language']} / {stats['test']['language']}")

    tools_present = Counter(r for r in stats["test"]["tools"] if not r.startswith("__"))
    all_tools = [r for r in stats["train"]["tools"] if r.startswith("__") is False]
    call_tools_test = set(stats["test"]["tools"]) - {"none", "clarify"}
    call_tools_all = set(stats["train"]["tools"]) | call_tools_test | set(stats["val"]["tools"])
    missing = sorted(call_tools_all - call_tools_test)
    print(f"\n  tools with >=1 example in test: {len(call_tools_test)}  missing: {missing or 'none'}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in ("train", "val", "test"):
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in splits[name]) + "\n"
        path = SPLITS_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_lines = [f"{digests[name]}  {name}.jsonl" for name in ("train", "val", "test")]
    (SPLITS_DIR / "sha256.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    summary = {
        "seed": SEED,
        "near_dup_threshold_masked": NEAR_DUP_THRESHOLD,
        "loaded": len(records),
        "dropped_exact": len(dropped_exact),
        "dropped_near": len(dropped_near),
        "kept": total,
        "splits": stats,
        "sha256": digests,
        "targets_check": {
            bucket: {"count": bucket_counts.get(bucket, 0),
                     "actual_pct": round(100 * bucket_counts.get(bucket, 0) / total, 1),
                     "target_pct": int(pct * 100)}
            for bucket, pct in TARGET_PCT.items()
        },
        "test_tools_missing": missing,
    }
    (SPLITS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n--- frozen ---")
    for name in ("train", "val", "test"):
        print(f"  {digests[name]}  {name}.jsonl")
    print("wrote data/splits/{train,val,test}.jsonl, sha256.txt, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
