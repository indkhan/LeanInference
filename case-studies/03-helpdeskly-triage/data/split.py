"""Stratified split of tickets_raw.jsonl into train/val/test = 4000/300/700.

Stratifies jointly on (intent, urgency, sentiment) by deterministic round-robin
over sorted buckets, seeded. Emits:
  data/tickets_train.jsonl / tickets_val.jsonl / tickets_test.jsonl
  data/manifest.json (counts, seed, split sizes, per-split distributions)
"""
import json
import random
from collections import Counter
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
SEED = 13
SIZES = {"train": 4000, "val": 300, "test": 700}


def main():
    rows = [json.loads(l) for l in (CASE / "data" / "tickets_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(SEED)
    rng.shuffle(rows)

    buckets = {}
    for row in rows:
        buckets.setdefault((row["intent"], row["urgency"], row["sentiment"]), []).append(row)

    splits = {k: [] for k in SIZES}
    # deal rows one at a time to whichever split is behind target ratio
    targets = {k: SIZES[k] / len(rows) for k in SIZES}
    counts = {k: 0 for k in SIZES}
    for key in sorted(buckets):
        for row in buckets[key]:
            ratios = {k: (counts[k] + 1) / max(1, sum(counts.values()) + 1) for k in SIZES}
            deficit = {k: targets[k] - (counts[k] / max(1, sum(counts.values()))) for k in SIZES}
            k = max(SIZES, key=lambda name: (deficit[name], name))
            if counts[k] >= SIZES[k]:
                k = next(n for n in SIZES if counts[n] < SIZES[n])
            splits[k].append(row)
            counts[k] += 1

    totals = {k: sum(vals.values()) for k, vals in ((k, Counter(r["intent"] for r in v)) for k, v in splits.items())}
    for name, rows_out in splits.items():
        path = CASE / "data" / f"tickets_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ci = Counter(r["intent"] for r in rows_out)
        cu = Counter(r["urgency"] for r in rows_out)
        cs = Counter(r["sentiment"] for r in rows_out)
        print(f"{name}: {len(rows_out)} rows -> {path.name}")
        print("  intent min/max:", min(ci.values()), "/", max(ci.values()), "| urgency:", dict(cu), "| sentiment:", dict(cs))

    manifest = {
        "seed": SEED,
        "source": "tickets_raw.jsonl",
        "rows": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "test_hardness": dict(Counter(h for r in splits["test"] for h in r["hardness"])),
        "test_contradictions": sum(1 for r in splits["test"] if r["notes"]["cues"].get("urgency_contradiction")),
    }
    (CASE / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest:", manifest["test_hardness"], "| high-urg test rows:",
          sum(1 for r in splits["test"] if r["urgency"] == "high"))


if __name__ == "__main__":
    main()
