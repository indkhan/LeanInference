"""Build the Needle fine-tuning JSONL from the train split.

Native Needle format (see needle/model/finetune.py::render_example):
  {"query": <document text>,
   "system": <the same system prompt used at inference>,
   "tools": [<the same record tool used at inference>],
   "reasoning": <one short line>,
   "answers": [{"name": "record_claim", "arguments": <ground truth>}]}

Fine-tuning data comes from the train split only. The val split checks
generalisation; the test split is never trained on.

Usage: python train/needle_data.py [--split train] [--out train/needle_train.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASE = Path(__file__).resolve().parent.parent
SPLITS = CASE / "data" / "splits"

REASONING = ("claim fields are read from wherever the document states them; "
             "fields the document does not state are null")


def build_rows(split: str) -> list[dict]:
    tools = json.loads((CASE / "eval" / "needle_record_tool.json").read_text(encoding="utf-8"))
    system = (CASE / "data" / "needle_system.txt").read_text(encoding="utf-8").strip()
    rows = []
    for line in (SPLITS / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        rows.append({
            "query": doc["text"],
            "system": system,
            "tools": tools,
            "reasoning": REASONING,
            "answers": [{"name": "record_claim", "arguments": doc["target"]}],
        })
    return rows


def stats(path: Path):
    """Token lengths of the exact training sequences, using Needle's own renderer."""
    from needle.model.finetune import render_example
    from needle.model.tokenizer import get_tokenizer

    tok = get_tokenizer()
    lens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prompt, target = render_example(json.loads(line))
        lens.append(len(tok.encode(prompt)) + len(tok.encode(target)) + 2)  # +BOS/+EOS
    lens.sort()
    n = len(lens)

    def pct(p):
        return lens[min(n - 1, int(round((p / 100) * (n - 1))))]

    print(f"{path.name}: n={n} p50={pct(50)} p90={pct(90)} p95={pct(95)} p99={pct(99)} max={lens[-1]} "
          f"over1024={sum(1 for x in lens if x > 1024)} over2048={sum(1 for x in lens if x > 2048)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", action="store_true", help="print token-length stats for the file")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (CASE / "train" / f"needle_{args.split}.jsonl")
    if args.stats:
        stats(out)
        return

    rows = build_rows(args.split)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
