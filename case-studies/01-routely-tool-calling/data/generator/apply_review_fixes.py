"""Apply the corrections found by the test-set review pass (2026-10-17).

The 500-example frozen test split was reviewed line by line by two independent
reviewers; 8 corrections were adjudicated and applied here to the SOURCE batch
files, after which assemble.py is re-run to regenerate the splits.

Fixes:
  gen03-0001  gold referenced order_id ORD-70012 not present in the request
  gen07-0159  dropoff=pickup had no pickup cue in the request
  gen12-0135  labelled clarify but was actionable under the dataset convention
              (cf. gen06-0273) -> invoice id removed so a required arg is truly missing
  gen12-0025  clarify reason omitted a required apply_credit arg
  gen12-0050  clarify reason omitted a required get_quote arg
  gen12-0096  clarify reason omitted a required schedule_pickup arg
  gen09-0285  status=missed with a future date resolved from "this Friday"
  gen09-0254  "booked" implied status=scheduled but gold had no status

Usage (from the case-study dir):
  uv run python data/generator/apply_review_fixes.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
BATCH_GLOB = str(ROOT / "data" / "batches" / "*.jsonl")

FIXES = {
    "gen03-0001": {
        "request": "the floor lamp from ORD-70012 landed here completely smashed, "
                   "can a driver swing by and collect it?",
    },
    "gen07-0159": {
        "request": "don't cancel anything - ORD-94334 needs a return raised and collected: "
                   "wrong item inside",
    },
    "gen12-0135": {
        "request": "dispute the $25 service fee",
        "gold.reason": "no invoice id",
    },
    "gen12-0025": {"gold.reason": "no customer id, amount or credit reason"},
    "gen12-0050": {"gold.reason": "no delivery address or weight"},
    "gen12-0096": {"gold.reason": "no address id, date or time window"},
    "gen09-0285": {
        "request_date": "2026-11-07",
        "request": "missed pickups at WH-002 yesterday",
    },
    "gen09-0254": {"request": "what pickups do we have this Thursday?"},
}


def apply_fix(rec: dict, changes: dict) -> bool:
    changed = False
    for key, value in changes.items():
        if key == "gold.reason":
            if rec.get("gold", {}).get("reason") != value:
                rec["gold"]["reason"] = value
                changed = True
        elif rec.get(key) != value:
            rec[key] = value
            changed = True
    return changed


def main():
    applied = {rid: 0 for rid in FIXES}
    files = sorted(Path(p) for p in glob.glob(BATCH_GLOB))
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        out = []
        touched = False
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid in FIXES:
                if apply_fix(rec, FIXES[rid]):
                    applied[rid] += 1
                touched = True
            out.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
        if touched:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(out) + "\n")
            print(f"updated: {path.name}")
    missing = [rid for rid, n in applied.items() if n != 1]
    for rid in sorted(FIXES):
        print(f"  {rid}: {'applied' if applied[rid] == 1 else f'SEE ({applied[rid]} changes)'}")
    if missing:
        print("PROBLEM: expected exactly one change each for:", missing)
        return 1
    print(f"result: OK ({len(FIXES)} records corrected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
