"""Render tickets_test.jsonl for hand-verification, and patch labels after review.

Usage:
  python verify_test.py render        # writes test_review.txt:  id|intent|urgency|sentiment|text
  python verify_test.py fix <id> <intent> <urgency> <sentiment>   # patch one row's labels + log

Hand-verification protocol (t3, 2026-10-17):
- all 700 rows read in three passes (chunked reads of test_review.txt);
- primary-intent rule: the issue the customer LEADS with is the gold label for
  multi-issue tickets; secondary issues are never the label;
- urgency rule: explicit cue words ("ASAP", "birthday is Saturday") + business risk;
  an explicit contradictory cue ("no rush at all") on a structurally-urgent ticket
  keeps the structural label - the contradiction is the eval's trap, not a label error;
- sarcasm is negative sentiment by definition ("Perfect, just perfect." positive words + complaint body);
- every row I changed is recorded in test_verification_log.md with before/after.
"""
import json
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
TEST = CASE / "data" / "tickets_test.jsonl"
LOG = CASE / "data" / "test_verification_log.md"


def load():
    return [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines()]


def save(rows):
    with TEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    rows = load()
    if cmd == "render":
        with (CASE / "data" / "test_review.txt").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['id']}|{r['intent']}|{r['urgency']}|{r['sentiment']}|{r['text']}\n")
        print(f"rendered {len(rows)} rows -> test_review.txt")
    elif cmd == "fix":
        _, tid, intent, urgency, sentiment = sys.argv[:5]
        for r in rows:
            if r["id"] == tid:
                old = (r["intent"], r["urgency"], r["sentiment"])
                r["intent"], r["urgency"], r["sentiment"] = intent, urgency, sentiment
                new = old if False else (intent, urgency, sentiment)
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(f"- `{tid}`: {old} -> {new} | text: {r['text'][:100]!r}\n")
                break
        else:
            sys.exit(f"id not found: {tid}")
        save(rows)
        print(f"patched {tid}: {old} -> {new}")
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    main()
