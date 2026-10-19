"""Convert tickets_train.jsonl into Needle 3 fine-tune format.

Format (per cactus-needle docs): one JSON object per line with
  query      - the ticket text
  tools      - [classify_ticket] record schema (identical render at train & eval)
  answers    - [{"name": "classify_ticket", "arguments": {...}}]
  reasoning  - one short line grounding EVERY argument in a verbatim text span

The reasoning rule is mechanical, so train/eval stay consistent:
  first clause of the ticket      -> intent
  explicit urgency cue, or "no explicit deadline" (then label)
  explicit tone cue, or "plain request" (then label)

Run:  python make_needle_jsonl.py data/tickets_train.jsonl data/needle_train.jsonl
(stdlib only)
"""
import json
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_lib as CL  # paths + label order

SCHEMA = {
    "name": "classify_ticket",
    "description": "Route one support ticket: intent, urgency, sentiment.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": CL.INTENTS},
            "urgency": {"type": "string", "enum": CL.URGENCY},
            "sentiment": {"type": "string", "enum": CL.SENTIMENT},
        },
        "required": ["intent", "urgency", "sentiment"],
    },
}


def first_clause(text):
    t = text.replace("\n", " ").strip()
    cut = min((p if p > 0 else 999) for p in
              (t.find("."), t.find("?"), t.find("!"), 40))
    return t[:cut].strip()


def reasoning_for(row):
    cues = row["notes"]["cues"]
    seg = [f"{first_clause(row['text'])} -> intent {row['intent']}"]
    if cues.get("urgency"):
        seg.append(f"'{cues['urgency']}' -> urgency {row['urgency']}")
    else:
        seg.append(f"no explicit deadline -> urgency {row['urgency']}")
    if cues.get("sentiment"):
        seg.append(f"'{cues['sentiment']}' -> tone {row['sentiment']}")
    else:
        seg.append(f"plain request -> tone {row['sentiment']}")
    return "; ".join(seg)


def main(src, dst):
    rows = [json.loads(l) for l in (CASE / "data" / src).read_text(encoding="utf-8").splitlines()]
    with (CASE / "data" / dst).open("w", encoding="utf-8") as f:
        for r in rows:
            args = {"intent": r["intent"], "urgency": r["urgency"], "sentiment": r["sentiment"]}
            line = {
                "query": r["text"],
                "tools": [SCHEMA],
                "answers": [{"name": "classify_ticket", "arguments": args}],
                "reasoning": reasoning_for(r),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> data/{dst}")
    print("sample:", json.dumps(line, ensure_ascii=False)[:420])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tickets_train.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "needle_train.jsonl")
