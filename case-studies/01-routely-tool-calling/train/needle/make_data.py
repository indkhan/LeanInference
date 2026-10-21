"""Render the frozen splits into Needle 3 fine-tune JSONL (train/needle/data/*.jsonl).

Needle's engine only emits calls to tools it is given, so it cannot drop the tool list the way the
Qwen no-schema variant does. It gets the *compact* list instead: the 40 tools with argument names,
types and enums, no descriptions (7.9K chars vs 19.2K for the full schemas). The same list and query
format are used at eval time (eval/adapters/needle_lib.py imports from here).

Gold mapping: call -> answers = the calls; clarify -> a pseudo-tool `ask_clarification(reason)`;
none -> answers = [] (Needle's native "no call").

Usage: python train/needle/make_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "train" / "needle" / "data"
CLARIFY = {"name": "ask_clarification",
           "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}


def compact_tools():
    tools = []
    for t in json.loads((ROOT / "data" / "tools.json").read_text(encoding="utf-8")):
        f = t.get("function", t)
        p = f.get("parameters", {})
        props = {k: {"type": v.get("type", "string"), **({"enum": v["enum"]} if "enum" in v else {})}
                 for k, v in p.get("properties", {}).items()}
        tools.append({"name": f["name"], "parameters": {"type": "object", "properties": props,
                                                         "required": p.get("required", [])}})
    return tools + [CLARIFY]


def query(rec):
    return f"Today: {rec['request_date']}\nRequest: {rec['request']}"


def answers(gold):
    if gold["type"] == "call":
        return [{"name": c["name"], "arguments": c["arguments"]} for c in gold["calls"]]
    if gold["type"] == "clarify":
        return [{"name": "ask_clarification", "arguments": {"reason": gold.get("reason", "")}}]
    return []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tools = compact_tools()
    for split in ("train", "val"):
        rows = [json.loads(l) for l in (ROOT / "data" / "splits" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps({"query": query(r), "tools": tools, "answers": answers(r["gold"])},
                                    ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows -> {OUT / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
