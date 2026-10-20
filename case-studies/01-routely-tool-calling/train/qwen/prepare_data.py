"""Render the frozen splits into chat-format JSONL for QLoRA fine-tuning.

Uses the same system prompt as evaluation (single source: eval/prompt.py), so
training and inference inputs are byte-identical.

Usage (from the case-study dir):
  uv run python train/qwen/prepare_data.py

Output:
  train/qwen/data/train-chat.jsonl   (2,432 examples)
  train/qwen/data/val-chat.jsonl     (200 examples)
Each line: {"messages": [system, user, assistant]} where assistant is the compact-JSON gold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from prompt import SYSTEM_PROMPT  # noqa: E402


def build(split_path: Path, out_path: Path, system: str, route_k: int = 0) -> int:
    records = [json.loads(l) for l in
               split_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            if route_k:  # routed prompt; gold tools forced in (train/val only)
                from router import learned_router
                must = tuple(c["name"] for c in r["gold"].get("calls", []))
                system = SYSTEM_PROMPT.replace(
                    "<<TOOLS>>", learned_router().tools_doc(r["request"], route_k, must))
            gold_json = json.dumps(r["gold"], ensure_ascii=False, separators=(",", ":"))
            example = {"messages": [
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Today: {r['request_date']}\nRequest: {r['request']}"},
                {"role": "assistant", "content": gold_json},
            ]}
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(records)


def main():
    tools_doc = json.dumps(json.loads((ROOT / "data" / "tools.json").read_text(encoding="utf-8")),
                           ensure_ascii=False, separators=(",", ":"))
    system = SYSTEM_PROMPT.replace("<<TOOLS>>", tools_doc)

    k = int(sys.argv[sys.argv.index("--route-k") + 1]) if "--route-k" in sys.argv else 0
    sfx = f"-k{k}" if k else ""
    n_train = build(ROOT / "data" / "splits" / "train.jsonl",
                    ROOT / "train" / "qwen" / "data" / f"train-chat{sfx}.jsonl", system, k)
    n_val = build(ROOT / "data" / "splits" / "val.jsonl",
                  ROOT / "train" / "qwen" / "data" / f"val-chat{sfx}.jsonl", system, k)

    print(f"train: {n_train} examples -> train/qwen/data/train-chat.jsonl")
    print(f"val:   {n_val} examples -> train/qwen/data/val-chat.jsonl")
    print(f"system prompt: {len(system)} chars  (tools doc: {len(tools_doc)} chars)")
    print(f"rough size per example: ~{int(len(system) / 3.6) + 120} tokens (Qwen tokenizer will differ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
