"""Needle 3 adapter (in-process `cactus-needle` engine, CPU).

--model is the weights: a tuned .cact path, or "base" for the published Needle 3 archive.
The prompt is NOT the shared system prompt: Needle gets the same compact tool list and
"Today/Request" query it was fine-tuned on (train/needle/make_data.py), which is the point -
no 5K-token schema dump. The reply is converted back into the harness' JSON contract:
  call(s)                    -> {"type": "call", "calls": [...]}
  ask_clarification(reason)  -> {"type": "clarify", "reason": ...}
  no call                    -> {"type": "none"}
The engine moves calls whose values are not literally in the request (enum values, resolved
dates) to `suppressed_calls`; needle.extract() reads `function_calls or suppressed_calls`, and so
does this adapter.

  python eval/run_eval.py --adapter needle --model train/needle/out/tuned_fullkv.cact
  python eval/run_eval.py --adapter needle --model base --out eval/results/needle3-base.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train" / "needle"))
from make_data import compact_tools, query  # noqa: E402


class NeedleAdapter:
    def __init__(self, model: str, max_tokens: int = 256, **_ignored):
        import needle
        self.agent = needle.Needle(tools=compact_tools(), system=None, stateless=True,
                                   weights=None if model == "base" else model, auto_date=False)
        self.max_tokens = max_tokens

    def predict(self, record, system, tools_doc):
        t0 = time.perf_counter()
        r = self.agent.complete(query(record), max_new_tokens=self.max_tokens)
        ms = int((time.perf_counter() - t0) * 1000)
        calls = r.get("function_calls") or r.get("suppressed_calls") or []
        if calls and calls[0].get("name") == "ask_clarification":
            out = {"type": "clarify", "reason": (calls[0].get("arguments") or {}).get("reason", "")}
        elif calls:
            out = {"type": "call", "calls": [{"name": c["name"], "arguments": c.get("arguments") or {}}
                                             for c in calls]}
        else:
            out = {"type": "none"}
        return {"text": json.dumps(out, ensure_ascii=False), "latency_ms": ms,
                "usage": {"suppressed": bool(calls) and not r.get("function_calls")}}
