"""Candidate D predictions - Needle 3, one stage per process.

  --stage before   base needle3 weights, zero-shot on our classify_ticket record
  --stage after    tuned .cact from `needle build --lora adapter.safetensors`

A tuned archive and the base weights cannot share a process (cactus-needle
contract), which is why one run = one stage. Refusals (empty call) keep the
candidate's label as ("other","medium","neutral") for scoring but are counted
separately and reported - routing refusal is a real product behaviour.

Run with venv-needle python. Writes artifacts/preds_d_<stage>.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_lib import ARTIFACTS, INTENTS, load, log_experiment, metrics  # noqa: E402

SCHEMA = {
    "name": "classify_ticket",
    "description": "Route one support ticket: intent, urgency, sentiment.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": INTENTS},
            "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
            "sentiment": {"type": "string", "enum": ["negative", "neutral", "positive"]},
        },
        "required": ["intent", "urgency", "sentiment"],
    },
}
SYSTEM = ("Support-ticket triage for an online shop. intent = the customer's main ask; "
          "urgency = low/medium/high from explicit time cues and business risk; "
          "sentiment = tone, not the situation.")
GOLD_DEFAULTS = {"intent": "other", "urgency": "medium", "sentiment": "neutral"}


def to_label(resp):
    """Merge one engine turn into (labels|None, kind, timing-fields)."""
    timing = {"prefill_tps": resp.get("prefill_tps"), "decode_tps": resp.get("decode_tps"),
              "peak_ram_mb": resp.get("peak_ram_mb")}
    calls = (resp or {}).get("function_calls") or []
    if not calls:
        return None, "refusal", timing
    args = calls[0].get("arguments", {})
    if (calls[0].get("name") not in ("classify_ticket",)
            or args.get("intent") not in INTENTS
            or args.get("urgency") not in ("low", "medium", "high")
            or args.get("sentiment") not in ("negative", "neutral", "positive")):
        return None, "invalid", timing
    return ({k: args[k] for k in ("intent", "urgency", "sentiment")}, resp.get("reasoning")),
           "ok", timing


def run_stage(stage, weights, split="test", tag=None):
    import needle
    ticket_rows = load(split)
    texts = [r["text"] for r in ticket_rows]
    golds = [{k: r[k] for k in ("intent", "urgency", "sentiment")} for r in ticket_rows]

    agent = needle.Needle(tools=[SCHEMA], system=SYSTEM,
                          weights=weights if stage == "after" else None)
    agent.complete("warmup: where is my order ORD-12345?")  # engine spin-up, not scored

    preds, raws, ok = [], [], {"ok": 0, "refusal": 0, "invalid": 0}
    lat = []
    for i, t in enumerate(texts):
        t0 = time.perf_counter()
        r = agent.complete(t)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        labels, kind, timing = to_label(r)
        ok[kind] += 1
        preds.append(labels[0] if labels else dict(GOLD_DEFAULTS))
        raws.append({"i": i, "kind": kind, "reasoning": labels[1] if labels else None,
                     "ms": round(ms, 1), **timing})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(texts)} ({ok['refusal']} refused, {ok['invalid']} invalid)")

    lat.sort()
    probe = {"p50_ms": round(lat[len(lat) // 2], 1),
             "p95_ms": round(lat[int(len(lat) * 0.95)], 1), "n": len(lat)}
    res = dict(metrics(preds, golds))
    tag = tag or f"d_needle_{stage}"
    log_experiment(tag, intent=res["intent"], urgency=res["urgency"],
                   sentiment=res["sentiment"], latency=probe, turns=ok,
                   mean_decode_tps=median(raws, "decode_tps"), peak_ram_mb=median(raws, "peak_ram_mb"),
                   weights=str(weights), config="needle3 record-as-tool, 1 call, batch-1", split=split, n=len(texts))
    (ARTIFACTS / f"preds_{tag}.json").write_text(json.dumps(
        {"name": tag, "predictions": preds, "gold": golds, "metrics": res,
         "latency": probe, "turns": ok, "raw": raws, "config": "1-tool record, batch-1",
         "weights": str(weights)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{tag}] {res} latency={probe} turns={ok}")


def median(rows, key):
    vals = sorted(r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] > 0)
    return round(vals[len(vals) // 2], 1) if vals else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["before", "after"], required=True)
    ap.add_argument("--weights", default="tuned.cact")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tag", default=None, help="output name (default d_needle_<stage>)")
    args = ap.parse_args()
    run_stage(args.stage, args.weights, args.split, args.tag)
