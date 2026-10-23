"""The Routely deliverable table: before -> after, cost at 2M calls/month, winner by the brief's rule.

Usage:  python eval/final_table.py        (reads eval/scores/*.json, cost/prices.json)
Writes: writeup/RESULTS.md

Cost rules
  hosted:       measured mean prompt + completion tokens per call (provider usage) x list price
  status quo:   frontier list price x the measured full-schema prompt (cost/prices.json token basis)
  self-hosted:  always-on machines to serve 2M calls/month at the measured single-request p50
                latency (serial; laptop hardware, so conservative) -> machines x $/h x 730 h
Winner rule (brief Step 4): cheapest option whose overall accuracy >= the baseline proxy's.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "cost" / "prices.json").read_text(encoding="utf-8"))
CALLS = P["volume_assumption"]["calls_per_month"]
HOURS = 730
MACHINES = {  # same reference machines as case studies 2 and 3 (sources in their prices.json)
    "gpu": ("NVIDIA L4 24GB (Runpod, on-demand)", 0.59),
    "cpu": ("AWS c7i.xlarge (4 vCPU)", 0.1785),
}
HOSTED = {  # $/MTok in, out - OpenRouter list prices fetched 2026-10-20
    "frontier": ("GPT-6 Astra", 10.0, 50.0),       # the bill being replaced (README locked decision)
    "candidate_a": ("MiniMax M3 (closest paid listing of minimax-m3.1-flash)", 0.3, 1.2),
}

# ponytail: the lineup is fixed by the brief, so it lives here, not in a config file.
SYSTEMS = [
    {"name": "Baseline proxy: Nemotron 3 Ultra 550B, all 40 schemas", "after": "opencode-nemotron-3-ultra-free",
     "price": "frontier", "size": "550B (API)", "where": "third-party API", "baseline": True},
    {"name": "Candidate A: MiniMax M3.1 Flash, all 40 schemas", "after": "openrouter-minimax-m3.1-flash-free",
     "price": "candidate_a", "size": "API", "where": "third-party API"},
    {"name": "B1: Qwen3.5-0.8B LoRA, no schemas in prompt", "before": "hf-qwen3.5-0.8b-base-noschema",
     "after": "hf-qwen3.5-0.8b-ft-noschema", "price": "gpu", "size": "0.8B, 4-bit ~0.6 GB + 25 MB LoRA",
     "where": "self-hosted GPU"},
    {"name": "B2: Qwen3.5-2B LoRA, no schemas in prompt", "before": "hf-qwen3.5-2b-base-noschema",
     "after": "hf-qwen3.5-2b-ft-noschema", "price": "gpu", "size": "2B, 4-bit ~1.5 GB + LoRA",
     "where": "self-hosted GPU"},
    {"name": "C: Needle 3 LoRA (.cact), compact tool list", "before": "needle3-base", "after": "needle3-ft",
     "price": "cpu", "size": "63 MB .cact (W4A8)", "where": "self-hosted CPU / on-device"},
]


def score(slug):
    p = ROOT / "eval" / "scores" / f"{slug}.json"
    return json.loads(p.read_text(encoding="utf-8")) if slug and p.exists() else None


def no_tool_acc(s):
    """Correct 'no tool' + 'ask clarification' decisions, pooled over both buckets."""
    b = s["by_bucket"]
    n = b["no_tool"]["n"] + b["clarify"]["n"]
    return (b["no_tool"]["acc"] * b["no_tool"]["n"] + b["clarify"]["acc"] * b["clarify"]["n"]) / n


def mean_tokens(slug):
    rows = [json.loads(l) for l in (ROOT / "eval" / "results" / f"{slug}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    u = [r["usage"] for r in rows if r.get("usage") and r["usage"].get("prompt_tokens")]
    return (sum(x["prompt_tokens"] for x in u) / len(u), sum(x.get("completion_tokens", 0) for x in u) / len(u)) if u else None


def cost(sd, s):
    if sd["price"] in HOSTED:
        name, pin, pout = HOSTED[sd["price"]]
        tok = mean_tokens(sd["after"])
        if not tok:
            return None, None, "no usage recorded"
        per = (tok[0] * pin + tok[1] * pout) / 1e6
        return per * 1e6, per * CALLS, f"{name} list (${pin}/${pout}) x {tok[0]:.0f} in + {tok[1]:.0f} out tokens/call (measured)"
    name, usd_h = MACHINES[sd["price"]]
    p50 = s["latency_ms"]["p50"] / 1000
    units = max(1, math.ceil(CALLS * p50 / (HOURS * 3600)))
    monthly = units * usd_h * HOURS
    return monthly / CALLS * 1e6, monthly, f"{units} x {name} (${usd_h}/h) always-on at measured p50 {p50:.2f} s"


def main():
    rows = []
    for sd in SYSTEMS:
        a, b = score(sd["after"]), score(sd.get("before"))
        complete = a is not None and a["n_scored"] == a["n_split"]
        r = {**sd, "a": a if complete else None, "b": b, "n": a["n_scored"] if a else 0}
        if complete:
            r["usd1m"], r["monthly"], r["basis"] = cost(sd, a)
        rows.append(r)

    tin, tout = P["token_assumptions"]["input_tokens_per_call"], P["token_assumptions"]["output_tokens_per_call"]
    fn, fin, fout = HOSTED["frontier"]
    sq = (tin * fin + tout * fout) / 1e6
    out = ["# Routely tool calling - results (500-example hand-verified test set)", "",
           f"Volume {CALLS:,} calls/month. Accuracy = the whole response correct (tool, every argument, or the right "
           "clarify / no-tool decision). Latency: hosted = API round trip; Qwen = RTX 3050 6 GB laptop GPU, 4-bit, batch 1; "
           "Needle = laptop CPU.", "",
           "| Model | Overall (before -> after) | Tool acc (before -> after) | Arg exact | No-tool / clarify acc | p95 latency | $/1M calls | Monthly $ | Size | Runs where |",
           "|---|---|---|---|---|---|---|---|---|---|",
           f"| Status quo: {fn} (cost only, not run) | - | - | - | - | - | {sq * 1e6:,.0f} | {sq * CALLS:,.0f} | API | third-party API |"]
    for r in rows:
        a, b = r["a"], r["b"]
        if not a:
            out.append(f"| {r['name']} | pending ({r['n']}/500) | | | | | | | {r['size']} | {r['where']} |")
            continue
        bo = f"{b['response_accuracy']}% -> " if b else ""
        bt = f"{b['tool_selection']['acc']}% -> " if b else ""
        out.append(f"| {r['name']} | {bo}**{a['response_accuracy']}%** | {bt}{a['tool_selection']['acc']}% | "
                   f"{a['args']['full_match_given_tool']}% | {no_tool_acc(a):.1f}% | {a['latency_ms']['p95'] / 1000:.1f} s | "
                   f"{r['usd1m']:,.0f} | {r['monthly']:,.0f} | {r['size']} | {r['where']} |")
    out += ["", f"Status quo = {fn} list price x {tin:,} input tokens (all 40 schemas, measured) + {tout} output tokens per call.",
            "Cost basis per row:"] + [f"- {r['name']}: {r['basis']}" for r in rows if r["a"]]
    out += ["", "## Accuracy by bucket", "",
            "| Model | clear | confusable | relative_date | multilingual | no_tool | clarify |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["a"]:
            bb = r["a"]["by_bucket"]
            out.append(f"| {r['name']} | " + " | ".join(f"{bb[k]['acc']}%" for k in
                       ("clear", "confusable", "relative_date", "multilingual", "no_tool", "clarify")) + " |")

    base = next(r for r in rows if r.get("baseline"))
    bar = base if base["a"] else next(r for r in rows if r["name"].startswith("Candidate A") and r["a"])
    provisional = bar is not base
    ok = [r for r in rows if r["a"] and r is not bar and r["a"]["response_accuracy"] >= bar["a"]["response_accuracy"]]
    winner = min(ok, key=lambda r: r["monthly"], default=None) if ok else None
    if winner is None and provisional:
        winner = bar  # the only system at the bar is the bar itself
    out += ["", "## Winner (brief Step 4 rule)", "",
            f"Bar = {'PROVISIONAL - baseline proxy incomplete, Candidate A used' if provisional else 'baseline proxy'}: "
            f"{bar['a']['response_accuracy']}% overall."]
    out += [f"- {r['name']}: {r['a']['response_accuracy']}% ({'>=' if r['a']['response_accuracy'] >= bar['a']['response_accuracy'] else '<'} bar), ${r['monthly']:,.0f}/month"
            for r in rows if r["a"] and r is not bar]
    out += ["", f"**Winner: {winner['name']}**" if winner else "**No option met the bar.**", ""]

    if winner:  # failure examples of the winner, one per distinct error type first
        res = {json.loads(l)["id"]: json.loads(l) for l in
               (ROOT / "eval" / "results" / f"{winner['after']}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        gold = {json.loads(l)["id"]: json.loads(l) for l in
                (ROOT / "data" / "splits" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        sys.path.insert(0, str(ROOT / "eval"))
        from score import evaluate_one
        fails, seen = [], set()
        for rid, r in res.items():
            correct, key, detail = evaluate_one(gold[rid]["gold"], r.get("prediction"), r.get("parse_status"))
            if not correct:
                fails.append((key in seen, key, rid, r))
                seen.add(key)
        fails.sort(key=lambda x: x[0])
        out += [f"## Failure examples - {winner['name']} ({len(fails)} of {len(res)} wrong)", ""]
        for _, key, rid, r in fails[:10]:
            out.append(f"- `{rid}` [{gold[rid]['bucket']}] {key}: \"{gold[rid]['request']}\" (today {gold[rid]['request_date']})")
            out.append(f"  - gold `{json.dumps(gold[rid]['gold'], ensure_ascii=False)[:200]}`")
            out.append(f"  - pred `{json.dumps(r.get('prediction'), ensure_ascii=False)[:200]}`")
        out.append("")

    text = "\n".join(out) + "\n"
    (ROOT / "writeup" / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
