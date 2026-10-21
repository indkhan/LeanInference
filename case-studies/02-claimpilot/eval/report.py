"""Print + write the ClaimPilot results table, per-field breakdown, winner and failure cases.

Usage:
  python eval/report.py

Inputs (all produced by real runs):
  eval/results/summary_<run>.json   - from eval/metrics.py (400-doc hand-verified test split)
  eval/prices.json                  - list prices with source URLs + retrieval date
Outputs:
  eval/results/final_table.md / final_table.json

Cost rules:
  - hosted:  measured mean tokens/doc x published $/MTok of the priced model
  - self-hosted: always-on reference machines needed to serve 500K docs/month at the
    measured p50 seconds/doc, serial single stream (no batching; measured on a slower
    RTX 3050 Laptop) -> monthly = machines x $/h x 730 h
Winner rule (brief, Step 4): cheapest system that runs inside the client's own cloud,
with field micro-F1 >= baseline and total_amount accuracy >= baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CASE = Path(__file__).resolve().parent.parent
RESULTS = CASE / "eval" / "results"
PRICES = json.loads((CASE / "eval" / "prices.json").read_text(encoding="utf-8"))
DOCS_PER_MONTH = PRICES["frontier_assumption"]["docs_per_month"]
HOURS_PER_MONTH = 730
FIELDS = ["claim_id", "policy_number", "claimant_name", "date_of_incident", "date_of_service",
          "provider_name", "line_items", "total_amount", "currency", "diagnosis_or_damage_code",
          "vehicle_vin", "claim_type", "payout_iban", "notes"]

# ponytail: the lineup is fixed by the brief, so it lives here, not in a config file.
SYSTEMS = [
    {"name": "Baseline proxy: beam-free", "after": "opencode__beam-free_final",
     "price": "frontier", "where": "third-party API", "own_cloud": False, "baseline": True},
    {"name": "Candidate A: DeepSeek V4.1 Flash", "after": "opencode__deepseek-v4.1-flash-free_final",
     "price": "DeepSeek V4.1 Flash", "where": "third-party API", "own_cloud": False},
    # B: cost sized from batch-8 throughput (latency_p50 of the after run = s/doc at batch 8);
    # the p95 column comes from a separate batch-1 run (single-request serving latency).
    {"name": "Candidate B: Qwen3.5-0.8B QLoRA + xgrammar JSON",
     "before": "local__Qwen3.5-0.8B_before-ft-xgr",
     "after": "local__Qwen3.5-0.8B__qwen3.5-0.8b-lora_after-ft-xgr",
     "latency": "local__Qwen3.5-0.8B__qwen3.5-0.8b-lora_latency-b1",
     "price": "local_gpu", "where": "self-hosted GPU (own cloud)", "own_cloud": True},
    {"name": "Candidate C: Needle 3 LoRA (.cact, full KV window)", "before": "needle3-base_before",
     "after": "needle3-ft-fullkv_after",
     "price": "local_cpu", "where": "self-hosted CPU / on-device", "own_cloud": True},
]


def load(run):
    p = RESULTS / f"summary_{run}.json" if run else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else None


def frontier_model():
    name = PRICES["client_current_model"]
    return next(m for m in PRICES["per_mtok_usd"]["frontier_baselines"] if m["name"] == name)


def hosted_per_doc(s, pin, pout, tokens_in=None):
    return ((tokens_in or s["tokens_in_mean"]) * pin + s["tokens_out_mean"] * pout) / 1e6


def cost(sd, s):
    """-> (usd_per_1k, monthly_usd, basis)"""
    p = sd["price"]
    if p is None:
        return None, None, "no published list price for this model"
    if s is None:
        return None, None, "not run"
    if p == "frontier":
        m = frontier_model()
        d = hosted_per_doc(s, m["input"], m["output"])
        return d * 1000, d * DOCS_PER_MONTH, f"priced as {m['name']} list price x measured tokens"
    if p in PRICES:  # self-hosted reference machine
        ref = PRICES[p]
        units = max(1, math.ceil(DOCS_PER_MONTH * s["latency_p50"] / (HOURS_PER_MONTH * 3600)))
        monthly = units * ref["usd_per_hour"] * HOURS_PER_MONTH
        return monthly / DOCS_PER_MONTH * 1000, monthly, (
            f"{units} x {ref['name']} (${ref['usd_per_hour']}/h) always-on at measured p50 {s['latency_p50']:.2f}s/doc")
    m = next(x for x in PRICES["per_mtok_usd"]["cheap_hosted"] if x["name"] == p)
    d = hosted_per_doc(s, m["input_peak"], m["output_peak"])
    return d * 1000, d * DOCS_PER_MONTH, f"{m['name']} peak list price x measured tokens"


def f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def money(v):
    return "-" if v is None else f"{v:,.0f}"


def main():
    rows = []
    for sd in SYSTEMS:
        b, a = load(sd.get("before")), load(sd["after"])
        usd1k, monthly, basis = cost(sd, a)
        lat = load(sd.get("latency")) or a
        rows.append({**sd, "b": b, "a": a, "usd1k": usd1k, "monthly": monthly, "basis": basis,
                     "p95": lat and lat["latency_p95"]})
    base = next(r for r in rows if r.get("baseline"))
    fm, fr = frontier_model(), PRICES["frontier_accuracy_assumed"]

    out = ["# ClaimPilot extraction - results (400-document hand-verified test split)", "",
           f"Prices: `eval/prices.json` (retrieved {PRICES['as_of']}). Volume: {DOCS_PER_MONTH:,} docs/month.", "",
           "| Model | Field F1 (before -> after) | Doc exact | Valid JSON | Amount acc | p95 (s) | $/1K docs | Monthly $ | Runs where | Docs |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    if base["a"]:  # status quo: accuracy assumed, cost at the brief's 1,500 input tokens
        d = hosted_per_doc(base["a"], fm["input"], fm["output"], tokens_in=1500)
        out.append(f"| Status quo: {fm['name']} | assumed* | - | - | - | - | {d * 1000:.2f} | "
                   f"{money(d * DOCS_PER_MONTH)} | third-party API | not run |")
    for r in rows:
        a, b = r["a"], r["b"]
        before = "n/a (no FT)" if "before" not in r else (f(b["micro_f1"]) if b else "-")
        out.append(
            f"| {r['name']} | {before} -> {f(a and a['micro_f1'])} | {pct(a and a['doc_exact_rate'])} | "
            f"{pct(a and a['valid_schema_rate'])} | {pct(a and a['total_amount_accuracy'])} | "
            f"{f(r['p95'], 1)} | {f(r['usd1k'], 2)} | {money(r['monthly'])} | {r['where']} | "
            f"{a['docs'] if a else 'not run'} |")
    out += ["", f"\\* {fr['statement']} Source: {fr['source']} (retrieved {fr['retrieved']}). Not measured here "
            f"($0 budget, no paid frontier access); the measured bar is the baseline proxy row. Status-quo cost = "
            f"{fm['name']} list price (${fm['input']}/${fm['output']} per MTok) x the brief's 1,500 input tokens + "
            f"the proxy's measured output tokens.", "", "Cost basis per row:"]
    out += [f"- {r['name']}: {r['basis']}" for r in rows]
    out.append("")

    shown = [r for r in rows if r["a"]]
    out += ["## Per-field F1 (final version of each system)", "",
            "| Field | " + " | ".join(r["name"].split(":")[0] for r in shown) + " |",
            "|---|" + "---|" * len(shown)]
    for fld in FIELDS:
        out.append(f"| {fld} | " + " | ".join(f(r["a"]["per_field"].get(fld, {}).get("f1")) for r in shown) + " |")
    out.append("| null discipline | " + " | ".join(f(r["a"]["null_discipline"]) for r in shown) + " |")
    out.append("")

    winner, bar = None, base["a"]
    if bar:
        ok = [r for r in rows if r["own_cloud"] and r["a"] and r["monthly"] is not None
              and r["a"]["micro_f1"] >= bar["micro_f1"]
              and r["a"]["total_amount_accuracy"] >= bar["total_amount_accuracy"]]
        winner = min(ok, key=lambda r: r["monthly"], default=None)
        out += ["## Winner (brief Step 4 rule)", "",
                f"Bar = baseline proxy: micro-F1 {f(bar['micro_f1'], 4)}, total_amount acc {pct(bar['total_amount_accuracy'])}."]
        for r in (r for r in rows if r["own_cloud"]):
            a = r["a"]
            out.append(f"- {r['name']}: " + ("not run" if not a else
                       f"F1 {f(a['micro_f1'], 4)} ({'>=' if a['micro_f1'] >= bar['micro_f1'] else '<'} bar), "
                       f"amount {pct(a['total_amount_accuracy'])} "
                       f"({'>=' if a['total_amount_accuracy'] >= bar['total_amount_accuracy'] else '<'} bar)"))
        out += ["", f"**Winner: {winner['name']}**" if winner else
                "**No self-hosted system met the bar** - see the write-up for the closest option.", ""]

    pick = winner or max((r for r in rows if r["own_cloud"] and r["a"]),
                         key=lambda r: r["a"]["micro_f1"], default=None)
    if pick:
        fails = json.loads((RESULTS / f"failures_{pick['after']}.json").read_text(encoding="utf-8"))
        seen, chosen = set(), []
        for fl in fails:  # one case per distinct failing-field pattern first, then fill
            key = tuple(sorted(d["field"] for d in fl["diffs"])) or ("error",)
            if key not in seen:
                seen.add(key)
                chosen.append(fl)
        chosen += [x for x in fails if x not in chosen]
        out += [f"## Failure cases - {pick['name']} ({len(fails)} of {pick['a']['docs']} docs not exact)", ""]
        for fl in chosen[:10]:
            out.append(f"- `{fl['doc_id']}` ({fl['family']})" + (f" error: {fl['error']}" if fl.get("error") else ""))
            for d in fl["diffs"]:
                out.append(f"  - {d['field']}: truth `{json.dumps(d['truth'], ensure_ascii=False)[:120]}` "
                           f"pred `{json.dumps(d['pred'], ensure_ascii=False)[:120]}`")
        out.append("")

    text = "\n".join(out) + "\n"
    (RESULTS / "final_table.md").write_text(text, encoding="utf-8")
    (RESULTS / "final_table.json").write_text(json.dumps(
        {"as_of": PRICES["as_of"], "winner": winner and winner["name"],
         "rows": [{k: v for k, v in r.items() if k not in ("a", "b")}
                  | {"micro_f1_before": r["b"] and r["b"]["micro_f1"], "micro_f1_after": r["a"] and r["a"]["micro_f1"]}
                  for r in rows]}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
