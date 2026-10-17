"""Build the final results table (markdown + json) from run summaries + prices.

Usage:
  python eval/report.py [--baseline-slug <summary suffix>]

Reads eval/results/summary_*.json (one per run), eval/prices.json (real list
prices with source URLs and fetch dates), and writes:
  eval/results/final_table.md    - the deliverable table
  eval/results/final_table.json  - same data, structured

Cost rules (all documented in the output):
  - hosted list-price cost: measured mean tokens/doc x published per-MTok prices
  - local deployment cost: reference GPU $/hr x measured p50 seconds/doc,
    serial single-stream (no batching) - conservative in both directions
  - "as run": what we actually paid ($0 for the free tier; own hardware for local)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASE = Path(__file__).resolve().parent.parent
RESULTS = CASE / "eval" / "results"
PRICES = json.loads((CASE / "eval" / "prices.json").read_text(encoding="utf-8"))

MONTHLY_DOCS = PRICES["frontier_assumption"]["docs_per_month"]


def load_summaries() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("summary_*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        s["_file"] = f.name
        rows.append(s)
    return rows


def hosted_cost_per_1k(s: dict, price_in: float, price_out: float) -> float:
    tin = s.get("tokens_in_mean") or 0.0
    tout = s.get("tokens_out_mean") or 0.0
    return (tin * price_in + tout * price_out) / 1e6 * 1000.0


def local_deploy_cost_per_1k(s: dict) -> float:
    gpu = PRICES["local_gpu"]["usd_per_hour"]
    p50 = s.get("latency_p50") or 0.0
    return gpu * (p50 * 1000.0 / 3600.0)


def as_run_cost(s: dict):
    t = s.get("transport")
    if t == "hosted_free":
        return 0.0
    if t == "local_transformers":
        return local_deploy_cost_per_1k(s)
    return None  # local_needle: CPU-only, no GPU/NPU needed - see notes


def where_it_runs(t: str) -> str:
    if t == "hosted_free":
        return "vendor API (free tier in this study)"
    if t == "local_transformers":
        return "own GPU, 4-bit + grammar"
    if t == "local_needle":
        return "own CPU/edge, 29 MB binary (no GPU)"
    return t or "?"


def role_of(run: str) -> str:
    for key in sorted(PRICES.get("run_roles", {}), key=len, reverse=True):
        if key in run:
            return PRICES["run_roles"][key]
    return ""


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-slug", default=None,
                    help="substring of the run to use for the frontier bill scenario (default: first hosted run)")
    args = ap.parse_args()

    summaries = load_summaries()
    if not summaries:
        raise SystemExit("no summary_*.json found - run eval/extract.py first")

    # pick the token profile for the frontier-bill scenario: prefer the baseline proxy run
    base = None
    if args.baseline_slug:
        base = next((s for s in summaries if args.baseline_slug in s["_file"]), None)
    if base is None:
        base = next((s for s in summaries if s.get("transport") == "hosted_free"), summaries[0])

    rows = []
    for s in summaries:
        run = s["_file"].replace("summary_", "").replace(".json", "")
        row = {
            "run": run,
            "role": role_of(run),
            "model": s.get("model"),
            "tag": s.get("tag"),
            "transport": s.get("transport"),
            "docs": s.get("docs"),
            "valid_json_schema_rate": s.get("valid_schema_rate"),
            "doc_exact_rate": s.get("doc_exact_rate"),
            "micro_f1": s.get("micro_f1"),
            "macro_f1": s.get("macro_f1"),
            "total_amount_accuracy": s.get("total_amount_accuracy"),
            "null_discipline": s.get("null_discipline"),
            "latency_p50": s.get("latency_p50"),
            "latency_p95": s.get("latency_p95"),
            "tokens_in_mean": s.get("tokens_in_mean"),
            "tokens_out_mean": s.get("tokens_out_mean"),
            "as_run_cost_per_1k": as_run_cost(s),
        }
        equiv = PRICES["run_cost_map"].get(s.get("model"))
        if equiv and s.get("transport") == "hosted_free":
            p1k = hosted_cost_per_1k(s, equiv["input"], equiv["output"])
            row["list_equiv_per_1k"] = p1k
            row["list_equiv_note"] = f"priced as {equiv['equivalence']} (${equiv['input']}/${equiv['output']} per MTok peak)"
            row["monthly_500k_list"] = p1k * MONTHLY_DOCS / 1000.0
        elif s.get("transport") in ("hosted_free", "local_transformers"):
            row["list_equiv_per_1k"] = None
        rows.append(row)

    # frontier scenario: same measured token profile, frontier prices
    tin, tout = base.get("tokens_in_mean"), base.get("tokens_out_mean")
    doc_tokens = PRICES["frontier_assumption"]
    frontier_rows = []
    for m in PRICES["per_mtok_usd"]["frontier_baselines"]:
        per_doc = ((tin or 0) * m["input"] + (tout or 0) * m["output"]) / 1e6
        per_doc_brief = (1500 * m["input"] + (tout or 0) * m["output"]) / 1e6
        frontier_rows.append({
            "model": m["name"], "vendor": m["vendor"],
            "usd_per_1k_docs_measured_tokens": round(per_doc * 1000, 4),
            "monthly_500k_measured_tokens": round(per_doc * MONTHLY_DOCS, 2),
            "monthly_500k_brief_1500tok": round(per_doc_brief * MONTHLY_DOCS, 2),
        })
    cheapest_local = min((r for r in rows if r.get("transport") == "local_transformers"),
                         key=lambda r: r["micro_f1"] or 0, default=None)

    lines = []
    lines.append("# ClaimPilot extraction - results table")
    lines.append("")
    lines.append(f"All numbers from real runs on {PRICES['as_of']}. Cost model: measured mean tokens/doc x "
                 f"published list prices (see `eval/prices.json` for source URLs). 500,000 docs/month = "
                 f"the client's volume assumption.")
    lines.append("")
    lines.append("| System | Role | Valid JSON+schema % | Doc exact % | Field micro-F1 | total_amount acc. | Null discipline | p95 latency (s) | $/1K docs (as run) | $/1K docs (list price) | Where it runs | Docs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r["micro_f1"] or 0)):
        lines.append(
            f"| {r['model']} [{r['tag']}] | {r['role']} | {fmt(r['valid_json_schema_rate'])} | {fmt(r['doc_exact_rate'])} | "
            f"{fmt(r['micro_f1'])} | {fmt(r['total_amount_accuracy'])} | {fmt(r['null_discipline'])} | "
            f"{fmt(r['latency_p95'], 1)} | {fmt(r['as_run_cost_per_1k'], 2) if r['as_run_cost_per_1k'] is not None else '-'} | "
            f"{fmt(r['list_equiv_per_1k'], 4) if r.get('list_equiv_per_1k') is not None else '-'} | "
            f"{where_it_runs(r['transport'])} | {r['docs']} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Hosted runs cost $0 as run (free tier models). List-price equivalents priced from the "
                 "model's own vendor page where a public price exists.")
    lines.append("- Local rows: deployment cost = reference GPU ("
                 f"{PRICES['local_gpu']['name']}, ${PRICES['local_gpu']['usd_per_hour']}/h) x measured p50 latency, "
                 "serial single-stream, no batching; our measured latencies come from an RTX 3050 Laptop 6GB "
                 "which is slower than the reference GPU. Batching would improve this further (not measured).")
    lines.append("")
    lines.append(f"## The client's current bill at {MONTHLY_DOCS:,} docs/month (frontier list prices)")
    lines.append("")
    lines.append(f"Token profile measured on our baseline proxy run ({base.get('model')}, tag {base.get('tag')}): "
                 f"~{fmt(tin, 0)} input tokens and ~{fmt(tout, 0)} output tokens per doc.")
    lines.append("")
    lines.append("| Frontier model | $/1K docs (measured tokens) | $/month (measured tokens) | $/month (brief's 1,500-token assumption) |")
    lines.append("|---|---|---|---|")
    for fr in frontier_rows:
        lines.append(f"| {fr['model']} ({fr['vendor']}) | {fr['usd_per_1k_docs_measured_tokens']} | "
                     f"${fr['monthly_500k_measured_tokens']:,.0f} | ${fr['monthly_500k_brief_1500tok']:,.0f} |")
    lines.append("")
    lines.append("## Cheapest hosted candidate (real list price)")
    for m in PRICES["per_mtok_usd"]["cheap_hosted"]:
        lines.append(f"- {m['name']} ({m['vendor']}): ${m['input_off_peak']}-${m['input_peak']} per MTok input, "
                     f"${m['output_off_peak']}-${m['output_peak']} per MTok output ({m['note']}).")
    lines.append("")

    (RESULTS / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "final_table.json").write_text(
        json.dumps({"as_of": PRICES["as_of"], "runs": rows, "frontier_scenarios": frontier_rows},
                   indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
