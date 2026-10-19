"""Baseline cost math: token counts x list prices -> $/call, $/1M calls, $/month.

No API runs - this is the "frontier bill" side of the case study: what the client
*was* paying (published list prices) against the candidates' (self-hosted) costs.

Usage:
  uv run python cost/compute_costs.py
  uv run python cost/compute_costs.py --calls 5000000 --output-tokens 120

Writes cost/baseline_costs.json (the receipt) and prints a table.
Cached scenario is computed only for models that declare a cached input price.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "cost" / "prices.json"


def money(v: float, dec: int = 2) -> str:
    return f"${v:,.{dec}f}"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prices", default=str(PRICES))
    ap.add_argument("--calls", type=int, default=None, help="calls per month")
    ap.add_argument("--input-tokens", type=int, default=None)
    ap.add_argument("--output-tokens", type=int, default=None)
    ap.add_argument("--cache-read-fraction", type=float, default=1.0,
                    help="fraction of input served from cache in the cached scenario")
    ap.add_argument("--out", default=str(ROOT / "cost" / "baseline_costs.json"))
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.prices).read_text(encoding="utf-8"))
    calls = args.calls or cfg["volume_assumption"]["calls_per_month"]
    tin = args.input_tokens or cfg["token_assumptions"]["input_tokens_per_call"]
    tout = args.output_tokens or cfg["token_assumptions"]["output_tokens_per_call"]
    cache_frac = max(0.0, min(1.0, args.cache_read_fraction))

    def per_call(p_in: float, p_out: float, p_cache: float | None) -> tuple[float, float | None]:
        uncached = tin / 1e6 * p_in + tout / 1e6 * p_out
        cached = None
        if p_cache is not None:
            eff_in = (1 - cache_frac) * p_in + cache_frac * p_cache
            cached = tin / 1e6 * eff_in + tout / 1e6 * p_out
        return uncached, cached

    rows = []
    receipt = {
        "generated_from": str(Path(args.prices).relative_to(ROOT)),
        "as_of": cfg["as_of"],
        "inputs": {"calls_per_month": calls, "input_tokens_per_call": tin,
                   "output_tokens_per_call": tout, "cache_read_fraction": cache_frac},
        "models": {},
        "note": "assumed list prices x token counts; no paid API runs",
    }
    for key, m in cfg["models"].items():
        uncached, cached = per_call(m["input_per_million"], m["output_per_million"],
                                    m.get("cached_input_per_million"))
        row = {
            "label": m["label"],
            "cost_per_call_uncached": round(uncached, 6),
            "cost_per_1m_calls_uncached": round(uncached * 1e6, 2),
            "cost_per_month_uncached": round(uncached * calls, 2),
            "cost_per_month_cached": round(cached * calls, 2) if cached is not None else None,
        }
        rows.append((key, row))
        receipt["models"][key] = row

    lines = ["| Model | $/call | $/1M calls | $/month uncached | $/month cached |",
             "|---|---|---|---|---|"]
    for key, r in rows:
        cached_s = money(r["cost_per_month_cached"], 0) if r["cost_per_month_cached"] is not None else "-"
        lines.append(f"| {r['label']} | {money(r['cost_per_call_uncached'], 4)} | "
                     f"{money(r['cost_per_1m_calls_uncached'], 0)} | "
                     f"{money(r['cost_per_month_uncached'], 0)} | {cached_s} |")
    lines.append("")
    lines.append(f"Assumed: {calls:,} calls/mo x {tin:,} input + {tout:,} output tokens "
                 f"(prices as of {cfg['as_of']}; cached scenario = {cache_frac:.0%} of input from cache)")
    lines.append("No paid API runs - assumed list prices x token counts.")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
