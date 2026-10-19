# Baseline assumptions — source + date record

**Why this file exists:** under the $0-API rule nobody ran the frontier baseline. Every
assumed number that appears in `RESULTS.md` must be traceable to this file. Anything not
listed here must come from `eval/results/` (a real run).

**All sources retrieved 2026-10-17.** Third-party aggregators — before publishing the
case study, re-check against the providers' own pages.

## 1. Frontier baseline (the bill being replaced)

| Item | Value | Status | Source |
|---|---|---|---|
| GPT-6 Astra tool-calling accuracy on our eval | **not measured** | — | $0 rule: no paid API runs |
| Frontier-class BFCL v4 accuracy (assumed range) | **85–88%** | assumed | benchlm.ai BFCL v4 catalog, leader at 88.5% (BTL-3), top entries 88.5→75% (2026-10-17) |
| Astra tool-use standing | **#1** of all tracked models | published | llm-stats.com tool-calling leaderboard: Astra 36.3 TrueSkill, followed by Claude Sonnet 5.5 (35.4), Oct 17 2026 |
| Astra list price | $10.00 in / $1.00 cached / $50.00 out per M | published | llm-stats.com model page for gpt-6-astra (released 2026-09-04), Oct 17 2026 |

Notes for the write-up: use the range, never a single invented point. In `RESULTS.md` the
baseline row is labeled `assumed` and the measured rows never merge with it.

## 2. Published zero-shot scores of OUR candidates' base models (the "before fine-tuning" line)

BFCL v4 leaderboard, llm-stats.com/benchmarks/bfcl-v4, retrieved 2026-10-17 (0–1 scale):

| Model | BFCL v4 (published, zero-shot) | Size |
|---|---|---|
| Qwen3.5-4B | **50.3%** | 4B |
| Qwen3.5-2B | **43.6%** | 2B |
| Qwen3.5-0.8B | **25.3%** | 800M |
| Qwen3.5-9B (context) | 66.1% | 9B |
| Qwen3.7 Max (leader) | 75.0% | — |

This is the reference for "what the base model does out of the box" — the fine-tuned
numbers we measure on our 500-example eval will be reported against these, plus the
assumed frontier range.

## 3. Candidate A (prompted GPT-6 Luna) — context only

| Item | Value | Status | Source |
|---|---|---|---|
| Luna tool-use standing | #50 (22.1 TrueSkill) | published | llm-stats tool-calling leaderboard, Oct 17 2026 |
| Luna BFCL v4 (published) | none found | — | not listed on llm-stats BFCL v4 (21 models) |
| Luna list price | $0.10 in / $0.01 cached / $0.50 out per M | published | llm-stats model page for gpt-6-luna (released 2026-09-22), Oct 17 2026 |
| Luna run on our eval | **not run** | — | $0 rule |

## 4. Cost assumptions

`cost/prices.json` (prices above; volume 2M calls/mo × 5,304 in + 100 out tokens —
token counts measured, see `data/splits/summary.json`). Computed receipts:
`cost/baseline_costs.json` (re-run `uv run python cost/compute_costs.py` to reproduce).
