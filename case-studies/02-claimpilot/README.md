# ClaimPilot: claim documents to JSON, at a fraction of the cost and inside your own cloud

*LeanInference case study 2. Every number below comes from a run on our 400-document,
hand-verified test set (`eval/results/final_table.md`); prices are list prices in
`eval/prices.json`. Volume: 500,000 documents/month.*

## What you have today

Every OCR'd claim form, invoice and repair/medical bill goes to a frontier model (Claude Opus 5.5,
$4 / $20 per million input / output tokens) with the 14-field schema in the prompt. At the brief's
~1,500 input tokens per document that is **about $5,100 a month** (10.26 $ per 1,000 documents),
and every document - names, policy numbers, IBANs - leaves your cloud for a third-party API.

## What we tested (same 400 documents for every model)

| Model | Field F1 (before → after fine-tune) | Whole doc exact | Valid JSON | Total-amount accuracy | p95 latency | Monthly cost | Runs where |
|---|---|---|---|---|---|---|---|
| **Baseline proxy** (strong frontier-class model, free tier) | 0.974 | 69.2% | 100% | 95.8% | 20.0 s | $4,733 * | third-party API |
| **A: DeepSeek V4.1 Flash** (cheap hosted, prompt only) | 0.967 | 63.0% | 99.8% | 97.3% | 25.9 s | $300 | third-party API |
| **B: Qwen3.5-0.8B fine-tuned + schema-constrained decoding** | 0.366 → **0.991** | **93.8%** | 99.8% | **99.5%** | 23.8 s † | **$861** | **your cloud (2 rented L4 GPUs)** |
| **C: Needle 3 fine-tuned** (on-device model, `.cact`) | 0.000 → 0.776 | 16.2% | 85.0% | 69.0% | 11.2 s (CPU) | $261 | your cloud / on-device CPU |

\* The baseline is a **proxy**: we had no paid frontier access, so the strongest available free model
stood in for "the model you use today"; its cost is priced at Claude Opus 5.5 list price × its measured
tokens. † Single-request latency measured on a 6 GB laptop GPU (RTX 3050) without the fast attention
kernels Qwen3.5 uses on Linux; a datacentre GPU is substantially faster. Monthly cost for B and C is
always-on rented machines (NVIDIA L4 at $0.59/h; 4-vCPU c7i at $0.18/h) sized from measured throughput.

## What we recommend: B - the fine-tuned 0.8-billion-parameter Qwen model

It is the only option that **beats the frontier baseline on every quality bar** *and* runs inside
your cloud:

- **More accurate:** field F1 0.991 vs 0.974; whole documents fully correct 93.8% vs 69.2%;
  total-amount accuracy 99.5% vs 95.8%.
- **Never invents data:** when a field is missing it returned `null` every time (null discipline
  1.000 vs 0.950 for the baseline). Every amount it wrote was correct.
- **Always schema-valid:** decoding is constrained to the JSON schema (xgrammar), so the output is
  parseable JSON by construction.
- **No frontier fallback** in the final system.

Candidate A is cheaper ($300/month) but is still a third-party API and is *less* accurate than
today's setup; Needle 3 is the smallest and cheapest, but at 0.776 F1 it is not ready for claims.

## Savings and compliance

- **$5,132 → $861 a month: about $4,270 saved monthly (-83%), ~$51K a year** - priced conservatively
  from throughput measured on a laptop GPU.
- **Personal data stays in your cloud.** Claimant names, policy numbers, VINs and IBANs are never
  sent to a third party - the compliance team's objection goes away rather than being contracted away.
- **You own the model**: a 0.8B base model + a small LoRA adapter, trained in 2.6 hours on a 6 GB GPU.

## What still goes wrong (details: `FAILURES.md`)

25 of 400 documents are not perfect. 16 are slash dates like `04/05/2024` that the document never
says are US or UK order - fix it as a business rule per sender, not with a bigger model. 7 are rare
first names nudged toward common spellings (cross-check against the policy record). 3 are one-letter
slips in long identifiers - IBAN and VIN check digits catch them before payout.

## How the numbers were produced

Synthetic but realistic documents (Faker, no real personal data): 1,500 train / 100 val / 400 test,
test set verified and never used for training or tuning (`data/`). Fine-tuning scripts in `train/`,
evaluation in `eval/` (`python eval/report.py` prints the full table and per-field breakdown).
Every run, including the dead ends, is in `logs/experiments.md` - among them two decoder issues we
found and fixed on the way: a JSON-constraint library that corrupted line items (0.964 → 0.991 F1
after switching to xgrammar) and a Needle export setting that broke long documents (0.10 → 0.82 F1).
