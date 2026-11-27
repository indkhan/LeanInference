# Case Study 2 — ClaimPilot: Implementation Plan

Status: **executed/done 2026-10-22** (results: `README.md`, `eval/results/final_table.md`)
Brief: `case-studies/02-claimpilot-extraction.md`
Working folder for all artifacts: `case-studies/02-claimpilot/`

---

## 0. Verified environment facts (what this plan is built on)

| Fact | Value | Consequence for the plan |
|---|---|---|
| Python | 3.11.9 + uv 0.13.0 | project venvs via `uv` |
| ML deps | none installed yet | install torch (CUDA), transformers, peft, trl, bitsandbytes, llama-cpp-python into a per-case-study venv |
| GPU | NVIDIA RTX 3050 Laptop, **6 GB VRAM** | QLoRA feasible up to ~4B model; **9B training is impossible** → target Qwen3-4B, documented fallback Qwen3-1.7B |
| Frontier API keys | **none** | baseline and hosted candidates must run on free access |
| Free hosted models | 8 models via the local `opencode` provider, **$0** (`opencode run -m opencode/…`, verified working) | baseline proxy + Candidate A runs are real runs at zero cost |
| vLLM | Linux-only (Windows unsupported) | constrained decoding via **llama.cpp GBNF grammar** (equivalent technique, deployable for ClaimPilot) |
| Disk | 579 GB free | fine for weights, datasets, caches |

Everything in the brief that requires paid resources gets an explicit, labeled substitution — no fake numbers, no "estimated" cells pretending to be measured.

---

## 1. How the brief's model lineup maps to what we can actually run

| Brief slot | What we will run | Deviation & justification |
|---|---|---|
| **Baseline** (frontier model ClaimPilot "currently uses") | After a 10-doc pilot across the free catalog, pick the **strongest free model** as a *frontier-class proxy*. Run it over the full test set. | No frontier key available. Accuracy numbers are a real run of the proxy; **cost** numbers are computed at published frontier list prices (fetched at build time, sources recorded in `eval/prices.json`). Clearly labeled "proxy" in every table. |
| **Candidate A** (cheaper hosted mini/flash, JSON mode) | 2–3 free flash-class models: `opencode/deepseek-v4.1-flash-free`, `opencode/glm-5.3-flash-free`, `opencode/mimo-v2.6-flash-free` (final set decided by the pilot) | Free tier stands in for paid mini/flash. Priced both "as run" ($0 free tier) and at the closest published list price for the savings model. |
| **Candidate B** (small Qwen, LoRA + constrained decoding) | **Qwen3-4B-Instruct QLoRA** on the 6 GB GPU; constrained decoding to the JSON schema with **llama.cpp GBNF** (GGUF export) | Within the brief's 4–9B range. vLLM→llama.cpp (Windows). If 4B training OOMs, fall back to **Qwen3-1.7B** and keep the trained-config evidence. |
| **Candidate C** (Needle 3, Cactus) | Time-boxed attempt: check license + weight availability on `cactus-compute/needle`; download and run if accessible, LoRA fine-tune + `.cact` export if the toolchain allows | If weights are gated/unavailable on this machine, deliver an honest "not evaluated — reason + evidence" entry instead of invented numbers. |

Pilot (Step M1) doubles as the experiment-log's first entry: all 8 free models, 10 documents, ranked → justifies baseline/candidate picks.

---

## 2. Data plan (deliverable 1 → `case-studies/02-claimpilot/data/`)

- **`schema.json`** — JSON Schema, exact 15-field contract from the brief: `claim_id`, `policy_number`, `claimant_name`, `date_of_incident`, `date_of_service`, `provider_name`, `line_items[] {description, amount}`, `total_amount`, `currency`, `diagnosis_or_damage_code`, `vehicle_vin`, `claim_type`, `payout_iban`, `notes`. Canonical values: ISO-8601 dates, ISO-4217 currency, decimal amounts as strings. Everything nullable except `claim_id`/`claim_type` (nulls are the *correct* answer for genuinely missing data).
- **`generate.py`** — seeded (reproducible) synthetic generator using Faker, **all fake personal data**:
  - 6+ document families: formal invoice with table; messy email thread; OCR-noisy scanned claim form (char confusions, broken wraps); auto repair estimate; medical bill with diagnosis code; terse fax/memo style.
  - Variation axes: date renderings (`03/01/2025`, `Jan 3rd, 2025`, …), currencies (USD/EUR/GBP), comma/dot decimals, line-item counts, IBAN renderings, typo/noise rate, ~15% missing fields (target `null`), VIN present only for auto claims.
  - Ground truth is exact by construction (generator emits text **and** canonical JSON).
- **Splits** — 1,500 train / 100 val / 400 test, disjoint seed ranges, stored as JSONL: `{id, text, target, meta}`.
- **`verify_test_report.md`** — test-set verification: (a) automatic round-trip check (re-parse rendered values, machine-check every field), (b) manual review pass over a random 20 + all ambiguous cases. Honestly labeled: these are synthetic docs, so "hand-verified" = construction-verified + sampled human review, not 400-doc human annotation. The limitation goes in the write-up.
- **Optional, time-boxed** — a small sample of real public receipts (CORD/SROIE) mapped to the schema as an external-realism check, clearly marked "external sanity check", never trained on. Skipped first if time gets tight.
- The 400-doc test set is touched exactly once per model per phase (logged in the experiment journal), never used for training or prompt iteration.

## 3. Training plan (deliverable 2 → `case-studies/02-claimpilot/train/`)

- `train_qlora.py` — transformers + peft + trl, bitsandbytes NF4 QLoRA.
  - Config: r=16, α=32, dropout 0.05, all attention+MLP linear targets, lr 1e-4 cosine, 2–3 epochs, seq len ~1024–1536, grad checkpointing, micro-batch 1 × accum 16, **fp16** (no bf16 on 3050).
  - Loss on JSON completion only (prompt masked). Fixed deterministic prompt format; canonical key order in targets (key-order-insensitive metric, but trains cleaner).
  - Eval on the 100-doc val set each epoch → pick best val field-F1.
  - Merge LoRA → convert to GGUF (q8_0) → serve through llama.cpp with GBNF grammar from `schema.json` for constrained decoding. This *is* also the deployment story for ClaimPilot's own infra.
- `configs/` + `logs/` — config files, loss curves, timings, GPU mem logs — all committed.
- Fallback ladder, in order: bnb on Windows problems → `triton-windows`/fp16 LoRA; 4B won't fit → Qwen3-1.7B (documented); last resort → llama.cpp CPU finetune for the small model.

## 4. Eval plan (deliverable 3 → `case-studies/02-claimpilot/eval/`)

- **`extract.py`** — one runner, pluggable transports:
  - `hosted_free`: `opencode run -m <model> --format json` per document, launched with cwd inside the case-study folder, which carries a minimal project `opencode.json` defining a **tools-disabled "extractor" agent** (system prompt = schema card + rules) so hosted calls behave like plain API calls. Retries + **on-disk cache** (`eval/cache/<model>/<docid>.json`) — every call made once, ever. Concurrency 2–4 to respect free-tier limits.
  - `local_llamacpp`: grammar-constrained generation; records tokens + latency.
  - Outputs: raw text, parsed JSON, latency, token counts → `eval/results/<model>_<phase>.jsonl` (`phase` = before/after FT).
- **`metrics.py`** — per-field normalizers (dates→ISO, amounts→Decimal, currency upper ISO, IBAN strip+upper, VIN upper, text trim/casefold), then:
  - field-level precision/recall/F1 overall **and per field** (null treated as a value class)
  - whole-document exact match %
  - valid JSON matching schema %
  - **null-discipline accuracy** (correct null vs invented value)
  - **`total_amount` accuracy** (amount + currency, reported separately)
  - p95 latency, $/1K docs, monthly cost @500K
- **`prices.json`** — every list price used, with URL + date fetched. Cost = measured tokens × list price; baseline priced at frontier list, A at mini/flash list (plus "as-run $0 free tier" column), B at $0 API + amortized self-host note.
- **`report.py`** — emits the brief's results table + per-field breakdown + picks 5–10 worst failure cases (with the model output and the correct value) into files the write-up cites.
- **`../logs/experiments.md`** — running journal: every run, timestamp, config hash, result summary. Satisfies "log every experiment".

## 5. Deliverable 6 — client one-pager (`README.md` in the folder)

Story format from the brief: what they were paying → what we tested → what we recommend → savings + privacy/compliance win. Numbers pasted from `eval/results/` only. Proxy/self-host caveats stated in one short "How the numbers were produced" box.

## 6. Execution order & gates

| # | Milestone | Gate / proof |
|---|---|---|
| M0 | Scaffold + schema + generator + splits | `verify_test_report.md` round-trip check passes |
| M1 | Pilot: 8 free models × 10 docs | ranked pilot table in experiment log → baseline + Candidate A chosen |
| M2 | Eval harness + full hosted runs (baseline proxy + A, before-FT) | results JSONLs + valid table rows |
| M3 | Local stack: Qwen3-4B GGUF + grammar, before-FT eval (fallback 1.7B if OOM) | local latency/tokens + before-FT metrics |
| M4 | QLoRA fine-tune → after-FT eval | training logs + after-FT metrics |
| M5 | Needle 3 attempt (time-boxed ~half a day) | either metrics or documented evidence of why not |
| M6 | Report: results table, failure analysis, one-pager, journal | all 6 deliverables present, every number traceable to `eval/results/` or `logs/` |
| M7 | Optional: CORD/SROIE realism sample | clearly-marked external check (first cut if time is tight) |

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 6 GB VRAM OOM on 4B QLoRA | probe first; fallback Qwen3-1.7B; keep evidence of both |
| bitsandbytes/triton friction on Windows | fp16 LoRA fallback ladder (§3) |
| Free-tier rate limits / flakiness | concurrency cap, retries, full on-disk cache, resumable runner |
| Needle 3 gated | honest "not evaluated" entry with fetched evidence (license/repo state) |
| "Hand-verified test set" is really generator-verified | round-trip verification + sampled manual review, limitation stated in write-up |
| Free models ≠ the frontier baseline | proxy labeled everywhere; costs split "as-run" vs "at list price" |

## 8. Budget

$0 spend (free tiers + local GPU). Wall-clock estimate: data 1–2 h; hosted evals ~1.5–3 h (dominated by API latency, parallelized); fine-tune 1–4 h; Needle 0.5 h; report 1–2 h. Total ≈ 1–2 working days of machine time.

## 9. Definition of done

- [x] `data/` schema + generator + splits + verification report
- [x] `train/` reproducible scripts/configs/logs; LoRA artifacts
- [x] `eval/` one-command results table + per-field breakdown + failure cases
- [x] Results table with all brief columns, each cell traceable
- [x] 5–10 failure examples from the winning model, explained
- [x] One-page client write-up (savings + compliance)
- [x] Experiment journal showing the testing journey
- [x] No frontier fallback in the final solution; all numbers from real runs
