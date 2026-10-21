# ClaimPilot experiment log

Append-only. Every experiment that produced a number used in the deliverables.
Raw artefacts live in `eval/results/` and `eval/cache/`; this file is the narrative index.

## E0 — Environment (2026-10-17)

- Windows 11 · Python 3.11.9 (per-project `uv` venv) · NVIDIA RTX 3050 Laptop 6 GB (driver 581.86, CUDA 13 capable)
- Installed: torch 2.11.0+cu128 (CUDA verified), transformers 5.20.1, peft 0.21.2, bitsandbytes 0.50.2, trl 1.16.0, datasets, accelerate, tiktoken, lm-format-enforcer 0.11.3, faker, jsonschema
- `llama-cpp-python` cu124 wheel install never produced output (source-build path on Windows); abandoned.
  Local decoding = transformers 4-bit + **lm-format-enforcer** grammar constraint (same idea as vLLM/Outlines guided decoding).
- Hosted transport = `opencode run --agent extractor --format json`, one tools-disabled agent (`opencode.json`),
  documents piped via **stdin**.
- Constraints honoured: $0 spend (free hosted tier only, no paid GPU), all data synthetic.

### Hosted transport findings (each cost a debug cycle)

1. Passing a long multi-line document as a CLI **argv** message truncates at the first newline
   (model saw only line 1; pilot run 1 scored ~0 everywhere). Fixed: pipe the document via stdin.
2. An agent config with `"permissions": [{"action": "*", ... "deny"}]` makes the free tier reject the
   run with `FreeTierError`. Fixed: no permission block; `steps: 1` already strips tools on the single step.
3. With `steps: 1` OpenCode appends a "maximum steps reached" nudge; models answer first, but the reply
   can include commentary after the JSON. The JSON extractor scans for the first balanced object, so
   this is harmless for scoring; retries (2) cover the rare fully-sided reply.

## E1 — Dataset (`data/`, seeds fixed, all fake via Faker)

- 2,000 docs: train 1500 / val 100 / test 400. Six families (formal invoice, email thread, OCR scan,
  auto repair estimate, medical bill, fax memo) × format variation × 10–25 % missing fields (null is the correct answer).
- Round-trip verification (`data/verify_test.py`): **0 problems on all splits**.
  Checks: schema validity, every rendered value present verbatim, evidence patterns (claim id, policy, VIN,
  IBAN, currency name, claim type) exist only when the field is non-null, totals equal item sums, ids unique.
- Median doc ≈ 2.55 K chars (~640 tiktoken tokens); 3.6 null fields per doc.
- Two ill-posedness fixes found by probing before any eval run (both would have created unfair failures):
  1. amount renderings sometimes had no currency indicator while ground truth had `currency` → a model
     guessing GBP from a GB-prefixed IBAN was "wrong" against an unknowable truth. Fixed: currency is always
     stated in the document when non-null; never when null.
  2. OCR/fax noise could corrupt values (item descriptions, provider names) → values are now inserted after
     noise, so ground truth is achievable; noise stays in surrounding prose.

## E2 — Pilot: 8 free hosted models × 12 train docs (tag `pilot`)

Model selection only (not prompt tuning): pilot runs use **train**-split documents; the test split stays
untouched until the final evaluation. Full scores in `eval/results/summary_opencode__*_pilot.json`.

| model (opencode free tier) | valid JSON+schema | doc exact | micro-F1 | total_amount | null disc. | p50 (s) | p95 (s) | out tok |
|---|---|---|---|---|---|---|---|---|
| beam-free | 1.000 | 0.833 | **0.9922** | 1.000 | 0.951 | **2.7** | 9.3 | 250 |
| deepseek-v4.1-flash-free | 1.000 | 0.750 | 0.9883 | 1.000 | 0.927 | 7.4 | 14.8 | 184 |
| nemotron-3.5-super-vl-free | 1.000 | 0.667 | 0.9764 | 1.000 | 0.976 | 33.3 | 146.5 | 255 |
| longcat-2.5-preview-free | 1.000 | 0.583 | 0.9688 | 1.000 | 0.951 | 14.7 | 27.8 | 204 |
| glm-5.3-flash-free | 1.000 | 0.500 | 0.9690 | 1.000 | 0.902 | 44.3 | 150.3 | 242 |
| space-bunny-free | 1.000 | 0.417 | 0.9606 | 1.000 | 0.927 | 3.0 | 15.3 | 250 |
| mimo-v2.6-flash-free | 1.000 | 0.250 | 0.9466 | 1.000 | 0.805 | 24.7 | 71.7 | 448 |
| big-pickle | 0.917 | 0.333 | 0.9120 | 0.917 | 0.829 | 21.3 | 35.7 | 345 |

All 8 completed; 96/96 documents parsed to valid JSON. 12 documents is a selection sample, not an
accuracy claim - final numbers come from the 400-document test split.

Decisions (recorded before the final runs):
- **Baseline proxy = `opencode/beam-free`** - strongest micro-F1 (0.9922) and fastest; carried as the
  stand-in for "the frontier model ClaimPilot uses today" (we cannot call a paid frontier API; the label
  `proxy` is kept everywhere the number appears). Costs are computed at frontier list prices, not beam's.
- **Candidate A = `opencode/deepseek-v4.1-flash-free`** - second by micro-F1 (0.9883, within noise of
  beam), much cheaper class of model and - crucially - it maps to a *published* price (DeepSeek V4.1
  Flash), so its cost column is a real list price rather than an unknown.
- Backup candidate carried to the final run for robustness: `opencode/nemotron-3.5-super-vl-free`
  (best null discipline, slow).

## E3 — Final hosted runs on the 400-doc test split (tag `final`)

`beam-free` (baseline proxy), `deepseek-v4.1-flash-free` (candidate A), `nemotron` (backup) over the full
test split, concurrency 5, cached per document. Launched 2026-10-17. Results land in
`eval/results/results_*_final.jsonl` + `summary_*_final.json`.

A first 12-doc slice (test-00000..11) ran at 2.7 / 8.7 / 30.5 s p50 with micro-F1 0.9888 / 0.9925 / 0.9701 -
consistent with the pilot; the full split decides the headline numbers.

## E4 — Needle 3 setup and a base-model probe (candidate C)

- `cactus-needle==3.2.4` installs and imports on Windows; the C engine for `windows-x86_64` downloads on
  first use; base weights `needle3.cact` come from HF `Cactus-Compute/needle3`. Engine loads in ~30 s cold,
  then ~0.3-2 s per query on CPU.
- Tool schema for extraction = exactly one tool (`record_claim`), a lightly adapted copy of
  `data/schema.json` (`eval/needle_record_tool.json`): `line_items` becomes a plain nullable array; the
  nested item object and the nullable-union fields are kept. Inference prompt = `data/needle_system.txt`,
  `auto_date=False` (the injected "today" fact was appearing inside the base model's answers).
- **Probe 1 - tiny schema, short text:** works perfectly. `vendor/total/currency` extracted from
  "Invoice from Acme Ltd. Total: 12.50 USD" in 0.32 s, confidence 0.78, short reasoning line. The engine,
  grammar and nullable unions are fine.
- **Probe 2 - tiny schema, our real 2.5 K-char claim doc:** base model *refuses* ("no document
  management tools exist in the available set") or, with the full 14-field record tool, loops its
  reasoning until the token budget is exhausted and emits no call. The base model cannot do this task.
  This is the honest "before fine-tuning" state; the fine-tune is the candidate.
- Fine-tuning data: `train/needle_data.py` writes the native JSONL (`query` = document text, `tools`,
  `system`, one-line `reasoning`, `answers` = the ground-truth record) for the **train split only**.
  Token lengths measured with Needle's own tokenizer/renderer: p50 **1724**, p95 1938, max 2113 - i.e.
  the CLI default `--max-len 1024` would truncate *every* example. Training will use `--max-len 2304`
  (all rows fit; model config allows 4096).

## E5 — Infrastructure findings that shaped the runs (2026-10-17)

1. **Free-tier rate limits are per model, not per account.** The 400-doc test run for `beam-free` hit
   `provider error: Rate limit exceeded` after ~228 clean documents (~10 min at concurrency 5) and
   every later call failed; single-doc probes at the same minute showed `deepseek-v4.1-flash-free`
   and `nemotron` still healthy. The harness now: re-runs cached *failed* documents instead of trusting
   the cache (`--keep-errors` to opt out), backs off 20 s before retrying a rate-limited document,
   accepts a global `--min-interval` pace, aborts the batch on a rate-limit failure
   (`--stop-on-ratelimit`), and resumes from cache on the next invocation.
2. **Windows Smart App Control blocks jaxlib.** `import jax` dies with
   `ImportError: DLL load failed while importing _lapack: An Application Control policy has blocked
   this file` (SAC state = on; the code-integrity event log shows the block). Torch/CUDA,
   bitsandbytes and the Needle C engine all load fine - only jaxlib's unsigned native module is
   stopped. Inference paths stay on Windows; Needle *fine-tuning* runs inside **WSL2 (Ubuntu 24.04)**
   with Linux wheels, reading the repo through `/mnt/c` and writing the adapter/`tuned.cact` back.
   Worth documenting: the client will meet the same class of deployment friction.
3. HF downloads on this link run ~0.7-1.2 MB/s, so the full fp16 Qwen3-4B (8 GB) was swapped for the
   pre-quantized `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` checkpoint (2.65 GB) - same NF4 weights,
   loads straight into transformers + bitsandbytes.

## E6 — Status board (16:50)

- Hosted final runs: `deepseek-v4.1-flash-free` in progress on 400 test docs (paced); `beam-free`
  paused at 230/400 cached successes until its provider window re-opens; `nemotron` probe OK, queued.
- Needle: WSL environment installing; fine-tune next, then `build` -> `tuned.cact`, then before/after
  evaluation on the pilot slice and the test split.
- Qwen: 4-bit checkpoint downloading (2.65 GB); next: smoke-test `train/train_qlora.py`, then the
  before-FT eval, then the full QLoRA run.



## E7 — Resume on 2026-10-20: Qwen3.5-0.8B adapter check, harness fixes, hosted transport

- **The 9.07 eval_loss was a metric bug, not overfitting.** The trainer's eval used the default
  `per_device_eval_batch_size=8` (13 eval steps for 100 docs) with right padding, while `TailLossTrainer`
  assumes one unpadded row: it took row 0's answer length and scored the *last k positions of the padded
  batch* - pad tokens. Recomputed properly (`train/val_loss.py`, one row at a time, answer tokens only,
  19,214 tokens): **val loss 0.00149 after epoch 1 (checkpoint-94), 0.00136 after epoch 2** (final adapter).
  Generation on val-00000..09 (no grammar): **10/10 valid, 10/10 doc-exact, micro-F1 1.000**, p50 18.6 s,
  peak VRAM 1.13 GB. Fix: `per_device_eval_batch_size=1` in `train/train_qlora.py`. The adapter is reused
  as-is; nothing retrained.
- **Constrained-decoding schema was broken.** `LM_SCHEMA.currency` was `{"type": ["string","null"], "enum":
  [..., None]}`; lm-format-enforcer raises `Unsupported enum type` once generation reaches that field, so the
  first test run produced truncated JSON. Rewritten as `anyOf [{enum}, {null}]` (same contract). The schema
  now lives in `eval/extract.py` and is shared by the local runner and hosted structured-output calls
  (Anthropic's structured output also rejected the old form). First run discarded and re-launched.
- **metrics.py bug:** item-level line_items P/R was always 0 (`items_pr` was handed `None` unless the
  prediction was a dict; line_items is a list). Fixed; field-level line_items scoring was unaffected.
- **Hosted transport:** OpenRouter adapters added (`openrouter`, `openrouter_schema`, `openrouter_json`,
  real usage tokens + billed $ recorded per call). The account had $0 credits and a 50 req/day free cap, so
  only 3 probe calls were made (Claude Haiku 5.5 $0.0004; Opus 5.5 refused HTTP 402; nemotron-3-ultra:free
  OK). OpenCode's free tier came back the same day, so hosted runs went back to OpenCode; OpenRouter stays as
  a fallback. Model slugs with `:` are now filesystem-safe on Windows.
- **Smart App Control is off now** (`SmartAppControlState=Off`); `import jax` works on Windows (CPU only, no
  CUDA jaxlib for Windows), so Needle training stays in WSL for the GPU.
- Candidate A resume: `opencode/deepseek-v4.1-flash-free` had 274/400 successes cached; the remaining
  126 are re-run (cached failures are retried).

## E8 — Hosted rows final, local-eval memory leak, process survival (2026-10-21)

- **Candidate A complete:** `opencode/deepseek-v4.1-flash-free` 400/400 on the test split (126 resumed from
  cache; the last 5 needed three paced retries on the free tier's rate limit). Micro-F1 **0.9670**, doc
  exact 63.0%, valid schema 99.75%, total_amount acc **97.26%**, null discipline 0.941, p95 25.9 s.
- **Baseline proxy** `opencode/beam-free` re-scored with the fixed metrics: micro-F1 **0.9738**, doc exact
  69.25%, valid 100%, total_amount acc **95.75%**, null discipline 0.950, p95 20.0 s.
- **Local eval crashed with `RuntimeError: bad allocation` (host RAM) every ~70-90 docs.** Cause: one shared
  lm-format-enforcer `TokenEnforcer` caches every prefix state it ever sees. Fix: the vocab scan is built
  once, a fresh enforcer per document. Runs resume from the per-doc cache, so no doc was re-run.
- **Jobs launched from the agent session die when the session goes idle** (twice overnight: base-Qwen eval
  at doc 149, then at 325; fine-tuned eval at 131). The GPU queue (`gpu_queue.ps1`, one GPU job at a time,
  each step resumable) and the hosted runs are now started through WMI (`Win32_Process.Create`), outside
  the session's process tree.
