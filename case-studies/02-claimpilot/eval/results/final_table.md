# ClaimPilot extraction - results (400-document hand-verified test split)

Prices: `eval/prices.json` (retrieved 2026-10-17). Volume: 500,000 docs/month.

| Model | Field F1 (before -> after) | Doc exact | Valid JSON | Amount acc | p95 (s) | $/1K docs | Monthly $ | Runs where | Docs |
|---|---|---|---|---|---|---|---|---|---|
| Status quo: Claude Opus 5.5 | assumed* | - | - | - | - | 10.26 | 5,132 | third-party API | not run |
| Baseline proxy: beam-free | n/a (no FT) -> 0.974 | 69.2% | 100.0% | 95.8% | 20.0 | 9.47 | 4,733 | third-party API | 400 |
| Candidate A: DeepSeek V4.1 Flash | n/a (no FT) -> 0.967 | 63.0% | 99.8% | 97.3% | 25.9 | 0.60 | 300 | third-party API | 400 |
| Candidate B: Qwen3.5-0.8B QLoRA + xgrammar JSON | 0.366 -> 0.991 | 93.8% | 99.8% | 99.5% | 23.8 | 1.72 | 861 | self-hosted GPU (own cloud) | 400 |
| Candidate C: Needle 3 LoRA (.cact, full KV window) | 0.000 -> 0.776 | 16.2% | 85.0% | 69.0% | 11.2 | 0.52 | 261 | self-hosted CPU / on-device | 400 |

\* Status-quo accuracy is ASSUMED to be at least the measured baseline proxy; the nearest published figure is Claude Opus 5.5 = 94.1% on Roboflow Vision Evals 'OCR - Structured JSON' (high effort), an image-OCR task, not our text-extraction task, so it is cited for context only. Source: https://playground.roboflow.com/models/anthropic/claude-opus-5-5 (Vision Evals updated 2026-10-13) (retrieved 2026-10-19). Not measured here ($0 budget, no paid frontier access); the measured bar is the baseline proxy row. Status-quo cost = Claude Opus 5.5 list price ($4.0/$20.0 per MTok) x the brief's 1,500 input tokens + the proxy's measured output tokens.

Cost basis per row:
- Baseline proxy: beam-free: priced as Claude Opus 5.5 list price x measured tokens
- Candidate A: DeepSeek V4.1 Flash: DeepSeek V4.1 Flash peak list price x measured tokens
- Candidate B: Qwen3.5-0.8B QLoRA + xgrammar JSON: 2 x NVIDIA L4 24GB (Runpod Secure Cloud, on-demand) ($0.59/h) always-on at measured p50 6.07s/doc
- Candidate C: Needle 3 LoRA (.cact, full KV window): 2 x AWS c7i.xlarge (4 vCPU, 8 GiB, us-east-1 on-demand Linux) ($0.1785/h) always-on at measured p50 5.64s/doc

## Per-field F1 (final version of each system)

| Field | Baseline proxy | Candidate A | Candidate B | Candidate C |
|---|---|---|---|---|
| claim_id | 1.000 | 0.998 | 0.998 | 0.914 |
| policy_number | 1.000 | 0.997 | 0.997 | 0.905 |
| claimant_name | 0.903 | 0.884 | 0.981 | 0.672 |
| date_of_incident | 0.944 | 0.955 | 0.955 | 0.736 |
| date_of_service | 0.942 | 0.952 | 0.988 | 0.692 |
| provider_name | 0.984 | 0.976 | 0.995 | 0.884 |
| line_items | 0.998 | 0.991 | 0.997 | 0.336 |
| total_amount | 0.974 | 0.983 | 0.997 | 0.786 |
| currency | 1.000 | 0.992 | 0.997 | 0.867 |
| diagnosis_or_damage_code | 1.000 | 0.996 | 0.996 | 0.844 |
| vehicle_vin | 1.000 | 0.990 | 0.990 | 0.817 |
| claim_type | 1.000 | 0.998 | 0.998 | 0.914 |
| payout_iban | 1.000 | 0.997 | 0.994 | 0.633 |
| notes | 0.709 | 0.587 | 1.000 | 0.779 |
| null discipline | 0.950 | 0.941 | 1.000 | 0.955 |

## Winner (brief Step 4 rule)

Bar = baseline proxy: micro-F1 0.9738, total_amount acc 95.8%.
- Candidate B: Qwen3.5-0.8B QLoRA + xgrammar JSON: F1 0.9911 (>= bar), amount 99.5% (>= bar)
- Candidate C: Needle 3 LoRA (.cact, full KV window): F1 0.7759 (< bar), amount 69.0% (< bar)

**Winner: Candidate B: Qwen3.5-0.8B QLoRA + xgrammar JSON**

## Failure cases - Candidate B: Qwen3.5-0.8B QLoRA + xgrammar JSON (25 of 400 docs not exact)

- `test-00022` (medical_bill)
  - claimant_name: truth `"Krystal Bean"` pred `"Kristal Bean"`
- `test-00029` (fax_memo)
  - date_of_incident: truth `"2024-04-05"` pred `"2024-05-04"`
- `test-00154` (medical_bill)
  - claim_id: truth `"CLM-2025-300154"` pred `"CLM-2025-300154"`
  - policy_number: truth `"POL-QD33-293852"` pred `"POL-QD33-293852"`
  - claimant_name: truth `"Stephen Gordon"` pred `"Stephen Gordon"`
  - date_of_incident: truth `"2024-06-24"` pred `"2024-06-24"`
  - date_of_service: truth `null` pred `null`
  - provider_name: truth `"Tracyshire Auto Body & Paint"` pred `"Tracyshire Auto Body & Paint"`
  - total_amount: truth `"898.00"` pred `"898.00"`
  - currency: truth `"USD"` pred `"USD"`
  - diagnosis_or_damage_code: truth `"D-214"` pred `"D-214"`
  - vehicle_vin: truth `"UTVLVSEAW28FEC3W1"` pred `"UTLVSEAW28FEC3W1"`
  - claim_type: truth `"auto"` pred `"auto"`
  - payout_iban: truth `"GB79JTVP96187197963834"` pred `"GB79JTVP96187197963834"`
  - notes: truth `null` pred `null`
  - line_items: truth `[{"description": "Tail lamp assembly", "amount": "898.00"}]` pred `[{"description": "Tail lamp assembly", "amount": "898.00"}]`
- `test-00211` (email_thread)
  - payout_iban: truth `"GB67WOFF46949981697780"` pred `"GB67WWOF46949981697780"`
- `test-00285` (auto_repair_estimate)
  - date_of_incident: truth `"2024-04-10"` pred `"2024-10-04"`
  - date_of_service: truth `"2024-04-10"` pred `"2024-10-04"`
- `test-00286` (medical_bill)
  - provider_name: truth `"Fieldshaven Urgent Care"` pred `"Fieldhaven Urgent Care"`
- `test-00342` (invoice_formal)
  - date_of_service: truth `"2024-10-03"` pred `"2024-09-10"`
- `test-00032` (ocr_scan)
  - claimant_name: truth `"Jonathan Peck"` pred `"Jonathan Pack"`
- `test-00042` (invoice_formal)
  - claimant_name: truth `"Roberto Lopez"` pred `"Robert Lopez"`
- `test-00044` (ocr_scan)
  - date_of_incident: truth `"2026-08-07"` pred `"2026-07-08"`

