# Failure analysis - the winner (Qwen3.5-0.8B QLoRA + xgrammar)

Source: `eval/results/failures_local__Qwen3.5-0.8B__qwen3.5-0.8b-lora_after-ft-xgr.json`
(400-document hand-verified test split). **25 of 400 documents are not exact (93.75% exact).**
Field-level errors in those 25: `date_of_incident` 16, `claimant_name` 7, `date_of_service` 3,
`provider_name` 2, `payout_iban` 2, and one schema-invalid record (below). Money fields: one
`total_amount` miss, and it is the schema-invalid record - every amount the model wrote was right.

## 1. Ambiguous slash dates read in the other convention (16 docs - the dominant error)

| doc | source text | truth | model |
|---|---|---|---|
| test-00029 (fax memo) | `Loss dt 04/05/2024` | 2024-04-05 | 2024-05-04 |
| test-00084 (formal invoice) | `Date of loss: 11/01/2025` | 2025-11-01 | 2025-01-11 |
| test-00169 (email thread) | `Incident: 05/10/2026` | 2026-05-10 | 2026-10-05 |
| test-00342 (formal invoice) | `Date of service: 03/10/2024` | 2024-10-03 | 2024-09-10 * |

`04/05/2024` is April 5 in the US and May 4 in the UK. The document itself does not say which, and
the generator's convention varies by document. This is a business rule, not a model defect: ClaimPilot
should fix the convention per sender/locale (or send the issuer country as an input field) - the
frontier-class baseline proxy makes the same kind of error (date_of_incident F1 0.944 vs the winner's
0.955). (*test-00342 is a genuine slip: it also shifted the day.)

## 2. Names "autocorrected" toward a more common spelling (7 docs)

| doc | truth | model |
|---|---|---|
| test-00022 (medical bill) | Krystal Bean | Kristal Bean |
| test-00042 (formal invoice) | Roberto Lopez | Robert Lopez |
| test-00048 (formal invoice) | Tamara Beck | Tamarae Beck |
| test-00057 (auto repair) | Renee Lewis | Reenie Lewis |

A small model's language prior pulls rare spellings toward familiar ones (and OCR-noise documents make
it worse). Still the best field result in the comparison: claimant_name F1 **0.981** vs 0.903 for the
baseline proxy. Mitigation: cross-check the name against the policy record ClaimPilot already holds
for `policy_number` (a lookup, not a model change).

## 3. Character-level slips in long identifiers (3 docs)

- test-00211 (email): IBAN `GB67WOFF4694...` -> `GB67WWOF4694...` (one letter doubled, one dropped).
- test-00286 (medical bill): provider `Fieldshaven Urgent Care` -> `Fieldhaven Urgent Care`.
- test-00154 (medical bill): VIN `UTVLVSEAW28FEC3W1` -> `UTLVSEAW28FEC3W1` - one character dropped,
  so the VIN is 16 characters, fails the schema's 17-character VIN pattern, and the metric marks the
  *whole record* wrong (every field shows up in the diff even though the other 13 are correct). This is
  the single schema-invalid output (99.75% valid). Mitigation: IBAN and VIN both carry check digits
  (IBAN mod-97; VIN position 9), so a validator can flag these for review before payout - recommended
  regardless of which model runs.

## What this means for the client

None of the 25 is a hallucinated value or a wrong amount: null discipline is 1.000 (the model never
invented a value for a missing field) and total_amount accuracy is 99.5%. The residual errors are
(1) a date convention the documents leave ambiguous, (2) rare-name spelling, (3) one-character slips
in identifiers that check-digit validation catches.
