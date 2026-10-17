# Test-set verification report — split `train`

- Documents: **1500**
- Problems found: **0**
- Families: auto_repair_estimate=250, email_thread=250, fax_memo=250, invoice_formal=250, medical_bill=250, ocr_scan=250
- Null fields: 5363 total, 3.58 avg/doc
- Median doc size: 2559 chars (~639 est. tokens)

## Checks performed

1. Every ground-truth object validates against `data/schema.json` (Draft 2020-12, format checks on).
2. Every non-null field's rendered form appears verbatim in the document text.
3. Evidence counts: exactly one claim id; policy/VIN/IBAN patterns exist only when the field is non-null.
4. `total_amount` equals the sum of `line_items` when both are present.
5. Doc ids and claim ids unique within the split.

**Scope honestly stated:** these documents are synthetic, so their ground truth is verified by
construction (generator renderings round-tripped against the text). This is not 400-document human
annotation. Human-style review of a sample and of all ambiguous cases is done separately.

## Result

All checks passed.
