# Test-set verification report — split `test`

- Documents: **400**
- Problems found: **0**
- Families: auto_repair_estimate=67, email_thread=67, fax_memo=66, invoice_formal=67, medical_bill=66, ocr_scan=67
- Null fields: 1454 total, 3.63 avg/doc
- Median doc size: 2566 chars (~641 est. tokens)

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
