"""Round-trip verification of generated splits.

Every ground-truth value in a generated document must be provably present in
the document text (in the exact rendering used), and every null field must be
provably absent evidence-wise. This is construction verification: it proves the
synthetic ground truth is consistent, it does not replace human review.

Checks per document:
  1. target validates against data/schema.json (Draft 2020-12, formats on)
  2. every rendered value appears verbatim in the text
  3. evidence counts: exactly one claim id; a policy/VIN/IBAN pattern exists
     only when that field is non-null
  4. total_amount == sum(item amounts) when both are present
  5. doc ids and claim ids are unique across the split
  6. claimant_name is present case-insensitively

Writes <split>.report.json + verify_<split>_report.md next to the split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SCHEMA = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)

CLAIM_RE = re.compile(r"CLM-\d{4}-\d{6}")
POLICY_RE = re.compile(r"\bPOL-[A-Z]{2}\d{2}-\d{6}\b")
VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b")


def verify_doc(doc: dict, problems: list[str]) -> None:
    did = doc["id"]
    target = doc["target"]
    text = doc["text"]
    rend = doc["renderings"]

    for err in VALIDATOR.iter_errors(target):
        problems.append(f"{did}: schema violation {list(err.path)}: {err.message}")

    # 1. rendered values present verbatim
    for field, rendered in rend.items():
        if field == "line_items":
            for amt in rendered:
                if amt not in text:
                    problems.append(f"{did}: rendered {field} amount {amt!r} not found in text")
        elif isinstance(rendered, str):
            if rendered not in text:
                problems.append(f"{did}: rendered {field} {rendered!r} not found in text")

    # 2. name present case-insensitively
    if target["claimant_name"]:
        if target["claimant_name"].casefold() not in text.casefold():
            problems.append(f"{did}: claimant_name {target['claimant_name']!r} not found (casefold)")

    # 2b. claim_type / currency evidence (word-bounded)
    ct = target["claim_type"]
    if not re.search(rf"\b{re.escape(ct)}\b", text, re.IGNORECASE):
        problems.append(f"{did}: claim_type {ct!r} not stated in text")
    if target["currency"] and not re.search(rf"\b{re.escape(target['currency'])}\b", text):
        problems.append(f"{did}: currency {target['currency']!r} not stated in text")
    if not target["currency"] and re.search(r"\b(USD|EUR|GBP|CHF)\b", text):
        problems.append(f"{did}: currency code found although currency is null")

    # 3. evidence counts
    claim_hits = CLAIM_RE.findall(text)
    if not claim_hits or set(claim_hits) != {target["claim_id"]}:
        problems.append(f"{did}: expected only claim id {target['claim_id']!r}, found {claim_hits}")

    pol_hits = POLICY_RE.findall(text)
    expect_pol = 1 if target["policy_number"] else 0
    if len(pol_hits) != expect_pol:
        problems.append(f"{did}: policy pattern count {len(pol_hits)} != {expect_pol} (policy={target['policy_number']!r})")

    vin_hits = VIN_RE.findall(text)
    expect_vin = 1 if target["vehicle_vin"] else 0
    if (1 if vin_hits else 0) != expect_vin:
        problems.append(f"{did}: VIN-like pattern count {len(vin_hits)} vs expected {expect_vin} (vin={target['vehicle_vin']!r})")
    if target["vehicle_vin"] and vin_hits and vin_hits[0] != target["vehicle_vin"]:
        problems.append(f"{did}: VIN in text {vin_hits[0]!r} != target {target['vehicle_vin']!r}")

    # IBAN: compare on text with whitespace removed (spaced renderings)
    compact = re.sub(r"\s+", "", text)
    if target["payout_iban"]:
        if target["payout_iban"] not in compact:
            problems.append(f"{did}: IBAN {target['payout_iban']!r} not found (whitespace-stripped)")
    else:
        if IBAN_RE.search(text):
            problems.append(f"{did}: IBAN-looking token found although payout_iban is null")

    # 4. money consistency
    if target["line_items"] and target["total_amount"]:
        item_cents = sum(round(float(it["amount"]) * 100) for it in target["line_items"])
        if item_cents != round(float(target["total_amount"]) * 100):
            problems.append(
                f"{did}: line items sum {item_cents} != total {round(float(target['total_amount']) * 100)}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    path = ROOT / "splits" / f"{args.split}.jsonl"
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    problems: list[str] = []
    ids = [d["id"] for d in docs]
    claims = [d["target"]["claim_id"] for d in docs]
    if len(set(ids)) != len(ids):
        problems.append("split: duplicate doc ids")
    if len(set(claims)) != len(claims):
        problems.append("split: duplicate claim ids")

    for d in docs:
        verify_doc(d, problems)

    families = {}
    nulls = 0
    chars = []
    for d in docs:
        families[d["family"]] = families.get(d["family"], 0) + 1
        nulls += sum(1 for v in d["target"].values() if v is None)
        chars.append(len(d["text"]))

    report = {
        "split": args.split,
        "docs": len(docs),
        "problems": len(problems),
        "problem_examples": problems[:20],
        "families": families,
        "nulls_total": nulls,
        "nulls_avg_per_doc": round(nulls / max(1, len(docs)), 2),
        "chars_p50": sorted(chars)[len(chars) // 2] if chars else 0,
        "est_tokens_p50": (sorted(chars)[len(chars) // 2] // 4) if chars else 0,
    }
    (ROOT / "splits" / f"verify_{args.split}.report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Test-set verification report — split `{args.split}`",
        "",
        f"- Documents: **{len(docs)}**",
        f"- Problems found: **{len(problems)}**",
        f"- Families: {', '.join(f'{k}={v}' for k, v in sorted(families.items()))}",
        f"- Null fields: {report['nulls_total']} total, {report['nulls_avg_per_doc']} avg/doc",
        f"- Median doc size: {report['chars_p50']} chars (~{report['est_tokens_p50']} est. tokens)",
        "",
        "## Checks performed",
        "",
        "1. Every ground-truth object validates against `data/schema.json` (Draft 2020-12, format checks on).",
        "2. Every non-null field's rendered form appears verbatim in the document text.",
        "3. Evidence counts: exactly one claim id; policy/VIN/IBAN patterns exist only when the field is non-null.",
        "4. `total_amount` equals the sum of `line_items` when both are present.",
        "5. Doc ids and claim ids unique within the split.",
        "",
        "**Scope honestly stated:** these documents are synthetic, so their ground truth is verified by",
        "construction (generator renderings round-tripped against the text). This is not 400-document human",
        "annotation. Human-style review of a sample and of all ambiguous cases is done separately.",
        "",
    ]
    if problems:
        lines += ["## Problems (first 20)", ""] + [f"- {p}" for p in problems[:20]]
    else:
        lines += ["## Result", "", "All checks passed."]

    (ROOT / "splits" / f"verify_{args.split}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
