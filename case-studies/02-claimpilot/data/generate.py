"""ClaimPilot synthetic claim-document generator.

Generates (document_text, ground_truth_json) pairs with exact, machine-checkable
ground truth: every non-null field is rendered into the text in one of several
format variants, every null field is genuinely absent. All data is fake
(Faker); no real personal data anywhere.

Six document families, format/noise variation, multiple currencies, varied
date formats, ~10-25% missing fields (correct answer: null).

Deterministic: doc seeds derive from a per-split seed base + doc index.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import timedelta
from pathlib import Path

from faker import Faker

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
FAKE = Faker()
FAKE.seed_instance(20261017)

FAMILIES = [
    "invoice_formal",
    "email_thread",
    "ocr_scan",
    "auto_repair_estimate",
    "medical_bill",
    "fax_memo",
]

CLAIM_TYPE_WEIGHTS = [
    ("medical", 0.38),
    ("auto", 0.32),
    ("property", 0.15),
    ("travel", 0.08),
    ("other", 0.07),
]

CURRENCIES = ["USD", "USD", "USD", "EUR", "EUR", "GBP", "CHF"]
CUR_SYMBOL = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "CHF": "CHF"}

ICD10 = ["M54.5", "S83.511", "J06.9", "R51.9", "K21.9", "L03.115", "S93.401", "H10.9", "M25.561", "S61.401"]
DAMAGE_CODES = ["D-214", "BMP-31A", "COL-REAR-02", "WTR-INT-07", "HAIL-04", "FIRE-ENG-01", "SCR-DOOR-L", "GLASS-WS"]

MED_ITEMS = [
    "Emergency department visit, level 3",
    "X-ray, 2 views",
    "Physical therapy session (45 min)",
    "MRI without contrast",
    "Laboratory panel, CBC",
    "Physician consultation",
    "Wound dressing and supplies",
    "Medication administration",
]
AUTO_ITEMS = [
    "Rear bumper cover",
    "Paint and refinish, rear panel",
    "Body labor (4.0 hrs)",
    "Wheel alignment",
    "Tail lamp assembly",
    "ADAS recalibration",
    "Radiator support bracket",
]
GENERIC_ITEMS = [
    "Replacement parts",
    "Labor (2.5 hrs)",
    "Diagnostic fee",
    "Materials and consumables",
    "Service call-out",
    "Towing and recovery",
    "Disposal fee",
    "Inspection fee",
    "Subcontractor labor",
    "Overtime surcharge",
]

NOTES_POOL = [
    "Insured requested expedited handling.",
    "Second occurrence at the same address.",
    "Adjuster site visit scheduled.",
    "Documents follow by courier.",
    "Customer asked for a call back before 5 pm.",
    "Photos available on request.",
]

PRIVACY_BLOCK = (
    "This document may contain confidential health or financial information. "
    "It is intended solely for the named recipient. If you received it in error, "
    "notify the sender and delete all copies. Processing of this claim is subject "
    "to the policy terms and applicable data protection regulations."
)

# Realistic long-form boilerplate that no model should extract values from.
# Deliberately free of claim references, IBAN-like and VIN-like tokens.
ENVELOPE_SECTIONS = [
    "How your claim is processed\nOur claims team reviews every submission within two business days of receipt. "
    "A case handler is assigned to your claim and will contact you if further documentation is required. "
    "Once the assessment is complete, a settlement decision is issued in writing. ",

    "Data protection notice\nWe process the information in this document for the purpose of assessing and "
    "administering your claim, in line with applicable data protection law. Data is retained only as long as "
    "necessary for the purpose it was collected for, after which it is deleted or anonymised. You may request "
    "a copy of the data we hold about you at any time by contacting our data protection office. ",

    "Fraud prevention\nWe take fraud seriously. Information provided in support of a claim may be checked "
    "against industry databases and shared with law enforcement where required. Submitting false or misleading "
    "information is a criminal offence and may result in the claim being declined and the policy being cancelled. ",

    "Questions about this document\nIf anything in this document is unclear, or if you believe a value has been "
    "captured incorrectly, contact the claims line quoting the claim reference shown on the first page. Please "
    "have the document to hand so that our team can locate your file quickly. ",

    "Coverage conditions summary\nSettlement is subject to the terms, conditions, limitations and exclusions of "
    "the policy under which the claim is made. Amounts shown may be subject to a policy excess or a deductible "
    "that is applied at settlement stage. Where the claim relates to a total loss, separate valuation rules apply. ",

    "Office hours and accessibility\nOur offices are open Monday to Friday during normal business hours. "
    "Assistance is available for customers who need documents in alternative formats or who require additional "
    "support when submitting a claim. Please ask when you contact us. ",

    "Document handling note\nThe document you submitted was captured electronically and processed by an "
    "automated intake pipeline. Scanned pages are checked for completeness and legibility; pages that fail "
    "quality checks are flagged for manual review. Reprints of this document can be requested at any time. ",

    "Payment processing\nApproved settlements are issued to the account of record. Payment processing times can "
    "vary depending on the destination bank and the currency of payment. Where payment details on file are no "
    "longer correct, contact the claims team before the settlement is issued. ",

    "Retention schedule\nClaim records are held in line with our regulatory retention schedule. After the "
    "retention period ends, records are destroyed in a controlled manner. Aggregate statistics may be retained "
    "for reporting and actuarial purposes; these do not identify individuals. ",

    "Regulatory information\nThis communication is issued by the handling office named above, which is "
    "authorised and regulated for the conduct of insurance business. Calls to our teams may be recorded for "
    "quality assurance and training purposes. Nothing in this document constitutes legal advice. ",
]

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def pick(rng: random.Random, pairs):
    r = rng.random()
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if r <= acc:
            return value
    return pairs[-1][0]


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------

def render_date(d, rng):
    style = rng.randrange(7)
    if style == 0:
        return d.strftime("%Y-%m-%d")
    if style == 1:
        return d.strftime("%d/%m/%Y")
    if style == 2:
        return d.strftime("%m/%d/%Y")
    if style == 3:
        return f"{d.strftime('%B')} {d.day}, {d.year}"
    if style == 4:
        return f"{d.day} {d.strftime('%b')} {d.year}"
    if style == 5:
        return d.strftime("%d.%m.%Y")
    return d.strftime("%b %d %Y")


def render_amount(cents: int, currency: str, style: int) -> str:
    whole, frac = divmod(cents, 100)
    if style == 0:
        return f"{whole}.{frac:02d}"
    if style == 1:
        return f"{whole:,}.{frac:02d}"
    if style == 2:
        # European style: 1.234,56
        return f"{whole:,}".replace(",", "X").replace(".", ",").replace("X", ".") + f",{frac:02d}"
    euro = f"{whole:,}".replace(",", "X").replace(".", ",").replace("X", ".") + f",{frac:02d}"
    if style == 3:
        return f"{CUR_SYMBOL[currency]}{whole:,}.{frac:02d}"
    if style == 4:
        return f"{CUR_SYMBOL[currency]} {whole:,}.{frac:02d}"
    if style == 5:
        return f"{currency} {whole:,}.{frac:02d}"
    if style == 6:
        return f"{euro} {CUR_SYMBOL[currency]}"
    return f"{currency} {whole:,}.{frac:02d}"


def render_iban(raw: str, rng) -> str:
    if rng.random() < 0.4:
        return raw
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def render_name(name: str, rng) -> str:
    style = rng.randrange(3)
    if style == 0:
        return name
    if style == 1:
        return name.upper()
    return f"Mr./Ms. {name}" if rng.random() < 0.5 else f"Patient: {name}"


def make_vin(rng: random.Random) -> str:
    return "".join(rng.choice(VIN_CHARS) for _ in range(17))


# --------------------------------------------------------------------------
# record builder
# --------------------------------------------------------------------------

def build_record(doc_id: str, rng: random.Random) -> dict:
    claim_type = pick(rng, CLAIM_TYPE_WEIGHTS)

    if claim_type == "medical":
        provider = rng.choice([
            f"St. {FAKE.last_name()} Medical Center",
            f"{FAKE.city()} General Hospital",
            f"{FAKE.last_name()} Clinic",
            f"{FAKE.city()} Urgent Care",
        ])
        code = rng.choice(ICD10) if rng.random() < 0.88 else None
    elif claim_type == "auto":
        provider = rng.choice([
            f"{FAKE.city()} Auto Body & Paint",
            f"{FAKE.last_name()} Collision Center",
            "Precision Auto Repair",
            "Midwest Motor Works",
        ])
        code = rng.choice(DAMAGE_CODES)
    elif claim_type == "property":
        provider = rng.choice([
            f"{FAKE.city()} Restoration Services",
            "HomeShield Contractors",
            "Apex Property Repair",
        ])
        code = rng.choice(DAMAGE_CODES) if rng.random() < 0.7 else None
    elif claim_type == "travel":
        provider = rng.choice([
            "SkyWay Airlines",
            f"Grand {FAKE.city()} Hotel",
            f"{FAKE.city()} Travel Partners",
        ])
        code = None
    else:
        provider = FAKE.company()
        code = None

    currency = rng.choice(CURRENCIES)
    amount_style = rng.randrange(6) if currency != "CHF" else rng.choice([0, 1, 5])

    # line items / total
    has_items = rng.random() > 0.22
    items = None
    total_cents = None
    if has_items:
        pool = MED_ITEMS if claim_type == "medical" else (AUTO_ITEMS if claim_type == "auto" else GENERIC_ITEMS)
        n = rng.randint(1, 6)
        items = []
        subtotal = 0
        for _ in range(n):
            cents = rng.randint(25, 1800) * 100 // 5 * 5
            items.append({"description": rng.choice(pool), "amount": cents})
            subtotal += cents
        if rng.random() < 0.3:
            tax = round(subtotal * rng.uniform(0.05, 0.1))
            items.append({"description": "Tax / VAT", "amount": tax})
            subtotal += tax
        total_cents = subtotal
    elif rng.random() < 0.75:
        total_cents = rng.randint(40, 4200) * 100 // 5 * 5

    fields = {
        "claim_id": doc_id,
        "policy_number": f"POL-{FAKE.lexify('??').upper()}{rng.randint(10, 99)}-{rng.randint(100000, 999999)}",
        "claimant_name": FAKE.name(),
        "date_of_incident": FAKE.date_between(start_date="-3y", end_date="-40d"),
        "date_of_service": None,
        "provider_name": provider,
        "line_items": items,
        "total_amount": total_cents,
        "currency": currency,
        "diagnosis_or_damage_code": code,
        "vehicle_vin": make_vin(rng) if claim_type == "auto" and rng.random() < 0.85 else None,
        "claim_type": claim_type,
        "payout_iban": FAKE.iban().replace(" ", ""),
        "notes": rng.choice(NOTES_POOL) if rng.random() < 0.45 else None,
    }
    if claim_type == "medical" and rng.random() < 0.85:
        fields["date_of_service"] = FAKE.date_between_dates(
            date_start=fields["date_of_incident"],
            date_end=fields["date_of_incident"] + timedelta(days=10),
        )
    elif claim_type in ("auto", "property") and rng.random() < 0.3:
        fields["date_of_service"] = FAKE.date_between_dates(
            date_start=fields["date_of_incident"],
            date_end=fields["date_of_incident"] + timedelta(days=20),
        )

    # null injection (claim_id and claim_type are never null)
    null_p = {
        "policy_number": 0.12,
        "claimant_name": 0.08,
        "date_of_incident": 0.10,
        "date_of_service": 0.55 if fields["date_of_service"] is None else 0.12,
        "provider_name": 0.07,
        "total_amount": 0.15,
        "currency": 0.10,
        "diagnosis_or_damage_code": 0.30 if code is not None else 0.0,
        "payout_iban": 0.15,
        "notes": 0.55 if fields["notes"] is not None else 0.0,
    }
    for key, p in null_p.items():
        if fields[key] is not None and rng.random() < p:
            fields[key] = None

    if fields["currency"] is None:
        amount_style = amount_style % 3  # no currency indicator when the document never states one
    elif fields["currency"] == "CHF":
        amount_style = rng.choice([5, 6])
    else:
        amount_style = 3 + amount_style % 4  # symbol or code renderings
    return fields, {"amount_style": amount_style}


# --------------------------------------------------------------------------
# rendering into text (segment model: literal text vs inserted values)
# --------------------------------------------------------------------------

def render_fields(fields: dict, opts: dict, rng: random.Random) -> dict:
    """Return dict field -> exact rendered string that will appear in the text."""
    r = {}
    if fields["policy_number"]:
        r["policy_number"] = fields["policy_number"]
    if fields["claimant_name"]:
        r["claimant_name"] = render_name(fields["claimant_name"], rng)
    if fields["date_of_incident"]:
        r["date_of_incident"] = render_date(fields["date_of_incident"], rng)
    if fields["date_of_service"]:
        r["date_of_service"] = render_date(fields["date_of_service"], rng)
    if fields["provider_name"]:
        r["provider_name"] = fields["provider_name"]
    if fields["line_items"]:
        r["line_items"] = [
            (it["description"], render_amount(it["amount"], fields["currency"] or "USD", opts["amount_style"]))
            for it in fields["line_items"]
        ]
    if fields["total_amount"] is not None:
        r["total_amount"] = render_amount(fields["total_amount"], fields["currency"] or "USD", opts["amount_style"])
    if fields["diagnosis_or_damage_code"]:
        r["diagnosis_or_damage_code"] = fields["diagnosis_or_damage_code"]
    if fields["vehicle_vin"]:
        r["vehicle_vin"] = fields["vehicle_vin"]
    if fields["payout_iban"]:
        r["payout_iban"] = render_iban(fields["payout_iban"], rng)
    if fields["currency"]:
        r["currency"] = fields["currency"]
    if fields["notes"]:
        r["notes"] = fields["notes"]
    return r


def noise_literal(text: str, level: int, rng: random.Random) -> str:
    if level <= 0 or not text:
        return text
    out = []
    for ch in text:
        if level >= 2 and rng.random() < 0.012:
            out.append({"l": "1", "O": "0", "o": "0", "S": "5", "i": "l", "e": "e "}.get(ch, ch))
        elif level >= 1 and ch == " " and rng.random() < 0.02:
            out.append("  ")
        else:
            out.append(ch)
    text = "".join(out)
    if level >= 2:
        # stray pipe artifacts and broken line wraps
        text = text.replace("\n", " \n" if rng.random() < 0.15 else "\n")
        if rng.random() < 0.5:
            text += "|"
    return text


def emit(segs, fields, r, level, rng) -> str:
    parts = []
    for kind, val in segs:
        parts.append(noise_literal(val, level, rng) if kind == "lit" else val)
    return "".join(parts)


def lit(s):
    return ("lit", s)


def val(v):
    return ("val", v)


def build_text(family: str, fields: dict, r: dict, rng: random.Random) -> tuple[str, int]:
    noise_level = {"ocr_scan": 2, "fax_memo": 1}.get(family, rng.choice([0, 0, 0, 1]))
    ct = fields["claim_type"]
    segs = []
    A = segs.append

    padded = rng.random() < 0.9

    if family == "invoice_formal":
        if "provider_name" in r:
            A(val(r["provider_name"])); A(lit("\n"))
        else:
            A(lit("AMERISAFE CLAIM SERVICES\n"))
        A(lit(f"{'Invoice' if rng.random() < 0.5 else 'Statement'}\n"))
        A(lit("Claim Ref: ")); A(val(fields["claim_id"])); A(lit("\n"))
        if "policy_number" in r:
            A(lit("Policy No.: ")); A(val(r["policy_number"])); A(lit("\n"))
        if "claimant_name" in r:
            A(lit("Insured: ")); A(val(r["claimant_name"])); A(lit("\n"))
        A(lit("Claim category: ")); A(val(ct)); A(lit("\n"))
        if "currency" in r:
            A(lit("Amounts in ")); A(val(r["currency"])); A(lit(".\n"))
        if "date_of_incident" in r:
            A(lit("Date of loss: ")); A(val(r["date_of_incident"])); A(lit("\n"))
        if "date_of_service" in r:
            A(lit("Date of service: ")); A(val(r["date_of_service"])); A(lit("\n"))
        if "diagnosis_or_damage_code" in r:
            A(lit("Code: ")); A(val(r["diagnosis_or_damage_code"])); A(lit("\n"))
        if "vehicle_vin" in r:
            A(lit("VIN: ")); A(val(r["vehicle_vin"])); A(lit("\n"))
        A(lit("\nDESCRIPTION                 AMOUNT\n" + "-" * 40 + "\n"))
        if "line_items" in r:
            for desc, amt in r["line_items"]:
                A(val(f"{desc:<28}")); A(val(amt)); A(lit("\n"))
        else:
            A(lit("Services as agreed under the policy terms and conditions.\n"))
        if "total_amount" in r:
            A(lit("\nTOTAL: ")); A(val(r["total_amount"])); A(lit("\n"))
        if "payout_iban" in r:
            A(lit("Payment to: ")); A(val(r["payout_iban"])); A(lit("\n"))
        if "notes" in r:
            A(lit("Notes: ")); A(val(r["notes"])); A(lit("\n"))
        A(lit("\nPage 1 of 1\n"))

    elif family == "email_thread":
        A(lit(f"From: {FAKE.email() if rng.random() < 0.5 else 'billing@' + FAKE.domain_name()}\n"))
        A(lit("To: claims@claimpilot.example\n"))
        A(lit("Subject: Claim ")); A(val(fields["claim_id"])); A(lit(f" - {ct} - documents attached\n"))
        A(lit("\nHi team,\n\n"))
        A(lit("Please find the details for the claim below.\n\n"))
        A(lit("Claim: ")); A(val(fields["claim_id"])); A(lit("\n"))
        if "policy_number" in r:
            A(lit("Policy: ")); A(val(r["policy_number"])); A(lit("\n"))
        if "claimant_name" in r:
            A(lit("Name: ")); A(val(r["claimant_name"])); A(lit("\n"))
        A(lit("Type: ")); A(val(ct)); A(lit("\n"))
        if "currency" in r:
            A(lit("Amounts stated in ")); A(val(r["currency"])); A(lit(".\n"))
        if "date_of_incident" in r:
            A(lit("Incident: ")); A(val(r["date_of_incident"])); A(lit("\n"))
        if "date_of_service" in r:
            A(lit("Service date: ")); A(val(r["date_of_service"])); A(lit("\n"))
        if "provider_name" in r:
            A(lit("Provider: ")); A(val(r["provider_name"])); A(lit("\n"))
        if "diagnosis_or_damage_code" in r:
            A(lit("Ref code: ")); A(val(r["diagnosis_or_damage_code"])); A(lit("\n"))
        if "vehicle_vin" in r:
            A(lit("VIN: ")); A(val(r["vehicle_vin"])); A(lit("\n"))
        A(lit("\n"))
        if "line_items" in r:
            A(lit("Breakdown:\n"))
            for desc, amt in r["line_items"]:
                A(lit("- ")); A(val(desc)); A(lit(": ")); A(val(amt)); A(lit("\n"))
        if "total_amount" in r:
            A(lit("Total: ")); A(val(r["total_amount"])); A(lit("\n"))
        if "payout_iban" in r:
            A(lit("Payout account: ")); A(val(r["payout_iban"])); A(lit("\n"))
        if "notes" in r:
            A(lit("Note: ")); A(val(r["notes"])); A(lit("\n"))
        A(lit("\nThanks,\n"))
        A(lit(f"{FAKE.first_name()}\n"))
        A(lit("Claims intake\n"))

    elif family == "ocr_scan":
        A(lit("SCANNED DOCUMENT - OCR OUTPUT (semi-automated capture)\n"))
        A(lit("=" * 44 + "\n"))
        A(lit("Doc ref  ")); A(val(fields["claim_id"])); A(lit("\n"))
        A(lit("Type of claim ")); A(val(ct)); A(lit("\n"))
        if "currency" in r:
            A(lit("Currency ")); A(val(r["currency"])); A(lit("\n"))
        if "provider_name" in r:
            A(lit("Issued by ")); A(val(r["provider_name"])); A(lit("\n"))
        if "claimant_name" in r:
            A(lit("Patient or insured ")); A(val(r["claimant_name"])); A(lit("\n"))
        if "policy_number" in r:
            A(lit("Policy ")); A(val(r["policy_number"])); A(lit("\n"))
        if "date_of_incident" in r:
            A(lit("Date of incident ")); A(val(r["date_of_incident"])); A(lit("\n"))
        if "date_of_service" in r:
            A(lit("Date of service ")); A(val(r["date_of_service"])); A(lit("\n"))
        if "diagnosis_or_damage_code" in r:
            A(lit("Code found: ")); A(val(r["diagnosis_or_damage_code"])); A(lit("\n"))
        if "vehicle_vin" in r:
            A(lit("Vehicle identification ")); A(val(r["vehicle_vin"])); A(lit("\n"))
        A(lit("-" * 44 + "\n"))
        if "line_items" in r:
            for desc, amt in r["line_items"]:
                A(val(desc)); A(lit("  ")); A(val(amt)); A(lit("\n"))
        else:
            A(lit("No itemised list captured on this page.\n"))
        if "total_amount" in r:
            A(lit("Amount due ")); A(val(r["total_amount"])); A(lit("\n"))
        if "payout_iban" in r:
            A(lit("Account ")); A(val(r["payout_iban"])); A(lit("\n"))
        if "notes" in r:
            A(lit("Handwritten note: ")); A(val(r["notes"])); A(lit("\n"))
        A(lit("End of OCR capture\n"))

    elif family == "auto_repair_estimate":
        A(lit("VEHICLE REPAIR ESTIMATE\n"))
        if "provider_name" in r:
            A(val(r["provider_name"])); A(lit("\n"))
        else:
            A(lit("Authorised Repair Facility\n"))
        A(lit("Estimate for claim ")); A(val(fields["claim_id"])); A(lit("\n"))
        A(lit("Claim type: ")); A(val(ct)); A(lit("\n"))
        if "currency" in r:
            A(lit("Estimate in ")); A(val(r["currency"])); A(lit("\n"))
        if "vehicle_vin" in r:
            A(lit("VIN: ")); A(val(r["vehicle_vin"])); A(lit("\n"))
        if "policy_number" in r:
            A(lit("Policy: ")); A(val(r["policy_number"])); A(lit("\n"))
        if "claimant_name" in r:
            A(lit("Customer: ")); A(val(r["claimant_name"])); A(lit("\n"))
        if "date_of_incident" in r:
            A(lit("Loss date: ")); A(val(r["date_of_incident"])); A(lit("\n"))
        if "date_of_service" in r:
            A(lit("Estimate date: ")); A(val(r["date_of_service"])); A(lit("\n"))
        if "diagnosis_or_damage_code" in r:
            A(lit("Damage code: ")); A(val(r["diagnosis_or_damage_code"])); A(lit("\n"))
        A(lit("\nPARTS AND LABOR\n"))
        if "line_items" in r:
            for desc, amt in r["line_items"]:
                A(val(desc)); A(lit(" ... ")); A(val(amt)); A(lit("\n"))
        if "total_amount" in r:
            A(lit("Estimate total: ")); A(val(r["total_amount"])); A(lit("\n"))
        if "payout_iban" in r:
            A(lit("Remit to: ")); A(val(r["payout_iban"])); A(lit("\n"))
        if "notes" in r:
            A(lit("Remarks: ")); A(val(r["notes"])); A(lit("\n"))
        A(lit("All work subject to parts availability.\n"))

    elif family == "medical_bill":
        if "provider_name" in r:
            A(val(r["provider_name"])); A(lit("\n"))
        else:
            A(lit("GENERAL HOSPITAL\n"))
        A(lit("STATEMENT OF ACCOUNT\n"))
        A(lit("Claim/billing ref: ")); A(val(fields["claim_id"])); A(lit("\n"))
        A(lit("Claim type: ")); A(val(ct)); A(lit("\n"))
        if "currency" in r:
            A(lit("Charges in ")); A(val(r["currency"])); A(lit("\n"))
        if "claimant_name" in r:
            A(lit("Patient: ")); A(val(r["claimant_name"])); A(lit("\n"))
        if "policy_number" in r:
            A(lit("Insurance policy: ")); A(val(r["policy_number"])); A(lit("\n"))
        if "date_of_service" in r:
            A(lit("Date of service: ")); A(val(r["date_of_service"])); A(lit("\n"))
        if "date_of_incident" in r:
            A(lit("Injury date: ")); A(val(r["date_of_incident"])); A(lit("\n"))
        if "vehicle_vin" in r:
            A(lit("Vehicle VIN: ")); A(val(r["vehicle_vin"])); A(lit("\n"))
        if "diagnosis_or_damage_code" in r:
            A(lit("Diagnosis (ICD-10): ")); A(val(r["diagnosis_or_damage_code"])); A(lit("\n"))
        A(lit("\nSERVICES RENDERED\n"))
        if "line_items" in r:
            for desc, amt in r["line_items"]:
                A(val(desc)); A(lit(": ")); A(val(amt)); A(lit("\n"))
        if "total_amount" in r:
            A(lit("\nTOTAL CHARGES: ")); A(val(r["total_amount"])); A(lit("\n"))
        if "payout_iban" in r:
            A(lit("Electronic payment: ")); A(val(r["payout_iban"])); A(lit("\n"))
        if "notes" in r:
            A(lit("Case note: ")); A(val(r["notes"])); A(lit("\n"))

    else:  # fax_memo
        A(lit("FAX TRANSMISSION\n"))
        A(lit("-" * 32 + "\n"))
        A(lit("TO: CLAIMS DEPARTMENT\n"))
        A(lit("RE: ")); A(val(ct.upper())); A(lit(" CLAIM ")); A(val(fields["claim_id"])); A(lit("\n"))
        if "currency" in r:
            A(lit("AMOUNTS IN ")); A(val(r["currency"])); A(lit("\n"))
        A(lit("-" * 32 + "\n"))
        if "claimant_name" in r:
            A(lit("Claimant: ")); A(val(r["claimant_name"])); A(lit("\n"))
        if "policy_number" in r:
            A(lit("Ref policy ")); A(val(r["policy_number"])); A(lit("\n"))
        if "date_of_incident" in r:
            A(lit("Loss dt ")); A(val(r["date_of_incident"])); A(lit("\n"))
        if "date_of_service" in r:
            A(lit("Service dt ")); A(val(r["date_of_service"])); A(lit("\n"))
        if "provider_name" in r:
            A(lit("From ")); A(val(r["provider_name"])); A(lit("\n"))
        if "diagnosis_or_damage_code" in r:
            A(lit("Code ")); A(val(r["diagnosis_or_damage_code"])); A(lit("\n"))
        if "vehicle_vin" in r:
            A(lit("VIN ")); A(val(r["vehicle_vin"])); A(lit("\n"))
        if "line_items" in r:
            for desc, amt in r["line_items"]:
                A(val(desc)); A(lit(" = ")); A(val(amt)); A(lit("\n"))
        if "total_amount" in r:
            A(lit("TOTAL ")); A(val(r["total_amount"])); A(lit("\n"))
        if "payout_iban" in r:
            A(lit("IBAN ")); A(val(r["payout_iban"])); A(lit("\n"))
        if "notes" in r:
            A(lit("Margin note: ")); A(val(r["notes"])); A(lit("\n"))
        A(lit("CONFIDENTIAL - FOR CLAIMS USE ONLY\n"))

    if padded:
        k = rng.randint(5, 8)
        chosen = rng.sample(ENVELOPE_SECTIONS, k=k)
        if rng.random() < 0.5:
            chosen.append(PRIVACY_BLOCK)
        for sec in chosen:
            A(lit("\n\n" + sec))

    text = emit(segs, fields, r, noise_level, rng).strip() + "\n"
    return text, noise_level


# --------------------------------------------------------------------------
# top-level generation
# --------------------------------------------------------------------------

def canonical_target(fields: dict) -> dict:
    return {
        "claim_id": fields["claim_id"],
        "policy_number": fields["policy_number"],
        "claimant_name": fields["claimant_name"],
        "date_of_incident": fields["date_of_incident"].isoformat() if fields["date_of_incident"] else None,
        "date_of_service": fields["date_of_service"].isoformat() if fields["date_of_service"] else None,
        "provider_name": fields["provider_name"],
        "line_items": (
            [{"description": it["description"], "amount": f"{it['amount'] // 100}.{it['amount'] % 100:02d}"}
             for it in fields["line_items"]]
            if fields["line_items"] else None
        ),
        "total_amount": f"{fields['total_amount'] // 100}.{fields['total_amount'] % 100:02d}" if fields["total_amount"] is not None else None,
        "currency": fields["currency"],
        "diagnosis_or_damage_code": fields["diagnosis_or_damage_code"],
        "vehicle_vin": fields["vehicle_vin"],
        "claim_type": fields["claim_type"],
        "payout_iban": fields["payout_iban"],
        "notes": fields["notes"],
    }


def generate_split(split: str, n: int, seed_base: int) -> list[dict]:
    docs = []
    for i in range(n):
        seed = seed_base + i
        rng = random.Random(seed)
        FAKE.seed_instance(seed ^ 0x5EED)
        doc_id = f"CLM-{2023 + (i % 4)}-{(seed % 900000):06d}"
        family = FAMILIES[i % len(FAMILIES)]
        fields, opts = build_record(doc_id, rng)
        renderings = render_fields(fields, opts, rng)
        text, noise = build_text(family, fields, renderings, rng)
        docs.append({
            "id": f"{split}-{i:05d}",
            "family": family,
            "noise": noise,
            "text": text,
            "target": canonical_target(fields),
            "renderings": {k: ([a for _, a in v] if isinstance(v, list) else v) for k, v in renderings.items()},
        })
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "splits"))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    plans = [
        ("train", 1500, 1_000_000),
        ("val", 100, 2_000_000),
        ("test", 400, 3_000_000),
    ]
    stats = {}
    for split, n, base in plans:
        docs = generate_split(split, n, base)
        path = outdir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        lengths = [len(d["text"]) for d in docs]
        nulls = [sum(1 for v in d["target"].values() if v is None) for d in docs]
        stats[split] = {
            "docs": n,
            "chars_median": sorted(lengths)[n // 2],
            "chars_p10": sorted(lengths)[int(n * 0.1)],
            "chars_p90": sorted(lengths)[int(n * 0.9)],
            "est_tokens_median": sorted(lengths)[n // 2] // 4,
            "nulls_avg": round(sum(nulls) / n, 2),
        }
        print(f"{split}: {n} docs -> {path}")

    (outdir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
