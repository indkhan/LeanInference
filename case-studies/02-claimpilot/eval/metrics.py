"""Score one extraction run against ground truth.

Usage:
  python eval/metrics.py --run eval/results/results_<slug>_<tag>.jsonl [--split test]

Produces eval/results/summary_<slug>_<tag>.json (all metrics + per-field
breakdown + per-family breakdown) and failures_<slug>_<tag>.json (detailed
failure cases for the write-up).

Metric conventions (documented so every number is reproducible):
  - Each field is scored as an extraction decision over the whole split.
  - TP: truth non-null and prediction matches it after normalization.
  - FP: prediction non-null and does not match truth.
  - FN: truth non-null and prediction does not match it.
  - TN: truth null and prediction null (correct abstention).
    A wrong non-null value on a non-null truth counts as both FP and FN.
  - field accuracy = (TP+TN)/N over all documents (null is a class).
  - overall F1 = micro-F1 over fields; macro-F1 = mean of per-field F1.
  - line_items: one field decision (multiset of (description, amount));
    item-level P/R also reported.
  - whole-document exact match: every field matches.
  - valid JSON+schema rate: parsed output validates against data/schema.json.
  - null discipline: share of truly-null fields where the model also said null.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import jsonschema

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CASE = ROOT.parent
SPLITS = CASE / "data" / "splits"
RESULTS = ROOT / "results"
SCHEMA = json.loads((CASE / "data" / "schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)

SCALAR_FIELDS = [
    "claim_id", "policy_number", "claimant_name", "date_of_incident", "date_of_service",
    "provider_name", "total_amount", "currency", "diagnosis_or_damage_code",
    "vehicle_vin", "claim_type", "payout_iban", "notes",
]
ALL_FIELDS = SCALAR_FIELDS + ["line_items"]
CLAIM_TYPES = ["auto", "medical", "property", "travel", "other"]

CUR_SYMS = ["$", "\u20ac", "\u00a3"]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%B %d, %Y", "%d %b %Y", "%b %d %Y"]


def norm_text(s) -> str:
    return " ".join(str(s).split()).casefold().strip()


def norm_id(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def norm_currency(s):
    if s is None:
        return None
    t = str(s).upper().strip()
    for cur in ["USD", "EUR", "GBP", "CHF"]:
        if cur in t:
            return cur
    return t


def norm_amount(s):
    """Return integer cents from any common rendering, or None if unparseable."""
    if s is None:
        return None
    t = str(s).strip()
    for sym in CUR_SYMS:
        t = t.replace(sym, "")
    t = re.sub(r"(?i)\b(usd|eur|gbp|chf)\b", "", t).strip()
    t = t.replace(" ", "").replace("\u00a0", "")
    t = t.replace("(", "-").replace(")", "")
    if not t or not re.search(r"\d", t):
        return None
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        t = t.replace(",", "").replace(".", "") if False else t
        # keep only the last separator as decimal point
        if dec == ",":
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        parts = t.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    elif "." in t:
        parts = t.split(".")
        if len(parts) == 2 and len(parts[1]) not in (1, 2):
            t = t.replace(".", "")
    try:
        return int((Decimal(t) * 100).to_integral_value())
    except InvalidOperation:
        return None


def norm_date(s):
    if s is None:
        return None
    t = str(s).strip()
    for fmt in DATE_FORMATS:
        try:
            import datetime
            return datetime.datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return norm_text(t)


def norm_claim_type(s):
    if s is None:
        return None
    t = norm_text(s)
    if t in CLAIM_TYPES:
        return t
    hits = [c for c in CLAIM_TYPES if c in t]
    if len(hits) == 1:
        return hits[0]
    return t


def norm_generic(s):
    return norm_text(s)


NORM = {
    "claim_id": norm_id,
    "policy_number": norm_id,
    "claimant_name": norm_generic,
    "date_of_incident": norm_date,
    "date_of_service": norm_date,
    "provider_name": norm_generic,
    "total_amount": norm_amount,
    "currency": norm_currency,
    "diagnosis_or_damage_code": lambda s: norm_text(s).upper() if s is not None else None,
    "vehicle_vin": norm_id,
    "claim_type": norm_claim_type,
    "payout_iban": norm_id,
    "notes": norm_generic,
}


def field_match(field: str, pred, truth) -> bool:
    if field == "line_items":
        return items_match(pred, truth)
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    return NORM[field](pred) == NORM[field](truth)


def items_match(pred, truth) -> bool:
    if pred is None and truth is None:
        return True
    if pred is None or truth is None or not isinstance(pred, list):
        return False
    def key(it):
        if not isinstance(it, dict):
            return ("?", None)
        d = it.get("description")
        a = it.get("amount")
        return (norm_text(d) if d is not None else None, norm_amount(a))
    return Counter(map(key, pred)) == Counter(map(key, truth))


def items_pr(pred, truth):
    if not isinstance(pred, list) or not isinstance(truth, list):
        return 0, 0, len(truth) if isinstance(truth, list) else 0
    def key(it):
        if not isinstance(it, dict):
            return ("?", None)
        d = it.get("description")
        a = it.get("amount")
        return (norm_text(d) if d is not None else None, norm_amount(a))
    pc, tc = Counter(map(key, pred)), Counter(map(key, truth))
    tp = sum((pc & tc).values())
    fp = sum((pc - tc).values())
    fn = sum((tc - pc).values())
    return tp, fp, fn


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 7) if xs else None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids-file", help="score only these doc ids (e.g. data/splits/test_subset50.txt); "
                    "outputs get the file's stem as a suffix")
    args = ap.parse_args()

    run_file = Path(args.run)
    if not run_file.is_absolute():
        run_file = CASE / run_file
    records = [json.loads(l) for l in run_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids_file:
        keep = set((CASE / args.ids_file).read_text(encoding="utf-8").split())
        records = [r for r in records if r["doc_id"] in keep]
    targets = {json.loads(l)["id"]: json.loads(l) for l in
               (SPLITS / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    stats = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in ALL_FIELDS}
    item_tp = item_fp = item_fn = 0
    n = 0
    doc_exact = 0
    valid_schema = 0
    valid_json_count = 0
    null_expected = 0
    null_kept = 0
    fam_stats = {}
    failures = []

    for rec in records:
        doc = targets.get(rec["doc_id"])
        if doc is None:
            continue
        n += 1
        truth = doc["target"]
        pred = rec.get("parsed")
        fam = rec["family"]
        fam_stats.setdefault(fam, {"n": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        fam_stats[fam]["n"] += 1

        schema_ok = False
        if isinstance(pred, dict):
            valid_json_count += 1
            schema_ok = VALIDATOR.is_valid(pred)
        if schema_ok:
            valid_schema += 1

        diffs = []
        for field in ALL_FIELDS:
            t = truth.get(field)
            p = pred.get(field) if isinstance(pred, dict) else None
            m = schema_ok and field_match(field, p, t)
            if field == "line_items":
                itp, ifp, ifn = items_pr(p, t)
                item_tp += itp
                item_fp += ifp
                item_fn += ifn
            if m:
                if t is None and p is None:
                    stats[field]["tn"] += 1
                    fam_stats[fam]["tn"] += 1
                elif t is not None:
                    stats[field]["tp"] += 1
                    fam_stats[fam]["tp"] += 1
            else:
                if p is not None:
                    stats[field]["fp"] += 1
                    fam_stats[fam]["fp"] += 1
                if t is not None:
                    stats[field]["fn"] += 1
                    fam_stats[fam]["fn"] += 1
                diffs.append({"field": field, "truth": t, "pred": p})
            if t is None:
                null_expected += 1
                if p is None:
                    null_kept += 1

        if not diffs:
            doc_exact += 1
            fam_stats[fam]["exact"] += 1
        elif len(failures) < 60:
            failures.append({
                "doc_id": rec["doc_id"], "family": fam, "error": rec.get("error"),
                "diffs": diffs,
                "raw": (rec.get("raw") or "")[:1200],
            })

    per_field = {}
    tp_all = fp_all = fn_all = tn_all = 0
    for field, s in stats.items():
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        acc = (s["tp"] + s["tn"]) / max(1, s["tp"] + s["tn"] + s["fp"] + s["fn"])
        per_field[field] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "accuracy": round(acc, 4), "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "tn": s["tn"],
        }
        tp_all += s["tp"]; fp_all += s["fp"]; fn_all += s["fn"]; tn_all += s["tn"]

    micro_p, micro_r, micro_f = prf(tp_all, fp_all, fn_all)
    macro_f = statistics.fmean(v["f1"] for v in per_field.values())
    ip, ir, _ = prf(item_tp, item_fp, item_fn)

    lats = sorted(r["latency_s"] for r in records)
    tin = [r["tokens_in_est"] for r in records]
    tout = [r["tokens_out_est"] for r in records]

    family_out = {}
    for fam, s in sorted(fam_stats.items()):
        _, _, ff = prf(s["tp"], s["fp"], s["fn"])
        family_out[fam] = {
            "docs": s["n"], "doc_exact_rate": round(s["exact"] / max(1, s["n"]), 4),
            "micro_f1": round(ff, 4),
        }

    summary = {
        "model": records[0]["model"] if records else None,
        "tag": records[0]["tag"] if records else None,
        "transport": records[0]["transport"] if records else None,
        "run_file": str(run_file.relative_to(CASE)),
        "split": args.split,
        "docs": n,
        "parsed_ok": valid_json_count,
        "valid_schema_rate": round(valid_schema / max(1, n), 4),
        "doc_exact_rate": round(doc_exact / max(1, n), 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f, 4),
        "macro_f1": round(macro_f, 4),
        "item_micro_precision": round(ip, 4),
        "item_micro_recall": round(ir, 4),
        "null_discipline": round(null_kept / max(1, null_expected), 4),
        "total_amount_accuracy": per_field["total_amount"]["accuracy"],
        "latency_p50": lats[len(lats) // 2] if lats else None,
        "latency_p95": lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else None,
        "tokens_in_mean": round(statistics.fmean(tin), 1) if tin else None,
        "tokens_out_mean": round(statistics.fmean(tout), 1) if tout else None,
        # provider-measured (OpenRouter usage); None for runs without it
        "usage_tokens_in_mean": _mean(r.get("usage", {}).get("tokens_in") for r in records),
        "usage_tokens_out_mean": _mean(r.get("usage", {}).get("tokens_out") for r in records),
        "cost_usd_mean": _mean(r.get("cost_usd") for r in records),
        "cost_usd_total": round(sum(r.get("cost_usd") or 0 for r in records), 4),
        "per_field": per_field,
        "per_family": family_out,
        "errors": sum(1 for r in records if r.get("error")),
    }

    slug = run_file.stem.replace("results_", "") + (f"__{Path(args.ids_file).stem}" if args.ids_file else "")
    out = RESULTS / f"summary_{slug}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / f"failures_{slug}.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_field", "per_family")}, indent=2))
    print("\nper-field F1:")
    for f, v in per_field.items():
        print(f"  {f:26s} f1={v['f1']:.3f}  acc={v['accuracy']:.3f}  tp={v['tp']} fp={v['fp']} fn={v['fn']} tn={v['tn']}")
    print(f"\nwrote {out.relative_to(CASE)}")


if __name__ == "__main__":
    main()
