"""Score a results file against a split. Implements FORMAT.md "Scoring normalization (v1)".

Usage:
  uv run python eval/score.py --results eval/results/mock-pipeline.jsonl
  uv run python eval/score.py --results eval/results/qwen3.5-2b-ft.jsonl --strict

Definitions (v1):
  overall          - full response correct (call name+args exact / clarify asks a question /
                     none emits no call), averaged over all examples
  tool_selection   - over gold=call: predicted call-set names match exactly (non-call = wrong)
  args_full        - among name-matched calls: all gold args present & equal, no extras
  args_values      - per-argument value accuracy among name-matched calls
  no_tool          - gold=none examples where no call is emitted
  clarify          - gold=clarify examples with no call and a question
  parse_validity   - strict-JSON share (call/clarify/none); non-JSON counts as wrong

Normalization: trim/collapse whitespace; strings case-insensitive (IDs uppercase);
numbers as numbers; null == missing; dates stay ISO strings.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[A-Za-z]{2,5}-\d{3,5}$")
QUESTION_HINTS = ("which", "what ", "could you", "can you provide", "please provide",
                  "please share", "please tell", "specify", "missing", "need to know",
                  "unclear", "provide the")


def norm_value(value):
    """FORMAT.md normalization applied to one argument value (recursive)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, str):
        s = " ".join(value.split())
        if ID_RE.match(s):
            return ("id", s.upper())
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return ("date", s)
        return ("str", s.casefold())
    if isinstance(value, dict):
        return ("obj", tuple(sorted(
            (k, norm_value(v)) for k, v in value.items() if v is not None)))
    if isinstance(value, list):
        return ("list", tuple(norm_value(v) for v in value))
    return ("raw", str(value))


def norm_args(args: dict) -> dict:
    return {k: norm_value(v) for k, v in (args or {}).items() if v is not None}


def compare_args(gold_args: dict, pred_args: dict):
    g, p = norm_args(gold_args), norm_args(pred_args)
    missing = sorted(set(g) - set(p))
    extra = sorted(set(p) - set(g))
    wrong = sorted(k for k in set(g) & set(p) if g[k] != p[k])
    wrong_dates = [k for k in wrong if k == "date" or k.endswith("_date")]
    return missing, extra, wrong, wrong_dates


def looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.endswith("?") or any(h in t for h in QUESTION_HINTS)


def evaluate_one(gold: dict, prediction, status: str):
    """Returns (correct: bool, category: str, detail: dict)."""
    gtype = gold.get("type")
    pred = prediction or {}
    is_call = status == "json_call" and pred.get("type") == "call" and pred.get("calls")

    if status == "adapter_error":
        return False, "adapter_error", {}
    if status == "invalid_json":
        return False, "invalid_output", {}

    if gtype == "call":
        gcall = gold["calls"][0]
        if not is_call:
            if status == "json_clarify":
                return False, "clarified_instead_of_call", {}
            if status == "json_none":
                return False, "none_instead_of_call", {}
            return False, "no_call_unparsed", {}
        names = [c["name"] for c in pred["calls"]]
        if sorted(names) != sorted([c["name"] for c in gold["calls"]]):
            return False, "wrong_tool", {"gold_tool": gcall["name"], "pred_tool": ",".join(names)}
        # single-call comparison (v1)
        pcall = pred["calls"][0]
        missing, extra, wrong, wrong_dates = compare_args(gcall.get("arguments", {}),
                                                          pcall.get("arguments", {}))
        detail = {"missing": missing, "extra": extra, "wrong": wrong,
                  "gold_tool": gcall["name"]}
        if not missing and not extra and not wrong:
            return True, "correct", detail
        if missing:
            return False, "missing_args", detail
        if extra:
            return False, "extra_args", detail
        if wrong and wrong_dates and len(wrong_dates) == len(wrong):
            return False, "wrong_date", detail
        return False, "wrong_value", detail

    if gtype == "clarify":
        if is_call:
            return False, "called_instead_of_clarify", {}
        if status == "json_clarify":
            return True, "correct", {}
        if status == "json_none":
            return False, "none_instead_of_clarify", {}
        if status == "free_text":
            if looks_like_question(pred.get("text", "")):
                return True, "correct", {}
            return False, "no_question", {}
        return False, "invalid_output", {}

    # gtype == "none"
    if is_call:
        return False, "called_when_none", {}
    return True, "correct", {}


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", default=str(ROOT / "data" / "splits" / "test.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if the results file does not cover the whole split")
    args = ap.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path

    res = {json.loads(l)["id"]: json.loads(l)
           for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    gold = {r["id"]: r for r in
            (json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    missing_ids = [i for i in gold if i not in res]
    if missing_ids and args.strict:
        print(f"ERROR: results cover {len(gold) - len(missing_ids)}/{len(gold)} examples "
              f"({len(missing_ids)} missing)")
        return 1
    joined = [(gold[i], res[i]) for i in gold if i in res]
    model = next(iter(res.values()))["model"] if res else "?"
    adapter = next(iter(res.values()))["adapter"] if res else "?"

    n = len(joined)
    correct = 0
    taxonomy = Counter()
    per_bucket = {}
    per_language = {}
    per_tool = {}
    call_total = tool_sel_ok = args_full_ok = 0
    args_ok = args_total = 0
    missing_total = extra_total = wrong_total = wrong_date_total = 0
    clarify_total = clarify_ok = none_total = none_ok = 0
    parse_ok = 0
    status_counts = Counter()
    latencies = []
    usage_tot = Counter()

    for g, r in joined:
        ok, category, detail = evaluate_one(g["gold"], r.get("prediction"), r["parse_status"])
        correct += ok
        if category != "correct":
            taxonomy[category] += 1
        status_counts[r["parse_status"]] += 1
        if r["parse_status"] in ("json_call", "json_clarify", "json_none"):
            parse_ok += 1
        if r.get("latency_ms"):
            latencies.append(r["latency_ms"])
        if r.get("usage"):
            for k, v in r["usage"].items():
                if isinstance(v, (int, float)):
                    usage_tot[k] += v

        b = per_bucket.setdefault(g["bucket"], {"n": 0, "ok": 0, "errors": Counter()})
        b["n"] += 1
        b["ok"] += ok
        if not ok:
            b["errors"][category] += 1
        lang = per_language.setdefault(g["language"], {"n": 0, "ok": 0})
        lang["n"] += 1
        lang["ok"] += ok

        if g["gold"]["type"] == "call":
            gcall = g["gold"]["calls"][0]
            t = per_tool.setdefault(gcall["name"], {"n": 0, "ok": 0, "sel": 0})
            t["n"] += 1
            t["ok"] += ok
            call_total += 1
            pred = r.get("prediction") or {}
            names = ([c["name"] for c in pred.get("calls", [])]
                     if r["parse_status"] == "json_call" else [])
            if sorted(names) == sorted(c["name"] for c in g["gold"]["calls"]):
                tool_sel_ok += 1
                t["sel"] += 1
                pc = pred["calls"][0] if pred.get("calls") else {}
                m, e, w, wd = compare_args(gcall.get("arguments", {}),
                                           pc.get("arguments", {}))
                missing_total += len(m)
                extra_total += len(e)
                wrong_total += len(w)
                wrong_date_total += len(wd)
                gg = norm_args(gcall.get("arguments", {}))
                pp = norm_args(pc.get("arguments", {}))
                args_total += len(gg)
                args_ok += sum(1 for k in gg if k in pp and gg[k] == pp[k])
                if not m and not e and not w:
                    args_full_ok += 1
        elif g["gold"]["type"] == "clarify":
            clarify_total += 1
            clarify_ok += ok
        else:
            none_total += 1
            none_ok += ok

    latencies.sort()

    def lat_p(p):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return latencies[idx]

    score = {
        "model": model, "adapter": adapter,
        "results_file": str(results_path.relative_to(ROOT)),
        "split": str(split_path.relative_to(ROOT)),
        "n_scored": n, "n_split": len(gold), "n_missing": len(missing_ids),
        "response_accuracy": pct(correct, n),
        "tool_selection": {"n": call_total, "acc": pct(tool_sel_ok, call_total)},
        "args": {
            "full_match_given_tool": pct(args_full_ok, tool_sel_ok),
            "value_acc": pct(args_ok, args_total),
            "missing_total": missing_total, "extra_total": extra_total,
            "wrong_value_total": wrong_total, "wrong_date_total": wrong_date_total,
        },
        "no_tool": {"n": none_total, "acc": pct(none_ok, none_total)},
        "clarify": {"n": clarify_total, "acc": pct(clarify_ok, clarify_total)},
        "parse_validity": pct(parse_ok, n),
        "status_counts": dict(status_counts.most_common()),
        "by_bucket": {k: {"n": v["n"], "acc": pct(v["ok"], v["n"]),
                          "top_errors": dict(v["errors"].most_common(4))}
                      for k, v in sorted(per_bucket.items())},
        "by_language": {k: {"n": v["n"], "acc": pct(v["ok"], v["n"])}
                        for k, v in sorted(per_language.items())},
        "by_tool": {k: {"n": v["n"], "acc": pct(v["ok"], v["n"])}
                    for k, v in sorted(per_tool.items())},
        "taxonomy": dict(taxonomy.most_common()),
        "latency_ms": {"mean": round(statistics.mean(latencies)) if latencies else None,
                       "p50": lat_p(0.50), "p95": lat_p(0.95)},
        "usage_totals": dict(usage_tot),
    }

    out_path = Path(args.out) if args.out else ROOT / "eval" / "scores" / f"{results_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"model: {model} ({adapter})  scored {n}/{len(gold)}"
          + (f"  MISSING {len(missing_ids)}" if missing_ids else ""))
    print(f"overall: {score['response_accuracy']}%   tool-selection: {score['tool_selection']['acc']}%"
          f"   args-full: {score['args']['full_match_given_tool']}%   parse-valid: {score['parse_validity']}%")
    print("by bucket: " + "  ".join(f"{k}={v['acc']}%" for k, v in score["by_bucket"].items()))
    print("top errors: " + ", ".join(f"{k}={v}" for k, v in score["taxonomy"].items() if k != "correct"))
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
