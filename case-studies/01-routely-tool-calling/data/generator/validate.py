"""Validate Routely dataset batches against data/tools.json.

Usage:
  uv run --with jsonschema --with tiktoken python data/generator/validate.py FILE.jsonl [FILE2.jsonl ...]

Checks
  - record shape and required fields
  - gold calls: tool exists, arguments validate against the tool's parameter schema
  - date-typed arguments are ISO YYYY-MM-DD
  - ids unique (across files), bucket/language known
  - exact duplicates and near-duplicates (normalized request text)
  - stats: bucket/language/gold-type distribution, tool usage, token estimate

Exit code 1 if any errors were found.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    print("ERROR: jsonschema not installed.")
    print("Run: uv run --with jsonschema --with tiktoken python data/generator/validate.py FILE...")
    raise SystemExit(2)

try:
    import tiktoken

    ENC = tiktoken.get_encoding("o200k_base")
except Exception:  # pragma: no cover
    ENC = None

try:  # optional accelerator for the near-duplicate check on large sets
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz import process as rf_process
except Exception:  # pragma: no cover
    rf_fuzz = None
    rf_process = None

ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "data" / "tools.json"

BUCKETS = {"clear", "confusable", "relative_date", "multilingual", "no_tool", "clarify"}
LANGUAGES = {"en", "es", "hi"}
GOLD_TYPES = {"call", "clarify", "none"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def load_tools():
    tools = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))
    by_name = {}
    for t in tools:
        fn = t["function"]
        by_name[fn["name"]] = fn["parameters"]
    return tools, by_name


def is_date_param(name: str, schema: dict) -> bool:
    if name == "date" or name.endswith("_date"):
        return True
    desc = schema.get("description")
    return isinstance(desc, str) and "ISO format" in desc


def normalize_request(text: str) -> str:
    """Lowercase, drop punctuation, keep letters/digits in any script + combining marks."""
    out = []
    for ch in text.lower():
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            out.append(ch)
        else:
            out.append(" ")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def check_records(path: Path, tools_by_name, tools_json, seen_ids, errors, warnings,
                  stats, requests, near_dup_pool):
    validator_cache = {}

    def validator_for(name):
        if name not in validator_cache:
            validator_cache[name] = Draft202012Validator(tools_by_name[name])
        return validator_cache[name]

    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        where = f"{path.name}:{lineno}"
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{where}: invalid JSON ({e})")
            continue

        rid = rec.get("id", "?")
        where = f"{path.name}:{lineno} [{rid}]"

        # --- shape ---
        for key in ("id", "bucket", "language", "request_date", "request", "gold", "source"):
            if key not in rec:
                errors.append(f"{where}: missing field '{key}'")

        if not isinstance(rid, str) or not ID_RE.match(rid):
            errors.append(f"{where}: bad id format")
        if rid in seen_ids:
            errors.append(f"{where}: duplicate id (also in {seen_ids[rid]})")
        else:
            seen_ids[rid] = where

        bucket = rec.get("bucket")
        if bucket not in BUCKETS:
            errors.append(f"{where}: unknown bucket {bucket!r}")
        language = rec.get("language")
        if language not in LANGUAGES:
            errors.append(f"{where}: unknown language {language!r}")
        rdate = rec.get("request_date")
        if not isinstance(rdate, str) or not DATE_RE.match(rdate):
            errors.append(f"{where}: request_date must be YYYY-MM-DD")

        request = rec.get("request")
        if not isinstance(request, str) or not request.strip():
            errors.append(f"{where}: empty request")

        gold = rec.get("gold") or {}
        gtype = gold.get("type")
        if gtype not in GOLD_TYPES:
            errors.append(f"{where}: gold.type must be one of {sorted(GOLD_TYPES)}")
        stats["records"] += 1
        stats["bucket"][bucket] += 1
        stats["language"][language] += 1
        stats["gold_type"][gtype] += 1

        if gtype == "call":
            calls = gold.get("calls") or []
            if not calls:
                errors.append(f"{where}: gold.type=call but no calls")
            if len(calls) > 1:
                warnings.append(f"{where}: multi-call example (scoring v1 assumes single call)")
            for call in calls:
                name = call.get("name")
                args = call.get("arguments")
                if name not in tools_by_name:
                    errors.append(f"{where}: unknown tool {name!r}")
                    continue
                stats["tool"][name] += 1
                if not isinstance(args, dict):
                    errors.append(f"{where}: arguments must be an object")
                    continue
                errs = sorted(validator_for(name).iter_errors(args), key=lambda e: list(e.path))
                for e in errs:
                    errors.append(f"{where}: {name}.{'.'.join(map(str, e.path))}: {e.message}")
                # explicit ISO date check (schemas describe dates as strings)
                schema = tools_by_name[name]
                for k, v in args.items():
                    ps = (schema.get("properties") or {}).get(k) or {}
                    if isinstance(v, str) and is_date_param(k, ps) and not DATE_RE.match(v):
                        errors.append(f"{where}: {name}.{k} should be YYYY-MM-DD, got {v!r}")
            # rule: request should not literally name the tool
            if isinstance(request, str):
                stripped = request.lower().replace("-", "_")
                for call in calls:
                    n = (call.get("name") or "").lower()
                    if n and n in stripped:
                        warnings.append(f"{where}: request literally contains tool name {n!r}")
        elif gtype in {"clarify", "none"}:
            if not gold.get("reason"):
                warnings.append(f"{where}: gold.type={gtype} has no reason")

        # --- duplicates ---
        if isinstance(request, str) and request.strip():
            norm = normalize_request(request)
            if norm in requests:
                errors.append(f"{where}: exact duplicate request of {requests[norm]}")
            else:
                requests[norm] = where
            near_dup_pool.append((len(norm), norm, where))


def near_duplicates(pool, threshold=0.92, max_len_diff=30):
    if rf_process is not None and len(pool) > 400:
        norms = [n for _, n, _ in pool]
        wheres = [w for _, _, w in pool]
        try:
            scores = rf_process.cdist(norms, norms, scorer=rf_fuzz.ratio, workers=-1)
        except Exception:
            scores = None
        if scores is not None:
            dupes = []
            n = len(norms)
            for i in range(n):
                row = scores[i]
                for j in range(i + 1, n):
                    if row[j] >= threshold * 100:
                        dupes.append((row[j] / 100.0, wheres[i], wheres[j]))
            return dupes

    if len(pool) > 1500:
        print("note: near-duplicate check skipped "
              "(install rapidfuzz+numpy for large sets)")
        return []

    dupes = []
    by_len = defaultdict(list)
    for length, norm, where in pool:
        by_len[length].append((norm, where))
    lengths = sorted(by_len)
    for i, l1 in enumerate(lengths):
        for l2 in lengths[i:]:
            if l2 - l1 > max_len_diff:
                break
            for n1, w1 in by_len[l1]:
                for n2, w2 in by_len[l2]:
                    if w1 == w2 or (l1 == l2 and w1 > w2):
                        continue
                    ratio = SequenceMatcher(None, n1, n2).ratio()
                    if ratio >= threshold:
                        dupes.append((ratio, w1, w2))
    return dupes


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    tools, tools_by_name = load_tools()
    print(f"tools.json: {len(tools_by_name)} tools")

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: dict[str, str] = {}
    requests: dict[str, str] = {}
    near_dup_pool: list[tuple[int, str, str]] = []
    stats = {
        "records": 0,
        "bucket": Counter(),
        "language": Counter(),
        "gold_type": Counter(),
        "tool": Counter(),
        "req_tokens": [],
        "gold_tokens": [],
    }

    files: list[Path] = []
    for fname in argv:
        matches = sorted(Path(p) for p in glob.glob(fname))
        if not matches:
            print(f"missing file: {fname}")
            return 2
        files.extend(matches)
    for path in files:
        check_records(path, tools_by_name, tools, seen_ids, errors, warnings, stats,
                      requests, near_dup_pool)
        print(f"checked: {path.name}")

    dupes = near_duplicates(near_dup_pool)

    # --- report ---
    print("\n--- stats ---")
    print(f"records: {stats['records']}")
    print("by bucket:  " + ", ".join(f"{k}={v}" for k, v in stats["bucket"].most_common()))
    print("by language:" + ", ".join(f"{k}={v}" for k, v in stats["language"].most_common()))
    print("by gold:    " + ", ".join(f"{k}={v}" for k, v in stats["gold_type"].most_common()))
    print("top tools:  " + ", ".join(f"{k}={v}" for k, v in stats["tool"].most_common(12)))

    if ENC is not None:
        tools_tokens = len(ENC.encode(json.dumps(tools, ensure_ascii=False)))
        req_tokens = [len(ENC.encode(r)) for r in requests]
        print(f"token estimate (o200k): tools.json={tools_tokens}")
        if req_tokens:
            print(f"  request avg={sum(req_tokens)/len(req_tokens):.0f} max={max(req_tokens)} tokens")
        print(f"  approx input per call = tools.json + request (+~150 system) = "
              f"{tools_tokens + (sum(req_tokens)/len(req_tokens) if req_tokens else 0) + 150:.0f} tokens")
    else:
        print("(tiktoken unavailable - no token estimate)")

    if dupes:
        warnings.append(f"{len(dupes)} near-duplicate request pair(s) (>=0.92 similarity)")

    print("\n--- findings ---")
    if warnings:
        for w in warnings[:30]:
            print(f"WARN  {w}")
        if len(warnings) > 30:
            print(f"WARN  ... and {len(warnings) - 30} more warnings")
    if errors:
        for e in errors[:40]:
            print(f"ERROR {e}")
        if len(errors) > 40:
            print(f"ERROR ... and {len(errors) - 40} more errors")
    print(f"\nresult: {'FAIL' if errors else 'OK'} "
          f"({len(errors)} errors, {len(warnings)} warnings)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
