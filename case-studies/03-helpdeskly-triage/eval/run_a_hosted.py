"""Candidate A - cheaper hosted model, classification prompt (no training).

Model: opencode/deepseek-v4.1-flash-free (free hosted tier, $0 spend).
The free tier has no list price, so cost is priced from DeepSeek V4.1 Flash list
prices - the cost is NOT the free-tier price.

Stages:
  zero_shot  intent definitions + rules, no examples
  few_shot   same prompt + K examples per intent drawn from TRAIN (seed fixed)
Both stages are run on val (prompt selection) and test (reported). The "after"
row in the results table is whichever stage wins on VAL intent macro-F1.
Test rows are never shown to the model or used for prompt choice.

Usage:
  python eval/run_a_hosted.py --stage zero_shot --split val
  python eval/run_a_hosted.py --stage few_shot --split test --workers 2
  python eval/run_a_hosted.py --model opencode/beam-free --role base --stage zero_shot --split test

Responses are cached per ticket in artifacts/a_<slug>_<stage>_<split>.jsonl, so a
rerun only calls the model for missing ids.
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common_lib import (ARTIFACTS, CASE, INTENTS, INTENT_DEFS, URGENCY, SENTIMENT,  # noqa: E402
                        load, log_experiment, metrics)

K_PER_INTENT = 2
SEED = 7

RULES = """Rules:
- intent: the issue the customer LEADS WITH. Secondary issues (e.g. "Also, ...") never override the lead issue.
- urgency: high = needs action today (explicit time pressure, or a business risk such as a missed event or trip).
  medium = this week / soon. low = no rush. Ignore polite phrases like "no rush" if the situation is clearly urgent.
- sentiment: the customer's tone. Sarcastic openers ("Perfect, just perfect.") over a complaint are negative.
"""


def intent_block():
    return "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFS.items())


def build_system(stage, shots):
    s = ("You label customer-support tickets for an e-commerce helpdesk.\n"
         f"Intent labels (choose exactly one):\n{intent_block()}\n\n"
         f"Urgency labels: {', '.join(URGENCY)}. Sentiment labels: {', '.join(SENTIMENT)}.\n\n"
         f"{RULES}\n"
         'Reply with ONLY a JSON object: {"intent": ..., "urgency": ..., "sentiment": ...}. No prose.\n')
    if stage == "few_shot":
        ex = "\n".join(
            f"Ticket: {r['text']}\nAnswer: " + json.dumps({k: r[k] for k in ("intent", "urgency", "sentiment")})
            for r in shots)
        s += f"\nLabelled examples:\n{ex}\n"
    return s


def pick_shots():
    rng = random.Random(SEED)
    train = load("train")
    shots = []
    for intent in INTENTS:
        pool = [r for r in train if r["intent"] == intent and len(r["text"]) < 300]
        shots += rng.sample(pool, min(K_PER_INTENT, len(pool)))
    rng.shuffle(shots)
    return shots


def find_opencode():
    exe = shutil.which("opencode")
    if not exe:
        raise FileNotFoundError("opencode not on PATH")
    return exe


def parse(reply):
    t = reply.strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON")
    d = json.loads(m.group(0))
    out = {"intent": str(d.get("intent", "")).strip(),
           "urgency": str(d.get("urgency", "")).strip().lower(),
           "sentiment": str(d.get("sentiment", "")).strip().lower()}
    return out


def call_model(model, system, text, timeout=180):
    prompt = f"{system}\nTicket: {text}\nAnswer:"
    argv = [find_opencode(), "run", "--agent", "classifier", "-m", model, "--format", "json"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=str(CASE / "eval"), input=prompt, capture_output=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None, "timeout", timeout
    lat = time.perf_counter() - t0
    chunks, err = [], None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "text":
            chunks.append(ev.get("part", {}).get("text", ""))
        elif ev.get("type") == "error":
            err = (ev.get("error", {}) or {}).get("message") or "provider error"
    raw = "".join(chunks)
    if not raw:
        return None, err or "empty", lat
    return raw, None, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opencode/deepseek-v4.1-flash-free")
    ap.add_argument("--stage", choices=["zero_shot", "few_shot"], required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attempts", type=int, default=6, help="tries per ticket (rate-limit backoff)")
    ap.add_argument("--role", default="a", help="output tag prefix: a = candidate A, base = baseline proxy")
    args = ap.parse_args()

    rows = load(args.split)
    if args.limit:
        rows = rows[: args.limit]
    shots = pick_shots() if args.stage == "few_shot" else []
    system = build_system(args.stage, shots)
    slug = args.model.replace("/", "_")
    cache_path = ARTIFACTS / f"a_{slug}_{args.stage}_{args.split}.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["pred"] is not None:  # failed calls (outage, rate limit) are retried
                done[rec["id"]] = rec

    todo = [r for r in rows if r["id"] not in done]
    print(f"[A] {args.model} {args.stage} {args.split}: {len(todo)} to call, {len(done)} cached", flush=True)

    def work(r):
        # Free tier rate-limits: retry with backoff. Only successful records are cached.
        for attempt in range(args.attempts):
            raw, err, lat = call_model(args.model, system, r["text"])
            pred, parse_err = None, None
            if raw is not None:
                try:
                    pred = parse(raw)
                except Exception as e:  # noqa: BLE001
                    parse_err = f"parse: {e}"
            if pred is not None or (err and "rate limit" not in err.lower()):
                break
            time.sleep(min(60, 5 * 2 ** attempt))
        return {"id": r["id"], "raw": raw, "pred": pred, "error": err or parse_err,
                "latency_s": round(lat, 3), "attempts": attempt + 1}

    with cache_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["id"]] = rec
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    # Score. Failed calls count as wrong: fall back to a labelled default so the
    # metric is never inflated by dropping rows.
    DEFAULT = {"intent": "other", "urgency": "medium", "sentiment": "neutral"}
    preds, golds, lats, n_fail = [], [], [], 0
    for r in rows:
        rec = done[r["id"]]
        if rec["pred"] is None:
            n_fail += 1
            p = dict(DEFAULT)
        else:
            p = rec["pred"]
            if p["intent"] not in INTENTS:
                p["intent"] = "other"
            if p["urgency"] not in URGENCY:
                p["urgency"] = "medium"
            if p["sentiment"] not in SENTIMENT:
                p["sentiment"] = "neutral"
        preds.append(p)
        golds.append({k: r[k] for k in ("intent", "urgency", "sentiment")})
        lats.append(rec["latency_s"])
    res = dict(metrics(preds, golds))
    lats.sort()
    p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))] if lats else None
    p50 = lats[len(lats) // 2] if lats else None
    tag = f"{args.role}_{args.stage}_{args.split}"
    log_experiment(tag, model=args.model, intent=res["intent"], urgency=res["urgency"],
                   sentiment=res["sentiment"], failed_calls=n_fail, n=len(rows),
                   latency={"p50_s": p50, "p95_s": p95}, config={
                       "k_per_intent": K_PER_INTENT if args.stage == "few_shot" else 0,
                       "system_chars": len(system), "transport": "opencode run (CLI, per call)"})
    (ARTIFACTS / f"preds_{tag}.json").write_text(json.dumps(
        {"name": tag, "model": args.model, "predictions": preds, "gold": golds,
         "ids": [r["id"] for r in rows], "metrics": res, "failed_calls": n_fail,
         "latency_s": {"p50": p50, "p95": p95}}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res), "failed:", n_fail, "p95_s:", p95)


if __name__ == "__main__":
    main()
