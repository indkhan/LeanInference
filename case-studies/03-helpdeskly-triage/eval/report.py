"""Print + write the Helpdeskly results table, the unseen-subset view, the winner and its
intent confusion matrix (PNG).

Usage:  python eval/report.py
Inputs (all from real runs): artifacts/preds_*.json, artifacts/a_*_test.jsonl (hosted raw replies
for token counts), eval/prices.json, data/test_unseen_ids.txt.
Outputs: results/final_table.md, results/final_table.json, results/confusion_intent_<winner>.png

Cost rules
  hosted:       (measured prompt + reply tokens, tiktoken cl100k over the exact prompts sent) x list
                price, scaled to 1M tickets/month
  self-hosted:  always-on machines to serve 1M tickets/month at the measured single-request p50
                latency (serial) -> machines x $/h x 730 h
Winner rule (brief Step 4): cheapest approach whose intent macro-F1 AND high-urgency recall are
>= the baseline proxy's.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE / "train"))
sys.path.insert(0, str(CASE / "eval"))
from common_lib import ARTIFACTS, INTENTS, RESULTS, confusion_matrix, load, metrics  # noqa: E402

PRICES = json.loads((CASE / "eval" / "prices.json").read_text(encoding="utf-8"))
N = PRICES["tickets_per_month"]
HOURS = 730

# ponytail: the lineup is fixed by the brief, so it lives here, not in a config file.
SYSTEMS = [
    {"name": "Baseline proxy: Nemotron 3 Ultra 550B (zero-shot)", "after": "base_zero_shot_test",
     "hosted": ("nvidia_nemotron-3-ultra-550b-a55b_free", "zero_shot"), "price": "frontier",
     "where": "third-party API", "size": "550B (API)", "train": "none", "baseline": True},
    {"name": "Candidate A: MiniMax M3.1 Flash (few-shot)", "after": "a_few_shot_test",
     "hosted": ("minimax_minimax-m3.1-flash_free", "few_shot"), "price": "candidate_a",
     "where": "third-party API", "size": "API", "train": "none (prompt + 24 shots)"},
    {"name": "B1: MiniLM-L6 embeddings + LR (no generation)", "before": "b_minilm_before",
     "after": "b_minilm_tuned", "price": "local_cpu", "where": "self-hosted CPU", "size": "22M params (~90 MB)",
     "train": "CPU: embed 4,000 tickets 367 s + fit 30 s"},
    {"name": "B2: Needle 3 embeddings + LR (no generation)", "before": "b_needle_before",
     "after": "b_needle_tuned", "price": "local_cpu", "where": "self-hosted CPU / on-device",
     "size": "Needle 3 (63 MB .cact)", "train": "CPU: embed + fit 616 s"},
    {"name": "C: Qwen3.5-0.8B LoRA label generator", "before": "c_qwen_before", "after": "c_qwen_after",
     "price": "local_gpu", "where": "self-hosted GPU", "size": "0.8B fp16 (1.6 GB) + 25 MB LoRA",
     "train": "RTX 3050: 34 min, 2 epochs"},
    {"name": "D: Needle 3 LoRA classifier (.cact)", "before": "d_needle_before", "after": "d_needle_after",
     "price": "local_cpu", "where": "self-hosted CPU / on-device", "size": "63 MB .cact (W4A8)",
     "train": "RTX 3050 (WSL): 33 min, 2 epochs"},
]


def preds(tag):
    p = ARTIFACTS / f"preds_{tag}.json" if tag else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else None


def latency_s(d):
    if "latency_s" in d:
        return d["latency_s"]["p50"], d["latency_s"]["p95"]
    lat = d["latency"]
    return lat["p50_ms"] / 1000, lat["p95_ms"] / 1000


def hosted_cost(slug, stage, rate):
    """Mean tokens per ticket from the prompts actually sent and the replies received."""
    import tiktoken
    from run_a_hosted import build_system, pick_shots
    enc = tiktoken.get_encoding("cl100k_base")
    system = build_system(stage, pick_shots() if stage == "few_shot" else [])
    replies = {}
    for line in (ARTIFACTS / f"a_{slug}_{stage}_test.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["pred"] is not None:
            replies[r["id"]] = r["raw"]
    rows = [r for r in load("test") if r["id"] in replies]
    tin = sum(len(enc.encode(system)) + len(enc.encode(f"Ticket: {r['text']}\nAnswer:")) for r in rows) / len(rows)
    tout = sum(len(enc.encode(replies[r["id"]])) for r in rows) / len(rows)
    per = (tin * rate["input"] + tout * rate["output"]) / 1e6
    return per * 1e6, per * N, f"{rate['name']} list x {tin:.0f} in + {tout:.0f} out tokens/ticket (measured)"


def self_hosted_cost(p50, ref):
    units = max(1, math.ceil(N * p50 / (HOURS * 3600)))
    monthly = units * ref["usd_per_hour"] * HOURS
    return monthly / N * 1e6, monthly, f"{units} x {ref['name']} (${ref['usd_per_hour']}/h) at p50 {p50 * 1000:.0f} ms"


def f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    test = load("test")
    unseen_ids = set((CASE / "data" / "test_unseen_ids.txt").read_text(encoding="utf-8").split())
    keep = [i for i, r in enumerate(test) if r["id"] in unseen_ids]
    rows = []
    for sd in SYSTEMS:
        a, b = preds(sd["after"]), preds(sd.get("before"))
        r = {**sd, "a": a, "b": b, "usd1m": None, "monthly": None, "basis": "not complete"}
        if a and len(a["predictions"]) == len(test):
            r["m"] = dict(metrics(a["predictions"], a["gold"]))
            r["mu"] = dict(metrics([a["predictions"][i] for i in keep], [a["gold"][i] for i in keep]))
            r["p50"], r["p95"] = latency_s(a)
            if "hosted" in sd:
                r["usd1m"], r["monthly"], r["basis"] = hosted_cost(*sd["hosted"], PRICES[sd["price"]])
            else:
                r["usd1m"], r["monthly"], r["basis"] = self_hosted_cost(r["p50"], PRICES[sd["price"]])
        if b:
            r["mb"] = dict(metrics(b["predictions"], b["gold"]))
        rows.append(r)

    out = ["# Helpdeskly ticket triage - results (700-ticket hand-verified test set)", "",
           f"Prices: `eval/prices.json` ({PRICES['as_of']}). Volume: {N:,} tickets/month.", "",
           "| Approach | Intent macro-F1 (before -> after) | High-urgency recall | Urgency acc | Sentiment acc | p95 | $/1M | Monthly $ | Size | Training | Runs where |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    sq = PRICES["frontier"]
    sq_per = (200 * sq["input"] + 16 * sq["output"]) / 1e6  # brief: ~200 tokens/ticket; ~16-token label reply (measured on A)
    out.append(f"| Status quo: {sq['name']} (cost only, not run) | - | - | - | - | - | {sq_per * 1e6:,.0f} | "
               f"{sq_per * N:,.0f} | API | none | third-party API |")
    for r in rows:
        if "m" not in r:
            out.append(f"| {r['name']} | pending | - | - | - | - | - | - | {r['size']} | {r['train']} | {r['where']} |")
            continue
        m, before = r["m"], (f(r["mb"]["intent"]["macro_f1"]) + " -> ") if "mb" in r else ""
        out.append(f"| {r['name']} | {before}{f(m['intent']['macro_f1'])} | {f(m['urgency']['high_recall'])} | "
                   f"{f(m['urgency']['acc'])} | {f(m['sentiment']['acc'])} | {r['p95']:.2f} s | "
                   f"{r['usd1m']:,.0f} | {r['monthly']:,.0f} | {r['size']} | {r['train']} | {r['where']} |")
    status_quo = PRICES["frontier"]
    base = next(r for r in rows if r.get("baseline"))
    out += ["", "Cost basis per row:"] + [f"- {r['name']}: {r['basis']}" for r in rows]
    out += ["", f"Status quo = {status_quo['name']} at ${status_quo['input']}/${status_quo['output']} per MTok; "
                "the baseline proxy row is priced at that list price with the proxy's measured tokens.",
            "p95 = single-request latency (laptop RTX 3050 for C; laptop CPU for B1/B2/D; API round trip for hosted).", ""]

    out += [f"## Unseen-ticket view ({len(keep)} of {len(test)} test tickets)", "",
            "129 test tickets duplicate a training ticket once digits are masked (49 verbatim). The test set is "
            "kept as-is; this view drops those tickets so a trained model cannot win by memorising.", "",
            "| Approach | Intent macro-F1 | High-urgency recall | Sentiment acc |", "|---|---|---|---|"]
    out += [f"| {r['name']} | {f(r['mu']['intent']['macro_f1'])} | {f(r['mu']['urgency']['high_recall'])} | "
            f"{f(r['mu']['sentiment']['acc'])} |" for r in rows if "m" in r]
    out.append("")

    bar = base if "m" in base else next(r for r in rows if r["name"].startswith("Candidate A") and "m" in r)
    provisional = bar is not base
    bi, bh = bar["m"]["intent"]["macro_f1"], bar["m"]["urgency"]["high_recall"]
    ok = [r for r in rows if "m" in r and r is not bar and "hosted" not in r
          and r["m"]["intent"]["macro_f1"] >= bi and r["m"]["urgency"]["high_recall"] >= bh]
    winner = min(ok, key=lambda r: r["monthly"], default=None)
    out += ["## Winner (brief Step 4 rule)", "",
            f"Bar = {'PROVISIONAL - baseline proxy incomplete, Candidate A used' if provisional else 'baseline proxy'}: "
            f"intent macro-F1 {bi:.3f}, high-urgency recall {bh:.3f}."]
    for r in rows:
        if "m" in r and r is not bar:
            out.append(f"- {r['name']}: intent {r['m']['intent']['macro_f1']:.3f}, high-urgency recall "
                       f"{r['m']['urgency']['high_recall']:.3f}, ${r['monthly']:,.0f}/month")
    out += ["", f"**Winner: {winner['name']}**" if winner else "**No approach met the bar.**", ""]

    if winner:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix([g["intent"] for g in winner["a"]["gold"]],
                              [p["intent"] for p in winner["a"]["predictions"]], labels=INTENTS)
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(INTENTS)), INTENTS, rotation=60, ha="right")
        ax.set_yticks(range(len(INTENTS)), INTENTS)
        for i in range(len(INTENTS)):
            for j in range(len(INTENTS)):
                if cm[i][j]:
                    ax.text(j, i, cm[i][j], ha="center", va="center",
                            color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Intent confusion - {winner['name']} (700 test tickets)")
        fig.tight_layout()
        png = RESULTS / f"confusion_intent_{winner['after']}.png"
        fig.savefig(png, dpi=120)
        out += [f"Confusion matrix: `results/{png.name}`", ""]

    text = "\n".join(out) + "\n"
    (RESULTS / "final_table.md").write_text(text, encoding="utf-8")
    (RESULTS / "final_table.json").write_text(json.dumps(
        {"as_of": PRICES["as_of"], "winner": winner and winner["name"], "bar_provisional": provisional,
         "rows": [{k: v for k, v in r.items() if k not in ("a", "b")} for r in rows]}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
