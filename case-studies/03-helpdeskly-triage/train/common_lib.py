"""Shared helpers for the Helpdeskly case study: paths, labels, metrics, logging."""
import json
import time
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
DATA = CASE / "data"
TRAIN = CASE / "train"
EVAL = CASE / "eval"
RESULTS = CASE / "results"
ARTIFACTS = CASE / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

INTENTS = list(json.loads((DATA / "intents.json").read_text(encoding="utf-8")).keys())
INTENT_DEFS = {k: v["definition"] for k, v in
               json.loads((DATA / "intents.json").read_text(encoding="utf-8")).items()}
URGENCY = ["low", "medium", "high"]
SENTIMENT = ["negative", "neutral", "positive"]


def load(split):
    """Load tickets_{split}.jsonl. Model code must use only id/text/intent/urgency/sentiment."""
    rows = [json.loads(l) for l in (DATA / f"tickets_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("notes", None)  # cues are for verification/reasoning-lines only
    return rows


def log_experiment(name, **fields):
    """Append one experiment record. Every run in this case study must go through here."""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": name, **fields}
    with (RESULTS / "experiment_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[log] {name}: {json.dumps({k: v for k, v in fields.items() if k != 'config'})[:200]}")


from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix  # noqa: E402


def metrics(preds, golds):
    """preds/golds: list of dicts with intent/urgency/sentiment keys."""
    for key in ("intent", "urgency", "sentiment"):
        acc = accuracy_score([g[key] for g in golds], [p[key] for p in preds])
        mf1 = f1_score([g[key] for g in golds], [p[key] for p in preds],
                       average="macro", zero_division=0)
        out = {"acc": round(acc, 4), "macro_f1": round(mf1, 4)}
        if key == "urgency":
            out["high_recall"] = round(recall_score(
                [g["urgency"] for g in golds], [p["urgency"] for p in preds],
                labels=["high"], average="macro", zero_division=0), 4)
        yield key, out
