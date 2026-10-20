"""Candidate B - embeddings + simple classifier (no generation at all).

Two stages ("before -> after" for the results table):
  before  zero-shot prototype matching: embed intent definitions & label-anchor
          sentences, cosine-1NN to intent / urgency / sentiment. Zero training.
  after   embed 4,000 train tickets, train one logistic-regression head and one
          kNN per label (hyper-params picked on the 300-row val set).

Two embedder arms with the same code:
  minilm  sentence-transformers/all-MiniLM-L6-v2 (standard open embedding model)
  needle  Needle 3 retrieval-head embeddings via agent.embed(text)  (run with venv-needle)

Usage:
  python candidate_b_clf.py --embedder minilm --stage both
  python candidate_b_clf.py --embedder needle --stage both

Writes artifacts/preds_b_<embedder>_<stage>.json and logs every experiment.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_lib import (ARTIFACTS, CASE, INTENTS, INTENT_DEFS, URGENCY, SENTIMENT,
                        load, log_experiment, metrics)

URG_ANCHORS = {
    "low": "no rush at all, whenever you can get to it, not urgent",
    "medium": "sometime this week would be good, soon-ish, before the weekend",
    "high": "urgent, needs to happen today at the latest, time-sensitive",
}
SENT_ANCHORS = {
    "negative": "this is unacceptable, I am honestly furious, very disappointed, ridiculous",
    "neutral": "my order number is 48213, I wanted to check the stock of an item please",
    "positive": "thanks in advance, long-time customer generally happy, love your shop otherwise",
}

PROTO_SOURCE = {}  # filled at runtime; used to explain the zero-shot arm


def get_embedder(name):
    if name == "minilm":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        def embed(texts, batch_size=64):
            v = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=False)
            return v
        dim_info = {"model": "all-MiniLM-L6-v2", "device": "cpu"}
        return embed, dim_info
    elif name == "needle":
        import needle
        agent = needle.Needle()

        def embed(texts, batch_size=None):
            return [agent.embed(t) for t in texts]
        dim_info = {"model": "needle3-retrieval-head", "device": "cpu"}
        return embed, dim_info
    raise ValueError(name)


def cosine_nn(train_vecs, train_labels, queries, k=1):
    import numpy as np
    tv = np.asarray(train_vecs, dtype="float32")
    qv = np.asarray(queries, dtype="float32")
    if tv.ndim == 1:
        tv = tv[None, :]
    t = tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9)
    q = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    sims = q @ t.T
    out = []
    for row in sims:
        idx = np.argpartition(-row, min(k, len(row) - 1))[:k]
        best = idx[np.argmax(row[idx])]
        out.append((train_labels[best], float(row[best])))
    return out


def stage_before(texts, embed, embed_info):
    t0 = time.time()
    intent_labels = list(INTENT_DEFS)
    ivec = embed(list(INTENT_DEFS.values()))
    uvec = embed(list(URG_ANCHORS.values()))
    svec = embed(list(SENT_ANCHORS.values()))
    tvec = embed(texts)
    embed_s = time.time() - t0

    preds = []
    for i, tv in enumerate(tvec):
        (intent, _), (urg, _), (sent, _) = (
            cosine_nn(ivec, intent_labels, [tv])[0],
            cosine_nn(uvec, list(URG_ANCHORS), [tv])[0],
            cosine_nn(svec, list(SENT_ANCHORS), [tv])[0])
        preds.append({"intent": intent, "urgency": urg, "sentiment": sent})
    return preds, {"mode": "zero-shot prototypes from label definitions", **embed_info}, embed_s


def stage_after(train, val, texts, embed, embed_info):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score

    t0 = time.time()
    etrain = embed([r["text"] for r in train])
    evalv = embed([r["text"] for r in val])
    etest = embed(texts)
    embed_s = time.time() - t0

    preds = [None] * len(etest)
    chosen = {}
    for head, classes in (("intent", INTENTS), ("urgency", URGENCY), ("sentiment", SENTIMENT)):
        ytr = [r[head] for r in train]
        yva = [r[head] for r in val]
        best, best_name, best_score = None, None, -1.0
        for name, clf in [
            ("lr_c0.25", LogisticRegression(C=0.25, max_iter=2000, n_jobs=-1)),
            ("lr_c1", LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1)),
            ("lr_c4", LogisticRegression(C=4.0, max_iter=2000, n_jobs=-1)),
            ("knn3", KNeighborsClassifier(3, metric="cosine")),
            ("knn7", KNeighborsClassifier(7, metric="cosine")),
        ]:
            clf.fit(etrain, ytr)
            va = clf.predict(evalv)
            score = f1_score(yva, va, average="macro", zero_division=0)
            if score > best_score:
                best, best_name, best_score = clf, name, score
        chosen[head] = (best_name, round(best_score, 4))
        test_pred = best.predict(etest)
        for i, v in enumerate(test_pred):
            preds[i] = preds[i] or {}
            preds[i][head] = v
    train_s = time.time() - t0
    meta = {"mode": "trained LR/kNN on 4000, val-selected", **embed_info,
            "embed_s": round(embed_s, 1), "train_s": round(train_s, 1), "heads": chosen}
    return preds, meta, embed_s + train_s


def latency_probe(embed, texts, n=100):
    """Batch-1 single-request latency: one embed + one trivial classifier forward."""
    ts = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        v = embed([t])
        _ = len(v[0])
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    p = lambda q: round(ts[min(len(ts) - 1, int(len(ts) * q))], 1)
    return {"p50_ms": p(0.50), "p95_ms": p(0.95), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["minilm", "needle"], required=True)
    ap.add_argument("--stage", choices=["before", "after", "both"], default="both")
    ap.add_argument("--latency", type=int, default=100)
    args = ap.parse_args()

    train, val = load("train"), load("val")
    test = load("test")
    texts = [r["text"] for r in test]
    golds = [{k: r[k] for k in ("intent", "urgency", "sentiment")} for r in test]

    embed, embed_info = get_embedder(args.embedder)
    probe = latency_probe(embed, texts, args.latency)

    stages = [args.stage] if args.stage != "both" else ["before", "after"]
    for stage in stages:
        tag = f"b_{args.embedder}_{stage}"
        if stage == "before":
            preds, meta, run_s = stage_before(texts, embed, embed_info)
        else:
            preds, meta, run_s = stage_after(train, val, texts, embed, embed_info)
        res = dict(metrics(preds, golds))
        log_experiment(tag, intent=res["intent"], urgency=res["urgency"],
                        sentiment=res["sentiment"], latency=probe, config=meta,
                        runtime_s=round(run_s, 1), test_n=len(test))
        (ARTIFACTS / f"preds_{tag}.json").write_text(json.dumps(
            {"name": tag, "predictions": preds, "gold": golds,
             "metrics": res, "latency": probe, "meta": meta, "probe": probe},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{tag}] intent={res['intent']} urgency={res['urgency']} sentiment={res['sentiment']}")


if __name__ == "__main__":
    main()
