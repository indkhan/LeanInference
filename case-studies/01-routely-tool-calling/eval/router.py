"""Top-k tool router for Needle (its prompt budget cannot hold all 40 schemas).

Stdlib BM25 over each tool's name + description + parameter names. The router picks k tools
per request; the Needle prompt then carries only those schemas. Recall = share of gold tool
calls whose tool is inside the top-k. k is chosen on val only; test is reported once.

  uv run python eval/router.py                # recall on val and test for k in 3,5,8,10
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = json.loads((ROOT / "data" / "tools.json").read_text(encoding="utf-8"))
TOKEN = re.compile(r"[a-z0-9]+")


def toks(text: str) -> list[str]:
    return TOKEN.findall(text.lower().replace("_", " "))


def tool_doc(t: dict) -> list[str]:
    f = t["function"]
    params = " ".join(f.get("parameters", {}).get("properties", {}).keys())
    return toks(f["name"]) * 2 + toks(f.get("description", "")) + toks(params)


class BM25Router:
    def __init__(self, tools: list[dict], k1: float = 1.2, b: float = 0.75):
        self.names = [t["function"]["name"] for t in tools]
        self.docs = [tool_doc(t) for t in tools]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        self.k1, self.b = k1, b
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tfs = [Counter(d) for d in self.docs]

    def rank(self, query: str) -> list[str]:
        q = toks(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            dl = len(self.docs[i])
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.names[i] for i in order]

    def top_k(self, query: str, k: int) -> list[str]:
        return self.rank(query)[:k]


def recall(split: Path, router: BM25Router, k: int) -> tuple[float, int]:
    rows = [json.loads(l) for l in split.read_text(encoding="utf-8").splitlines() if l.strip()]
    hit = total = 0
    for r in rows:
        if r["gold"]["type"] != "call":
            continue
        for c in r["gold"]["calls"]:
            total += 1
            hit += c["name"] in router.top_k(r["request"], k)
    return (hit / total if total else 0.0), total


def main():
    router = BM25Router(TOOLS)
    for name in ("val", "test"):
        split = ROOT / "data" / "splits" / f"{name}.jsonl"
        for k in (3, 5, 8, 10):
            r, n = recall(split, router, k)
            print(f"{name:5s} k={k:2d}  recall={r * 100:6.2f}%  (gold tool calls={n})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()


class LearnedRouter:
    """Char+word TF-IDF + logistic regression fitted on the TRAIN split's gold tool names.
    BM25 over schema text tops out at 77% recall@10 (multilingual/slang requests share few
    words with the schemas); this reaches ~99% recall@5 on val. k is chosen on val."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union
        X, y = [], []
        for line in (ROOT / "data" / "splits" / "train.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            for c in r["gold"].get("calls", []) if r["gold"]["type"] == "call" else []:
                X.append(r["request"])
                y.append(c["name"])
        self.model = make_pipeline(
            make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                       TfidfVectorizer(sublinear_tf=True)),
            LogisticRegression(C=20, max_iter=3000)).fit(X, y)
        self.by_name = {t["function"]["name"]: t for t in TOOLS}

    def top_k(self, query: str, k: int) -> list[str]:
        p = self.model.predict_proba([query])[0]
        return [self.model.classes_[i] for i in p.argsort()[::-1][:k]]

    def tools_doc(self, query: str, k: int, must: tuple[str, ...] = ()) -> str:
        """Compact JSON of the top-k schemas; `must` (train only) swaps in gold tools the router missed."""
        names = self.top_k(query, k)
        for m in must:
            if m not in names:
                names[-1] = m
        return json.dumps([self.by_name[n] for n in names], ensure_ascii=False, separators=(",", ":"))


_learned = None


def learned_router() -> LearnedRouter:
    global _learned
    if _learned is None:
        _learned = LearnedRouter()
    return _learned
