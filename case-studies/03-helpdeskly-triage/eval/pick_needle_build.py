"""Print the Needle archive with the higher VAL intent macro-F1 (default vs full-KV build)."""
import json
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "artifacts"
f1 = {t: json.loads((A / f"preds_d_needle_val_{t}.json").read_text(encoding="utf-8"))["metrics"]["intent"]["macro_f1"]
      for t in ("default", "fullkv")}
print("tuned_fullkv.cact" if f1["fullkv"] >= f1["default"] else "tuned.cact")
