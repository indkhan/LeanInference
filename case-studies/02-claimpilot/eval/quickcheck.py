"""Print the key columns of every run summary. Small convenience for the log book."""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = Path(__file__).resolve().parent / "results"

rows = []
for f in sorted(RESULTS.glob("summary_*.json")):
    s = json.loads(f.read_text(encoding="utf-8"))
    rows.append(s)

for s in rows:
    print(
        f"{s.get('model', '?'):42s} tag={s.get('tag', '?'):8s} "
        f"valid={s.get('valid_schema_rate', 0):.3f} exact={s.get('doc_exact_rate', 0):.3f} "
        f"f1={s.get('micro_f1', 0):.3f} amt={s.get('total_amount_accuracy', 0):.3f} "
        f"null={s.get('null_discipline', 0):.3f} p50={s.get('latency_p50', 0):.1f}s "
        f"p95={s.get('latency_p95', 0):.1f}s tin={s.get('tokens_in_mean', 0):.0f} tout={s.get('tokens_out_mean', 0):.0f}"
    )
print(f"\n{len(rows)} summaries")
