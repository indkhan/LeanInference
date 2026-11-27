"""Structural probe: can the base engine extract at all on a tiny schema?

Distinguishes "engine/tool-schema problem" from "base model cannot handle the
14-field claim record" before we invest in fine-tuning.
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
from needle import Needle

small_tool = [{
    "name": "record_probe",
    "description": "Record the invoice fields.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["vendor", "total", "currency"],
        "properties": {
            "vendor": {"type": "string"},
            "total": {"type": "string", "pattern": "^-?[0-9]+\\.[0-9]{2}$"},
            "currency": {"type": ["string", "null"], "enum": ["USD", "EUR", "GBP", "CHF", None]},
        },
    },
}]

probe_text = "Invoice from Acme Ltd.\nTotal: 12.50 USD\nThank you."

agent = Needle(tools=small_tool, system="Extract the invoice record.", stateless=True,
               auto_date=False)
t0 = time.perf_counter()
resp = agent.complete(probe_text, max_new_tokens=256)
print(f"small schema: {time.perf_counter() - t0:.2f}s")
print(json.dumps(resp, ensure_ascii=False)[:1500])
agent.close()

# Second probe: same tiny schema, but with a real (long) claim document.
docs = [json.loads(line) for line in
        open("data/splits/train.jsonl", encoding="utf-8").read().splitlines() if line.strip()]
agent2 = Needle(tools=small_tool, system="Extract the invoice record.", stateless=True,
                auto_date=False)
t0 = time.perf_counter()
resp2 = agent2.complete(docs[0]["text"], max_new_tokens=256)
print(f"\nlong doc, small schema: {time.perf_counter() - t0:.2f}s")
print(json.dumps(resp2, ensure_ascii=False)[:1200])
agent2.close()
