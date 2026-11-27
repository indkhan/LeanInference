"""Mock adapter - exercises the whole eval pipeline with zero cost and zero compute.

Not a model. Produces deterministic, bucket-aware synthetic predictions with a realistic
error mix, so run_eval -> score -> report can be tested end-to-end before any real run.

Mock results must never be presented as measured model performance; the report flags
them with "(mock)".

Accuracy model (per bucket): clear 0.96, confusable 0.86, relative_date 0.90,
multilingual 0.93, no_tool 0.97, clarify 0.89. Wrong answers are drawn from the same
taxonomy the scorer counts (wrong twin tool, missing/extra/wrong arg, off-by-one date,
called when it should clarify/none, or no call at all). Latency is simulated.
"""
from __future__ import annotations

import json
import random
import re
from datetime import date, timedelta

ACC = {"clear": 0.96, "confusable": 0.86, "relative_date": 0.90,
       "multilingual": 0.93, "no_tool": 0.97, "clarify": 0.89}
LATENCY = {"clear": (140, 80), "confusable": (190, 90), "relative_date": (240, 110),
           "multilingual": (220, 100), "no_tool": (120, 60), "clarify": (200, 90)}

TWIN_SETS = [
    {"cancel_order", "cancel_pickup", "cancel_return"},
    {"update_address", "update_customer_address"},
    {"track_shipment", "get_eta"},
    {"get_quote", "create_order"},
    {"request_refund", "dispute_charge"},
    {"assign_driver", "reassign_driver"},
]
FALLBACK_TOOLS = [
    ("track_shipment", {"shipment_id": "SHP-10234"}),
    ("get_order", {"order_id": "ORD-55678"}),
    ("cancel_order", {"order_id": "ORD-34019"}),
    ("get_eta", {"shipment_id": "SHP-88120"}),
]
ID_RE = re.compile(r"^[A-Z]{2,5}-\d{3,5}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ENUM_SWAP = {
    "standard": "express", "express": "standard", "overnight": "freight", "freight": "overnight",
    "pickup": "dropoff_point", "dropoff_point": "pickup",
    "item_damaged": "wrong_item", "wrong_item": "item_damaged",
    "billing": "default_pickup", "default_pickup": "billing",
    "scheduled": "in_progress", "in_progress": "scheduled", "missed": "cancelled",
    "08:00-12:00": "12:00-16:00", "12:00-16:00": "16:00-20:00", "16:00-20:00": "08:00-12:00",
    "email": "sms", "sms": "whatsapp", "whatsapp": "email",
}


class MockAdapter:
    def __init__(self, model: str = "mock", seed: int = 42, **_ignored):
        self.model = model
        self.seed = seed

    def predict(self, record, system, tools_doc):
        rng = random.Random(f"{self.seed}:{record['id']}")
        bucket = record.get("bucket", "clear")
        mean, sd = LATENCY.get(bucket, (200, 90))
        latency = max(40, int(rng.gauss(mean, sd)))
        prompt_chars = len(system) + len(record["request"]) + 10
        if rng.random() < ACC.get(bucket, 0.9):
            text = json.dumps(record["gold"], ensure_ascii=False)
        else:
            text = self._wrong(rng, record)
        usage = {"prompt_tokens": int(prompt_chars / 3.6),
                 "completion_tokens": int(len(text) / 3.6)}
        return {"text": text, "latency_ms": latency, "usage": usage}

    # ------------------------------------------------------------------ errors
    def _wrong(self, rng, record):
        gold = record["gold"]
        gtype = gold.get("type")
        if gtype == "call":
            mode = rng.choices(
                ["wrong_tool", "missing_arg", "extra_arg", "wrong_value", "wrong_date", "free_text"],
                weights=[30, 15, 10, 20, 15, 10])[0]
            calls = json.loads(json.dumps(gold["calls"]))
            call = calls[0]
            args = call.setdefault("arguments", {})
            if mode == "wrong_tool":
                call["name"] = self._twin(rng, call["name"])
            elif mode == "missing_arg":
                if args:
                    del args[rng.choice(sorted(args))]
                else:
                    call["name"] = self._twin(rng, call["name"])
            elif mode == "extra_arg":
                args["reference"] = "call-back later"
            elif mode == "wrong_value":
                k = rng.choice(sorted(args))
                args[k] = self._mutate(rng, args[k])
            elif mode == "wrong_date":
                date_keys = [k for k in args if k == "date" or k.endswith("_date")]
                if date_keys:
                    k = rng.choice(sorted(date_keys))
                    args[k] = self._shift(args[k], rng.choice([-1, 1]))
                else:
                    k = rng.choice(sorted(args))
                    args[k] = self._mutate(rng, args[k])
            else:  # free_text - no call at all
                return "Sure, I have passed that on to the operations team."
            return json.dumps({"type": "call", "calls": calls}, ensure_ascii=False)
        if gtype == "clarify":
            if rng.random() < 0.6:
                name, args = rng.choice(FALLBACK_TOOLS)
                return json.dumps({"type": "call", "calls": [{"name": name, "arguments": args}]},
                                  ensure_ascii=False)
            return json.dumps({"type": "none"})
        # gold none -> invent a call
        name, args = rng.choice(FALLBACK_TOOLS)
        return json.dumps({"type": "call", "calls": [{"name": name, "arguments": args}]},
                          ensure_ascii=False)

    @staticmethod
    def _twin(rng, name):
        for group in TWIN_SETS:
            if name in group:
                return rng.choice(sorted(group - {name}))
        return rng.choice([n for n, _ in FALLBACK_TOOLS])

    @staticmethod
    def _shift(iso, days):
        if not isinstance(iso, str) or not DATE_RE.match(iso):
            return iso
        return (date.fromisoformat(iso) + timedelta(days=days)).isoformat()

    @staticmethod
    def _mutate(rng, value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return int(value * 1.5) if isinstance(value, int) else value * 1.5
        if isinstance(value, str):
            if ID_RE.match(value):
                prefix, digits = value.rsplit("-", 1)
                return f"{prefix}-{int(digits) + 1:0{len(digits)}d}"
            if DATE_RE.match(value):
                return MockAdapter._shift(value, rng.choice([-1, 1]))
            if value in ENUM_SWAP:
                return ENUM_SWAP[value]
            return value + " (unspecified)"
        if isinstance(value, dict):
            value = dict(value)
            if value:
                k = rng.choice(sorted(value))
                value[k] = MockAdapter._mutate(rng, value[k])
            return value
        return value
