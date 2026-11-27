# Failure analysis - Qwen3.5-2B fine-tuned, no tool schemas in the prompt

Source: `eval/results/hf-qwen3.5-2b-ft-noschema.jsonl` scored by `eval/score.py` on the 500-example,
hand-verified test split. **410/500 correct (82.0%)**; tool selection 97.9%; every reply was valid
JSON. This is the best self-hosted model (see `RESULTS.md`; until the baseline proxy completes, the
provisional bar is Candidate A at 83.6%).

The 90 misses by type: wrong argument value 60, wrong relative date 10, missing argument 5,
clarified instead of calling 4, said "no tool" instead of calling 3, wrong tool 2, extra argument 2,
no-tool / called instead of clarifying 4.

## 1. Free-text wording, not wrong behaviour (50 of the 90)

The scorer compares arguments exactly, including free-text fields like `reason` and `notes`. 50 of
the 60 "wrong value" misses differ **only** in such a field:

| request | gold `reason` | model `reason` |
|---|---|---|
| "we won't ship this after all: ORD-61544" | shipment no longer needed | no longer needed |
| "dispute INV-96090 in full, services were never delivered" | services not delivered | services never delivered |
| "CUS-1035: $95 credit, the customs docs delay was our fault" | customs docs delayed by us | customs docs delay was our fault |

The tool, the IDs and the amounts are right; the call would do the right thing in Routely. Every
model has this (scoring free text loosely: Candidate A 83.6 -> 95.8%, Qwen 2B 82.0 -> 92.0%, Qwen 0.8B
78.8 -> 89.0%), so the ranking does not change - we report the strict number as the headline because
that is the frozen scoring rule (`data/FORMAT.md`).

## 2. Relative dates (10)

| request (today) | gold | model |
|---|---|---|
| "this Friday at WH-004 midday - who's free?" (Sat 2026-06-20) | 2026-06-26 | 2026-06-21 |
| "WH-011 opening times this Wednesday" (Fri 2026-06-05) | 2026-06-10 | 2026-06-09 |
| "what pickups are on the boards for the 18th?" (2026-08-25) | 2026-09-18 | 2026-08-18 |

Date arithmetic is the one thing the frontier-class model does clearly better (relative_date bucket:
Candidate A 96.0%, Qwen 2B 78.0%). The fix is cheap and deterministic: resolve "this Friday" / "the 18th"
in code (the rules are already written down in the system prompt) and pass the ISO date in, instead of
asking a 2B model to count weekdays.

## 3. Optional details dropped (5) or added (2)

"collect 15 shoeboxes from ADR-1157 on May 8, before noon - buzz unit 12B" -> correct pickup, but the
`notes: "buzz unit 12B"` argument is missing. Conversely, "don't send anyone for PCK-14456" ->
correct `cancel_pickup`, plus an invented `reason: "no one assigned"`. Both are optional fields; the
call succeeds either way, but a dropped delivery note matters operationally.

## 4. Tool / no-tool decisions (13)

- Look-alike tools (2): "CUS-5334, return history please" -> `get_return_history` instead of
  `list_returns`; "how many drivers are available at WH-023" -> `driver_availability` instead of
  `list_drivers(status=available)`. Still better than Candidate A on the confusable bucket (68.8% vs 62.3%).
- Too cautious (7): "let's move 111 SKU-7110 from WH-006 to WH-052" -> asks for the source warehouse that
  is in the request; "anything in Omaha?" -> "no tool" instead of `list_warehouses(city=Omaha)`.
- Clarify cases missed (4): 2 acted anyway - "move PCK-22110 to the 18th" -> reschedules without the
  time window the gold label says to ask for; 2 answered "no tool" - "you know what to do" -> `none`
  instead of a clarifying question.

## What this means for Routely

The 2B model knows the 40 tools without seeing their schemas: 97.9% right tool, 90% clarify accuracy,
100% on "no tool applies". What remains is (1) free-text wording that does not change the action,
(2) relative dates - best moved into deterministic code, and (3) a handful of over/under-cautious
decisions that more examples of those patterns would address.
