# Dataset format & rules (v1)

## Canonical record (one JSON object per line)

```json
{
  "id": "routely-0001",
  "bucket": "clear",
  "language": "en",
  "request_date": "2026-03-02",
  "request": "where is my order ORD-88213?",
  "gold": { "type": "call", "calls": [ { "name": "track_shipment", "arguments": { "order_id": "ORD-88213" } } ] },
  "source": "agent-v1",
  "notes": ""
}
```

- `gold.type` — one of:
  - `call` — one or more tool calls, `calls[].name` must exist in `data/tools.json`, arguments must
    validate against that tool's parameter schema.
  - `clarify` — the correct answer is to ask a clarifying question (no tool call). Add `"reason"`.
  - `none` — no tool applies (chit-chat / off-topic). Add `"reason"`.
- `request_date` — the "today" for the example. Relative dates in the request resolve against it.
  All gold dates are ISO (`YYYY-MM-DD`).
- `notes` — free text for verifiers (optional, omit when empty).

## Buckets (targets for 3,000 examples)

| Bucket | Share | What it tests | Target count |
|---|---|---|---|
| `clear` | 55% | single obvious call | 1,650 |
| `confusable` | 15% | near-twin tools (cancel_order vs cancel_pickup vs cancel_return; update_address vs update_customer_address; track_shipment vs get_eta; get_quote vs create_order; request_refund vs dispute_charge; assign_driver vs reassign_driver) | 450 |
| `relative_date` | 10% | "next Friday", "the 15th", "tomorrow", month/year rollovers | 300 |
| `multilingual` | 5% | Spanish + Hindi (mixed into request text) | 150 |
| `no_tool` | 7% | chit-chat, off-topic -> `gold.type = none` | 210 |
| `clarify` | 8% | missing/ambiguous info -> `gold.type = clarify` | 240 |

## Difficulty rules (what makes the eval real)

- Requests must vary: typos, slang, short/long, polite/rude, one issue vs several intertwined.
- IDs must match the tool's pattern (ORD-#####, SHP-#####, PCK-#####, RET-#####, CUS-####, DRV-###,
  WH-###, INV-#####, ADR-####, SKU-####).
- No request may contain the literal tool name.
- Confusable examples are only "confusable" if the wrong twin is a plausible reading of the request.
- Multilingual requests are written in the target language (code-switching allowed), with the same
  gold call as the English equivalent would have.
- Clarify examples must genuinely lack the information needed (e.g. "cancel my order" with no ID).
  The reason records what is missing.

### Time-window mapping (v1, use consistently)

| Phrase in request | `time_window` |
|---|---|
| morning / before noon | `08:00-12:00` |
| midday / noon / lunchtime / afternoon | `12:00-16:00` |
| evening / late afternoon / "after 5" | `16:00-20:00` |
| explicit times | map to the containing window (e.g. "5-8pm" -> `16:00-20:00`) |

If a request does not clearly indicate a window for a tool that requires one, it belongs in the
`clarify` bucket, not anywhere else.

### Relative-date convention (v1)

- "today" = request_date. "tomorrow" = +1 day. "day after tomorrow" = +2 days.
- "this <weekday>", bare "<weekday>", and "next <weekday>" = the first occurrence of that weekday
  strictly after request_date (said on Wed, "next Friday" = in 2 days — "next" does not skip a week).
- "a week on <weekday>" / "<weekday> next week" = that weekday in the following calendar week.
- "the <N>" = day N of request_date's month if N >= day-of-month, otherwise of the following month.
- "in N days/weeks", "end of the month", "start of next month" resolve literally; month/year rollovers
  are the classic edge cases (Jan 31 + "tomorrow" = Feb 1; Dec 30 + "Jan 2" = next year).
- Prefer phrasings whose date is plainly derivable. Avoid examples where two readings are equally
  plausible, and avoid "next <weekday>" when request_date is the day before that weekday.

## Splits & integrity

- train 2,300 / val 200 / test 500, stratified by bucket + language + tool.
- Test set is **frozen**: after verification, record a SHA-256 in `data/splits/sha256.txt`.
- No test or val example may appear in any training corpus — including public data and generation
  seeds. Dedup is enforced by the validator (normalized-text exact + near-duplicate check).
- Only val may influence prompt engineering / early stopping. Test runs are final runs.

## Scoring normalization (v1) — implement exactly this in `eval/score.py`

Applied to both gold and predicted arguments before comparison:

1. Trim whitespace; collapse internal whitespace.
2. String values: compare case-insensitively, except IDs (compare as-is, uppercase for safety).
3. Dates: normalize to ISO `YYYY-MM-DD` (already ISO in gold).
4. Enums: exact string after case normalization.
5. Numbers: compare as numbers (int vs float allowed), tolerance 0.
6. Missing vs `null` argument: both count as absent.

Metric definitions:

- **Tool selection accuracy** — predicted call set's tool name(s) match gold exactly (for `call`).
  Single-call comparison for scoring v1 (multi-call is a stretch bucket, excluded).
- **Argument exact match** — for correctly selected tools: all gold arguments present and equal
  after normalization, no extra arguments. Report per-argument and full-call.
- **No-tool accuracy** — `none` examples where the model emits no call.
- **Clarify accuracy** — `clarify` examples where the model emits no call AND asks a question
  (manual rule: output contains no tool call and ends with "?" or states the missing info).
- **Parse validity** — fraction of outputs that are strict JSON tool calls (report separately;
  malformed = wrong for accuracy).

## Baseline accuracy policy (the $0 rule)

The frontier baseline is NOT run on this test set (no API spend). Its accuracy is **assumed** from
published scores and MUST be labeled as an assumption wherever reported:

- Source: Berkeley Function Calling Leaderboard (BFCL v4) and vendor model cards, retrieved on the
  build date. Example reference point already on record: Qwen3.5-9B reports BFCL-V4 66.1 and
  Qwen3.5-4B reports 50.3 on their model cards (general benchmark, not Routely-specific).
- The write-up states the assumption, the source, the retrieval date, and the caveat that
  domain-specific accuracy may differ.

The baseline **cost** needs no runs at all:

```
monthly_usd = 2,000,000 calls x (t_in x p_in + t_out x p_out)
```

`t_in` measured by tokenizing tools.json + system prompt + request; `t_out` from candidate runs
(or assumed, and labeled). Prices live in `cost/prices.json` with a retrieval date.
