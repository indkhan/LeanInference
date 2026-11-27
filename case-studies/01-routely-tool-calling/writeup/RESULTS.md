# Routely tool calling - results (500-example hand-verified test set)

Volume 2,000,000 calls/month. Accuracy = the whole response correct (tool, every argument, or the right clarify / no-tool decision). Latency: hosted = API round trip; Qwen = RTX 3050 6 GB laptop GPU, 4-bit, batch 1; Needle = laptop CPU.

| Model | Overall (before -> after) | Tool acc (before -> after) | Arg exact | No-tool / clarify acc | p95 latency | $/1M calls | Monthly $ | Size | Runs where |
|---|---|---|---|---|---|---|---|---|---|
| Status quo: GPT-6 Astra (cost only, not run) | - | - | - | - | - | 58,040 | 116,080 | API | third-party API |
| Baseline proxy: Nemotron 3 Ultra 550B, all 40 schemas | pending (0/500) | | | | | | | 550B (API) | third-party API |
| Candidate A: MiniMax M3.1 Flash, all 40 schemas | **83.6%** | 98.6% | 82.2% | 97.5% | 5.0 s | 1,502 | 3,005 | API | third-party API |
| B1: Qwen3.5-0.8B LoRA, no schemas in prompt | 7.6% -> **78.8%** | 0.0% -> 95.7% | 78.7% | 97.5% | 5.5 s | 646 | 1,292 | 0.8B, 4-bit ~0.6 GB + 25 MB LoRA | self-hosted GPU |
| B2: Qwen3.5-2B LoRA, no schemas in prompt | 15.8% -> **82.0%** | 1.7% -> 97.9% | 81.3% | 94.9% | 5.1 s | 646 | 1,292 | 2B, 4-bit ~1.5 GB + LoRA | self-hosted GPU |
| C: Needle 3 LoRA (.cact), compact tool list | 2.0% -> **43.2%** | 12.8% -> 59.4% | 59.6% | 84.8% | 0.8 s | 65 | 130 | 63 MB .cact (W4A8) | self-hosted CPU / on-device |

Status quo = GPT-6 Astra list price x 5,304 input tokens (all 40 schemas, measured) + 100 output tokens per call.
Cost basis per row:
- Candidate A: MiniMax M3.1 Flash, all 40 schemas: MiniMax M3 (closest paid listing of minimax-m3.1-flash) list ($0.3/$1.2) x 4524 in + 121 out tokens/call (measured)
- B1: Qwen3.5-0.8B LoRA, no schemas in prompt: 3 x NVIDIA L4 24GB (Runpod, on-demand) ($0.59/h) always-on at measured p50 3.15 s
- B2: Qwen3.5-2B LoRA, no schemas in prompt: 3 x NVIDIA L4 24GB (Runpod, on-demand) ($0.59/h) always-on at measured p50 3.01 s
- C: Needle 3 LoRA (.cact), compact tool list: 1 x AWS c7i.xlarge (4 vCPU) ($0.1785/h) always-on at measured p50 0.52 s

## Accuracy by bucket

| Model | clear | confusable | relative_date | multilingual | no_tool | clarify |
|---|---|---|---|---|---|---|
| Candidate A: MiniMax M3.1 Flash, all 40 schemas | 84.0% | 62.3% | 96.0% | 76.9% | 100.0% | 95.1% |
| B1: Qwen3.5-0.8B LoRA, no schemas in prompt | 82.5% | 67.5% | 60.0% | 53.8% | 100.0% | 95.1% |
| B2: Qwen3.5-2B LoRA, no schemas in prompt | 83.6% | 68.8% | 78.0% | 73.1% | 100.0% | 90.2% |
| C: Needle 3 LoRA (.cact), compact tool list | 45.9% | 19.5% | 14.0% | 15.4% | 97.4% | 73.2% |

## Winner (brief Step 4 rule)

Bar = PROVISIONAL - baseline proxy incomplete, Candidate A used: 83.6% overall.
- B1: Qwen3.5-0.8B LoRA, no schemas in prompt: 78.8% (< bar), $1,292/month
- B2: Qwen3.5-2B LoRA, no schemas in prompt: 82.0% (< bar), $1,292/month
- C: Needle 3 LoRA (.cact), compact tool list: 43.2% (< bar), $130/month

**Winner: Candidate A: MiniMax M3.1 Flash, all 40 schemas**

## Failure examples - Candidate A: MiniMax M3.1 Flash, all 40 schemas (82 of 500 wrong)

- `gen12-0173` [clarify] none_instead_of_clarify: "you know what to do" (today 2026-04-08)
  - gold `{"type": "clarify", "reason": "no ids or issue; request unclear"}`
  - pred `{"type": "none"}`
- `gen12-0062` [clarify] called_instead_of_clarify: "cancel the one I set up this morning" (today 2026-02-26)
  - gold `{"type": "clarify", "reason": "ambiguous: which order or pickup"}`
  - pred `{"type": "call", "calls": [{"name": "list_pickups", "arguments": {"date": "2026-02-26", "status": "scheduled"}}]}`
- `gen06-0175` [clear] wrong_value: "we won't ship this after all: ORD-61544" (today 2026-06-09)
  - gold `{"type": "call", "calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-61544", "reason": "shipment no longer needed"}}]}`
  - pred `{"type": "call", "calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-61544", "reason": "Customer no longer wants to ship this order"}}]}`
- `gen02-0036` [clear] extra_args: "Can you get 6 crates collected from ADR-7708 on May 22 in the afternoon? Seller will be on site until 4." (today 2026-05-20)
  - gold `{"type": "call", "calls": [{"name": "schedule_pickup", "arguments": {"address_id": "ADR-7708", "date": "2026-05-22", "time_window": "12:00-16:00", "package_count": 6}}]}`
  - pred `{"type": "call", "calls": [{"name": "schedule_pickup", "arguments": {"address_id": "ADR-7708", "date": "2026-05-22", "time_window": "12:00-16:00", "package_count": 6, "notes": "Seller will be on site `
- `gen05-0084` [clear] clarified_instead_of_call: "okay, walk me through the whole network" (today 2026-04-14)
  - gold `{"type": "call", "calls": [{"name": "list_warehouses", "arguments": {}}]}`
  - pred `{"type": "clarify", "reason": "Do you want an overview of the entire Routely network (warehouses, drivers, shipment flow, and coverage), or the current activity for your account? If you mean your ship`
- `gen01-0266` [clear] wrong_tool: "what does order ORD-67011 look like right now?" (today 2026-01-18)
  - gold `{"type": "call", "calls": [{"name": "get_order", "arguments": {"order_id": "ORD-67011"}}]}`
  - pred `{"type": "call", "calls": [{"name": "track_shipment", "arguments": {"order_id": "ORD-67011"}}]}`
- `gen09-0130` [relative_date] wrong_date: "clone ORD-37025, pickup Friday next week" (today 2026-12-28)
  - gold `{"type": "call", "calls": [{"name": "duplicate_order", "arguments": {"order_id": "ORD-37025", "new_pickup_date": "2027-01-08"}}]}`
  - pred `{"type": "call", "calls": [{"name": "duplicate_order", "arguments": {"order_id": "ORD-37025", "new_pickup_date": "2027-01-01"}}]}`
- `gen06-0165` [clear] wrong_value: "we want to dispute INV-96090 in full, services were never delivered" (today 2026-11-02)
  - gold `{"type": "call", "calls": [{"name": "dispute_charge", "arguments": {"invoice_id": "INV-96090", "reason": "services not delivered"}}]}`
  - pred `{"type": "call", "calls": [{"name": "dispute_charge", "arguments": {"invoice_id": "INV-96090", "reason": "Services were never delivered"}}]}`
- `gen06-0209` [clear] wrong_value: "CUS-1035: $95 credit, the customs docs delay was our fault" (today 2026-12-13)
  - gold `{"type": "call", "calls": [{"name": "apply_credit", "arguments": {"customer_id": "CUS-1035", "amount_usd": 95, "reason": "customs docs delayed by us"}}]}`
  - pred `{"type": "call", "calls": [{"name": "apply_credit", "arguments": {"customer_id": "CUS-1035", "amount_usd": 95, "reason": "Customs docs delay was our fault"}}]}`
- `gen02-0088` [clear] wrong_value: "Please cancel PCK-81234. Tenant moved out, no parcels to collect." (today 2026-02-13)
  - gold `{"type": "call", "calls": [{"name": "cancel_pickup", "arguments": {"pickup_id": "PCK-81234", "reason": "tenant moved out, no parcels to collect"}}]}`
  - pred `{"type": "call", "calls": [{"name": "cancel_pickup", "arguments": {"pickup_id": "PCK-81234", "reason": "Tenant moved out, no parcels to collect."}}]}`

