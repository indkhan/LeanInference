# Helpdeskly ticket triage - results (700-ticket hand-verified test set)

Prices: `eval/prices.json` (2026-10-20). Volume: 1,000,000 tickets/month.

| Approach | Intent macro-F1 (before -> after) | High-urgency recall | Urgency acc | Sentiment acc | p95 | $/1M | Monthly $ | Size | Training | Runs where |
|---|---|---|---|---|---|---|---|---|---|---|
| Status quo: Claude Opus 5.5 (cost only, not run) | - | - | - | - | - | 1,120 | 1,120 | API | none | third-party API |
| Baseline proxy: gpt-oss-120b (zero-shot) | 0.969 | 0.943 | 0.833 | 0.779 | 2.76 s | 2,202 | 2,202 | 117B (API) | none | third-party API |
| Candidate A: MiniMax M3.1 Flash (few-shot) | 1.000 | 0.936 | 0.953 | 0.814 | 3.66 s | 563 | 563 | API | none (prompt + 24 shots) | third-party API |
| B1: MiniLM-L6 embeddings + LR (no generation) | 0.608 -> 0.972 | 0.936 | 0.740 | 0.837 | 0.02 s | 130 | 130 | 22M params (~90 MB) | CPU: embed 4,000 tickets 367 s + fit 30 s | self-hosted CPU |
| B2: Needle 3 embeddings + LR (no generation) | 0.184 -> 0.952 | 0.901 | 0.849 | 0.896 | 0.22 s | 130 | 130 | Needle 3 (63 MB .cact) | CPU: embed + fit 616 s | self-hosted CPU / on-device |
| C: Qwen3.5-0.8B LoRA label generator | 0.551 -> 1.000 | 0.943 | 0.989 | 1.000 | 0.98 s | 431 | 431 | 0.8B fp16 (1.6 GB) + 25 MB LoRA | RTX 3050: 34 min, 2 epochs | self-hosted GPU |
| D: Needle 3 LoRA classifier (.cact) | 0.244 -> 0.576 | 0.858 | 0.904 | 0.967 | 0.51 s | 130 | 130 | 63 MB .cact (W4A8) | RTX 3050 (WSL): 33 min, 2 epochs | self-hosted CPU / on-device |

Cost basis per row:
- Baseline proxy: gpt-oss-120b (zero-shot): Claude Opus 5.5 list x 469 in + 16 out tokens/ticket (measured)
- Candidate A: MiniMax M3.1 Flash (few-shot): MiniMax M3 (closest paid listing; minimax-m3.1-flash is free-only on OpenRouter) list x 1813 in + 16 out tokens/ticket (measured)
- B1: MiniLM-L6 embeddings + LR (no generation): 1 x AWS c7i.xlarge (4 vCPU, 8 GiB, us-east-1 on-demand Linux) ($0.1785/h) at p50 12 ms
- B2: Needle 3 embeddings + LR (no generation): 1 x AWS c7i.xlarge (4 vCPU, 8 GiB, us-east-1 on-demand Linux) ($0.1785/h) at p50 96 ms
- C: Qwen3.5-0.8B LoRA label generator: 1 x NVIDIA L4 24GB (Runpod Secure Cloud, on-demand) ($0.59/h) at p50 679 ms
- D: Needle 3 LoRA classifier (.cact): 1 x AWS c7i.xlarge (4 vCPU, 8 GiB, us-east-1 on-demand Linux) ($0.1785/h) at p50 372 ms

Status quo = Claude Opus 5.5 at $4.0/$20.0 per MTok; the baseline proxy row is priced at that list price with the proxy's measured tokens.
p95 = single-request latency (laptop RTX 3050 for C; laptop CPU for B1/B2/D; API round trip for hosted).

## Unseen-ticket view (571 of 700 test tickets)

129 test tickets duplicate a training ticket once digits are masked (49 verbatim). The test set is kept as-is; this view drops those tickets so a trained model cannot win by memorising.

| Approach | Intent macro-F1 | High-urgency recall | Sentiment acc |
|---|---|---|---|
| Baseline proxy: gpt-oss-120b (zero-shot) | 0.973 | 0.937 | 0.762 |
| Candidate A: MiniMax M3.1 Flash (few-shot) | 1.000 | 0.929 | 0.814 |
| B1: MiniLM-L6 embeddings + LR (no generation) | 0.964 | 0.945 | 0.806 |
| B2: Needle 3 embeddings + LR (no generation) | 0.940 | 0.898 | 0.881 |
| C: Qwen3.5-0.8B LoRA label generator | 1.000 | 0.937 | 1.000 |
| D: Needle 3 LoRA classifier (.cact) | 0.550 | 0.843 | 0.962 |

## Winner (brief Step 4 rule)

Bar = baseline proxy: intent macro-F1 0.969, high-urgency recall 0.943.
- Candidate A: MiniMax M3.1 Flash (few-shot): intent 1.000, high-urgency recall 0.936, $563/month
- B1: MiniLM-L6 embeddings + LR (no generation): intent 0.972, high-urgency recall 0.936, $130/month
- B2: Needle 3 embeddings + LR (no generation): intent 0.952, high-urgency recall 0.901, $130/month
- C: Qwen3.5-0.8B LoRA label generator: intent 1.000, high-urgency recall 0.943, $431/month
- D: Needle 3 LoRA classifier (.cact): intent 0.576, high-urgency recall 0.858, $130/month

**Winner: C: Qwen3.5-0.8B LoRA label generator**

Confusion matrix: `results/confusion_intent_c_qwen_after.png`

