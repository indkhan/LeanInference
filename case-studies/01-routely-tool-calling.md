# Case Study 1 — Routely: Tool Calling

## Your task
You own this case study from start to finish. You'll build a realistic demo showing how LeanInference took a (fake) company that sent every request to an expensive frontier model, researched and tested cheaper options, and landed on the best solution for them. The final deliverable is the story, the numbers, and the code that produced the numbers.

The story we tell the client:
> "You were using a frontier model for every tool call. We built an eval set from your real usage, tested several alternatives, and found that a fine-tuned small model gives you the same (or better) accuracy at a fraction of the cost and latency. Here's the proof."

## The fake company
- **Name:** Routely, a logistics SaaS (shipments, pickups, drivers, warehouses).
- **Feature:** an in-app assistant that turns user requests ("move my Tuesday pickup to Thursday 3pm", "where is order 88213?") into function calls against Routely's API.
- **Current setup:** every request goes to a frontier model (pick a current top model, e.g. GPT/Claude/Gemini flagship) with all tool schemas in the prompt.
- **Volume:** assume 2M tool calls/month. Write down the current monthly bill using real list pricing.

## Step 1 — Get the data
1. Define **~40 tools** as JSON schemas, e.g. `track_shipment(order_id)`, `reschedule_pickup(pickup_id, date, time_window)`, `create_return(order_id, reason)`, `assign_driver(...)`, `get_eta(...)`, `cancel_order(...)`, `update_address(...)`, `request_refund(...)`. Include a few that look alike on purpose (e.g. `cancel_order` vs `cancel_pickup`) so the eval actually tests something.
2. Generate **~3,000 synthetic user requests** with the correct tool call (name + arguments). Use a frontier model to generate them, then fix mistakes by hand. Mix in:
   - typos, slang, multilingual requests (some Spanish/Hindi)
   - relative dates ("next Friday")
   - requests that need **no tool** or are ambiguous (the correct answer is to ask a clarifying question)
3. Also check public datasets for extra or pre-training data: **BFCL v4**, **DroidCall**, **xLAM / APIGen function-calling**, **Glaive function-calling** on Hugging Face.
4. Split: **train 2,300 / val 200 / test 500**. The test set must be **hand-verified** and never used for training or prompt tuning.

## Step 2 — Get the models
- **Baseline:** the frontier model Routely "currently uses".
- **Candidate A:** a cheaper hosted model (a mini/flash-class model) with a well-engineered prompt.
- **Candidate B:** a small Qwen model (use the newest small Qwen with open weights; check Hugging Face for Qwen 3.8 small sizes, otherwise use the newest Qwen 3.x in the 4–9B range), fine-tuned with LoRA/QLoRA.
- **Candidate C:** **Needle 3** from Cactus Compute (https://cactuscompute.com/needle). Install with `pip install cactus-needle`. Fine-tune with LoRA on the frozen base, export to `.cact`. Check the license on their GitHub before we use it in client material.

## Step 3 — Test
Run every model on the same 500-example test set **before and after** fine-tuning (where it applies). Measure:
- Tool selection accuracy (%)
- Exact-match arguments (%)
- Correct "no tool / ask clarification" (%)
- p50 / p95 latency (state the hardware used)
- Cost per 1M calls, and monthly cost at 2M calls
- Model size (API / GB / MB)

## Step 4 — Pick the winner
Choose the best solution for Routely: the cheapest option that meets or beats baseline accuracy. If the winner isn't the one we expected, say so. An honest result is the selling point.

## Deliverables
1. `data/`: tool schemas, train/val/test files, plus the generation script.
2. `train/`: fine-tuning scripts/configs for Qwen and Needle (reproducible).
3. `eval/`: one script that runs every model on the test set and prints the results table.
4. **Results table** (markdown):
   | Model | Tool acc (before → after) | Arg exact | No-tool acc | p95 latency | $/1M | Monthly $ | Size |
5. **5–10 failure examples** from the winning model, explained.
6. **A one-page client write-up:** what they were using → what we tested → what we recommend → savings ($/month, % cheaper, latency gain).

## Rules
- No fallback to the frontier model in the final solution. The winner has to stand alone.
- Don't fake numbers. The company and data are synthetic, but every metric must come from a real run.
- Log every experiment (model, data size, hyperparameters, score) so we can show the "we tested many things" journey.
