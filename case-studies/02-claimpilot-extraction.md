# Case Study 2 — ClaimPilot: Structured Extraction

## Your task
You own this case study from start to finish. You'll build a realistic demo showing how LeanInference took a (fake) company that sent every document to an expensive frontier model, researched and tested cheaper options, and landed on the best solution for them. The final deliverable is the story, the numbers, and the code that produced the numbers.

The story we tell the client:
> "You were paying frontier-model prices to turn claim documents into JSON, and sending customer personal data to a third-party API. We built an eval set from your documents, tested several alternatives, and found a fine-tuned small model that extracts just as accurately, costs a fraction as much, and runs inside your own infrastructure."

## The fake company
- **Name:** ClaimPilot, an insurtech that processes health and auto insurance claims.
- **Feature:** incoming claim forms, invoices, and repair/medical bills (as text, after OCR) are converted into a fixed JSON schema for their claims system.
- **Current setup:** every document goes to a frontier model with the schema in the prompt.
- **Volume:** assume 500K documents/month, about 1,500 tokens each. Write down the current monthly bill using real list pricing.
- **Extra pain point:** compliance doesn't like personal data leaving their cloud.

## Step 1 — Get the data
1. Define a **fixed schema of ~15 fields**: `claim_id`, `policy_number`, `claimant_name`, `date_of_incident`, `date_of_service`, `provider_name`, `line_items[] {description, amount}`, `total_amount`, `currency`, `diagnosis_or_damage_code`, `vehicle_vin` (auto only), `claim_type`, `payout_iban` (fake), `notes`.
2. Generate **~2,000 synthetic documents** with their ground-truth JSON. Vary the formats a lot: formal invoices, messy emails, handwritten-style OCR noise, tables, different date formats, multiple currencies, missing fields (the correct value is `null`).
3. Check public datasets for realism and extra training data: **CORD**, **SROIE** (receipts), **FUNSD**, **DocILE** (invoices). Use the OCR text versions.
4. Use only fake personal data (generate it with Faker). No real personal data anywhere.
5. Split: **train 1,500 / val 100 / test 400**. The test set must be **hand-verified** and never used for training or prompt tuning.

## Step 2 — Get the models
- **Baseline:** the frontier model ClaimPilot "currently uses".
- **Candidate A:** a cheaper hosted model (mini/flash class) with structured output / JSON mode.
- **Candidate B:** a small Qwen model (newest small open-weight Qwen; check for Qwen 3.8 small sizes, otherwise the newest Qwen 3.x in the 4–9B range), fine-tuned with LoRA/QLoRA, using constrained decoding to the JSON schema (e.g. vLLM guided decoding or Outlines).
- **Candidate C:** **Needle 3** from Cactus Compute (https://cactuscompute.com/needle). It supports structured extraction. Fine-tune with LoRA and export to `.cact`. Check the license on their GitHub.

## Step 3 — Test
Run every model on the same 400-document test set **before and after** fine-tuning. Measure:
- Field-level precision / recall / F1 (overall and per field)
- Whole-document exact match (%)
- Valid JSON that matches the schema (%)
- Accuracy on `null` fields (doesn't invent missing data)
- Accuracy on `total_amount` (money fields matter most, so report them separately)
- p95 latency, cost per 1K documents, monthly cost at 500K documents
- Where it runs (third-party API vs. self-hosted vs. on-device)

## Step 4 — Pick the winner
Choose the best solution for ClaimPilot: the cheapest option that meets or beats baseline F1, with total-amount accuracy at least equal to the baseline, and that can run inside their own cloud. If the result surprises us, report it honestly.

## Deliverables
1. `data/`: schema, generator script, train/val/test files.
2. `train/`: reproducible fine-tuning scripts/configs.
3. `eval/`: one script that prints the results table and the per-field breakdown.
4. **Results table:**
   | Model | Field F1 (before → after) | Doc exact | Valid JSON | Amount acc | p95 | $/1K docs | Monthly $ | Runs where |
5. **5–10 failure examples** from the winning model, explained.
6. **A one-page client write-up:** what they were using → what we tested → what we recommend → savings plus the privacy/compliance benefit.

## Rules
- No fallback to the frontier model in the final solution.
- Every number must come from a real run.
- Log every experiment so we can show the testing journey.
