# Case Study 3 — Helpdeskly: Ticket Triage / Classification

## Your task
You own this case study from start to finish. You'll build a realistic demo showing how LeanInference took a (fake) company that used an expensive frontier model just to label support tickets, researched and tested cheaper options, and landed on the best solution for them. The final deliverable is the story, the numbers, and the code that produced the numbers.

The story we tell the client:
> "You were paying a frontier model to output a label. We built an eval set from your tickets, tested several approaches, and found that a small model, possibly without any text generation at all, routes tickets as accurately as before for a tiny fraction of the cost."

## The fake company
- **Name:** Helpdeskly, a customer support platform for e-commerce brands.
- **Feature:** every incoming ticket is tagged with **intent** (12 classes), **urgency** (low/medium/high), and **sentiment** (neg/neutral/pos), then routed to a team.
- **Current setup:** every ticket goes to a frontier model with a classification prompt.
- **Volume:** assume 1M tickets/month, about 200 tokens each. Write down the current monthly bill using real list pricing.

## Step 1 — Get the data
1. Define **12 intents**, e.g. `order_status`, `refund_request`, `return_exchange`, `damaged_item`, `wrong_item`, `payment_issue`, `account_access`, `shipping_address_change`, `product_question`, `cancel_order`, `complaint_escalation`, `other`. Write a one-line definition for each.
2. Generate **~5,000 synthetic tickets** with labels. Make them realistic: angry customers, multiple issues in one ticket, very short tickets ("where is my stuff??"), long rambling ones, non-English ones, and an uneven class distribution like real traffic.
3. Check public datasets: **Bitext customer support**, **BANKING77**, **CLINC150**, **SNIPS** on Hugging Face.
4. Split: **train 4,000 / val 300 / test 700**. The test set must be **hand-verified** and never used for training or prompt tuning.

## Step 2 — Get the models
- **Baseline:** the frontier model Helpdeskly "currently uses".
- **Candidate A:** a cheaper hosted model (mini/flash class) with a tuned prompt and few-shot examples.
- **Candidate B:** **embeddings + a simple classifier.** Use **Needle 3 embeddings** (https://cactuscompute.com/needle) and one standard open embedding model, then put logistic regression or kNN on top (scikit-learn). No generation at all.
- **Candidate C:** a small Qwen model (newest small open-weight Qwen; check for Qwen 3.8 small sizes, otherwise the newest Qwen 3.x in the 1–4B range), fine-tuned with LoRA to output the three labels.
- **Candidate D:** **Needle 3**, fine-tuned with LoRA as a classifier, exported to `.cact`. Check the license on their GitHub.

## Step 3 — Test
Run every model on the same 700-ticket test set **before and after** fine-tuning or training. Measure:
- Intent accuracy and macro-F1 (macro matters because the classes are imbalanced)
- Urgency accuracy, and specifically **recall on high urgency** (missing an urgent ticket is the costly error)
- Sentiment accuracy
- Confusion matrix for intent
- p95 latency, cost per 1M tickets, monthly cost at 1M tickets
- Training cost and time for each approach

## Step 4 — Pick the winner
Choose the best solution for Helpdeskly: the cheapest option whose intent macro-F1 and high-urgency recall meet or beat the baseline. If the simple embeddings + classifier approach wins, that's a great result, so show it proudly.

## Deliverables
1. `data/`: label definitions, generator script, train/val/test files.
2. `train/`: classifier and fine-tuning scripts (reproducible).
3. `eval/`: one script that prints the results table and saves the confusion matrix image.
4. **Results table:**
   | Approach | Intent macro-F1 (before → after) | High-urgency recall | Sentiment acc | p95 | $/1M | Monthly $ | Size |
5. **5–10 failure examples** from the winning model, explained.
6. **A one-page client write-up:** what they were using → what we tested → what we recommend → savings.

## Rules
- No fallback to the frontier model in the final solution.
- Every number must come from a real run.
- Log every experiment so we can show the testing journey.
