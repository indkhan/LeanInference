# Failure analysis - the winner (C: Qwen3.5-0.8B LoRA label generator)

Source: `artifacts/preds_c_qwen_after.json` (700-ticket hand-verified test set).
Intent: **700/700 correct** (macro-F1 1.000). Sentiment: **700/700 correct**. Every output was in the
`intent|urgency|sentiment` format (0 unparseable). **All 8 errors are the same kind: urgency `high`
predicted as `low`.** That is the costly direction (a missed urgent ticket), so it is worth reading them.

## The one failure mode: an urgent situation written with a polite "no rush"

The labelling rule (the generator `data/make_tickets.py`, the hand-verification log and the hosted
classification prompt) is:
*ignore polite phrases like "no rush" when the situation itself is urgent* - a locked account, a demand
for a manager, an address change before dispatch. In all 8 misses the ticket contains such a phrase and
the model followed the phrase instead of the situation.

| ticket | gold | model | text (abridged) |
|---|---|---|---|
| t04407 | high | low | "account locked help ... And note: **no rush at all**." |
| t01640 | high | low | "third broken item in a row ... I want a manager, now. **(not urgent, just asking)** I'm honestly furious." |
| t00846 | high | low | "Enough. Every answer I get is a template. I demand someone in charge ... **no rush at all**. this is unacceptable." |
| t03667 | high | low | "arrived with a torn seam ... What now, order ORD-62381? **when you get a minute**. very disappointed." |
| t02442 | high | low | "Money back pls, order ORD-85177 ... **whenever you can**." |
| t02007 | high | low | "exchange the kettle ... change the delivery address on order ORD-13975 ... **when you get a minute**." |
| t00699 | high | low | "I need to change the delivery address on order ORD-47529 ... **(whenever you can)**" |
| t00717 | high | low | "You sent me the black yoga mat but I ordered the blue one ... **when you get a minute**." |

Reading them as a support lead would: t04407, t01640, t00846 and the two address changes (t02007,
t00699 - the parcel ships before "whenever" arrives) are clearly urgent despite the wording. t03667,
t02442 and t00717 are borderline - a customer who says "when you get a minute" about a wrong item
might reasonably be routed as low. Every other model makes this error too: high-urgency recall is
0.943 for this model, 0.936 for Candidate A and the tuned MiniLM classifier, 0.901 for Needle
embeddings.

## What to do about it

1. **Business rule first:** decide whether "account locked" and "change delivery address on an open
   order" are *always* high regardless of wording. If yes, that is a 2-line keyword override in front
   of any model and it removes 3 of the 8 misses (5 of 8 if "I want a manager / someone in charge"
   is added as an escalation rule).
2. **More training examples of the pattern:** of the 1,930 training tickets that contain a softener
   ("no rush", "whenever you can", "when you get a minute", "not urgent"), only 59 (3%) are labelled
   high - the model learned "softener means low". Twenty to fifty extra examples of it are cheap to generate and fit in one retrain (34 min).
3. Keep measuring high-urgency recall on the unseen-ticket view (571 tickets): 0.937 there, so the
   result is not driven by the 129 test tickets that duplicate training tickets.
