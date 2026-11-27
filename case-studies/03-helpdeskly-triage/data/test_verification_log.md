# Test-set hand-verification log — case study 3 (Helpdeskly)

**Protocol** (defined before reading, in `verify_test.py` header):
- All 700 test rows read in full, in three chunked passes, twice (two generator revisions).
- **Primary-intent rule** — for multi-issue tickets the issue the customer *leads with* is the gold label; secondary issues ("Also, ...") never override.
- **Urgency rule** — explicit time cues + business risk decide the label. An explicit *contradictory* cue ("no rush at all") on a structurally urgent ticket is kept as the structural label — the built-in contradiction traps are part of the eval by design (~8% of tickets), mirroring real traffic where passive-aggressive politeness hides urgency.
- **Sentiment rule** — tone-based; sarcastic openers ("Perfect, just perfect.", "Fantastic.") over complaint bodies negate the literal words.

## Pass 1 (v1 generator, 2026-10-17)
Read 700/700. All labels sound. **Rejected & fixed at the source 3 data-quality defects** (fixed in `make_tickets.py`, data regenerated — no downstream run had consumed the data yet):
1. Urgency cue pool leaked order-domain phrasing ("the last one I ordered never arrived in time", "we fly out Friday") onto account/pre-sale/other tickets where it is semantically nonsense → made the high-urgency cue pool **intent-aware** (order-shape intents only get birthday/flight/move cues).
2. Non-English templates injected English day slots mid-sentence ("Wo bleibt meine Bestellung ORD-…? Bestellt two days ago.") → removed, foreign templates are self-contained.
3. Positive-sentiment cues read as post-help gratitude on tickets nothing had helped yet → reworded to pre-help phrasing ("thanks in advance", "long-time customer, generally happy").

## Pass 2 (v2 generator, current locked files)
Read 700/700 again. **Zero label changes required** — every row interrogated against the three rules above. Two rows deliberately kept with notes:
- `t02656` wrong_item/low: quotes identical sizes ("I ordered M... the box says M") — contradictory in the same way rare real tickets are; kept as realistic noisemakers.
- `t04706`-return / `t03270`-class tickets swapping a colour to itself appear off-hand but swap templates sample independent colours; no actual contradiction found beyond the size case above.

Conclusion: **test set locked.** 700 rows, 12 intents (min class support 22), 141 high-urgency rows (20.2%), 192 negative / 415 neutral / 93 positive.

— t3, recorded 2026-10-17 during the case-3 build.
