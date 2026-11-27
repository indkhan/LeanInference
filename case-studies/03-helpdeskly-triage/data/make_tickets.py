"""Synthetic support-ticket generator for the Helpdeskly case study.

Produces ~5,000 realistic tickets with ground-truth labels:
  intent (12 classes), urgency (low/medium/high), sentiment (negative/neutral/positive).

Design notes
- Labels are assigned by construction: the primary intent template chosen for a
  multi-issue ticket defines the gold label ("primary = the issue the customer
  leads with"), urgency/sentiment come from per-intent distributions.
- Hard cases by design: ultra-short messages, rambles, all-caps anger, sarcasm,
  multi-issue tickets, non-English (es/de/fr), code-switched English, and
  urgency/sentiment contradiction flips (text cue says the opposite of the label,
  ~8% of tickets) - these are where frontier-style zero-shot models lose recall.
- Every ticket carries a `notes.cues` object: the verbatim snippets that justify
  each label. Used for the Needle reasoning lines and for hand-verification only;
  model code must never read it.

Run:  python make_tickets.py --n 5000 --out tickets_raw.jsonl
"""
import argparse
import json
import random
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
SEED = 13

ITEMS = [
    "wireless earbuds", "kettle", "yoga mat", "desk lamp", "phone case",
    "charging cable", "sneakers", "bluetooth speaker", "coffee maker",
    "backpack", "smartwatch band", "hair dryer", "computer mouse",
    "water bottle", "garden hose", "air fryer", "mechanical keyboard",
    "over-ear headphones", "plant pot", "electric shaver",
]
ITEMS2 = ["kitchen scale", "toaster", "desk chair", "monitor stand", "bookends"]
COLORS = ["black", "white", "blue", "green", "red", "grey"]
SIZES = ["S", "M", "L", "XL", "9", "10", "41", "42"]
AMOUNTS = ["$17.25", "$34.50", "$59.90", "$89", "$120.99", "$239"]
DAYS = ["two days ago", "last week", "this morning", "yesterday", "on the 12th", "last Friday"]
NTHS = ["second", "third", "second", "third"]
EMAILS = ["my hotmail", "my gmail", "my work email", "the email I signed up with"]
ADDRS_2 = ["Birch Lane 14 in Rotterdam", "12 Maple Court in Leeds", "402 Oak Avenue, Portland"]
PHONES = []

# ----------------------------------------------------------------------------- templates
# Each template: format string using slot names. `notes` keys point at cue words.
T = {}

T["order_status"] = [
    "Hi, I placed order {oid} {day} and there has been no tracking update since. Where is it?",
    "Order {oid} still says processing {day}. Any idea when it actually ships?",
    "The tracking for {oid} hasn't moved in five days. Is the parcel lost?",
    "Could you check what happened to order {oid}? The courier never picked it up as far as I can tell.",
    "Where is my stuff?? {oid}",
    "My order {oid} was supposed to arrive {day} and the tracking page still shows nothing.",
]
T["refund_request"] = [
    "I want a refund for the {item}, order {oid}. It's still unopened and I changed my mind.",
    "Please refund {amount} for order {oid}: the {item} simply doesn't do what the page said it would.",
    "I'd like my money back for the {item} from order {oid}. It arrived fine but it's not what I expected.",
    "Money back pls, order {oid}, the {item}.",
    "The {item} stopped working after two uses. Refund order {oid} please.",
]
T["return_exchange"] = [
    "I need to return the {item} from order {oid}. Can you tell me the process and where the return label is?",
    "Can I exchange the {item} for a size {size}? I clearly ordered the wrong size.",
    "The {item} doesn't fit. How do I set up a return for order {oid}?",
    "I want to swap the {color} {item} for the same thing in {color2}, order {oid}.",
]
T["damaged_item"] = [
    "The {item} arrived broken: the box was crushed and the lid is cracked straight through. Order {oid}.",
    "Just opened the package and the {item} is shattered. This is the {nth} time something like this happens!",
    "There is a huge dent on the {item} and one cable won't even connect. Order {oid}.",
    "The {item} arrived with a torn seam and stuffing coming out. What now, order {oid}?",
]
T["wrong_item"] = [
    "You sent me the {color2} {item} but I ordered the {color} one. Order {oid}.",
    "I received a {item2} instead of the {item} I actually ordered. {oid}",
    "Wrong size arrived: I ordered {size} and the box says {size2}. Order {oid}.",
    "The box contained two of the same {item2} and none of the {item}. {oid}",
]
T["payment_issue"] = [
    "My card was declined at checkout twice, and there is definitely money on it. Can you look into it?",
    "I was charged {amount} twice for order {oid}. One of them needs to be reversed.",
    "Checkout kept failing with payment error 3 {day}. Is something wrong with my card on your side?",
    "I paid {amount} but the site says transaction failed. Did the money actually leave my account?",
]
T["account_access"] = [
    "I can't log in: the password reset emails never arrive in {email}.",
    "My account is locked after too many attempts. What do I do? {email}",
    "Two-factor codes keep getting rejected. I've been locked out since {day}. {email}",
    "Forgot my password, and now it says the account is locked too. {email}",
]
T["shipping_address_change"] = [
    "I need to change the delivery address on order {oid}: it should go to my new place, {addr}.",
    "Just realised I used my old address for {oid}. Please update it to {addr} before it ships.",
    "Can you reroute order {oid}? The address on it is half missing, correct one is {addr}.",
]
T["product_question"] = [
    "Does the {item} come with batteries or do I need to buy them separately?",
    "Will the {item} charging cable work with a Pixel 9? Thinking of ordering tonight.",
    "Is the {item} available in size {size} again? Page says out of stock.",
    "What is the difference between the {item} and the {item2}? Which one is better for daily use?",
    "Does the {item} shrink in the wash? Sizing chart is unclear.",
]
T["cancel_order"] = [
    "Please cancel order {oid}. I ordered the wrong thing and reordered seconds later.",
    "Can I still cancel {oid}? It hasn't shipped yet as far as I can see.",
    "Cancel the {item} order please ({oid}), found it cheaper elsewhere, sorry!",
]
T["complaint_escalation"] = [
    "This is the {nth} broken item in a row and your support keeps replying with the same copied answer. I want a manager, now.",
    "I've emailed five times about order {oid} and nobody fixed anything. I want a supervisor to call me.",
    "If nobody gets back to me about my case today I will dispute the charge and post about this everywhere. Get me a human.",
    "Enough. Every answer I get is a template. I demand someone in charge looks at {oid} personally.",
]
T["other"] = [
    "Do your gift cards expire? Thinking of getting one for my sister.",
    "I would like to apply as a wholesale partner. Who do I talk to?",
    "My loyalty points disappeared after my last purchase. Where did they go?",
    "Feedback: the returns label page prints badly in Firefox, worth checking.",
    "Please remove me from the marketing list for good.",
]

# ultra-short variants, <= ~10 words
SHORT = {
    "order_status": ["order 48213 where?", "my stuff where??", "tracking never moved, why"],
    "refund_request": ["money back pls", "refund my order", "I want a refund now"],
    "return_exchange": ["how do I return this", "exchange size pls", "return label where?"],
    "damaged_item": ["it arrived broken", "kettle came shattered", "item broke on arrival"],
    "wrong_item": ["you sent the wrong thing", "wrong size came", "this isn't what I ordered"],
    "payment_issue": ["charged twice, fix it", "card declined twice today", "payment error 3??"],
    "account_access": ["cant log in", "account locked help", "reset email never comes"],
    "shipping_address_change": ["change my address please", "wrong address on order", "reroute parcel pls"],
    "product_question": ["does the kettle come with batteries?", "size 9 back in stock?", "will the cable fit pixel 9?"],
    "cancel_order": ["cancel order 48213", "cancel pls", "cancel it, reorder"],
    "complaint_escalation": ["manager. NOW.", "get me a human", "supervisor, today, or I dispute"],
    "other": ["gift cards expire?", "loyalty points gone", "unsubscribe me"],
}

# non-English: es / de / fr, one or two variants per intent, gold label unchanged
FOREIGN = {
    "order_status": {
        "es": ["¿Dónde está mi pedido {oid}?", "Sigo sin recibir el pedido {oid}."],
        "de": ["Wo bleibt meine Bestellung {oid}?", "Ich habe mein Paket {oid} immer noch nicht."],
        "fr": ["Où est ma commande {oid} ?", "Je n'ai toujours pas reçu la commande {oid}."],
    },
    "refund_request": {
        "es": ["Quiero el reembolso del {item}, pedido {oid}."],
        "de": ["Ich möchte das Geld für den {item} zurückerstatten bekommen, Bestellung {oid}."],
        "fr": ["Je veux un remboursement pour le {item}, commande {oid}."],
    },
    "return_exchange": {
        "es": ["Quiero devolver el {item} y cambiarlo por talla {size}."],
        "de": ["Ich möchte den {item} zurückschicken und die Größe {size} bekommen."],
        "fr": ["Je voudrais retourner le {item} et l'échanger contre la taille {size}."],
    },
    "damaged_item": {
        "es": ["El {item} llegó roto, pedido {oid}."],
        "de": ["Der {item} ist kaputt angekommen, Bestellung {oid}."],
        "fr": ["Le {item} est arrivé cassé, commande {oid}."],
    },
    "wrong_item": {
        "es": ["Me enviaron el {item2} en vez del {item} que pedí."],
        "de": ["Ich habe den {item2} statt des bestellten {item} erhalten."],
        "fr": ["J'ai reçu le {item2} au lieu du {item} commandé."],
    },
    "payment_issue": {
        "es": ["Me cobraron {amount} dos veces."],
        "de": ["Mir wurde {amount} doppelt abgebucht."],
        "fr": ["On m'a débité deux fois {amount}."],
    },
    "account_access": {
        "es": ["No puedo iniciar sesión, cuenta bloqueada."],
        "de": ["Ich kann mich nicht einloggen, Konto gesperrt."],
        "fr": ["Je ne peux pas me connecter, compte bloqué."],
    },
    "shipping_address_change": {
        "es": ["Cambien la dirección del pedido {oid} por favor."],
        "de": ["Bitte ändern Sie die Lieferadresse für Bestellung {oid}."],
        "fr": ["Veuillez changer l'adresse de livraison de la commande {oid}."],
    },
    "product_question": {
        "es": ["¿La {item} incluye pilas?"],
        "de": ["Hat der {item} Batterien dabei?"],
        "fr": ["Le {item} est-il livré avec des piles ?"],
    },
    "cancel_order": {
        "es": ["Cancela el pedido {oid} por favor."],
        "de": ["Bitte stornieren Sie Bestellung {oid}."],
        "fr": ["Veuillez annuler la commande {oid}."],
    },
    "complaint_escalation": {
        "es": ["¡Inaceptable! Quiero hablar con un responsable AHORA."],
        "de": ["Ungenügend! Ich verlange sofort mit einem Vorgesetzten zu sprechen."],
        "fr": ["Inacceptable ! Je veux parler à un responsable MAINTENANT."],
    },
    "other": {
        "es": ["¿Vencen las tarjetas de regalo?"],
        "de": ["Verfallen die Geschenkgutscheine?"],
        "fr": ["Les cartes cadeaux expirent-elles ?"],
    },
}

# Code-switched closers, appended to an English ticket (labels unchanged).
SWITCH = [
    ("es", " Por favor respóndanme pronto, gracias."),
    ("de", " Bitte antworten Sie mir bald, danke."),
    ("fr", " Merci de me répondre rapidement."),
]

# urgency / sentiment distribution per intent (primary drives base distribution)
DIST = {
    #            urgency(lo,med,hi)            sentiment(neg,neu,pos)
    "order_status":    [(0.62, 0.30, 0.08), (0.16, 0.70, 0.14)],
    "refund_request":  [(0.38, 0.42, 0.20), (0.34, 0.56, 0.10)],
    "product_question": [(0.88, 0.10, 0.02), (0.06, 0.72, 0.22)],
    "return_exchange": [(0.42, 0.44, 0.14), (0.22, 0.66, 0.12)],
    "damaged_item":    [(0.16, 0.40, 0.44), (0.62, 0.32, 0.06)],
    "payment_issue":   [(0.22, 0.40, 0.38), (0.40, 0.52, 0.08)],
    "cancel_order":    [(0.40, 0.44, 0.16), (0.20, 0.68, 0.12)],
    "wrong_item":      [(0.20, 0.42, 0.38), (0.48, 0.44, 0.08)],
    "account_access":  [(0.18, 0.38, 0.44), (0.28, 0.60, 0.12)],
    "shipping_address_change": [(0.24, 0.40, 0.36), (0.14, 0.70, 0.16)],
    "other":           [(0.82, 0.14, 0.04), (0.12, 0.74, 0.14)],
    "complaint_escalation": [(0.05, 0.25, 0.70), (0.94, 0.06, 0.00)],
}

HI_CUES = ["ASAP", "urgent", "today at the latest", "this is time-sensitive"]
# order-shape intents get birthday/flight/move-deadline cues; they make no sense
# on account or pre-sale tickets (caught during the first hand-review pass)
ORDERY = {"order_status", "refund_request", "return_exchange", "damaged_item",
          "wrong_item", "cancel_order", "payment_issue", "shipping_address_change"}
HI_CUES_ORDERY = HI_CUES + ["my daughter's birthday is Saturday",
                            "we fly out on vacation Friday",
                            "I need it before my move on Thursday",
                            "the event is on Sunday"]
MED_CUES = ["this week if possible", "soon-ish", "before the weekend", "in the next few days"]
LO_CUES = ["no rush at all", "whenever you can", "not urgent, just asking", "when you get a minute"]
NEG_CUES = ["this is unacceptable", "very disappointed", "I'm honestly furious",
            "ridiculous", "seems like I always have bad luck with you"]
POS_CUES = ["thanks in advance", "I appreciate your help", "love your shop otherwise",
            "you've been great every time before", "long-time customer, generally happy"]


def sample(rng, dist):
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(dist):
        acc += p
        if r <= acc:
            return i
    return len(dist) - 1


class Gen:
    """One instantiated ticket plus the cue-notes that explain its labels."""

    def __init__(self, rng, intent):
        self.rng = rng
        self.intent = intent
        self.cues = {"intent": None, "urgency": None, "sentiment": None}

    def slots(self):
        r = self.rng
        return {
            "oid": f"ORD-{r.randrange(10000, 99999)}",
            "item": r.choice(ITEMS),
            "item2": r.choice(ITEMS2),
            "color": r.choice(COLORS),
            "color2": r.choice(COLORS),
            "size": r.choice(SIZES),
            "size2": r.choice(SIZES),
            "amount": r.choice(AMOUNTS),
            "day": r.choice(DAYS),
            "nth": r.choice(NTHS),
            "email": r.choice(EMAILS),
            "addr": r.choice(ADDRS_2),
        }

    def foreign(self):
        r = self.rng
        lang = r.choice(["es", "de", "fr"])
        tmpl = r.choice(FOREIGN[self.intent][lang])
        return lang, tmpl.format(**self.slots())

    def short(self, text):
        r = self.rng
        base = r.choice(SHORT[self.intent])
        return base

    def anger(self, text):
        r = self.rng
        words = text.split()
        for _ in range(min(3, len(words) // 6 + 1)):
            i = r.randrange(len(words))
            words[i] = words[i].upper()
        return " ".join(words) + " " + "!" * r.randrange(2, 4)

    def ramble(self, text):
        r = self.rng
        pre = r.choice([
            "Sorry in advance for the long email, I've been a customer since 2019 and I honestly enjoy your products, ",
            "First of all I want to say your packaging quality is usually great. ",
            "I don't usually write to support, so I hope you read this fully. ",
        ])
        post = r.choice([
            " Also, unrelated: the loyalty programme interface confuses me a bit, but that's for another day I guess.",
            " Also I ordered from your summer collection before and that was perfect, to be fair.",
            " Also my colleague placed an order last month and had zero problems, which makes this all the more confusing.",
        ])
        return pre + text + post

    def sarcastic(self, text):
        pre = self.rng.choice(["Oh great, another one. ", "Fantastic. ", "Perfect, just perfect. "])
        return pre + text

    def cuefor(self, kind, value):
        pools = {"urgency": {"high": HI_CUES_ORDERY if self.intent in ORDERY else HI_CUES,
                             "medium": MED_CUES, "low": LO_CUES},
                 "sentiment": {"negative": NEG_CUES, "neutral": [], "positive": POS_CUES}}
        pool = pools[kind][value]
        if not pool:
            return None
        return self.rng.choice(pool)

    def attach(self, text, urgency, sentiment, contradiction=False):
        segs = [text]
        u_cue = self.cuefor("urgency", ("low" if contradiction and urgency == "high" else urgency)
                            if contradiction else urgency)
        s_cue = self.cuefor("sentiment", sentiment)
        if u_cue:
            segs.append(self.rng.choice([f"{u_cue}.", f"({u_cue})", f"And note: {u_cue}."]))
            self.cues["urgency"] = u_cue
        if s_cue:
            self.cues["sentiment"] = s_cue
            segs.append(s_cue + ".")
        if contradiction:
            self.cues["urgency_contradiction"] = True
        return " ".join(segs)

    def build(self):
        r = self.rng
        hardness = []

        # 5% fully foreign, 3% code-switched english
        lang = "en"
        if r.random() < 0.05:
            lang, text = self.foreign()
            hardness.append("non_english")
        else:
            tmpl = r.choice(T[self.intent])
            slots = self.slots()
            text = tmpl.format(**slots)
            self.cues["intent"] = tmpl[:44].replace("{", "").replace("}", "")
            if r.random() < 0.12:
                text = self.short(text)
                hardness.append("short")
            if r.random() < 0.15:
                text = self.ramble(text)
                hardness.append("rambling")
            if self.intent in ("damaged_item", "wrong_item", "order_status", "complaint_escalation") and r.random() < 0.18:
                text = self.anger(text)
                hardness.append("angry")
            if self.intent in ("damaged_item", "wrong_item", "order_status") and r.random() < 0.10:
                text = self.sarcastic(text)
                hardness.append("sarcastic")
            if r.random() < 0.10:  # multi-issue: later sentence from another intent
                second = r.choice([i for i in T if i != self.intent])
                st = r.choice(T[second]).format(**self.slots())
                text = f"{text} Also, {lower_first(st)}"
                hardness.append(f"multi_issue_with_{second}")

            if r.random() < 0.03:
                lang2, tail = r.choice(SWITCH)
                text += tail
                lang = "en+" + lang2
                hardness.append("code_switched")

        urg_i = sample(r, DIST[self.intent][0])
        urg = ["low", "medium", "high"][urg_i]
        sent_i = sample(r, DIST[self.intent][1])
        sent = ["negative", "neutral", "positive"][sent_i]

        contrad = r.random() < 0.08 and urgency_flip_possible(urg)
        text = self.attach(text, urg, sent, contradiction=contrad)

        return {
            "intent": self.intent,
            "urgency": urg,
            "sentiment": sent,
            "lang": lang,
            "hardness": hardness,
            "text": text,
            "notes": {"cues": {k: v for k, v in self.cues.items() if v} },
        }


def lower_first(s):
    return s[0].lower() + s[1:] if s and s[0].isupper() else s


def urgency_flip_possible(urg):
    return urg == "high"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="tickets_raw.jsonl")
    args = ap.parse_args()

    rng = random.Random(SEED)
    intents = json.loads((CASE / "data" / "intents.json").read_text(encoding="utf-8"))
    names = list(intents)
    weights = [intents[k]["weight"] for k in names]

    rows, idx = [], 0
    for _ in range(args.n):
        intent = rng.choices(names, weights=weights, k=1)[0]
        row = Gen(rng, intent).build()
        row["id"] = f"t{idx:05d}"
        rows.append(row)
        idx += 1

    out = CASE / "data" / args.out
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # quick distribution report
    from collections import Counter
    ci, cu, cs = Counter(), Counter(), Counter()
    for row in rows:
        ci[row["intent"]] += 1
        cu[row["urgency"]] += 1
        cs[row["sentiment"]] += 1
    print(f"wrote {len(rows)} tickets -> {out.name}")
    print("intent:", dict(ci.most_common()))
    print("urgency:", dict(cu))
    print("sentiment:", dict(cs))


if __name__ == "__main__":
    main()
