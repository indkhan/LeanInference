"""Answer-only val loss of a saved adapter, one unpadded row at a time (what the
trainer's eval should have reported). Usage: python train/val_loss.py <adapter> [--model ...]"""
import argparse, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_qlora import VAL_JSONL, build_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("adapter")
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--max-len", type=int, default=2048)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
docs = [json.loads(l) for l in VAL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
ds = build_dataset(tok, docs, a.max_len)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(
    a.model, quantization_config=bnb, device_map={"": 0}, dtype=torch.float16), a.adapter).eval()
tot = n = 0
with torch.no_grad():
    for r in ds:
        ids = torch.tensor([r["input_ids"]], device=0)
        k = sum(1 for x in r["labels"] if x != -100)
        logits = m(input_ids=ids, logits_to_keep=k + 1).logits[0, :-1].float()
        tot += torch.nn.functional.cross_entropy(logits, ids[0, -k:], reduction="sum").item()
        n += k
print(json.dumps({"adapter": a.adapter, "val_docs": len(ds), "answer_tokens": n,
                  "val_loss": round(tot / n, 5), "ppl": round(math.exp(tot / n), 4)}))
