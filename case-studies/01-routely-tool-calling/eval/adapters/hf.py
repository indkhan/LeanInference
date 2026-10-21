r"""Local in-process inference (transformers + bitsandbytes 4-bit + PEFT) for candidate B.

Windows has no vLLM, so this replaces the vllm adapter for local runs. Greedy decoding,
non-thinking chat template (same as training), batch 1. Latency = full prefill + decode
(the no-schema prompts are short, so no prefix-cache trick). Unsloth was dropped: it is
the part that kept failing on this 6 GB laptop (see train/experiments.csv).

  .venv\Scripts\python.exe eval/run_eval.py --adapter hf --model Qwen/Qwen3.5-0.8B \
      --prompt-file eval/prompt_noschema.md
  .venv\Scripts\python.exe eval/run_eval.py --adapter hf --model qwen3.5-0.8b-ft \
      --adapter-path train/qwen/out/Qwen__Qwen3.5-0.8B-noschema/final --prompt-file eval/prompt_noschema.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class HFAdapter:
    def __init__(self, model: str, adapter_path: str | None = None, max_tokens: int = 256,
                 base_model: str | None = None, **_ignored):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        if adapter_path:
            base = base_model or json.loads((Path(adapter_path) / "adapter_config.json").read_text())[
                "base_model_name_or_path"]
        else:
            base = base_model or model
        self.model = AutoModelForCausalLM.from_pretrained(
            base, device_map={"": 0}, dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16,
                                                   bnb_4bit_use_double_quant=True))
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        tok = AutoTokenizer.from_pretrained(base)
        self.tok = getattr(tok, "tokenizer", tok)  # multimodal checkpoints ship a Processor
        self.max_tokens = max_tokens

    def predict(self, record, system, tools_doc):
        torch = self.torch
        user = f"Today: {record['request_date']}\nRequest: {record['request']}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, max_new_tokens=self.max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        ms = int((time.perf_counter() - t0) * 1000)
        new = out[0, ids.shape[1]:]
        return {"text": self.tok.decode(new, skip_special_tokens=True), "latency_ms": ms,
                "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": int(new.shape[0])}}
