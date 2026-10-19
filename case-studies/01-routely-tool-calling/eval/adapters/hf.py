r"""Local in-process inference (Unsloth 4-bit) for candidate B, base or + LoRA adapter.

Windows has no vLLM, so this replaces the vllm adapter for local runs. Greedy decoding,
non-thinking chat template (same as training). The shared system prompt is prefilled once
and its KV cache reused for every request, so latency = request prefill + decode only;
that is what a prefix-caching server (vLLM/llama.cpp) would do in production.

  .venv\Scripts\python.exe eval/run_eval.py --adapter hf --model Qwen/Qwen3-4B-Instruct-2507
  .venv\Scripts\python.exe eval/run_eval.py --adapter hf --model qwen3-4b-ft \
      --adapter-path train/qwen/out/Qwen__Qwen3-4B-Instruct-2507/final
"""
from __future__ import annotations

import copy
import time


class HFAdapter:
    def __init__(self, model: str, adapter_path: str | None = None, max_tokens: int = 256,
                 base_model: str | None = None, **_ignored):
        import torch
        from unsloth import FastLanguageModel
        self.torch = torch
        name = adapter_path or base_model or model
        self.model, tok = FastLanguageModel.from_pretrained(
            model_name=name, max_seq_length=8192, dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(self.model)
        self.tok = getattr(tok, "tokenizer", tok)  # multimodal checkpoints ship a Processor
        self.max_tokens = max_tokens
        self.prefix_ids = None
        self.prefix_cache = None

    def _render(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def predict(self, record, system, tools_doc):
        torch = self.torch
        user = f"Today: {record['request_date']}\nRequest: {record['request']}"
        ids = self.tok(self._render(system, user), return_tensors="pt").input_ids.to("cuda")
        if self.prefix_ids is None:  # common prefix = everything up to the user text
            probe = self.tok(self._render(system, "\x00"), return_tensors="pt").input_ids[0]
            n = 0
            while n < min(len(probe), ids.shape[1]) and probe[n] == ids[0, n]:
                n += 1
            self.prefix_ids = ids[:, :n - 1]
            with torch.no_grad():
                self.prefix_cache = self.model(self.prefix_ids, use_cache=True).past_key_values
        n = self.prefix_ids.shape[1]
        assert torch.equal(ids[:, :n], self.prefix_ids), "prompt prefix drifted"
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, past_key_values=copy.deepcopy(self.prefix_cache),
                                      max_new_tokens=self.max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        ms = int((time.perf_counter() - t0) * 1000)
        new = out[0, ids.shape[1]:]
        return {"text": self.tok.decode(new, skip_special_tokens=True), "latency_ms": ms,
                "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": int(new.shape[0])}}
