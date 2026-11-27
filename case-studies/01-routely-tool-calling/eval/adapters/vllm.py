"""Adapter for a local OpenAI-compatible server (vLLM, llama.cpp, LM Studio).

Example (Qwen3.5 fine-tune served with vLLM):
  python -m vllm.entrypoints.openai.api_server --model ./merged --port 8000
  uv run python eval/run_eval.py --adapter vllm --model qwen3.5-2b-ft \\
      --base-url http://localhost:8000/v1 --no-thinking

--no-thinking adds {"chat_template_kwargs": {"enable_thinking": false}} to the request
body. Qwen3.5 thinks by default; the benchmark runs it in non-thinking mode and the
fine-tune is trained the same way, so keep this flag on for those models.
"""
from __future__ import annotations

from .openai import OpenAIAdapter


class VLLMAdapter(OpenAIAdapter):
    default_base_url = "http://localhost:8000/v1"
    requires_key = False

    def __init__(self, model: str, base_url: str | None = None, no_thinking: bool = False,
                 extra_body: dict | None = None, **kw):
        extra_body = dict(extra_body or {})
        if no_thinking:
            extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        super().__init__(model=model, base_url=base_url, extra_body=extra_body, **kw)
