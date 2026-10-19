"""OpenAI-compatible chat-completions adapter (stdlib only - no dependencies).

Works with api.openai.com and any compatible server (vLLM, llama.cpp --api, LM Studio, ...).

Examples:
  uv run python eval/run_eval.py --adapter openai --model gpt-6-luna --api-key-env OPENAI_API_KEY
  uv run python eval/run_eval.py --adapter openai --model local-ft --base-url http://localhost:8000/v1

No paid calls happen unless a key is present in the environment. The benchmark's
$0 rule applies: only candidate B (local) runs are actually executed.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAIAdapter:
    default_base_url = "https://api.openai.com/v1"
    requires_key = True

    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", extra_body: dict | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024, timeout: int = 120,
                 **_ignored):
        self.model = model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        key = os.environ.get(api_key_env, "")
        if self.requires_key and not key:
            raise SystemExit(
                f"adapter 'openai': environment variable {api_key_env} is not set. "
                "Set it, or use a local server (--adapter vllm / --base-url).")
        self.api_key = key
        self.extra_body = dict(extra_body or {})
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def predict(self, record, system, tools_doc):
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Today: {record['request_date']}\nRequest: {record['request']}"},
        ]
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        body.update(self.extra_body)
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            return {"text": "", "error": f"HTTP {e.code}: {detail}", "latency_ms": None}
        except Exception as e:  # noqa: BLE001 - connection errors etc.
            return {"text": "", "error": f"{type(e).__name__}: {e}", "latency_ms": None}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        return {"text": text, "usage": data.get("usage") or {}, "latency_ms": None}
