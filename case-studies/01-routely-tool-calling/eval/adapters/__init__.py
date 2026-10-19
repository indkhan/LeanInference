"""Adapter registry for eval/run_eval.py.

Every adapter shares one interface:

    adapter = SomeAdapter(model="...", **options)
    result = adapter.predict(record, system_prompt, tools_doc)
    # -> {"text": "<raw model output>", "latency_ms": int|None, "usage": {...}|None}
    #    or {"text": "", "error": "<message>", "latency_ms": None} on failure

`record` is one dataset record (see data/FORMAT.md).
`system_prompt` is the shared system prompt; `tools_doc` is the JSON text of
data/tools.json as embedded in it.
"""
from __future__ import annotations


def get_adapter(name: str):
    name = name.lower()
    if name == "mock":
        from .mock import MockAdapter
        return MockAdapter
    if name == "openai":
        from .openai import OpenAIAdapter
        return OpenAIAdapter
    if name == "vllm":
        from .vllm import VLLMAdapter
        return VLLMAdapter
    if name == "needle":
        from .needle import NeedleAdapter
        return NeedleAdapter
    raise SystemExit(f"unknown adapter {name!r} (choose: mock, openai, vllm, needle)")
