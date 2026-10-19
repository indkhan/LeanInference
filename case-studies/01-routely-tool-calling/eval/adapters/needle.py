"""Adapter for the Needle 3 CLI/library (pip install cactus-needle).

Thin subprocess wrapper around a command template, so it works with whatever the
Needle CLI looks like once installed (verified in the training phase; adjust the
template to match - the only contract is: prompt via a temp file ({prompt_file}
placeholder), raw answer on stdout, non-zero exit = error).

Example:
  uv run python eval/run_eval.py --adapter needle --model tuned.cact \\
      --needle-cmd "needle chat --model {model} --prompt {prompt_file}"
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class NeedleAdapter:
    def __init__(self, model: str, needle_cmd: str | None = None, timeout: int = 120,
                 **_ignored):
        if not needle_cmd:
            raise SystemExit("adapter 'needle' needs --needle-cmd (see eval/adapters/needle.py)")
        self.model = model
        self.cmd = needle_cmd
        self.timeout = timeout

    def predict(self, record, system, tools_doc):
        prompt = f"{system}\n\nToday: {record['request_date']}\nRequest: {record['request']}\n"
        tmp = Path(tempfile.gettempdir()) / f"needle_prompt_{record['id']}.txt"
        tmp.write_text(prompt, encoding="utf-8")
        cmd = self.cmd.format(prompt_file=str(tmp), model=self.model)
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return {"text": "", "error": "needle CLI timeout", "latency_ms": None}
        if proc.returncode != 0:
            return {"text": "",
                    "error": f"needle CLI exit {proc.returncode}: {proc.stderr[:200]}",
                    "latency_ms": None}
        return {"text": proc.stdout}
