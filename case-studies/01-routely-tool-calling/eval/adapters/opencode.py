"""Free hosted models through the local OpenCode CLI (`opencode run`) - $0 spend.

The system prompt is fixed in ../opencode.json (agent "routely", file eval/opencode_system.md,
rendered from eval/prompt.py by this adapter so it can never drift). Latency includes CLI
start-up (~1-2 s), so hosted p50/p95 here are upper bounds.

  uv run python eval/run_eval.py --adapter opencode --model opencode/deepseek-v4.1-flash-free
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

CASE = Path(__file__).resolve().parents[2]
SYSTEM_FILE = CASE / "eval" / "opencode_system.md"


def find_opencode() -> list[str]:
    exe = shutil.which("opencode.exe")
    if exe:
        return [exe]
    shim = shutil.which("opencode")
    if shim and shim.lower().endswith(".ps1"):
        cand = Path(shim).parent / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
        if cand.exists():
            return [str(cand)]
    cmd = shutil.which("opencode.cmd")
    if cmd:
        return ["cmd", "/c", cmd]
    raise SystemExit("opencode executable not found")


class OpenCodeAdapter:
    def __init__(self, model: str, timeout: int = 180, **_ignored):
        self.model = model
        self.timeout = timeout
        self.argv = find_opencode() + ["run", "--agent", "routely", "-m", model, "--format", "json"]
        self._system_written = False

    def predict(self, record, system, tools_doc):
        for attempt in range(4):  # free tier drops connections; back off and retry
            r = self._once(record, system)
            if not r.get("error"):
                return r
            time.sleep(15 * (attempt + 1))
        return r

    def _once(self, record, system):
        if not self._system_written:
            SYSTEM_FILE.write_text(system, encoding="utf-8", newline="\n")
            self._system_written = True
        msg = f"Today: {record['request_date']}\nRequest: {record['request']}"
        try:  # stdin: argv messages get cut at the first newline on Windows
            proc = subprocess.run(self.argv, cwd=str(CASE), input=msg, capture_output=True,
                                  timeout=self.timeout, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"text": "", "error": "timeout", "latency_ms": None}
        chunks, err, usage = [], None, None
        for line in (proc.stdout or "").splitlines():
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "text":
                chunks.append(ev.get("part", {}).get("text", ""))
            elif ev.get("type") == "error":
                e = ev.get("error", {})
                err = e.get("message") or json.dumps(e)[:300]
            elif ev.get("type") == "step_finish":
                usage = ev.get("part", {}).get("tokens")
        text = "".join(chunks)
        if not text:
            tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
            return {"text": "", "error": err or f"empty response: {tail}", "latency_ms": None}
        return {"text": text, "usage": usage, "latency_ms": None}
