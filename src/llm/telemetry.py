"""Per-question latency and token telemetry for the agentic harnesses.

Written to a sidecar JSONL (never into the answer file the scorers read) so
metrics_based_eval.py / comparative_eval.py's parsing is never at risk from an
unrecognized field -- deliberately not verified against their source, so kept
out of the path they read.

Token counts come from ADK's per-event `usage_metadata`
(google.genai.types.GenerateContentResponseUsageMetadata), confirmed present
for all three model paths these harnesses drive (native Gemini and both
LiteLlm-routed partner models) with live calls against
project-amer-scs-sandbox before this was wired in. Some events in a run carry
usage_metadata and some carry None (ADK appears to split a single model turn
across multiple events, e.g. a function-call event and a text event); this
sums whichever events are non-None, which is every real model call in the run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field


@dataclass
class QuestionTelemetry:
    question_id: str
    start: float = field(default_factory=time.monotonic)
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0
    llm_calls: int = 0
    elapsed_sec: float | None = None

    def record_event(self, usage_metadata) -> None:
        if usage_metadata is None:
            return
        self.llm_calls += 1
        self.prompt_tokens += usage_metadata.prompt_token_count or 0
        self.output_tokens += usage_metadata.candidates_token_count or 0
        self.total_tokens += usage_metadata.total_token_count or 0
        self.thoughts_tokens += usage_metadata.thoughts_token_count or 0

    def finish(self) -> dict:
        self.elapsed_sec = time.monotonic() - self.start
        return {
            "question_id": self.question_id,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "thoughts_tokens": self.thoughts_tokens,
        }


def telemetry_path_for(output_path: str) -> str:
    """Derive the sidecar telemetry path from a harness's --output path."""
    if output_path.endswith(".jsonl"):
        return output_path[: -len(".jsonl")] + ".telemetry.jsonl"
    return output_path + ".telemetry.jsonl"


def append_telemetry(path: str, record: dict, lock: threading.Lock) -> None:
    with lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
