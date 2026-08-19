# Handoff: 60-question full redo (Flash / Mistral / Llama, both harnesses)

Session moved from local (macOS) execution to cloud continuation mid-run.
This directory has everything generated so far — the point is to resume, not
regenerate, since generation and scoring cost real time and Vertex spend.

## Question set

60 questions (source_types=confluence+jira, corpus_cap=800), a superset of the
original 30 (deterministic seeded shuffle — see `select_questions()` in
`vertex_retrieval.py`). **The embedding cache had to be rebuilt for this**
(see SETUP.md's cache-rebuild note) — the old 30-question cache does not
guarantee gold-document coverage for the new 30 questions. If regenerating
anything for these 60 questions on a fresh machine, rebuild the cache first:
delete `answer_evaluation/_vertex_cache/`, then run `vertex_retrieval.py
--question-limit 60 ...` before any agentic harness call.

## Generation: COMPLETE, all 6 cells, 60/60 rows each

`system_{adk,gleantoolkit}_{flash,mistral-medium-3,llama-4-maverick}.jsonl`
— all fully generated, verified 60 unique question_ids each, no duplicates.
One row (`qst_0262`, mistral/adk) hit a transient litellm content-parsing bug
on the first pass and was repaired via one `--resume` retry — succeeded, real
answer, not blank.

Each has a sidecar `<name>.telemetry.jsonl` — per-question wall-clock latency
and token usage (prompt/output/total/thoughts + llm_calls), captured via
`src/llm/telemetry.py`, verified live for all three model paths before this
run. Real data, not proxies.

## Scoring: 4 of 6 cells COMPLETE, 2 STALLED mid-run

| Cell | Status |
|---|---|
| `results_adk_flash.json` | done, 60/60 |
| `results_gleantoolkit_flash.json` | done, 60/60 |
| `results_adk_mistral-medium-3.json` | done, 59/59 -- **missing qst_0262**, needs one more scoring pass after the answer-file repair (see above); the answer row itself is fine, only its score is missing |
| `results_adk_llama-4-maverick.json` | done, 60/60 |
| `results_gleantoolkit_mistral-medium-3.json` | **PARTIAL, 34/60** -- process hung (see below), killed, never resumed |
| `results_gleantoolkit_llama-4-maverick.json` | **PARTIAL, 18/60** -- same |

### The hang, so it doesn't repeat silently

Both stalled `metrics_based_eval.py` processes showed 4 established HTTPS
connections to a Google endpoint but near-zero CPU growth over ~70 minutes,
and their results files hadn't been written to in that entire window despite
scoring incrementally under normal operation. Looks like a hung read with no
client-side timeout, or a retry loop that leaks connections without ever
succeeding -- root cause not identified. Killed both (`kill -9`); progress up
to that point (34/60, 18/60) is preserved in the results files since they
write incrementally.

**Resume, don't restart**, using the scorer's own `--resume` (skips
question_ids already in the results file):

```bash
export LLM_PROVIDER=vertex LLM_MODEL_NAME=gemini-2.5-pro   # see SETUP.md's trap
export VERTEX_PROJECT=<project>

python -m src.scripts.answer_evaluation.metrics_based_eval \
  --answers-file system_adk_mistral-medium-3.jsonl \
  --results-file results_adk_mistral-medium-3.json \
  --resume --no-correction --parallelism 4   # picks up just qst_0262

python -m src.scripts.answer_evaluation.metrics_based_eval \
  --answers-file system_gleantoolkit_mistral-medium-3.jsonl \
  --results-file results_gleantoolkit_mistral-medium-3.json \
  --resume --no-correction --parallelism 4   # picks up the remaining 26

python -m src.scripts.answer_evaluation.metrics_based_eval \
  --answers-file system_gleantoolkit_llama-4-maverick.jsonl \
  --results-file results_gleantoolkit_llama-4-maverick.json \
  --resume --no-correction --parallelism 4   # picks up the remaining 42
```

If it hangs again, add a hard wall-clock watchdog around the invocation
(e.g. `timeout 1800 python -m ...` in a retry loop) rather than trusting the
script's own error handling -- it produced no error, no traceback, no log
line on hang, just silence.

## Not yet done at all

- The 4 `results_comparative_adk_vs_gleantoolkit_*.json` files for this
  60-question set (ADK vs Glean-toolkit head-to-head, one per model) --
  none have been run yet. Needs `comparative_eval.py` with `--answer-file-1`
  / `--answer-file-2` (see SETUP.md for the exact flags -- they differ from
  `metrics_based_eval.py`'s).
- Tool-call integrity was checked once for Llama
  (`results_tool_call_integrity_llama_q60.json`, included) but not for
  Flash/Mistral at n=60 -- run `tool_call_integrity.py` over all 6
  `system_*.jsonl` files.
- The full accuracy + latency + token + cost + integrity report the user
  asked for. Everything needed to build it is in this directory once scoring
  finishes; nothing has been synthesized into a final table yet.

## Known findings already confirmed at this sample size (won't change)

- Llama's tool-call clean rate replicated almost exactly between n=30 and
  n=60 (ADK: 40%->36.7%, Glean-toolkit: 26.7%->28.3%) -- this is a stable,
  real defect rate, not small-sample noise.
- Real per-question latency: Flash ~50s, Mistral ~150-165s, Llama ~95-145s.
  Mistral/Llama are genuinely ~3x slower than native Gemini per call --
  confirmed real, not a parallelism artifact (measures each call's own
  duration).
- Llama's Vertex-specific per-token price was never found despite two direct
  fetches of Google's own pricing docs -- don't fabricate one.
