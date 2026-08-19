# Session Summary — Glean → Gemini Migration Eval

What this session did, what it found, and exactly where everything lives.
Two companion documents go deeper on specific parts:

- **[SETUP.md](SETUP.md)** — how to stand up the environment from scratch (upstream corpus, venv, credentials, the `LLM_PROVIDER` trap, the two scorers' different flag names).
- **[results/matrix_v2_q60/HANDOFF.md](results/matrix_v2_q60/HANDOFF.md)** — exact state and resume commands for the in-progress 60-question redo.

## Repo

**https://github.com/ShreyasPatel031/enterprise-rag-glean-gemini** (public)

Everything below is committed on `main`. Latest commit at time of writing: `e03ddca`.

This repo is a partial extract — it does not vendor `src/llm/interface.py`, `src/paths.py`, `src/utils/*`, `src/prompts/*`, the two upstream scorers, or the corpus. Those come from cloning [onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) and copying this repo's `src/` on top — see SETUP.md.

## The question

Glean's real agent toolkit (not their hosted search index — no tenant was available) driven by Gemini, compared against a hand-rolled ADK agent, a naive single-shot retrieval baseline, and — added this session — Mistral Medium 3 and Llama 4 Maverick as generator models, to see whether the toolkit-vs-harness comparison and "does an open-weight model work as the reasoning layer" hold up under scrutiny.

## What was built this session (new files, all in `src/`)

| File | What it does |
|---|---|
| `llm/agentic_model.py` | Routes `--model` through ADK's `LiteLlm` so Mistral/Llama can drive the agentic harnesses (they couldn't before — see Bugs below). Also documents a tried-and-rejected fix (forced `tool_choice`) so it isn't retried. |
| `llm/telemetry.py` | Per-question latency + real token usage (via ADK's `usage_metadata`), written as a sidecar `<answers>.telemetry.jsonl` next to each harness's output. Verified live against all three model paths before shipping. |
| `scripts/answer_evaluation/significance.py` | Paired McNemar on correctness, exact binomial on judge preference, method-disagreement detection. Reads only `results/*.json` — no corpus or credentials needed. |
| `scripts/answer_evaluation/harness_divergence.py` | Decomposes a head-to-head by retrieval overlap, computes the real effective sample size, a noise floor, split-half robustness, and exact statistical power (verified against brute force). |
| `scripts/answer_evaluation/tool_call_integrity.py` | Flags answers where the agent failed to make a real structured tool call: no citations, leaked chat-template tokens, narrated pseudo-calls, stringified `document_ids`, missing `dsid_` prefix. |
| `scripts/answer_evaluation/repair_failed_answers.py` | Triages blank rows (infra failure vs. genuine model non-answer, distinguished via the harness log) and repairs the two malformed-`document_ids` shapes above — verified against the real corpus index so it can only ever recover a real document, never invent one. |
| `scripts/run_model_toolkit_matrix.sh` | Runs the full model × toolkit matrix: per-model parallelism (partner MaaS models throttled to serial — they rate-limit at parallelism 4), a smoke test before committing to a full run, triage, integrity check, and a pooled ADK-vs-Glean verdict across models. |

Two harnesses were modified (not rewritten) to use `agentic_model.py` and `telemetry.py`: `scripts/answer_generation/adk_harness.py` and `glean_migration/glean_gemini_agent.py`.

## Key findings

### Original 30-question run (in `results/`)

- **No pairwise correctness difference across any of 15 system-pairs reaches significance** (exact McNemar, best p=0.125). Point-estimate comparisons in the original writeup were being over-read.
- **ADK vs. Glean-toolkit is a genuine dead heat** (14–15 judge preference, p=1.00) — and it survives scrutiny: retrieval overlap is 73% identical, so the *effective* sample for the toolkit comparison is only 8 questions, not 30; there's a 9.1% noise floor even on identical evidence; split-half is non-significant in all four halves and flips direction. This is the best-supported finding in the repo.
- **Single-shot's apparent recall lead is a mirage** — it returns a fixed top-10 every time (11% precision) vs. ~80% precision for the agentic harnesses.
- **Power analysis**: 23 decisive judge picks gives 80% power to detect only a 79/21 preference. Detecting 60/40 needs ~265 questions per pair.

### Extending to Mistral and Llama (also n=30 initially, then redone at n=60 — see below)

- **Mistral Medium 3 drove both harnesses with zero tool-calling defects** — 30/30 (then 60/60) clean, live-verified against Glean's actual `glean_search`/`glean_read_document` tool specs (not toy tools). Its correctness sits between Flash and Flash-Lite. This is the strongest "open-weight model works as the reasoning layer" result in the repo.
- **Llama 4 Maverick has a real, replicated tool-calling defect**, not just weak reasoning: it emits textual pseudo tool calls (sometimes leaking raw chat-template tokens) instead of structured calls, and fabricates the "tool result" it never received. Clean rate: 12/30 → 22/60 on ADK (~37-40%), 8/30 → 17/60 on Glean-toolkit (~27-28%) — the rate **replicated almost exactly** between n=30 and n=60, so this is a stable defect, not small-sample noise. On its clean answers only, Llama scores ~75% correctness — competitive with Flash and Mistral. The defect, not the model's reasoning, is what's broken.
- **A forced-`tool_choice="required"` fix was tried and rejected**, with live evidence: Llama's Vertex endpoint returns `400 INVALID_ARGUMENT` for it outright, and testing the same mechanism on Mistral made it loop forever calling `search` and never reach `finish_answer` — would have broken a model that already worked. A prompt-level reminder was tried instead; it recovered only 2 of 18 previously-failed questions, not a real fix. This is documented in `agentic_model.py` so it isn't re-attempted blind.

### Real telemetry (partial — see below for what's complete)

- Flash: ~50s/question, ~10K input tokens, ~530 output tokens.
- Mistral and Llama are **genuinely ~3x slower per call than native Gemini** (~95-165s/question) — confirmed real per-call latency, not a parallelism artifact.
- Llama's low LLM-call count on Glean-toolkit (2.0 vs ~3.5-3.8 elsewhere) tracks its tool-call defect: many answers end early on a failed call rather than completing the full loop.
- Verified Vertex pricing: Flash $0.30/$2.50 per M tok (in/out), Flash-Lite $0.10/$0.40, Mistral Medium 3 $0.40/$2.00. **Llama's Vertex-specific price was never found** despite two direct fetches of Google's own pricing docs — not fabricated, left blank.

## Bugs found and fixed this session (worth knowing about before trusting any number)

1. **`LLM_PROVIDER` defaults to `"openai"`** (`src/llm/factory.py`), and upstream's own quickstart tells you to set it to `"openai"`. On a Vertex-only setup with no OpenAI key, the judge fails on every question, the eval script swallows the errors, and you get a *complete-looking* results file with 0.0% correctness/completeness while recall (computed without an LLM) still looks perfectly normal. Caught because Mistral — independently verified clean — cannot score 0%. Now hard-guarded in `run_model_toolkit_matrix.sh`.
2. **Partner MaaS models rate-limit at `--parallelism 4`**, Gemini doesn't. The harnesses swallow the exception and write a *blank-answer* row, which then silently scores as a wrong answer. Fixed: partner models run at parallelism 1.
3. **A blank answer is not always an infra failure** — `repair_failed_answers.py` originally assumed it always was; a real case (Gemini returning genuinely empty output, likely a thinking-budget exhaustion) proved that wrong. Fixed to attribute via the harness log and stop retrying degenerate model outputs forever.
4. **`comparative_eval.py`'s flags were guessed wrong initially** (`--answers-file-1/2` instead of the real `--answer-file-1/2`, and a nonexistent `--no-correction`) — corrected once the upstream repo was available to check against.
5. **Two silent `document_ids` corruptions**, found only because `comparative_eval.py` validates more strictly than the pointwise scorer: Llama returning `document_ids` as a stringified Python list, and Flash-Lite dropping a required `dsid_` prefix on an otherwise-real document. Both repaired and verified against the real corpus index (`repair_failed_answers.py --fix-document-ids`), with the repair function unit-tested to confirm it can never invent a document that doesn't exist.
6. **`vertex_retrieval.py`'s embedding-cache loader only checks whether the cache exists, never whether it covers the current question selection.** Increasing the question count from 30 to 60 without forcing a rebuild would have silently degraded recall for the new questions as a measurement artifact. Caught before it corrupted anything; cache was rebuilt.
7. **A reused `--output` path without truncation produced 31 duplicate rows** in a throwaway single-shot baseline file — caught immediately, not present in anything scored.

## Current state: the 60-question full redo (Flash / Mistral / Llama, both harnesses)

Started because 30 questions has too little power to resolve close comparisons (see the power analysis above). Full detail and exact resume commands: **[results/matrix_v2_q60/HANDOFF.md](results/matrix_v2_q60/HANDOFF.md)**. Short version:

- **Generation: complete.** All 6 cells (`{flash,mistral-medium-3,llama-4-maverick} × {adk,gleantoolkit}`), 60/60 rows each, real telemetry captured for every row.
- **Scoring: 4 of 6 done.** Flash (both harnesses), Llama/ADK, Mistral/ADK (missing one question's score, not its answer — `qst_0262`).
- **2 cells stalled mid-scoring and were killed**: Mistral/Glean-toolkit (34/60) and Llama/Glean-toolkit (18/60). Both held open HTTPS connections with near-zero CPU growth and zero file writes for ~70 minutes — cause not identified, no error was ever raised. Progress is preserved (the scorer writes incrementally); resume with `--resume`, do not restart from scratch.
- **Not started**: the 4 ADK-vs-Glean-toolkit comparatives at n=60, tool-call integrity for Flash/Mistral at n=60 (only run for Llama so far), and the final synthesized accuracy + latency + token + cost + integrity report.

All generated data (answers, telemetry, partial scores) for this redo is committed under `results/matrix_v2_q60/`, not left on the local machine only — a fresh environment can resume from it directly.

## Environment note

This entire session ran on a local macOS machine (not a cloud sandbox) with real `gcloud` credentials against Vertex project `project-amer-scs-sandbox`. A cloud Claude Code session needs the repo explicitly attached (private repos 403 otherwise, which is why this repo was made public) and will need to redo the environment setup in SETUP.md — the corpus and venv do not travel with the git history.
