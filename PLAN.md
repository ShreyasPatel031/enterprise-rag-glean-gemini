# Run the real Glean Agent Toolkit on Gemini

## Context

No Glean tenant is available (no self-service trial; sales-only, ~$70K PoC), so Glean's hosted index is unreachable. But the toolkit's own transport layer is designed for exactly this swap — its docstring: *"swapping a tool's backend is a one-line declaration change."*

**Verified working just now:**
- `configure(client=<stub>)` — `get_client()` returns a pre-set client immediately, before any token check. No credentials needed.
- `register_backend("glean_search", LocalBackend())` — replaces the backend for the real tool.
- Calling the unmodified `search()` returned `status: ok` with local data, and `search.as_adk_tool()` still builds.

So we run **Glean's actual toolkit** — real tool specs, real descriptions, real `ToolResult` envelope, real ADK adapters, real agent contract — driven by **Gemini**, over the EnterpriseRAG-Bench corpus. The only substituted piece is the index backend, which we have no access to regardless.

## Plan

### 1. `src/glean_migration/local_backends.py`
Backends satisfying the toolkit's `Backend` protocol (`execute` / `execute_async`), reusing the existing cached corpus embeddings from `vertex_retrieval.py`:

- **`glean_search`** → embedding search over the corpus. Must return Glean's exact shape: `{results: [{title, url, snippets, datasource, document_id}], result_count, has_more_results}`. Honor `datasources` (maps to corpus source types) and `page_size`.
- **`glean_read_document`** → full doc by `document_id`, via `extract_document_content`.
- **`glean_code_search`** → scoped to the `github` source type.
- **`glean_gmail_search`** → scoped to `gmail`; **`glean_employee_search`** → `generated_data/employee_directory.yaml`.
- `web_search` / `calendar_search` / `outlook_search`: no corpus equivalent — leave unregistered so they return the toolkit's own structured "no backend" error.

### 2. `src/glean_migration/glean_gemini_agent.py`
The harness: `configure(client=stub)` → register backends → `tool.as_adk_tool()` for each → ADK `LlmAgent` on Gemini → run the 30-question set → write answers JSONL in the standard eval format.

Reuse the same corpus cache, same question sample (same seed/`select_questions`), same output contract as `adk_harness.py` so results are directly comparable. Track `document_id`s returned through the Glean tools for document-recall scoring.

### 3. Score and compare
Run the repo's unmodified `metrics_based_eval.py` and `comparative_eval.py` (Gemini 2.5 Pro judge) against the existing baselines:

| System | Status |
|---|---|
| Single-shot retrieval | done — 93.3 / 89.3 / 91.7, 8.9 junk docs |
| Custom ADK agent | done — 83.3 / 83.2 / 79.4, 0.23 junk docs |
| **Glean toolkit + Gemini** | **this plan** |

Head-to-head: Glean-toolkit-harness vs. custom ADK harness, both on Gemini. That isolates the *harness/tool design* — Glean's tool contract vs. a hand-rolled one — with model and corpus held constant.

### 4. Swap-in test for a real tenant
Keep a flag that skips `register_backend` and uses real credentials when `GLEAN_API_TOKEN` is set. Same file runs against a real tenant unchanged the day one exists.

## Files

| File | Change |
|---|---|
| `src/glean_migration/local_backends.py` | new — Glean-shaped backends over the corpus |
| `src/glean_migration/glean_gemini_agent.py` | new — toolkit + ADK + Gemini harness |
| `src/scripts/answer_generation/vertex_retrieval.py` | reuse (corpus cache, `select_questions`) |
| `src/llm/vertex_llm.py` | reuse as-is |
| `src/scripts/answer_evaluation/*` | reuse **unmodified** |

## Verification

- Assert each backend's return shape matches the Glean SDK field names the tools' shapers expect (`title`/`url`/`snippets`/`datasource`/`document_id`).
- Smoke-test on 3 questions, confirm the agent actually calls `glean_search` + `glean_read_document` and returns real corpus content.
- Full 30-question run → both eval scripts → compare to the two existing baselines.
- Confirm the real-tenant path still errors correctly on auth when backends aren't registered (proves the swap-out is clean).

## Honest labeling (must appear in any writeup)

This is Glean's **real agent toolkit and tool contract** running on Gemini — not Glean's search index, ranking, or permissions layer. It measures how well Gemini performs *as the reasoning layer inside Glean's own agent harness*. It is **not** a quality benchmark against the Glean product.
