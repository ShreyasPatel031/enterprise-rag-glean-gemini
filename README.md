# Glean → Gemini migration harnesses

Work-in-progress evaluation of enterprise-RAG harnesses on
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench),
comparing Glean's real agent toolkit against Gemini-native alternatives.

Only the new code and eval outputs live here. The benchmark corpus
(~3.7 GB, 512k documents) is **not** included — clone the upstream repo for
that and drop these files in on top.

## What's here

| Path | What it is |
|---|---|
| `src/glean_migration/local_backends.py` | Serves Glean's built-in tools from the benchmark corpus via the toolkit's own `register_backend` seam |
| `src/glean_migration/glean_gemini_agent.py` | Glean's real toolkit (real tool specs + `ADKAdapter`) driven by Gemini through Google ADK. `--live-glean` targets a real tenant instead |
| `src/llm/vertex_llm.py` | Vertex provider: Gemini native, Mistral Medium 3 (`europe-west4` rawPredict), Llama 4 Maverick (`us-east5` OpenAI-compatible) |
| `src/scripts/answer_generation/vertex_retrieval.py` | Single-shot embed → retrieve → generate baseline; builds the shared embedding cache |
| `src/scripts/answer_generation/adk_harness.py` | Hand-rolled ADK agent (search / read_document / finish_answer) |
| `results/` | Scored outputs from the repo's own `metrics_based_eval.py` / `comparative_eval.py` |

## Results so far

30 questions, Confluence + Jira subset, identical corpus and retrieval
substrate. Gemini 2.5 Flash answers; Gemini 2.5 Pro judges.

| | Single-shot | Custom ADK | Glean toolkit |
|---|---|---|---|
| Correctness | **93.3%** | 83.3% | 80.0% |
| Completeness | **89.3%** | 83.2% | 85.5% |
| Document recall | **91.7%** | 79.4% | 81.1% |
| Invalid extra docs | 8.9 | **0.23** | 0.30 |

The two agentic harnesses land within noise of each other, which suggests the
agent harness and tool contract are reproducible — not where Glean's
differentiation lives. Single-shot's higher absolute recall comes from
returning a fixed top-10 every time (8.9 irrelevant documents per question)
rather than from better targeting; the agentic harnesses pull ~1.2 documents
per question with almost no noise.

## What this does and does not show

**Does:** Gemini works correctly as the reasoning layer inside Glean's own
agent harness and tool contract, and how that compares to a hand-rolled ADK
agent and a naive retrieval baseline.

**Does not:** benchmark Glean's search index, ranking, or permission model.
Those require a live Glean tenant, which was not available — the toolkit's
tools are HTTP clients against a hosted backend, so only the transport was
substituted. `--live-glean` runs the identical agent against a real tenant
once credentials exist.

## Running it

Needs Google Cloud credentials with Vertex AI access
(`gcloud auth application-default login`) and `VERTEX_PROJECT` set. No
credentials are stored in this repo.

```bash
# 1. build the shared corpus embedding cache + single-shot baseline
python -m src.scripts.answer_generation.vertex_retrieval \
  --source-types confluence jira --question-limit 30 --corpus-cap 800 \
  --model gemini-2.5-flash --output answer_evaluation/system_gemini_flash.jsonl

# 2. Glean toolkit on Gemini
python -m src.glean_migration.glean_gemini_agent \
  --source-types confluence jira --question-limit 30 --corpus-cap 800 \
  --model gemini-2.5-flash --output answer_evaluation/system_glean_toolkit_gemini.jsonl

# 3. score with the upstream harness
python -m src.scripts.answer_evaluation.metrics_based_eval \
  --answers-file answer_evaluation/system_glean_toolkit_gemini.jsonl \
  --results-file answer_evaluation/results_glean_toolkit.json \
  --no-correction --parallelism 4
```
