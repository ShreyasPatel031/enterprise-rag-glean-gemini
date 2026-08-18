# Running the harnesses for real

This repo is only the new code and eval outputs. The harnesses import upstream
modules and read the benchmark corpus, so they cannot run from this repo alone.
This file records the setup that actually works, because several of these
details are not guessable from the code.

## 1. Upstream repo (provides the corpus and the missing modules)

```bash
git clone https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git
```

~5.5 GB checked out (3.7 GB of it `generated_data/`). It supplies six modules
this repo's harnesses import but does not vendor:

```
src/llm/interface.py      src/utils/document_content.py
src/paths.py              src/utils/file_io.py
src/prompts/vector_search_answer_gen.py   src/utils/retrieval.py
```

plus the two scorers (`metrics_based_eval.py`, `comparative_eval.py`) and
`questions.jsonl` (500 questions).

Copy this repo's `src/` over the upstream checkout and run everything from
there.

## 2. Python environment

Upstream uses a `uv`-managed venv (Python 3.12, no `pip` binary inside it):

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

**`google-cloud-aiplatform>=1.38` is required for Mistral/Llama** and is not in
upstream's requirements. LiteLLM's Vertex partner route imports `vertexai`;
without it every partner-model call fails with a *misleading*
`litellm.BadRequestError` rather than an ImportError, which is a slow thing to
diagnose:

```bash
uv pip install --python .venv/bin/python "google-cloud-aiplatform>=1.38"
```

Note this downgrades `protobuf` (7.x to 6.x). ADK, LiteLLM, google-genai,
numpy and the Glean toolkit all still import fine afterwards — verified.

## 3. Credentials

```bash
gcloud auth application-default login
export VERTEX_PROJECT=your-project
```

Gemini is served from `us-central1`. The partner models are region-locked and
`agentic_model.py` overrides per call, because LiteLLM's Vertex default
(`us-central1`) is wrong for both:

| Model | Region | Endpoint |
|---|---|---|
| `mistral-medium-3` | `europe-west4` | `publishers/mistralai/models/…:rawPredict` |
| `meta/llama-4-maverick-17b-128e-instruct-maas` | `us-east5` | `endpoints/openapi/chat/completions` |

## 4. Embedding cache

Every harness shares one cache so they retrieve over identical documents. Build
it once with the single-shot harness:

```bash
python -m src.scripts.answer_generation.vertex_retrieval \
  --source-types confluence jira --question-limit 30 --corpus-cap 800 \
  --model gemini-2.5-flash --output answer_evaluation/system_gemini_flash.jsonl
```

Writes `answer_evaluation/_vertex_cache/{index,uuids,vectors}_confluence_jira_cap800.*`.
The agentic harnesses **fail fast** if it is absent rather than silently
rebuilding it.

## 5. Scorer flag names (they differ between the two scripts)

| Script | Answers flag | `--no-correction`? |
|---|---|---|
| `metrics_based_eval.py` | `--answers-file` (plural) | yes |
| `comparative_eval.py` | `--answer-file-1` / `--answer-file-2` (singular) | **no** |

## 6. Run the matrix

```bash
export VERTEX_PROJECT=your-project
export PYTHON=.venv/bin/python
bash src/scripts/run_model_toolkit_matrix.sh answer_evaluation/_matrix
```

To avoid re-paying for cells already done, seed them under the matrix's naming
(`system_<toolkit>_<slug>.jsonl`, slug = model minus the `gemini-2.5-` prefix)
and the script skips them.
