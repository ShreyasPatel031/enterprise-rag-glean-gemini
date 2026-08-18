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
| `src/llm/agentic_model.py` | Routes `--model` through ADK's `LiteLlm` so Mistral/Llama can drive the agentic harnesses; Gemini unchanged |
| `src/scripts/answer_evaluation/significance.py` | Significance layer over the scored outputs — paired McNemar, exact binomial, method-disagreement check. Needs no corpus or credentials |
| `src/scripts/answer_evaluation/harness_divergence.py` | Validates a two-harness head-to-head: retrieval overlap, identical-vs-divergent decomposition, noise floor, split-half, exact power |
| `src/scripts/run_model_toolkit_matrix.sh` | Runs the model × toolkit bake-off matrix, smoke-tests partner models, pools the result |
| `results/` | Scored outputs from the repo's own `metrics_based_eval.py` / `comparative_eval.py` |

## Results so far

30 questions, Confluence + Jira subset, identical corpus and retrieval
substrate. Gemini 2.5 Pro judges throughout.

| Harness | Model | Correct | 95% CI | Recall | Docs | Junk | Precision |
|---|---|---|---|---|---|---|---|
| Single-shot | Gemini 2.5 Flash | 93.3% | [78.7, 98.2] | 91.7% | 10.0 | 8.9 | 11% |
| Single-shot | Mistral Medium 3 | 93.3% | [78.7, 98.2] | 91.7% | 10.0 | 8.9 | 11% |
| Single-shot | Gemini 2.5 Flash-Lite | 83.3% | [66.4, 92.7] | 91.7% | 10.0 | 8.9 | 11% |
| Single-shot | Llama 4 Maverick | 83.3% | [66.4, 92.7] | 91.7% | 10.0 | 8.9 | 11% |
| Custom ADK | Gemini 2.5 Flash | 83.3% | [66.4, 92.7] | 79.4% | 1.17 | 0.23 | **80%** |
| Glean toolkit | Gemini 2.5 Flash | 80.0% | [62.7, 90.5] | 81.1% | 1.23 | 0.30 | 76% |

### At n=30 the correctness column does not separate these systems

Every CI above is 19–28pp wide and they all overlap. Exact McNemar (paired —
same 30 questions per system) across all 15 pairs: **none reach p<0.05**, best
0.125. The 93.3% vs 80.0% spread is 4 questions. Treat that column as "no
measured difference," not a ranking.

### The pairwise judge is more sensitive, and finds two real effects

| Comparison | Preference | p | Verdict |
|---|---|---|---|
| Flash vs Flash-Lite | 22 – 8 | **0.016** | real, generator quality |
| Flash vs Mistral Medium 3 | 21 – 9 | **0.043** | marginal (0.40 decisive-only) |
| Flash vs Llama 4 Maverick | 19 – 11 | 0.20 | not separated |
| Custom ADK vs Glean toolkit | 14 – 15 | 1.00 | **dead heat** |
| Single-shot vs Custom ADK | 10 – 20 | 0.099 | leans agentic — see below |

### The two methods disagree on single-shot vs. agentic

Pointwise favours single-shot (93.3% vs 83.3%); the pairwise judge favours the
ADK agent (10–20, and 7–17 on non-equivalent picks). Neither is significant and
they point opposite ways, so **that pair is unresolved.**

Mechanism: single-shot's entire lead is 3 questions, 2 of them pure retrieval
misses (the agent never surfaced the gold document) rather than reasoning
errors. And single-shot buys its 91.7% recall by returning a fixed top-10 every
time — 8.9 irrelevant documents per question, **11% precision** — against
**76–80%** for the agentic harnesses at ~1.2 documents. Which is "better"
depends on whether junk context costs you anything downstream; this metric does
not price it.

### Validating the dead heat

A dead heat can mean "equivalent" or "never did anything different." Comparing
what each harness actually *retrieved* separates those:

| | Questions | Correct (ADK / Glean) | Recall | Judge |
|---|---|---|---|---|
| Identical documents | 22 (73%) | 20/22 · 20/22 | 90.2% · 90.2% | 11–10 |
| Divergent documents | 8 (27%) | 5/8 · 4/8 | 50.0% · 56.2% | 3–5 |

Mean Jaccard 0.794, and on 4 questions they shared *no* documents — so the tie
is not an artifact of identical behaviour. But on 22 of 30 they handed the same
evidence to the same model, so **the effective sample for the toolkit question
is 8, not 30.** The divergent subset is also the hard subset (recall ~50% vs
~90%).

- **Noise floor.** On 2 of the 22 identical-retrieval questions (9.1%) the
  verdict flipped anyway — `qst_0327` (Glean omitted a file path) and
  `qst_0341` (ADK described a past verification instead of the method). Same
  documents, same model, different score, and they cancel. Any harness effect
  below ~9pp is inseparable from answer-phrasing variance.
- **Split-half.** Holds in all four halves (p = 0.42 / 0.61 / 1.00 / 0.79) and
  the halves lean *opposite* ways — a true-null signature, not a real effect
  buried in noise.

### Power: what this design could ever have detected

23 decisive judge picks gives 80% power to detect only a **79/21** preference.
No plausible two-toolkit difference is that lopsided, so non-significance was
close to predetermined.

| Design | Decisive picks | Smallest detectable |
|---|---|---|
| current: 1 model × 2 toolkits, n=30 | 23 | 79/21 |
| 2 models × 2 toolkits, n=30 | 46 | 72/28 |
| 4 models × 2 toolkits, n=30 | 92 | 65/35 |
| 4 models × 2 toolkits, n=100 | 307 | 59/41 |

Detecting 60/40 needs ~203 decisive picks ≈ **265 questions per toolkit pair**.
Note this supersedes a naive "~200 questions" estimate, which ignored that only
27% of questions produce divergent retrieval.

Reproduce both analyses (no corpus or credentials needed):

```bash
python -m src.scripts.answer_evaluation.significance
python -m src.scripts.answer_evaluation.harness_divergence
```

### The bake-off matrix: model × toolkit

Pooling the head-to-head across generator models is what buys the power above.
`src/scripts/run_model_toolkit_matrix.sh` runs every cell, smoke-tests the
partner models first, and pools the ADK-vs-Glean result across all four.

| Model | Single-shot | ADK | Glean toolkit |
|---|---|---|---|
| Gemini 2.5 Flash | done | done | done |
| Gemini 2.5 Flash-Lite | done | runnable | runnable |
| Mistral Medium 3 | done | runnable | runnable |
| Llama 4 Maverick | done | runnable | runnable |

**Mistral and Llama could not drive either agentic harness before.** Both
passed `--model` straight to ADK's `LlmAgent` as a bare string, which resolves
only against ADK's Gemini registry; and `src/llm/vertex_llm.py`'s own
`_generate_mistral` / `_generate_llama` never sent a `tools` field or parsed
`tool_calls` back — plain-text completion only, fine for single-shot, useless
for a tool-calling agent. `src/llm/agentic_model.py` fixes the routing through
ADK's `LiteLlm` against the Vertex partner endpoints.

**Verified with live calls** (project `project-amer-scs-sandbox`, 2026-08-18;
full detail in `results/verification_partner_models.json`):

| Check | Gemini 2.5 Flash | Mistral Medium 3 | Llama 4 Maverick |
|---|---|---|---|
| Raw endpoint returns `tool_calls` | n/a (native) | pass | pass |
| ADK loop, hand-rolled tools | pass | pass | pass |
| ADK loop, **Glean's real tool specs** | pass | pass | pass |

That last row is the one that matters for the bake-off: Glean's own
`glean_search` / `glean_read_document` specs (via `as_adk_tool()`, toolkit
v0.8.0) are much richer than hand-rolled ones, and all three models completed
`search → read → submit_answer` against them.

Two things only a live run surfaced:

- **`google-cloud-aiplatform>=1.38` is required.** LiteLLM's Vertex partner
  route imports `vertexai`; without it every Mistral/Llama call fails with a
  misleading `BadRequestError` rather than an import error.
- **Llama 4 Maverick intermittently emits a textual pseudo tool call** carrying
  raw chat-template markers (`<|header_start|>assistant<|header_end|>…`) before
  the real structured call. It recovered every time observed, but a turn that
  does not would yield an answer citing no documents — exactly what the
  script's smoke step checks.

Running any of this for real requires the upstream corpus and a few
non-obvious environment details — see **[SETUP.md](SETUP.md)**. Most load-bearing
of them: `google-cloud-aiplatform>=1.38` is required for the partner models and
is *not* in upstream's requirements, and without it every Mistral/Llama call
fails with a misleading `BadRequestError` rather than an import error.

## What this does and does not show

**Does:** Gemini works correctly as the reasoning layer inside Glean's own
agent harness and tool contract, and how that compares to a hand-rolled ADK
agent and a naive retrieval baseline.

**Does not:** support ranking these systems by accuracy. 30 questions is too
few — one question moves a score 3.3pp and no pairwise correctness difference
is significant. For the toolkit comparison specifically the effective sample is
8 questions, not 30, because retrieval is identical on the other 22.

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

The full bake-off matrix (4 models × 2 toolkits, smoke-tested and pooled).
Partner models additionally need `google-cloud-aiplatform>=1.38`:

```bash
VERTEX_PROJECT=your-project bash src/scripts/run_model_toolkit_matrix.sh
```

The two analysis modules read only `results/` and need no corpus or
credentials:

```bash
python -m src.scripts.answer_evaluation.significance
python -m src.scripts.answer_evaluation.harness_divergence
```
