#!/usr/bin/env bash
#
# Run the generator-model x agentic-toolkit bake-off matrix.
#
# The point of the matrix: the ADK-vs-Glean-toolkit head-to-head currently
# rests on a single generator model (Gemini 2.5 Flash) at n=30, which has 80%
# power to detect only a ~79/21 preference. Pooling the head-to-head across
# four models is what makes the comparison capable of resolving anything --
# see src/scripts/answer_evaluation/harness_divergence.py.
#
# MATRIX
#
#   Gemini 2.5 Flash       x ADK / Glean toolkit   already in results/
#   Gemini 2.5 Flash-Lite  x ADK / Glean toolkit   run by this script
#   Mistral Medium 3       x ADK / Glean toolkit   run by this script
#   Llama 4 Maverick       x ADK / Glean toolkit   run by this script
#
# MISTRAL / LLAMA ON THE AGENTIC TOOLKITS
#
# Previously impossible: both agentic harnesses passed --model straight to
# ADK's LlmAgent as a bare string, which only resolves against ADK's Gemini
# registry, and src/llm/vertex_llm.py's own _generate_mistral/_generate_llama
# never sent a tools field or parsed tool_calls (plain-text completion only --
# fine for single-shot, useless for a tool-calling agent).
#
# src/llm/agentic_model.py fixes this: resolve_agentic_model() passes Gemini
# names through unchanged and wraps mistral-medium-3 / llama-4-maverick in
# ADK's LiteLlm against the Vertex partner endpoints.
#
# VERIFIED LIVE against project-amer-scs-sandbox (2026-08-18):
#   * raw endpoints: both return HTTP 200 with finish_reason="tool_calls"
#     (mistral europe-west4 :rawPredict; llama us-east5 endpoints/openapi)
#   * full ADK loop with hand-rolled tools: all 3 models completed
#     search -> tool result -> finish_answer
#   * full ADK loop with GLEAN'S REAL tool specs (glean_search /
#     glean_read_document via as_adk_tool, toolkit v0.8.0): all 3 models
#     completed search -> read -> submit_answer
#
# Requires google-cloud-aiplatform (litellm's Vertex partner route imports
# `vertexai`); without it every partner call fails with a confusing
# BadRequestError. Installed in .venv here.
#
# KNOWN QUIRK -- Llama 4 Maverick sometimes emits a *textual* pseudo tool call
# with raw chat-template markers, e.g.
#     <|header_start|>assistant<|header_end|>\n\nglean_read_document(document_id="dsid_001")
# before issuing the real structured call. It recovered every time in testing,
# but it means Llama answers may carry template noise, and a run where it does
# NOT recover yields an answer with no document_ids. The smoke step below
# catches that case.
#
# Usage:  bash src/scripts/run_model_toolkit_matrix.sh [OUT_DIR]
# Needs:  VERTEX_PROJECT set, ADC available, benchmark corpus + embedding
#         cache present (build it with vertex_retrieval.py first).

set -euo pipefail

OUT_DIR="${1:-answer_evaluation}"
SOURCE_TYPES="confluence jira"
QUESTION_LIMIT=30
CORPUS_CAP=800
PARALLELISM=4
# Partner MaaS endpoints have far tighter quota than Gemini: mistral on
# europe-west4 and llama on us-east5 return 429s at PARALLELISM=4, and the
# harnesses swallow the exception and write a blank-answer row, which the
# scorer then counts against the model. Run those serially.
PARTNER_PARALLELISM=1
SMOKE_LIMIT=3

PY="${PYTHON:-python}"

MODELS=("gemini-2.5-flash" "gemini-2.5-flash-lite" "mistral-medium-3" "llama-4-maverick")
# Models resolve_agentic_model() routes through LiteLlm rather than handing
# straight to ADK.
PARTNER_MODELS=("mistral-medium-3" "llama-4-maverick")

declare -A TOOLKITS=(
  [adk]="src.scripts.answer_generation.adk_harness"
  [gleantoolkit]="src.glean_migration.glean_gemini_agent"
)

if [[ -z "${VERTEX_PROJECT:-}" ]]; then
  echo "VERTEX_PROJECT is not set; the harnesses need it." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

slug_for() { echo "${1//gemini-2.5-/}"; }

is_partner_model() {
  local needle="$1"
  for m in "${PARTNER_MODELS[@]}"; do
    [[ "$m" == "$needle" ]] && return 0
  done
  return 1
}

parallelism_for() {
  if is_partner_model "$1"; then echo "$PARTNER_PARALLELISM"; else echo "$PARALLELISM"; fi
}

# --- smoke test: partner models only, few questions, before the real run -----
# A model that free-texts instead of calling tools produces an answer with no
# document_ids rather than an error, so this is checked explicitly.
for model in "${MODELS[@]}"; do
  is_partner_model "$model" || continue
  for toolkit in "${!TOOLKITS[@]}"; do
    smoke_out="$OUT_DIR/_smoke_${toolkit}_${model}.jsonl"
    echo "== smoke: $toolkit x $model ($SMOKE_LIMIT questions) -> $smoke_out"
    rm -f "$smoke_out"
    $PY -m "${TOOLKITS[$toolkit]}" \
      --source-types $SOURCE_TYPES \
      --question-limit "$SMOKE_LIMIT" \
      --corpus-cap "$CORPUS_CAP" \
      --model "$model" \
      --parallelism 1 \
      --output "$smoke_out"
    empty=$($PY -c "
import json
rows=[json.loads(l) for l in open('$smoke_out') if l.strip()]
print(sum(1 for r in rows if not r.get('document_ids')))
")
    if [[ "$empty" -gt 0 ]]; then
      echo "   WARNING: $empty/$SMOKE_LIMIT smoke answers cite no documents." >&2
      echo "   The model likely answered without calling a tool. Inspect" >&2
      echo "   $smoke_out before trusting the full run." >&2
    else
      echo "   OK: all $SMOKE_LIMIT smoke answers cite >=1 document."
    fi
  done
done

# --- generation --------------------------------------------------------------
for model in "${MODELS[@]}"; do
  slug=$(slug_for "$model")
  for toolkit in "${!TOOLKITS[@]}"; do
    answers="$OUT_DIR/system_${toolkit}_${slug}.jsonl"
    if [[ -s "$answers" ]]; then
      echo "== skip (exists): $answers"; continue
    fi
    par=$(parallelism_for "$model")
    echo "== generate: $toolkit x $model (parallelism $par) -> $answers"
    # --resume so a re-invocation after a rate-limit repair refills only the
    # dropped questions instead of regenerating the whole cell.
    $PY -m "${TOOLKITS[$toolkit]}" \
      --source-types $SOURCE_TYPES \
      --question-limit "$QUESTION_LIMIT" \
      --corpus-cap "$CORPUS_CAP" \
      --model "$model" \
      --parallelism "$par" \
      --resume \
      --output "$answers"
  done
done

# --- triage: drop rows that failed for infrastructure reasons ----------------
# A rate-limited or timed-out question lands as a blank-answer row rather than
# an error. Scoring it would penalise the model for a quota problem. Dropping
# it here means the loop above (with --resume) refills it on the next pass.
echo
echo "============ answer row triage ============"
$PY -m src.scripts.answer_evaluation.repair_failed_answers \
  "$OUT_DIR"/system_*.jsonl || true
echo "If rows were listed above, re-run this script to refill them before"
echo "trusting the scores; --resume makes that cheap."

# --- tool-call integrity over the full answer files ----------------------------
# A model that fails to emit a structured tool call does not error -- it
# free-texts, and some models narrate a tool call and invent its result. Those
# answers still get scored, which silently prices an integration defect as a
# quality difference. Check before scoring so the numbers can be read honestly.
echo
echo "============ tool-call integrity ============"
$PY -m src.scripts.answer_evaluation.tool_call_integrity \
  "$OUT_DIR"/system_*.jsonl \
  --json "$OUT_DIR/results_tool_call_integrity.json" || true

# --- pointwise scoring -------------------------------------------------------
for model in "${MODELS[@]}"; do
  slug=$(slug_for "$model")
  for toolkit in "${!TOOLKITS[@]}"; do
    answers="$OUT_DIR/system_${toolkit}_${slug}.jsonl"
    scores="$OUT_DIR/results_${toolkit}_${slug}.json"
    if [[ -s "$scores" ]]; then
      echo "== skip (exists): $scores"; continue
    fi
    echo "== score: $answers"
    $PY -m src.scripts.answer_evaluation.metrics_based_eval \
      --answers-file "$answers" \
      --results-file "$scores" \
      --no-correction --parallelism "$PARALLELISM"
  done
done

# --- per-model head-to-head --------------------------------------------------
# Flags verified against the upstream scripts: comparative_eval takes
# --answer-file-1/-2 (singular "answer") and has no --no-correction;
# metrics_based_eval takes --answers-file (plural) and does have it.
for model in "${MODELS[@]}"; do
  slug=$(slug_for "$model")
  comparative="$OUT_DIR/results_comparative_adk_vs_gleantoolkit_${slug}.json"
  if [[ -s "$comparative" ]]; then
    echo "== skip (exists): $comparative"; continue
  fi
  echo "== compare: adk vs gleantoolkit on $model"
  $PY -m src.scripts.answer_evaluation.comparative_eval \
    --answer-file-1 "$OUT_DIR/system_adk_${slug}.jsonl" \
    --answer-file-2 "$OUT_DIR/system_gleantoolkit_${slug}.jsonl" \
    --results-file "$comparative" \
    --updated-questions-file "$OUT_DIR/questions_updated_${slug}.jsonl" \
    --parallelism "$PARALLELISM"
done

# --- per-model validation ----------------------------------------------------
for model in "${MODELS[@]}"; do
  slug=$(slug_for "$model")
  echo
  echo "===================== divergence: $model ====================="
  $PY -m src.scripts.answer_evaluation.harness_divergence \
    --answers-a "$OUT_DIR/system_adk_${slug}.jsonl" \
    --answers-b "$OUT_DIR/system_gleantoolkit_${slug}.jsonl" \
    --scores-a "$OUT_DIR/results_adk_${slug}.json" \
    --scores-b "$OUT_DIR/results_gleantoolkit_${slug}.json" \
    --comparative "$OUT_DIR/results_comparative_adk_vs_gleantoolkit_${slug}.json" \
    --json "$OUT_DIR/results_harness_divergence_${slug}.json"
done

# --- pooled verdict ----------------------------------------------------------
# The payoff of running four models instead of one.
echo
echo "============ pooled ADK vs Glean toolkit, all models ============"
$PY -c "
import json, sys
sys.path.insert(0, '.')
from src.scripts.answer_evaluation.harness_divergence import (
    exact_binomial_two_sided, min_detectable_rate)

# slugs must match slug_for() above
models = ['flash', 'flash-lite', 'mistral-medium-3', 'llama-4-maverick']
pref_a = pref_b = dec_a = dec_b = 0
rows = []
for slug in models:
    path = f'$OUT_DIR/results_comparative_adk_vs_gleantoolkit_{slug}.json'
    try:
        qs = json.load(open(path))['questions']
    except FileNotFoundError:
        print(f'  missing (skipped): {path}')
        continue
    c = [q['comparison'] for q in qs]
    a  = sum(1 for x in c if x.get('preferred_system') == '1')
    b  = sum(1 for x in c if x.get('preferred_system') == '2')
    da = sum(1 for x in c if x.get('preferred_system') == '1' and not x.get('effectively_equivalent'))
    db = sum(1 for x in c if x.get('preferred_system') == '2' and not x.get('effectively_equivalent'))
    pref_a += a; pref_b += b; dec_a += da; dec_b += db
    rows.append((slug, a, b, da, db))

for slug, a, b, da, db in rows:
    print(f'  {slug:<20} all {a}-{b}    decisive {da}-{db}')

if pref_a + pref_b:
    p = exact_binomial_two_sided(min(pref_a, pref_b), pref_a + pref_b)
    print(f'\n  pooled all judged:  ADK {pref_a} - {pref_b} Glean   p={p:.4f}'
          + ('  SIG' if p < 0.05 else '  (not significant)'))
if dec_a + dec_b:
    n = dec_a + dec_b
    p = exact_binomial_two_sided(min(dec_a, dec_b), n)
    mde = min_detectable_rate(n)
    print(f'  pooled decisive:    ADK {dec_a} - {dec_b} Glean   p={p:.4f}'
          + ('  SIG' if p < 0.05 else '  (not significant)'))
    if mde:
        print(f'\n  {n} pooled decisive picks -> 80%% power to detect a '
              f'{100*mde:.0f}/{100*(1-mde):.0f} preference.')
        print('  A null here means \"no difference larger than that\", not \"identical\".')
"
