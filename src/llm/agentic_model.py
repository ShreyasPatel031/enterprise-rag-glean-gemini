"""Resolve a --model string into whatever ADK's LlmAgent needs to drive it.

ADK's LlmAgent accepts model in two forms: a bare string, resolved against
Google's own Gemini registry, or a google.adk.models.lite_llm.LiteLlm
instance, which routes through LiteLLM to any provider LiteLLM supports.

Gemini models pass straight through as a string, exactly as both agentic
harnesses (adk_harness.py, glean_gemini_agent.py) did before this existed.
Mistral and Llama did not work in either harness before this module: neither
harness's model plumbing recognized non-Gemini names, and separately,
src/llm/vertex_llm.py's own Mistral/Llama code paths (_generate_mistral,
_generate_llama) never sent a `tools` field or parsed `tool_calls` -- they
only did plain-text completion, which is why the single-shot harness could
use them but the agentic ones could not.

This resolves those two through LiteLLM's Vertex AI partner-model route
instead of hand-rolling tool-call translation a second time: LiteLLM's
MistralConfig declares real tools/tool_choice support and targets the exact
publishers/mistralai/...:rawPredict endpoint vertex_llm.py already calls;
Llama routes to the same endpoints/openapi/chat/completions endpoint via the
standard OpenAI-compatible path.

Per-partner regions mirror src/llm/vertex_llm.py's MISTRAL_VERTEX_LOCATION /
LLAMA_VERTEX_LOCATION -- LiteLLM's own Vertex default region (us-central1) is
wrong for both partner routes and must be overridden per call.

Model IDs and tool-calling support were verified with live calls against
project-amer-scs-sandbox (2026-08-18). Both endpoints returned HTTP 200 with
finish_reason="tool_calls" and a well-formed function call:
    mistral-medium-3                              europe-west4, :rawPredict
    meta/llama-4-maverick-17b-128e-instruct-maas  us-east5, endpoints/openapi
What is still unverified is the full multi-turn agent loop through ADK (tool
result feedback, parallel calls, the toolkit's own richer specs) -- run the
smoke-test step in src/scripts/run_model_toolkit_matrix.sh before a full run.
"""

from __future__ import annotations

import os

MISTRAL_VERTEX_LOCATION = os.environ.get("MISTRAL_VERTEX_LOCATION", "europe-west4")
LLAMA_VERTEX_LOCATION = os.environ.get("LLAMA_VERTEX_LOCATION", "us-east5")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")

# --model value the caller passes in -> (litellm model id, region). The
# left-hand keys are the same short names src/llm/vertex_llm.py's is_mistral
# / is_llama checks match, so callers can carry over whatever they already
# passed to the single-shot harness.
_PARTNER_MODELS = {
    "mistral-medium-3": ("vertex_ai/mistral-medium-3", MISTRAL_VERTEX_LOCATION),
    "llama-4-maverick": (
        "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
        LLAMA_VERTEX_LOCATION,
    ),
}


# --- forced tool_choice: tried and rejected, kept here so it is not retried ---
#
# The obvious fix for Llama 4 Maverick's textual-pseudo-tool-call failure is
# forcing tool_choice="required" (ADK: tool_config.function_calling_config.mode
# = ANY). Verified live against project-amer-scs-sandbox and disqualified on
# both ends it would need to help:
#
#   * Llama 4 Maverick's Vertex MaaS endpoint rejects it outright:
#       400 INVALID_ARGUMENT "forced function calling (mode = ANY) is not
#       supported for this model"
#     -- so it cannot be applied to the model with the problem.
#   * Mistral Medium 3's endpoint accepts it, but forcing a tool call on every
#     turn removes the model's ability to ever decide it is done: in testing
#     it called `search` on 6+ consecutive turns and never reached
#     `finish_answer`. Mistral was already 30/30 clean without this, so
#     applying it would trade a working model for a hung one.
#
# Do not re-attempt this without a per-turn stopping mechanism (e.g. force
# ANY only until the model has called at least one non-search tool, then
# drop back to AUTO) -- naive blanket forcing is worse than the defect it
# targets.


# Extra instruction text for models observed to emit tool calls as plain
# text instead of through the structured mechanism (see the rejected
# forced-tool_choice section above for why the request-level fix doesn't
# work). Appended to a harness's system instruction, never replacing it.
_TOOL_CALL_REMINDERS = {
    "llama-4-maverick": (
        "\n\nCRITICAL: call tools ONLY through the function-calling mechanism "
        "your API exposes. Never write a call as plain text, e.g. "
        '`read_document(doc_id="...")` or inside brackets or code fences -- '
        "text like that does not execute and produces no result. If you find "
        "yourself about to type a function name followed by parentheses, stop "
        "and use a real tool call instead. Never invent, guess, or continue as "
        "if a tool had returned a result you were not actually given -- always "
        "wait for the genuine tool response before using its content."
    ),
}


def tool_call_reminder(model: str) -> str:
    """Extra instruction text to append for models prone to textual pseudo
    tool calls. Empty string for every other model."""
    return _TOOL_CALL_REMINDERS.get(model, "")


def resolve_agentic_model(model: str):
    """Turn a --model string into what LlmAgent(model=...) expects.

    Gemini names pass through unchanged (the pre-existing behavior of both
    agentic harnesses). Mistral/Llama names return a LiteLlm instance wired
    to the correct Vertex partner region; anything else is passed through
    unchanged on the assumption ADK or LiteLLM already knows what to do with
    it, rather than rejected here.
    """
    if model not in _PARTNER_MODELS:
        return model

    if not VERTEX_PROJECT:
        raise ValueError("VERTEX_PROJECT env var required to resolve a partner model.")

    from google.adk.models.lite_llm import LiteLlm

    litellm_model, location = _PARTNER_MODELS[model]
    return LiteLlm(
        model=litellm_model,
        vertex_project=VERTEX_PROJECT,
        vertex_location=location,
    )
