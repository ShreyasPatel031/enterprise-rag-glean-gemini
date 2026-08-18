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
