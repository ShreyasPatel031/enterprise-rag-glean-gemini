"""Vertex AI (Gemini + partner MaaS models) implementation of the LLM interface."""

import json
import os
from collections.abc import Generator

import google.auth
import google.auth.transport.requests
import requests
from google import genai
from google.genai import types

from src.llm.interface import LLMInterface, Message, ReasoningLevel, ToolCall


VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
# Partner MaaS models on Vertex are only served from specific regions,
# independent of VERTEX_LOCATION used for Gemini.
MISTRAL_VERTEX_LOCATION = os.environ.get("MISTRAL_VERTEX_LOCATION", "europe-west4")
LLAMA_VERTEX_LOCATION = os.environ.get("LLAMA_VERTEX_LOCATION", "us-east5")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-pro")
CHEAP_LLM_MODEL_NAME = os.environ.get("CHEAP_LLM_MODEL_NAME", "gemini-2.5-flash-lite")


def _get_access_token() -> str:
    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _generate_mistral(model: str, messages: list[Message]) -> str:
    """Call a Mistral MaaS model on Vertex via its native rawPredict endpoint."""
    token = _get_access_token()
    url = (
        f"https://{MISTRAL_VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{VERTEX_PROJECT}/locations/{MISTRAL_VERTEX_LOCATION}/publishers/mistralai/"
        f"models/{model}:rawPredict"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system" if m.role == "system" else m.role, "content": m.content}
            for m in messages
        ],
        "stream": False,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def _generate_llama(model: str, messages: list[Message]) -> str:
    """Call a Meta Llama MaaS model on Vertex via its OpenAI-compatible endpoint."""
    token = _get_access_token()
    url = (
        f"https://{LLAMA_VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/"
        f"{VERTEX_PROJECT}/locations/{LLAMA_VERTEX_LOCATION}/endpoints/openapi/"
        "chat/completions"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system" if m.role == "system" else m.role, "content": m.content}
            for m in messages
        ],
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


class VertexLLM(LLMInterface):
    """Gemini-on-Vertex-AI implementation of the LLM interface.

    Uses Application Default Credentials rather than an API key, so this
    provider is selected via LLM_PROVIDER=vertex with no LLM_API_KEY required.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        quiet: bool = False,
        reasoning_level: ReasoningLevel = "medium",
    ):
        if not VERTEX_PROJECT:
            raise ValueError("VERTEX_PROJECT env var required for the vertex provider.")
        self.model = model or LLM_MODEL_NAME
        self.tools = tools
        self.quiet = quiet
        self.reasoning_level = reasoning_level
        self.is_mistral = self.model.startswith("mistral")
        self.is_llama = "llama" in self.model.lower()
        if not (self.is_mistral or self.is_llama):
            self.client = genai.Client(
                vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
            )

    def generate(
        self, messages: list[Message]
    ) -> Generator[str | ToolCall, None, None]:
        if not self.quiet:
            print(f"Waiting on LLM (Vertex: {self.model})...", flush=True)

        if self.is_mistral:
            yield _generate_mistral(self.model, messages)
            return
        if self.is_llama:
            yield _generate_llama(self.model, messages)
            return

        system_parts = [m.content for m in messages if m.role == "system"]
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))

        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )

        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        text = response.text or ""
        yield text
