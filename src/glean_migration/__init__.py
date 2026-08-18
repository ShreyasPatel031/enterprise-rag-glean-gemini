"""Glean -> Gemini migration harness.

Runs Glean's real agent toolkit (tool specs, descriptions, ToolResult
envelope, ADK adapters) against the EnterpriseRAG-Bench corpus by swapping
only the toolkit's backend transport layer, which the toolkit itself
supports via ``register_backend``.
"""
