"""Local corpus backends for Glean's real agent toolkit.

The toolkit routes every built-in tool through a backend registry
(``glean.agent_toolkit.tools._transport``), whose own docstring notes that
"swapping a tool's backend is a one-line declaration change rather than a
plumbing edit in the tool module". This module supplies backends that serve
Glean's tools from the EnterpriseRAG-Bench corpus instead of a hosted Glean
tenant, so the real tool specs / descriptions / ``ToolResult`` envelope /
ADK adapters stay untouched while the data comes from local documents.

Payload shapes intentionally mirror what each tool's shaper produces
against the real API, so an agent cannot tell the difference from the
shape alone.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from google import genai

from src.paths import SOURCES_DIR
from src.utils.document_content import extract_document_content
from src.utils.file_io import load_json_file

EMBEDDING_MODEL = "gemini-embedding-001"
_SNIPPET_MAX_CHARS = 500


class LocalBackend:
    """Backend protocol impl that dispatches to a plain local callable.

    Mirrors ``glean.agent_toolkit.tools._transport.Backend``: the ``client``
    argument is accepted and ignored, since no Glean tenant is involved.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def execute(self, client: Any, arguments: Mapping[str, Any]) -> Any:
        return self.fn(**arguments)

    async def execute_async(self, client: Any, arguments: Mapping[str, Any]) -> Any:
        return self.fn(**arguments)


class CorpusIndex:
    """Embedding index over a slice of the benchmark corpus.

    Loads the vector/uuid/path caches written by
    ``src.scripts.answer_generation.vertex_retrieval`` so every harness in
    this repo searches exactly the same documents.
    """

    def __init__(
        self,
        source_types: list[str],
        corpus_cap: int,
        cache_dir: str = "answer_evaluation/_vertex_cache",
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        cache_key = "_".join(sorted(source_types)) + f"_cap{corpus_cap}"
        index_path = os.path.join(cache_dir, f"index_{cache_key}.json")
        uuids_path = os.path.join(cache_dir, f"uuids_{cache_key}.json")
        vectors_path = os.path.join(cache_dir, f"vectors_{cache_key}.npy")

        if not (os.path.exists(index_path) and os.path.exists(vectors_path)):
            raise SystemExit(
                f"No cached corpus at {index_path}. Run vertex_retrieval.py first "
                "with the same --source-types/--corpus-cap."
            )

        self.path_index: dict[str, str] = load_json_file(index_path)
        self.uuids: list[str] = load_json_file(uuids_path)
        vectors = np.load(vectors_path)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vectors / norms

        self.client = genai.Client(
            vertexai=True,
            project=project or os.environ.get("VERTEX_PROJECT", "project-amer-scs-sandbox"),
            location=location or os.environ.get("VERTEX_LOCATION", "us-central1"),
        )

    def datasource_of(self, doc_id: str) -> str | None:
        """Return the corpus source type (slack, jira, ...) for a document."""
        path = self.path_index.get(doc_id)
        if not path:
            return None
        rel = os.path.relpath(path, SOURCES_DIR)
        return rel.split(os.sep)[0]

    def load(self, doc_id: str) -> tuple[str, str]:
        return extract_document_content(load_json_file(self.path_index[doc_id]))

    def search(
        self,
        query: str,
        datasources: list[str] | None = None,
        page_size: int = 10,
    ) -> list[str]:
        """Return document ids ranked by cosine similarity to *query*."""
        resp = self.client.models.embed_content(model=EMBEDDING_MODEL, contents=[query])
        qv = np.array(resp.embeddings[0].values, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        sims = self.vectors @ qv

        wanted = {d.lower() for d in datasources} if datasources else None
        hits: list[str] = []
        for idx in np.argsort(-sims):
            doc_id = self.uuids[idx]
            if wanted is not None and (self.datasource_of(doc_id) or "") not in wanted:
                continue
            hits.append(doc_id)
            if len(hits) >= page_size:
                break
        return hits


def _shaped_search(index: CorpusIndex, doc_ids: list[str], more: bool) -> dict[str, Any]:
    """Build the payload shape produced by ``_shape_search_response``."""
    results = []
    for doc_id in doc_ids:
        title, content = index.load(doc_id)
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > _SNIPPET_MAX_CHARS:
            snippet = snippet[:_SNIPPET_MAX_CHARS] + "..."
        results.append(
            {
                "title": title,
                "url": f"local://{index.datasource_of(doc_id)}/{doc_id}",
                "snippets": [snippet],
                "datasource": index.datasource_of(doc_id),
                "document_id": doc_id,
            }
        )
    return {
        "results": results,
        "result_count": len(results),
        "has_more_results": more,
    }


def register_local_backends(
    index: CorpusIndex,
    *,
    employee_directory_path: str = "generated_data/employee_directory.yaml",
) -> list[str]:
    """Point Glean's built-in tools at the local corpus.

    Returns the list of tool names that were re-registered. Tools with no
    corpus equivalent (web/calendar/outlook search) are deliberately left
    alone so they keep returning the toolkit's own structured errors.
    """
    from glean.agent_toolkit.tools._transport import register_backend

    def search(
        query: str,
        datasources: list[str] | None = None,
        filters: list[dict[str, Any]] | None = None,
        page_size: int = 10,
    ) -> dict[str, Any]:
        doc_ids = index.search(query, datasources=datasources, page_size=page_size)
        return _shaped_search(index, doc_ids, more=len(doc_ids) >= page_size)

    def read_document(
        document_id: str | None = None, url: str | None = None
    ) -> dict[str, Any]:
        doc_id = document_id
        if doc_id is None and url:
            doc_id = url.rstrip("/").split("/")[-1]
        if not doc_id or doc_id not in index.path_index:
            return {"error": f"Document '{doc_id}' not found in the corpus."}
        title, content = index.load(doc_id)
        return {
            "document_id": doc_id,
            "title": title,
            "datasource": index.datasource_of(doc_id),
            "content": content,
        }

    def _scoped_search(source_type: str) -> Callable[..., dict[str, Any]]:
        def _search(query: str, page_size: int = 10, **_: Any) -> dict[str, Any]:
            doc_ids = index.search(query, datasources=[source_type], page_size=page_size)
            return _shaped_search(index, doc_ids, more=len(doc_ids) >= page_size)

        return _search

    def employee_search(query: str, page_size: int = 10, **_: Any) -> dict[str, Any]:
        """Grep the generated employee directory (it is not in the vector index)."""
        try:
            with open(employee_directory_path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError as exc:
            return {"error": f"Employee directory unavailable: {exc}"}

        terms = [t for t in query.lower().split() if t]
        matches = [ln.strip() for ln in lines if any(t in ln.lower() for t in terms)]
        return {
            "results": [{"snippets": [m]} for m in matches[:page_size]],
            "result_count": min(len(matches), page_size),
            "has_more_results": len(matches) > page_size,
        }

    registered = []
    for name, fn in (
        ("glean_search", search),
        ("glean_read_document", read_document),
        ("glean_code_search", _scoped_search("github")),
        ("glean_gmail_search", _scoped_search("gmail")),
        ("glean_employee_search", employee_search),
    ):
        register_backend(name, LocalBackend(fn))
        registered.append(name)
    return registered
