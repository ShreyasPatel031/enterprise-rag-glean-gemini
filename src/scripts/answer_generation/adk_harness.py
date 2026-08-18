"""Agentic RAG harness using Google's Agent Development Kit (ADK).

Represents the migration-target architecture: an ADK LlmAgent iteratively
searches and reads documents via tools (rather than a single fixed
retrieve-then-generate pass) before submitting a final answer. Reuses the
same cached corpus embeddings built by vertex_retrieval.py so the two
harnesses are compared on identical retrieval substrate and the identical
30-question sample (same source_types + question_limit + RANDOM_SEED).

Usage:
    python -m src.scripts.answer_generation.adk_harness \
        --source-types confluence jira \
        --question-limit 30 \
        --corpus-cap 800 \
        --model gemini-2.5-flash \
        --output answer_evaluation/system_gemini_adk.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading

import numpy as np
from google import genai
from google.genai import types

from src.llm.agentic_model import resolve_agentic_model
from src.scripts.answer_generation.vertex_retrieval import select_questions
from src.paths import QUESTIONS_PATH
from src.utils.document_content import extract_document_content
from src.utils.file_io import load_json_file
from src.utils.retrieval import append_result, load_existing_question_ids, load_questions

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "project-amer-scs-sandbox")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_MODEL = "gemini-embedding-001"

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", VERTEX_PROJECT)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_LOCATION)

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions.in_memory_session_service import InMemorySessionService  # noqa: E402

SYSTEM_INSTRUCTION = """\
You are an enterprise search agent with access to a company's internal document corpus \
(Confluence wikis and Jira tickets). You do NOT know the answer to the user's question in \
advance -- you must find it by searching and reading real documents.

Process:
1. Call `search` with a query describing what you're looking for. It returns up to 10 \
candidate documents (ID, title, snippet).
2. Call `read_document` on promising candidates to read their full content.
3. Refine your search and read more documents if the first results are not sufficient. \
You may call `search` multiple times with different queries.
4. Once you have enough information, you MUST call the `finish_answer` tool as your final \
action, with your final answer and the list of document IDs you actually used. \
NEVER respond with a plain text message as your final turn -- a plain text final response \
is treated as a failure. The task is only complete once finish_answer has been called.

Base your answer purely on documents you have read. Do not make up information. If the \
answer is not present in the corpus after a reasonable search, call finish_answer saying so.
"""


def make_search_tool(path_index: dict[str, str], uuids: list[str], normed_vectors: np.ndarray, client: genai.Client):
    def search(query: str) -> str:
        """Search the enterprise document corpus for documents relevant to a query.

        Args:
            query: A search query describing the information you're looking for.

        Returns:
            Up to 10 candidate documents, one per line, as "ID | Title | snippet".
        """
        resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=[query])
        qv = np.array(resp.embeddings[0].values, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        sims = normed_vectors @ qv
        top_idx = np.argsort(-sims)[:10]
        lines = []
        for i in top_idx:
            uid = uuids[i]
            doc = load_json_file(path_index[uid])
            title, content = extract_document_content(doc)
            snippet = content[:200].replace("\n", " ")
            lines.append(f"{uid} | {title} | {snippet}")
        return "\n".join(lines)

    return search


def make_read_tool(path_index: dict[str, str], state: dict):
    def read_document(doc_id: str) -> str:
        """Read the full title and content of one document.

        Args:
            doc_id: The document ID (e.g. dsid_xxxx) as returned by search.

        Returns:
            The document's full title and content, or an error message if the ID is invalid.
        """
        if doc_id not in path_index:
            return f"Error: document '{doc_id}' not found. Use an ID returned by search."
        state["read_ids"].append(doc_id)
        doc = load_json_file(path_index[doc_id])
        title, content = extract_document_content(doc)
        return f"Title: {title}\n\n{content}"

    return read_document


def make_finish_tool(state: dict):
    def finish_answer(answer: str, document_ids: list[str]) -> str:
        """Submit your final answer and end the task. Call this exactly once, when done.

        Args:
            answer: Your final answer to the user's question.
            document_ids: The document IDs you used to construct the answer.

        Returns:
            Confirmation message.
        """
        state["answer"] = answer
        state["document_ids"] = document_ids
        return "Answer submitted."

    return finish_answer


async def process_question(
    question: dict,
    model: str,
    path_index: dict[str, str],
    uuids: list[str],
    normed_vectors: np.ndarray,
    embed_client: genai.Client,
    output: str,
    write_lock: threading.Lock,
    semaphore: asyncio.Semaphore,
) -> None:
    qid = question["question_id"]
    async with semaphore:
        state: dict = {"answer": None, "document_ids": None, "read_ids": []}
        tools = [
            make_search_tool(path_index, uuids, normed_vectors, embed_client),
            make_read_tool(path_index, state),
            make_finish_tool(state),
        ]
        agent = LlmAgent(
            name="enterprise_search_agent",
            model=resolve_agentic_model(model),
            instruction=SYSTEM_INSTRUCTION,
            tools=tools,
        )
        session_service = InMemorySessionService()
        runner = Runner(agent=agent, app_name="adk_harness", session_service=session_service)
        user_id = "eval_user"
        await session_service.create_session(app_name="adk_harness", user_id=user_id, session_id=qid)

        last_text = ""
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=qid,
                new_message=types.Content(role="user", parts=[types.Part(text=question["question"])]),
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            last_text = part.text
        except Exception as e:
            print(f"  {qid} ADK run error: {e}")

        answer = state["answer"] if state["answer"] is not None else last_text
        if state["document_ids"] is not None:
            document_ids = state["document_ids"]
        else:
            # finish_answer was never called (agent ended on plain text) --
            # fall back to every document it actually opened via read_document,
            # deduped, preserving order.
            seen: set[str] = set()
            document_ids = []
            for did in state["read_ids"]:
                if did not in seen:
                    seen.add(did)
                    document_ids.append(did)

        append_result(
            output,
            {"question_id": qid, "answer": answer, "document_ids": document_ids},
            write_lock,
        )
        print(f"  {qid} done (finish_answer called: {state['answer'] is not None})")


async def main_async(args: argparse.Namespace) -> None:
    questions = load_questions(QUESTIONS_PATH)
    selected = select_questions(questions, args.source_types, args.question_limit)
    print(f"Selected {len(selected)} questions for source_types={args.source_types}")

    if args.resume:
        existing_ids = load_existing_question_ids(args.output)
        selected = [q for q in selected if q["question_id"] not in existing_ids]
        print(f"  Resuming: {len(selected)} questions remaining")

    cache_key = "_".join(sorted(args.source_types)) + f"_cap{args.corpus_cap}"
    cache_dir = args.cache_dir
    index_cache_path = os.path.join(cache_dir, f"index_{cache_key}.json")
    vectors_cache_path = os.path.join(cache_dir, f"vectors_{cache_key}.npy")
    uuids_cache_path = os.path.join(cache_dir, f"uuids_{cache_key}.json")

    if not (os.path.exists(index_cache_path) and os.path.exists(vectors_cache_path)):
        raise SystemExit(
            f"No cached corpus found at {index_cache_path}. Run vertex_retrieval.py first "
            "with the same --source-types/--corpus-cap to build the shared embedding cache."
        )

    path_index = load_json_file(index_cache_path)
    uuids = load_json_file(uuids_cache_path)
    vectors = np.load(vectors_cache_path)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_vectors = vectors / norms

    embed_client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_lock = threading.Lock()
    semaphore = asyncio.Semaphore(args.parallelism)

    await asyncio.gather(
        *[
            process_question(
                q, args.model, path_index, uuids, normed_vectors, embed_client,
                args.output, write_lock, semaphore,
            )
            for q in selected
        ]
    )

    print(f"Done. Answers written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-types", nargs="+", required=True)
    parser.add_argument("--question-limit", type=int, default=30)
    parser.add_argument("--corpus-cap", type=int, default=800)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="answer_evaluation/_vertex_cache")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
