"""Glean's real agent toolkit, driven by Gemini via Google ADK.

Builds an ADK agent whose tools are Glean's own built-in tools -- unmodified
specs, descriptions, parameters, and ``ToolResult`` envelope, converted via
the toolkit's own ``ADKAdapter`` -- and answers EnterpriseRAG-Bench
questions with them.

By default the tools are served from the local benchmark corpus (see
``local_backends``) because no Glean tenant is available. Set
``GLEAN_API_TOKEN`` (plus ``GLEAN_SERVER_URL`` or ``GLEAN_INSTANCE``) and
pass ``--live-glean`` to run the identical agent against a real tenant.

Usage:
    python -m src.glean_migration.glean_gemini_agent \
        --source-types confluence jira \
        --question-limit 30 \
        --corpus-cap 800 \
        --model gemini-2.5-flash \
        --output answer_evaluation/system_glean_toolkit_gemini.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
from typing import Any

from google.genai import types

from src.glean_migration.local_backends import CorpusIndex, register_local_backends
from src.llm.agentic_model import resolve_agentic_model, tool_call_reminder
from src.paths import QUESTIONS_PATH
from src.scripts.answer_generation.vertex_retrieval import select_questions
from src.utils.retrieval import append_result, load_existing_question_ids, load_questions

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "project-amer-scs-sandbox")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", VERTEX_PROJECT)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_LOCATION)

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions.in_memory_session_service import InMemorySessionService  # noqa: E402

SYSTEM_INSTRUCTION = """\
You are an enterprise search assistant for a company's internal knowledge \
(wikis, tickets, docs). Answer the user's question using ONLY information you \
retrieve with your tools -- you do not know the answer in advance.

Process:
1. Call `glean_search` to find candidate documents. Search again with different \
phrasings if the first results are thin; internal jargon and codenames matter.
2. Call `glean_read_document` with a `document_id` from the search results to read \
promising documents in full before relying on them.
3. When you have enough grounded information, call `submit_answer` exactly once \
with your final answer and the document_ids you actually used.

Never invent facts. If the corpus does not contain the answer, say so via \
`submit_answer`.
"""


class _StubGleanClient:
    """Placeholder client.

    ``GleanContext.get_client()`` returns a pre-supplied client before any
    credential check, and the local backends ignore the client entirely, so
    this is only here to satisfy that resolution step.
    """


def configure_glean(live: bool) -> list[str]:
    """Configure the toolkit's client resolution and report the tool names."""
    from glean.agent_toolkit import configure
    from glean.agent_toolkit.tools.read_document import read_document
    from glean.agent_toolkit.tools.search import search

    if live:
        configure()  # resolve real credentials from the environment
    else:
        configure(client=_StubGleanClient())

    return [search.tool_spec.name, read_document.tool_spec.name]


def build_glean_tools() -> list[Any]:
    """Build a fresh set of ADK tools from Glean's built-in tool specs.

    Built per question: the read-tracking wrapper below rebinds ``tool.func``,
    so sharing tool objects across concurrently-running questions would leak
    one question's document ids into another's answer.
    """
    from glean.agent_toolkit.tools.read_document import read_document
    from glean.agent_toolkit.tools.search import search

    return [search.as_adk_tool(), read_document.as_adk_tool()]


async def answer_question(
    question: dict,
    model: str,
    output: str,
    write_lock: threading.Lock,
    semaphore: asyncio.Semaphore,
) -> None:
    qid = question["question_id"]
    async with semaphore:
        state: dict = {"answer": None, "document_ids": None, "seen": []}

        def submit_answer(answer: str, document_ids: list[str]) -> str:
            """Submit the final answer and finish. Call exactly once.

            Args:
                answer: The final answer to the user's question.
                document_ids: IDs of documents actually used for the answer.

            Returns:
                Confirmation string.
            """
            state["answer"] = answer
            state["document_ids"] = document_ids
            return "Answer recorded."

        # Observe which documents the agent actually pulled through Glean's
        # tools, as a fallback when the model answers without calling
        # submit_answer.
        tools = build_glean_tools()
        for tool in tools:
            if tool.name != "glean_read_document":
                continue
            inner = tool.func

            async def _tracking(*args: Any, __inner=inner, **kwargs: Any) -> Any:
                result = await __inner(*args, **kwargs)
                doc_id = kwargs.get("document_id")
                # Only credit a document the agent actually read. A model can
                # invent plausible-looking ids; those come back as an error
                # payload and must not enter the answer's document set.
                failed = isinstance(result, dict) and "error" in result
                if doc_id and not failed and doc_id not in state["seen"]:
                    state["seen"].append(doc_id)
                return result

            _tracking.__name__ = inner.__name__
            _tracking.__doc__ = inner.__doc__
            _tracking.__signature__ = inner.__signature__  # type: ignore[attr-defined]
            _tracking.__annotations__ = inner.__annotations__
            tool.func = _tracking

        agent = LlmAgent(
            name="glean_toolkit_agent",
            model=resolve_agentic_model(model),
            instruction=SYSTEM_INSTRUCTION + tool_call_reminder(model),
            tools=[*tools, submit_answer],
        )
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent, app_name="glean_migration", session_service=session_service
        )
        await session_service.create_session(
            app_name="glean_migration", user_id="eval_user", session_id=qid
        )

        last_text = ""
        try:
            async for event in runner.run_async(
                user_id="eval_user",
                session_id=qid,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=question["question"])]
                ),
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            last_text = part.text
        except Exception as exc:  # noqa: BLE001
            print(f"  {qid} run error: {exc}")

        answer = state["answer"] if state["answer"] is not None else last_text
        doc_ids = (
            state["document_ids"] if state["document_ids"] is not None else state["seen"]
        )

        append_result(
            output,
            {"question_id": qid, "answer": answer, "document_ids": doc_ids},
            write_lock,
        )
        print(
            f"  {qid} done (submit_answer={state['answer'] is not None}, "
            f"docs={len(doc_ids)})"
        )


async def main_async(args: argparse.Namespace) -> None:
    index = None
    if not args.live_glean:
        index = CorpusIndex(
            source_types=args.source_types,
            corpus_cap=args.corpus_cap,
            cache_dir=args.cache_dir,
        )
        registered = register_local_backends(index)
        print(f"Local backends registered for: {', '.join(registered)}")
    else:
        print("Using LIVE Glean tenant (GLEAN_API_TOKEN / GLEAN_SERVER_URL).")

    tool_names = configure_glean(live=args.live_glean)
    print(f"Glean toolkit tools wired into ADK: {', '.join(tool_names)}")

    questions = load_questions(QUESTIONS_PATH)
    selected = select_questions(questions, args.source_types, args.question_limit)
    print(f"Selected {len(selected)} questions for source_types={args.source_types}")

    if args.resume:
        existing = load_existing_question_ids(args.output)
        selected = [q for q in selected if q["question_id"] not in existing]
        print(f"  Resuming: {len(selected)} remaining")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_lock = threading.Lock()
    semaphore = asyncio.Semaphore(args.parallelism)

    await asyncio.gather(
        *[
            answer_question(q, args.model, args.output, write_lock, semaphore)
            for q in selected
        ]
    )
    print(f"Done. Answers written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-types", nargs="+", default=["confluence", "jira"])
    parser.add_argument("--question-limit", type=int, default=30)
    parser.add_argument("--corpus-cap", type=int, default=800)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="answer_evaluation/_vertex_cache")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--live-glean",
        action="store_true",
        help="Skip local backends and use a real Glean tenant from the environment.",
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
