"""Retrieve documents via in-memory Vertex AI embeddings and generate answers.

A Docker/Qdrant/OpenSearch-free alternative to vector_retrieval.py, scoped to a
small subset of source types so a full comparison run stays cheap and fast.
Embeds every candidate document once per (source_types, cap) combination and
caches the vectors to disk; each system-under-test then reuses that cache.

Usage:
    python -m src.scripts.answer_generation.vertex_retrieval \
        --source-types confluence jira \
        --question-limit 30 \
        --corpus-cap 800 \
        --model gemini-2.5-flash \
        --output answer_evaluation/system_gemini_flash.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from google import genai
from tqdm import tqdm

from src.llm.interface import Message
from src.llm.vertex_llm import VertexLLM
from src.paths import QUESTIONS_PATH, SOURCES_DIR
from src.prompts.vector_search_answer_gen import ANSWER_GEN_PROMPT
from src.utils.document_content import extract_document_content
from src.utils.file_io import load_json_file
from src.utils.retrieval import append_result, load_existing_question_ids, load_questions

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "project-amer-scs-sandbox")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_MODEL = "gemini-embedding-001"
RANDOM_SEED = 42


def select_questions(
    questions: list[dict], source_types: list[str], limit: int
) -> list[dict]:
    wanted = set(source_types)
    matching = [
        q for q in questions if set(q.get("source_types") or []) and set(q.get("source_types") or []) <= wanted
    ]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(matching)
    return matching[:limit]


def build_scoped_corpus(
    source_types: list[str], gold_doc_ids: set[str], corpus_cap: int
) -> dict[str, str]:
    """Return {dataset_doc_uuid: absolute_path} for a capped subset of the given source types.

    All documents referenced by gold_doc_ids are guaranteed to be included;
    the remainder is filled with a random sample of distractors.
    """
    all_paths: list[str] = []
    for st in source_types:
        for root, _dirs, files in os.walk(os.path.join(SOURCES_DIR, st)):
            for fn in files:
                if fn.endswith(".json"):
                    all_paths.append(os.path.join(root, fn))

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_paths)

    index: dict[str, str] = {}
    remaining_gold = set(gold_doc_ids)
    extra: list[str] = []

    for path in all_paths:
        if not remaining_gold and len(index) >= corpus_cap:
            break
        try:
            doc = load_json_file(path)
        except Exception:
            continue
        uid = doc.get("dataset_doc_uuid")
        if not uid:
            continue
        if uid in remaining_gold:
            index[uid] = path
            remaining_gold.discard(uid)
        elif len(index) + len(extra) < corpus_cap:
            extra.append(path)

    for path in extra:
        doc = load_json_file(path)
        uid = doc.get("dataset_doc_uuid")
        if uid and uid not in index:
            index[uid] = path

    return index


def embed_texts(client: genai.Client, texts: list[str], batch_size: int = 16) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
        batch = texts[i : i + batch_size]
        resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        vectors.extend(e.values for e in resp.embeddings)
    return np.array(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-types", nargs="+", required=True)
    parser.add_argument("--question-limit", type=int, default=30)
    parser.add_argument("--corpus-cap", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", required=True, help="Gemini model name for answer generation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="answer_evaluation/_vertex_cache")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    questions = load_questions(QUESTIONS_PATH)
    selected = select_questions(questions, args.source_types, args.question_limit)
    print(f"Selected {len(selected)} questions for source_types={args.source_types}")

    gold_doc_ids: set[str] = set()
    for q in selected:
        gold_doc_ids.update(q.get("expected_doc_ids") or [])

    cache_key = "_".join(sorted(args.source_types)) + f"_cap{args.corpus_cap}"
    index_cache_path = os.path.join(args.cache_dir, f"index_{cache_key}.json")
    vectors_cache_path = os.path.join(args.cache_dir, f"vectors_{cache_key}.npy")
    uuids_cache_path = os.path.join(args.cache_dir, f"uuids_{cache_key}.json")

    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)

    if os.path.exists(index_cache_path) and os.path.exists(vectors_cache_path):
        print("Loading cached corpus index + embeddings...")
        path_index = load_json_file(index_cache_path)
        uuids = load_json_file(uuids_cache_path)
        vectors = np.load(vectors_cache_path)
    else:
        print("Building scoped corpus...")
        path_index = build_scoped_corpus(args.source_types, gold_doc_ids, args.corpus_cap)
        print(f"  Corpus size: {len(path_index)} documents")

        uuids = list(path_index.keys())
        texts = []
        for uid in uuids:
            doc = load_json_file(path_index[uid])
            title, content = extract_document_content(doc)
            texts.append(f"{title}\n\n{content}"[:8000])

        vectors = embed_texts(client, texts)

        with open(index_cache_path, "w") as f:
            json.dump(path_index, f)
        with open(uuids_cache_path, "w") as f:
            json.dump(uuids, f)
        np.save(vectors_cache_path, vectors)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_vectors = vectors / norms

    if args.resume:
        existing_ids = load_existing_question_ids(args.output)
        selected = [q for q in selected if q["question_id"] not in existing_ids]
        print(f"  Resuming: {len(selected)} questions remaining")

    write_lock = threading.Lock()

    def process_question(question: dict) -> str:
        qid = question["question_id"]
        query = question["question"]

        q_resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=[query])
        q_vec = np.array(q_resp.embeddings[0].values, dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1.0)

        sims = normed_vectors @ q_vec
        top_idx = np.argsort(-sims)[: args.top_k]
        doc_uuids = [uuids[i] for i in top_idx]

        context_parts = []
        for i, uid in enumerate(doc_uuids, 1):
            doc = load_json_file(path_index[uid])
            title, content = extract_document_content(doc)
            context_parts.append(f"--- Document {i} (ID: {uid}) ---\nTitle: {title}\n\n{content}")
        context = "\n\n".join(context_parts)

        prompt = ANSWER_GEN_PROMPT.format(context_documents=context, question=query)
        llm = VertexLLM(model=args.model, quiet=True)
        answer = "".join(
            c for c in llm.generate([Message(role="user", content=prompt)]) if isinstance(c, str)
        ).strip()

        append_result(
            args.output,
            {"question_id": qid, "answer": answer, "document_ids": doc_uuids},
            write_lock,
        )
        return qid

    print(f"Generating answers with model={args.model} ({args.parallelism} workers)...")
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = {executor.submit(process_question, q): q["question_id"] for q in selected}
        with tqdm(total=len(selected), desc="Questions") as pbar:
            for future in as_completed(futures):
                qid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  Question {qid} failed: {e}")
                pbar.update(1)

    print(f"Done. Answers written to {args.output}")


if __name__ == "__main__":
    main()
