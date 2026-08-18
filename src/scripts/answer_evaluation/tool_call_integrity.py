"""Detect answers where the agent failed to use tools properly.

Motivation: a tool-calling agent that *fails* to emit a structured tool call
does not error. It free-texts instead, and some models then narrate a tool
call and invent its result. The answer still lands in the JSONL and still gets
scored, so the failure is silently priced as a quality difference when it is
really a model/harness integration defect. That distinction matters a lot when
comparing a Gemini-native harness against non-Gemini models routed through
LiteLLM.

Observed concretely with Llama 4 Maverick on this benchmark (qst_0023): it
emitted `[read_document(doc_id="dsid_...")]` as plain text, leaked raw Llama
chat-template markers, then wrote out document content it had never received,
complete with `[Name], [Name]` placeholders where the real document had actual
reviewer names. It cited no documents and never called the finish tool.

Three signals, all cheap and syntactic:

* **no citations** -- `document_ids` empty. With these harnesses that almost
  always means the finish/submit tool was never called, so the recorded answer
  is whatever plain text the model happened to end on.
* **chat-template leakage** -- raw special tokens in the answer text. Never
  legitimate; a decoding/templating failure surfacing in output.
* **textual pseudo tool call** -- the answer narrates a call (`read_document(`,
  `glean_search(`) instead of making one. Strong sign the model intended a tool
  call and the structured path failed.

An answer flagged by any of these should not be read as a quality datapoint
for the model until the integration issue is ruled out.

Usage:
    python -m src.scripts.answer_evaluation.tool_call_integrity \
        answer_evaluation/_matrix/system_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

# Raw special tokens from common chat templates. These are never valid answer
# content -- if one appears, templating or decoding leaked into the output.
TEMPLATE_MARKERS = [
    "<|python_start|>",
    "<|python_end|>",
    "<|header_start|>",
    "<|header_end|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|begin_of_text|>",
    "[/INST]",
    "<s>[INST]",
]

# A narrated tool call: a known tool name immediately followed by "(".
# Anchored to the tool names these harnesses actually expose, so ordinary
# prose mentioning a function name in passing is unlikely to trip it.
TOOL_NAMES = [
    "search",
    "read_document",
    "finish_answer",
    "submit_answer",
    "glean_search",
    "glean_read_document",
]
PSEUDO_CALL = re.compile(
    r"(?<![\w.])(" + "|".join(sorted(TOOL_NAMES, key=len, reverse=True)) + r")\s*\(",
)


def scan_answer(row: dict) -> list:
    """Return the list of integrity problems for one answer row.

    Two of these signatures were only found by running comparative_eval.py
    against real output, because it validates document_ids strictly where
    metrics_based_eval.py and this scanner's earlier version did not:

    * malformed_document_ids -- document_ids came back as a string
      (sometimes literally `str(list)`, e.g. "['dsid_a', 'dsid_b']") instead
      of a JSON/Python list. A non-empty string is truthy, so the earlier
      no_citations check missed this entirely; comparative_eval.py's document
      lookup then fails with "expected list, got str".
    * missing_dsid_prefix -- a document id lost its required "dsid_" prefix
      (e.g. "1980f45c..." instead of "dsid_1980f45c..."). This is not a
      hallucinated id -- the underlying document is real -- so it is
      repairable via repair_failed_answers.py's --fix-document-ids rather
      than something to drop and re-run.
    """
    problems = []
    text = row.get("answer") or ""
    doc_ids = row.get("document_ids")

    if isinstance(doc_ids, str):
        problems.append("malformed_document_ids")
    elif not doc_ids:
        problems.append("no_citations")
    elif isinstance(doc_ids, list) and any(
        isinstance(d, str) and d and not d.startswith("dsid_") for d in doc_ids
    ):
        problems.append("missing_dsid_prefix")

    found_markers = [m for m in TEMPLATE_MARKERS if m in text]
    if found_markers:
        problems.append("template_leak")

    if PSEUDO_CALL.search(text):
        problems.append("pseudo_tool_call")

    return problems


def scan_file(path: str) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    flagged = {}
    counts = {"no_citations": 0, "template_leak": 0, "pseudo_tool_call": 0, "malformed_document_ids": 0, "missing_dsid_prefix": 0}
    for row in rows:
        problems = scan_answer(row)
        if problems:
            flagged[row["question_id"]] = problems
            for p in problems:
                counts[p] += 1
    return {
        "path": path,
        "total": len(rows),
        "flagged": flagged,
        "counts": counts,
        "clean": len(rows) - len(flagged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="answer JSONL files (globs ok)")
    parser.add_argument("--json", help="write the full report here")
    parser.add_argument(
        "--fail-on-flag",
        action="store_true",
        help="exit 1 if any answer is flagged (for use as a gate in a run script)",
    )
    args = parser.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no matching answer files")

    print("=" * 84)
    print("TOOL-CALL INTEGRITY")
    print("=" * 84)
    print(f"{'file':<44}{'n':>4}{'clean':>7}{'nocite':>8}{'tmpl':>6}{'pseudo':>8}{'strlist':>9}{'noprefix':>10}")
    print("-" * 84)

    reports = []
    for path in paths:
        r = scan_file(path)
        reports.append(r)
        c = r["counts"]
        print(
            f"{os.path.basename(path):<44}{r['total']:>4}{r['clean']:>7}"
            f"{c['no_citations']:>8}{c['template_leak']:>6}{c['pseudo_tool_call']:>8}"
            f"{c['malformed_document_ids']:>9}{c['missing_dsid_prefix']:>10}"
        )

    detail = [r for r in reports if r["flagged"]]
    if detail:
        print()
        print("Flagged questions:")
        for r in detail:
            print(f"\n  {os.path.basename(r['path'])}")
            for qid, problems in sorted(r["flagged"].items()):
                print(f"    {qid}: {', '.join(problems)}")
        print()
        print("An answer flagged here reflects a tool-calling integration failure,")
        print("not necessarily model answer quality. Exclude these before reading")
        print("the run as a quality comparison, or report them alongside it.")
    else:
        print("\nNo integrity problems found.")
    print()

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(reports, handle, indent=2)
        print(f"wrote {args.json}")

    if args.fail_on_flag and detail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
