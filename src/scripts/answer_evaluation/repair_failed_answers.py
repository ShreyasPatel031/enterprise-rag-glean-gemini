"""Find and drop answer rows that failed for infrastructure reasons.

The agentic harnesses catch per-question exceptions and still append a row, so
a question killed by a 429, a timeout, or a transport error lands in the JSONL
as a row with an empty answer rather than as a visible failure. Scored as-is it
counts against the model, which is wrong -- nothing about the model produced it.

The partner MaaS endpoints make this common: Mistral on europe-west4 and Llama
on us-east5 have far tighter quota than Gemini, so --parallelism 4 draws 429s
that Gemini never sees.

The discriminator, and an important limit on it:

    blank answer        -> either an infrastructure failure (the exception
                           handler ran before any text was produced) OR the
                           model genuinely returned nothing. Both look
                           identical in the JSONL, because the harness records
                           no error flag. Retry distinguishes them: a transient
                           failure succeeds on a second pass, a model that
                           returns empty output does so again.
    text but no docs    -> model behaviour. The model answered without calling
                           the finish tool (see tool_call_integrity.py). Keep;
                           it is a real datapoint about the model.

Do NOT read a blank row as automatically infrastructural. Observed here:
qst_0395 came back blank for gemini-2.5-flash-lite on the adk harness twice --
once under rate limiting and again on a clean serial pass with no exception
logged at all (`finish_answer called: False`, no error line). That is the model
returning nothing, which for a thinking-enabled Gemini model can happen when
the thinking budget consumes the entire output allowance. Retrying it forever
will not help; it is a degenerate but real model outcome.

So: drop blank rows and re-run ONCE. Anything still blank afterwards is a model
result, not a quota problem, and should be reported rather than retried. Pass
--run-log to attribute causes directly when the harness log is available.

Re-running is:

    python -m <harness> ... --resume --parallelism 1

since --resume skips question_ids already present, and the dropped ones no
longer are.

Usage:
    python -m src.scripts.answer_evaluation.repair_failed_answers \
        answer_evaluation/_matrix/system_*.jsonl            # report only
    python -m src.scripts.answer_evaluation.repair_failed_answers \
        answer_evaluation/_matrix/system_*.jsonl --apply    # rewrite files
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def classify(row: dict) -> str:
    """blank_answer | model_no_citation | ok

    blank_answer is deliberately NOT named "infrastructure_failure": the row
    shape cannot distinguish a transient error from a model that returned
    nothing. See the module docstring.
    """
    text = (row.get("answer") or "").strip()
    if not text:
        return "blank_answer"
    if not row.get("document_ids"):
        return "model_no_citation"
    return "ok"


def attribute_from_log(log_path: str) -> dict:
    """Map question_id -> 'exception' | 'no_exception' using a harness log.

    The harnesses print "<qid> ADK run error:" / "<qid> run error:" when an
    exception was caught, and a "<qid> done (...)" line otherwise. A blank row
    whose qid has a done-line but no error-line was the model returning
    nothing, not a transport failure.
    """
    errored, completed = set(), set()
    for line in open(log_path, errors="replace"):
        m = re.search(r"(qst_[0-9a-zA-Z_]+)\s+(?:ADK )?run error", line)
        if m:
            errored.add(m.group(1))
            continue
        m = re.search(r"(qst_[0-9a-zA-Z_]+)\s+done", line)
        if m:
            completed.add(m.group(1))
    return {
        **{q: "exception" for q in errored},
        **{q: "no_exception" for q in completed - errored},
    }


def scan(path: str) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    buckets = {"blank_answer": [], "model_no_citation": [], "ok": []}
    for row in rows:
        buckets[classify(row)].append(row)
    return {"path": path, "rows": rows, "buckets": buckets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite each file without its blank-answer rows",
    )
    parser.add_argument(
        "--run-log",
        help="harness log, used to attribute each blank row to an exception or not",
    )
    args = parser.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no matching files")

    print("=" * 82)
    print("ANSWER ROW TRIAGE")
    print("=" * 82)
    print(f"{'file':<46}{'rows':>5}{'ok':>5}{'blank':>7}{'nocite':>8}")
    print("-" * 82)

    total_infra = 0
    reports = []
    for path in paths:
        r = scan(path)
        b = r["buckets"]
        total_infra += len(b["blank_answer"])
        reports.append(r)
        print(
            f"{os.path.basename(path):<46}{len(r['rows']):>5}{len(b['ok']):>5}"
            f"{len(b['blank_answer']):>7}{len(b['model_no_citation']):>8}"
        )

    attribution = attribute_from_log(args.run_log) if args.run_log else {}

    for r in reports:
        blanks = r["buckets"]["blank_answer"]
        if not blanks:
            continue
        print(f"\n  {os.path.basename(r['path'])}")
        for row in blanks:
            qid = row["question_id"]
            cause = attribution.get(qid)
            if cause == "exception":
                note = "exception logged -> transient, worth one re-run"
            elif cause == "no_exception":
                note = "NO exception logged -> model returned nothing; re-running will not help"
            else:
                note = "cause unknown (pass --run-log to attribute)"
            print(f"    {qid}: blank answer, {note}")

    if not total_infra:
        print("\nNo blank rows. Nothing to repair.")
        return

    print()
    if args.apply:
        for r in reports:
            infra = r["buckets"]["blank_answer"]
            if not infra:
                continue
            keep = [row for row in r["rows"] if classify(row) != "blank_answer"]
            with open(r["path"], "w") as handle:
                for row in keep:
                    handle.write(json.dumps(row) + "\n")
            print(f"rewrote {r['path']}: dropped {len(infra)}, kept {len(keep)}")
        print()
        print("Now re-run each affected harness ONCE with --resume --parallelism 1.")
        print("Anything still blank after that is a model result, not a quota")
        print("problem -- report it rather than retrying.")
    else:
        print(f"{total_infra} row(s) would be dropped. Re-run with --apply to rewrite.")


if __name__ == "__main__":
    main()
