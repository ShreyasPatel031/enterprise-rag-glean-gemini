"""Find and drop answer rows that failed for infrastructure reasons.

The agentic harnesses catch per-question exceptions and still append a row, so
a question killed by a 429, a timeout, or a transport error lands in the JSONL
as a row with an empty answer rather than as a visible failure. Scored as-is it
counts against the model, which is wrong -- nothing about the model produced it.

The partner MaaS endpoints make this common: Mistral on europe-west4 and Llama
on us-east5 have far tighter quota than Gemini, so --parallelism 4 draws 429s
that Gemini never sees.

The useful discriminator:

    blank answer        -> infrastructure failure. The exception handler ran
                           before any text was produced. Drop and re-run.
    text but no docs    -> model behaviour. The model answered without calling
                           the finish tool (see tool_call_integrity.py). Keep;
                           it is a real datapoint about the model.

So this only ever removes rows with no answer text, which cannot be a
legitimate model output in these harnesses. Re-running is then just:

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


def classify(row: dict) -> str:
    """infrastructure_failure | model_no_citation | ok"""
    text = (row.get("answer") or "").strip()
    if not text:
        return "infrastructure_failure"
    if not row.get("document_ids"):
        return "model_no_citation"
    return "ok"


def scan(path: str) -> dict:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    buckets = {"infrastructure_failure": [], "model_no_citation": [], "ok": []}
    for row in rows:
        buckets[classify(row)].append(row)
    return {"path": path, "rows": rows, "buckets": buckets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite each file without its infrastructure-failure rows",
    )
    args = parser.parse_args()

    paths = sorted({p for pattern in args.files for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no matching files")

    print("=" * 82)
    print("ANSWER ROW TRIAGE")
    print("=" * 82)
    print(f"{'file':<46}{'rows':>5}{'ok':>5}{'infra':>7}{'nocite':>8}")
    print("-" * 82)

    total_infra = 0
    reports = []
    for path in paths:
        r = scan(path)
        b = r["buckets"]
        total_infra += len(b["infrastructure_failure"])
        reports.append(r)
        print(
            f"{os.path.basename(path):<46}{len(r['rows']):>5}{len(b['ok']):>5}"
            f"{len(b['infrastructure_failure']):>7}{len(b['model_no_citation']):>8}"
        )

    for r in reports:
        infra = r["buckets"]["infrastructure_failure"]
        if infra:
            qids = ", ".join(q["question_id"] for q in infra)
            print(f"\n  {os.path.basename(r['path'])}")
            print(f"    infrastructure failures (blank answer): {qids}")

    if not total_infra:
        print("\nNo infrastructure failures. Nothing to repair.")
        return

    print()
    if args.apply:
        for r in reports:
            infra = r["buckets"]["infrastructure_failure"]
            if not infra:
                continue
            keep = [row for row in r["rows"] if classify(row) != "infrastructure_failure"]
            with open(r["path"], "w") as handle:
                for row in keep:
                    handle.write(json.dumps(row) + "\n")
            print(f"rewrote {r['path']}: dropped {len(infra)}, kept {len(keep)}")
        print()
        print("Now re-run each affected harness with --resume --parallelism 1 to")
        print("refill the dropped questions.")
    else:
        print(f"{total_infra} row(s) would be dropped. Re-run with --apply to rewrite.")


if __name__ == "__main__":
    main()
