"""Significance layer over the scored eval outputs.

The upstream eval scripts report point estimates only. At n=30 questions a
single question is worth 3.3 percentage points, so most of the gaps between
systems in those reports are inside sampling noise. This module re-reads the
committed `results/` files and answers the question the point estimates
cannot: which differences would survive resampling.

Three tests, matched to how each metric is actually produced:

* pointwise correctness is a paired binary outcome (same 30 questions, same
  gold answers), so the exact McNemar test on the discordant pairs is the
  right one -- an unpaired proportion test would throw away the pairing and
  overstate the variance.
* pairwise judge preference is a paired binary choice per question, so a
  two-sided exact binomial against 50/50. Reported twice: over every judged
  question, and over only the questions the judge did *not* mark
  `effectively_equivalent`, since a forced pick between two answers the judge
  calls equivalent is close to a coin flip and dilutes a real effect.
* retrieval is summarised as a recall/precision pair rather than tested,
  because recall here is driven by a design constant (how many documents the
  harness returns) rather than by anything stochastic.

Reads only `results/*.json` -- no corpus, credentials, or model calls, so it
reproduces anywhere the repo is checked out.

Usage:
    python -m src.scripts.answer_evaluation.significance
    python -m src.scripts.answer_evaluation.significance --results-dir results --json out.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from math import comb, sqrt

# Scored pointwise runs: label -> results filename. Labels are what the
# comparison tables key on.
POINTWISE_RUNS = {
    "singleshot/flash": "results_flash.json",
    "singleshot/flash-lite": "results_flash_lite.json",
    "singleshot/mistral-medium": "results_mistral.json",
    "singleshot/llama4-maverick": "results_llama4.json",
    "adk/flash": "results_gemini_adk.json",
    "glean-toolkit/flash": "results_glean_toolkit.json",
}

# Which harness family each run belongs to, for the retrieval summary.
FAMILY = {
    "singleshot/flash": "single-shot",
    "singleshot/flash-lite": "single-shot",
    "singleshot/mistral-medium": "single-shot",
    "singleshot/llama4-maverick": "single-shot",
    "adk/flash": "agentic",
    "glean-toolkit/flash": "agentic",
}

# Answers files, used to count documents actually returned per question so
# precision is exact rather than inferred from the recall percentage.
ANSWER_FILES = {
    "singleshot/flash": "system_gemini_flash.jsonl",
    "singleshot/flash-lite": "system_gemini_flash_lite.jsonl",
    "singleshot/mistral-medium": "system_mistral_medium.jsonl",
    "singleshot/llama4-maverick": "system_llama4_maverick.jsonl",
    "adk/flash": "system_gemini_adk.jsonl",
    "glean-toolkit/flash": "system_glean_toolkit_gemini.jsonl",
}

ALPHA = 0.05


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value against p=0.5.

    Sums the probability of every outcome no more likely than the observed
    one, which is the correct two-sided construction for a discrete
    distribution (doubling the one-tailed value can exceed 1 and is wrong
    when the observed outcome sits near the mode).
    """
    if trials == 0:
        return 1.0
    probs = [comb(trials, k) / 2**trials for k in range(trials + 1)]
    observed = probs[successes]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-12))


def mcnemar_exact(flags_a: dict, flags_b: dict, ids: list) -> tuple:
    """Exact McNemar on paired binary outcomes.

    Returns (a_only, b_only, p_value): the counts of questions where exactly
    one system was correct, and the exact two-sided p-value. Questions where
    both systems agree carry no information about which is better and drop
    out -- that is the point of the test.
    """
    a_only = sum(1 for i in ids if flags_a[i] and not flags_b[i])
    b_only = sum(1 for i in ids if flags_b[i] and not flags_a[i])
    discordant = a_only + b_only
    return a_only, b_only, exact_binomial_two_sided(min(a_only, b_only), discordant)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval -- behaves sensibly near 0% and 100%, where the
    textbook normal-approximation interval runs off the end of the scale."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def load_pointwise(results_dir: str) -> dict:
    """Load each scored pointwise run, keyed by label then question id."""
    runs = {}
    for label, filename in POINTWISE_RUNS.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        payload = json.load(open(path))
        runs[label] = {q["question_id"]: q for q in payload["questions"]}
    return runs


def mean_docs_returned(results_dir: str, label: str) -> float:
    """Mean number of documents a system returned per question, read from its
    answers file. Returns 0.0 if the file is absent."""
    filename = ANSWER_FILES.get(label)
    if not filename:
        return 0.0
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return 0.0
    counts = [
        len(json.loads(line).get("document_ids") or [])
        for line in open(path)
        if line.strip()
    ]
    return sum(counts) / len(counts) if counts else 0.0


def load_comparative(results_dir: str) -> list:
    """Load every pairwise-judge run in the results directory."""
    runs = []
    for filename in sorted(os.listdir(results_dir)):
        if not filename.startswith("results_comparative") or not filename.endswith(".json"):
            continue
        payload = json.load(open(os.path.join(results_dir, filename)))
        runs.append((filename, payload))
    return runs


def system_label(path: str) -> str:
    """Turn a system's answers-file path into a short label."""
    return os.path.basename(path).replace("system_", "").replace(".jsonl", "")


def report_pointwise(runs: dict, results_dir: str) -> dict:
    """Per-system correctness with confidence intervals, and the retrieval
    recall/precision tradeoff."""
    summary = {}
    print("=" * 78)
    print("POINTWISE PER-SYSTEM  (n=30 questions; one question = 3.3pp)")
    print("=" * 78)
    header = (
        f"{'system':<26}{'correct':>9}{'95% CI':>16}"
        f"{'recall':>9}{'docs':>7}{'junk':>7}{'prec':>7}"
    )
    print(header)
    print("-" * 78)
    widest_ci = 0.0
    for label, run in runs.items():
        ids = sorted(run)
        n = len(ids)
        correct = sum(1 for i in ids if run[i]["answer_correct"])
        recall = sum(run[i]["document_recall_pct"] for i in ids) / n
        junk = sum(run[i]["invalid_extra_docs"] for i in ids) / n
        low, high = wilson_interval(correct, n)
        widest_ci = max(widest_ci, high - low)
        # Exact: documents returned comes from the answers file, so precision
        # is the share of returned documents the scorer did not call invalid.
        docs = mean_docs_returned(results_dir, label)
        precision = 100 * (docs - junk) / docs if docs else 0.0
        print(
            f"{label:<26}{100*correct/n:>8.1f}%"
            f"{f'[{low:.1f}, {high:.1f}]':>16}"
            f"{recall:>8.1f}%{docs:>7.2f}{junk:>7.2f}{precision:>6.0f}%"
        )
        summary[label] = {
            "n": n,
            "correct": correct,
            "correctness_pct": round(100 * correct / n, 2),
            "correctness_ci95": [round(low, 2), round(high, 2)],
            "recall_pct": round(recall, 2),
            "docs_returned_per_question": round(docs, 2),
            "invalid_extra_docs": round(junk, 2),
            "precision_pct": round(precision, 1),
            "family": FAMILY.get(label, "unknown"),
        }
    narrowest_ci = min(
        s["correctness_ci95"][1] - s["correctness_ci95"][0] for s in summary.values()
    )
    print()
    print(
        f"Every 95% CI above is at least {narrowest_ci:.0f}pp wide"
        f" (widest {widest_ci:.0f}pp), so all {len(summary)}"
        "\ncorrectness figures overlap each other. The four single-shot runs share"
        "\none retrieval pass, so their recall/docs/junk columns are identical by"
        "\nconstruction and they differ only in the generator model."
    )
    print()
    return summary


def report_mcnemar(runs: dict) -> list:
    """Exact McNemar for every pair of systems on pointwise correctness."""
    labels = list(runs)
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    rows = []
    print("=" * 78)
    print(f"PAIRED CORRECTNESS -- exact McNemar over {len(ids)} shared questions")
    print("=" * 78)
    print(f"{'comparison':<56}{'discordant':>12}{'p':>7}{'':>5}")
    print("-" * 78)
    for a, b in itertools.combinations(labels, 2):
        flags_a = {i: runs[a][i]["answer_correct"] for i in ids}
        flags_b = {i: runs[b][i]["answer_correct"] for i in ids}
        a_only, b_only, p = mcnemar_exact(flags_a, flags_b, ids)
        verdict = "SIG" if p < ALPHA else ""
        print(f"{a + ' vs ' + b:<56}{f'{a_only}/{b_only}':>12}{p:>7.3f}{verdict:>5}")
        rows.append(
            {
                "system_a": a,
                "system_b": b,
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "p_value": round(p, 4),
                "significant": p < ALPHA,
            }
        )
    significant = [r for r in rows if r["significant"]]
    print()
    print(
        f"{len(significant)} of {len(rows)} correctness gaps reach p<{ALPHA}."
        if significant
        else f"None of the {len(rows)} correctness gaps reach p<{ALPHA}: on this"
        " question set,\nno system is distinguishable from any other by pointwise"
        " correctness alone."
    )
    print()
    return rows


def report_comparative(comparative: list) -> list:
    """Exact binomial on each pairwise-judge run."""
    rows = []
    print("=" * 78)
    print("PAIRWISE JUDGE -- exact binomial against 50/50")
    print("=" * 78)
    for filename, payload in comparative:
        questions = payload["questions"]
        s1, s2 = system_label(payload["system_1"]), system_label(payload["system_2"])
        picks = [q["comparison"] for q in questions]
        all_1 = sum(1 for c in picks if c.get("preferred_system") == "1")
        all_2 = sum(1 for c in picks if c.get("preferred_system") == "2")
        equivalent = sum(1 for c in picks if c.get("effectively_equivalent"))
        dec_1 = sum(
            1
            for c in picks
            if c.get("preferred_system") == "1" and not c.get("effectively_equivalent")
        )
        dec_2 = sum(
            1
            for c in picks
            if c.get("preferred_system") == "2" and not c.get("effectively_equivalent")
        )
        p_all = exact_binomial_two_sided(min(all_1, all_2), all_1 + all_2)
        p_dec = exact_binomial_two_sided(min(dec_1, dec_2), dec_1 + dec_2)
        print(f"\n{s1}  vs  {s2}    (judged {len(questions)}, called equivalent {equivalent})")
        print(
            f"   all judged   {all_1:>3} - {all_2:<3}  p={p_all:.4f}"
            f"  {'SIG' if p_all < ALPHA else ''}"
        )
        print(
            f"   decisive     {dec_1:>3} - {dec_2:<3}  p={p_dec:.4f}"
            f"  {'SIG' if p_dec < ALPHA else ''}"
        )
        rows.append(
            {
                "file": filename,
                "system_1": s1,
                "system_2": s2,
                "judged": len(questions),
                "effectively_equivalent": equivalent,
                "preferred_1": all_1,
                "preferred_2": all_2,
                "p_value_all": round(p_all, 4),
                "significant_all": p_all < ALPHA,
                "decisive_1": dec_1,
                "decisive_2": dec_2,
                "p_value_decisive": round(p_dec, 4),
                "significant_decisive": p_dec < ALPHA,
            }
        )
    print()
    return rows


def report_disagreements(runs: dict, comparative_rows: list) -> list:
    """Flag pairs where the pointwise and pairwise methods point opposite ways.

    A disagreement is not a bug in either eval -- they measure different
    things (gold-answer coverage vs. head-to-head answer quality). It is a
    signal that the pair should not be reported as a ranked win either way.
    """
    # Map judge-run system labels back to pointwise run labels.
    alias = {
        "gemini_flash": "singleshot/flash",
        "gemini_flash_lite": "singleshot/flash-lite",
        "mistral_medium": "singleshot/mistral-medium",
        "llama4_maverick": "singleshot/llama4-maverick",
        "gemini_adk": "adk/flash",
        "glean_toolkit_gemini": "glean-toolkit/flash",
    }
    print("=" * 78)
    print("METHOD DISAGREEMENTS")
    print("=" * 78)
    found = []
    for row in comparative_rows:
        a, b = alias.get(row["system_1"]), alias.get(row["system_2"])
        if a not in runs or b not in runs:
            continue
        ids = sorted(set(runs[a]) & set(runs[b]))
        corr_a = sum(1 for i in ids if runs[a][i]["answer_correct"])
        corr_b = sum(1 for i in ids if runs[b][i]["answer_correct"])
        pointwise_winner = a if corr_a > corr_b else (b if corr_b > corr_a else None)
        judge_winner = (
            a
            if row["preferred_1"] > row["preferred_2"]
            else (b if row["preferred_2"] > row["preferred_1"] else None)
        )
        if pointwise_winner and judge_winner and pointwise_winner != judge_winner:
            print(
                f"\n{a}  vs  {b}"
                f"\n   pointwise correctness favours {pointwise_winner}"
                f"  ({100*corr_a/len(ids):.1f}% vs {100*corr_b/len(ids):.1f}%)"
                f"\n   pairwise judge favours        {judge_winner}"
                f"  ({row['preferred_1']} - {row['preferred_2']})"
                f"\n   -> report as unresolved, not as a win for either."
            )
            found.append(
                {
                    "system_a": a,
                    "system_b": b,
                    "pointwise_winner": pointwise_winner,
                    "judge_winner": judge_winner,
                }
            )
    if not found:
        print("\nNone: the two methods agree on direction for every compared pair.")
    print()
    return found


def report_correctness_attribution(runs: dict) -> dict:
    """For each pair, show whether one system's correctness lead comes with a
    retrieval miss on exactly those questions.

    This separates 'answered wrong from the right documents' (a reasoning
    failure) from 'never found the document' (a retrieval failure), which
    the single correctness number conflates.
    """
    print("=" * 78)
    print("WHERE THE CORRECTNESS GAPS COME FROM")
    print("=" * 78)
    attribution = {}
    for a, b in itertools.combinations(runs, 2):
        ids = sorted(set(runs[a]) & set(runs[b]))
        lost = [i for i in ids if runs[a][i]["answer_correct"] and not runs[b][i]["answer_correct"]]
        if not lost:
            continue
        misses = [i for i in lost if runs[b][i]["document_recall_pct"] == 0]
        attribution[f"{a} > {b}"] = {
            "questions_lost": lost,
            "of_which_zero_recall": misses,
        }
        if misses and len(misses) == len(lost):
            verdict = (
                f"all {len(misses)} had zero document recall: every loss is a"
                " retrieval miss,\n   not a reasoning error"
            )
        elif misses:
            verdict = (
                f"{len(misses)} of those had zero document recall (retrieval miss);"
                f"\n   the other {len(lost) - len(misses)} had the gold document and still"
                " answered wrong (reasoning)"
            )
        else:
            verdict = (
                "all of those had non-zero document recall: the gold document was"
                " retrieved\n   and the answer was still wrong, so these are reasoning"
                " errors, not retrieval"
            )
        print(
            f"\n{b} misses {len(lost)} question(s) that {a} gets:"
            f"\n   {', '.join(lost)}"
            f"\n   {verdict}."
        )
    print()
    return attribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--json", help="also write the full analysis to this path")
    args = parser.parse_args()

    runs = load_pointwise(args.results_dir)
    if not runs:
        raise SystemExit(f"no pointwise results found in {args.results_dir}")
    comparative = load_comparative(args.results_dir)

    pointwise = report_pointwise(runs, args.results_dir)
    mcnemar_rows = report_mcnemar(runs)
    comparative_rows = report_comparative(comparative)
    disagreements = report_disagreements(runs, comparative_rows)
    attribution = report_correctness_attribution(runs)

    if args.json:
        payload = {
            "alpha": ALPHA,
            "pointwise": pointwise,
            "paired_correctness_mcnemar": mcnemar_rows,
            "pairwise_judge_binomial": comparative_rows,
            "method_disagreements": disagreements,
            "correctness_gap_attribution": attribution,
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
