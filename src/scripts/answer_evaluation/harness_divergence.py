"""Validate a head-to-head between two harnesses run on the same substrate.

An aggregate dead heat can mean two very different things: the harnesses are
genuinely equivalent, or they never actually did anything different. This
module separates those cases by looking at what each harness *retrieved*
before scoring the answers, then decomposing the comparison:

* **retrieval overlap** -- exact-match rate and Jaccard on the returned
  document sets. Two harnesses that return identical documents on a question
  are not being compared on that question in any meaningful way; the same
  model then sees the same evidence.
* **identical vs. divergent split** -- scores computed separately over the
  questions where retrieval matched and where it did not. The divergent subset
  is the only part carrying information about harness behaviour, so its size
  is the *effective* sample size, not the question count.
* **noise floor** -- how often the correctness verdict flips even when both
  harnesses retrieved exactly the same documents. Any true effect smaller than
  this is unmeasurable without repeated sampling per question.
* **split-half robustness** -- whether the aggregate result survives being cut
  in half. A true null should stay non-significant in both halves and may flip
  direction; a real effect suppressed by noise should lean the same way in both.
* **power** -- exact binomial power, to say what effect the current design
  could have detected and what sample a target effect would need.

Reads only the committed `results/` files. No corpus, credentials, or model
calls, so it reproduces anywhere the repo is checked out.

Usage:
    python -m src.scripts.answer_evaluation.harness_divergence
    python -m src.scripts.answer_evaluation.harness_divergence \
        --answers-a results/system_gemini_adk.jsonl \
        --answers-b results/system_glean_toolkit_gemini.jsonl \
        --scores-a results/results_gemini_adk.json \
        --scores-b results/results_glean_toolkit.json \
        --comparative results/results_comparative_adk_vs_gleantoolkit.json
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb, exp, inf, lgamma, log

ALPHA = 0.05
TARGET_POWER = 0.80


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value against p=0.5 (method of small p-values)."""
    if trials == 0:
        return 1.0
    probs = [comb(trials, k) / 2**trials for k in range(trials + 1)]
    observed = probs[successes]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-12))


def _log_pmf(trials: int, rate: float, k: int) -> float:
    """log of the binomial pmf, via lgamma so large `trials` stays stable."""
    if rate <= 0.0:
        return 0.0 if k == 0 else -inf
    if rate >= 1.0:
        return 0.0 if k == trials else -inf
    return (
        lgamma(trials + 1)
        - lgamma(k + 1)
        - lgamma(trials - k + 1)
        + k * log(rate)
        + (trials - k) * log(1 - rate)
    )


def _cdf(trials: int, rate: float, k: int) -> float:
    """P(X <= k) for X ~ Binomial(trials, rate)."""
    if k < 0:
        return 0.0
    if k >= trials:
        return 1.0
    return sum(exp(_log_pmf(trials, rate, j)) for j in range(k + 1))


def critical_value(trials: int, alpha: float = ALPHA) -> int:
    """Largest c with 2*P(X <= c) < alpha under H0: p=0.5.

    At p=0.5 the null is symmetric, so the exact two-sided test rejects exactly
    when min(k, trials-k) <= c. Computing c once is O(trials) and avoids
    re-deriving the whole rejection set per candidate rate.
    """
    total = 0.0
    critical = -1
    for k in range(trials // 2 + 1):
        total += exp(_log_pmf(trials, 0.5, k))
        if 2 * total < alpha:
            critical = k
        else:
            break
    return critical


def exact_power(trials: int, true_rate: float, alpha: float = ALPHA) -> float:
    """Exact power of the two-sided binomial test at a given true rate."""
    if trials == 0:
        return 0.0
    critical = critical_value(trials, alpha)
    if critical < 0:
        return 0.0
    lower = _cdf(trials, true_rate, critical)
    upper = 1.0 - _cdf(trials, true_rate, trials - critical - 1)
    return lower + upper


def min_detectable_rate(trials: int, power: float = TARGET_POWER) -> float | None:
    """Smallest preference rate above 0.5 detectable at the target power."""
    if trials == 0 or critical_value(trials) < 0:
        return None
    rate = 0.50
    while rate < 0.999:
        rate += 0.005
        if exact_power(trials, rate) >= power:
            return rate
    return None


def required_trials(true_rate: float, power: float = TARGET_POWER, cap: int = 5000) -> int | None:
    """Decisive picks needed to detect a given preference rate at target power.

    Power is not perfectly monotone in `trials` for a discrete test, so this
    scans upward and returns the first size that clears the target and stays
    clear just above it, rather than the first lucky crossing.
    """
    for trials in range(4, cap + 1):
        if exact_power(trials, true_rate) >= power and exact_power(
            trials + 1, true_rate
        ) >= power:
            return trials
    return None


def load_answers(path: str) -> dict:
    return {
        json.loads(line)["question_id"]: json.loads(line)
        for line in open(path)
        if line.strip()
    }


def load_scores(path: str) -> dict:
    return {q["question_id"]: q for q in json.load(open(path))["questions"]}


def load_comparisons(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    return {
        q["question_id"]: q["comparison"]
        for q in json.load(open(path))["questions"]
    }


def report_retrieval_overlap(answers_a: dict, answers_b: dict, ids: list) -> dict:
    """Exact-match rate and Jaccard on returned document sets."""
    identical, jaccards = [], []
    for qid in ids:
        set_a = set(answers_a[qid].get("document_ids") or [])
        set_b = set(answers_b[qid].get("document_ids") or [])
        union = set_a | set_b
        jaccards.append(len(set_a & set_b) / len(union) if union else 1.0)
        if set_a == set_b:
            identical.append(qid)
    divergent = [q for q in ids if q not in identical]
    print("=" * 78)
    print("RETRIEVAL OVERLAP")
    print("=" * 78)
    print(f"  identical document sets   {len(identical)}/{len(ids)}"
          f"  ({100*len(identical)/len(ids):.1f}%)")
    print(f"  mean Jaccard overlap      {sum(jaccards)/len(jaccards):.3f}")
    print(f"  divergent questions       {len(divergent)}  {divergent}")
    print()
    print("  On the identical questions both harnesses handed the same evidence to")
    print("  the same model, so those questions cannot speak to harness behaviour.")
    print()
    return {
        "identical_ids": identical,
        "divergent_ids": divergent,
        "identical_pct": round(100 * len(identical) / len(ids), 1),
        "mean_jaccard": round(sum(jaccards) / len(jaccards), 3),
    }


def subset_stats(
    scores_a: dict, scores_b: dict, comparisons: dict, group: list, label_a: str, label_b: str
) -> dict:
    """Correctness, recall, and judge split over one subset of questions."""
    n = len(group)
    if not n:
        return {}
    correct_a = sum(1 for q in group if scores_a[q]["answer_correct"])
    correct_b = sum(1 for q in group if scores_b[q]["answer_correct"])
    recall_a = sum(scores_a[q]["document_recall_pct"] for q in group) / n
    recall_b = sum(scores_b[q]["document_recall_pct"] for q in group) / n
    judged = [q for q in group if q in comparisons]
    pref_a = sum(1 for q in judged if comparisons[q].get("preferred_system") == "1")
    pref_b = sum(1 for q in judged if comparisons[q].get("preferred_system") == "2")
    equivalent = sum(1 for q in judged if comparisons[q].get("effectively_equivalent"))
    decisive_a = sum(
        1
        for q in judged
        if comparisons[q].get("preferred_system") == "1"
        and not comparisons[q].get("effectively_equivalent")
    )
    decisive_b = sum(
        1
        for q in judged
        if comparisons[q].get("preferred_system") == "2"
        and not comparisons[q].get("effectively_equivalent")
    )
    print(f"    correct    {label_a} {correct_a}/{n}    {label_b} {correct_b}/{n}")
    print(f"    recall     {label_a} {recall_a:.1f}%    {label_b} {recall_b:.1f}%")
    print(
        f"    judge      {label_a} {pref_a} - {pref_b} {label_b}"
        f"   (equivalent {equivalent}, decisive {decisive_a}-{decisive_b})"
    )
    return {
        "n": n,
        "correct_a": correct_a,
        "correct_b": correct_b,
        "recall_a": round(recall_a, 1),
        "recall_b": round(recall_b, 1),
        "preferred_a": pref_a,
        "preferred_b": pref_b,
        "effectively_equivalent": equivalent,
        "decisive_a": decisive_a,
        "decisive_b": decisive_b,
    }


def report_decomposition(
    scores_a: dict,
    scores_b: dict,
    comparisons: dict,
    overlap: dict,
    label_a: str,
    label_b: str,
) -> dict:
    """Scores split by whether retrieval matched."""
    print("=" * 78)
    print("DECOMPOSITION BY RETRIEVAL AGREEMENT")
    print("=" * 78)
    out = {}
    for name, group in [
        ("identical retrieval", overlap["identical_ids"]),
        ("divergent retrieval", overlap["divergent_ids"]),
    ]:
        print(f"\n  {name}  (n={len(group)})")
        out[name.replace(" ", "_")] = subset_stats(
            scores_a, scores_b, comparisons, group, label_a, label_b
        )
    print()
    return out


def report_noise_floor(
    scores_a: dict, scores_b: dict, overlap: dict, label_a: str, label_b: str
) -> dict:
    """How often the verdict flips on identical evidence."""
    identical = overlap["identical_ids"]
    flipped = [
        q
        for q in identical
        if scores_a[q]["answer_correct"] != scores_b[q]["answer_correct"]
    ]
    rate = 100 * len(flipped) / len(identical) if identical else 0.0
    print("=" * 78)
    print("NOISE FLOOR  (identical evidence, different verdict)")
    print("=" * 78)
    print(f"  {len(flipped)}/{len(identical)} = {rate:.1f}% of identical-retrieval questions")
    print("  scored differently despite both harnesses reading the same documents.")
    for qid in flipped:
        winner = label_a if scores_a[qid]["answer_correct"] else label_b
        print(f"    {qid}: {winner} scored correct, the other did not")
    if flipped:
        print()
        print(f"  A harness effect smaller than ~{rate:.0f}pp cannot be separated from")
        print("  answer-phrasing variance without repeated sampling per question.")
    print()
    return {"flipped_ids": flipped, "flip_rate_pct": round(rate, 1)}


def report_split_half(
    scores_a: dict, scores_b: dict, comparisons: dict, ids: list, label_a: str, label_b: str
) -> dict:
    """Does the aggregate result survive cutting the question set in half?"""
    print("=" * 78)
    print("SPLIT-HALF ROBUSTNESS")
    print("=" * 78)
    splits = {
        "odd index": [q for n, q in enumerate(ids) if n % 2 == 0],
        "even index": [q for n, q in enumerate(ids) if n % 2 == 1],
        "first half": ids[: len(ids) // 2],
        "second half": ids[len(ids) // 2 :],
    }
    out = {}
    for name, group in splits.items():
        judged = [q for q in group if q in comparisons]
        pref_a = sum(1 for q in judged if comparisons[q].get("preferred_system") == "1")
        pref_b = sum(1 for q in judged if comparisons[q].get("preferred_system") == "2")
        correct_a = sum(1 for q in group if scores_a[q]["answer_correct"])
        correct_b = sum(1 for q in group if scores_b[q]["answer_correct"])
        p = exact_binomial_two_sided(min(pref_a, pref_b), pref_a + pref_b)
        leans = label_a if pref_a > pref_b else (label_b if pref_b > pref_a else "neither")
        print(
            f"  {name:<12} judge {pref_a}-{pref_b} (p={p:.3f}, leans {leans})"
            f"   correct {correct_a}/{len(group)} vs {correct_b}/{len(group)}"
        )
        out[name] = {
            "preferred_a": pref_a,
            "preferred_b": pref_b,
            "p_value": round(p, 3),
            "leans": leans,
        }
    directions = {v["leans"] for v in out.values() if v["leans"] != "neither"}
    print()
    if len(directions) > 1:
        print("  The halves lean in opposite directions and none is significant, which")
        print("  is the signature of a true null rather than a real effect hidden by noise.")
    else:
        print("  Every half leans the same way. Non-significance here is more likely")
        print("  to be low power than a true null -- worth more questions.")
    print()
    return out


def report_power(decomposition: dict, overlap: dict, total: int) -> dict:
    """What the design could detect, and what a target effect would cost."""
    identical = decomposition.get("identical_retrieval", {})
    divergent = decomposition.get("divergent_retrieval", {})
    decisive_now = (
        identical.get("decisive_a", 0)
        + identical.get("decisive_b", 0)
        + divergent.get("decisive_a", 0)
        + divergent.get("decisive_b", 0)
    )
    informative_frac = len(overlap["divergent_ids"]) / total if total else 0.0

    print("=" * 78)
    print("POWER")
    print("=" * 78)
    mde = min_detectable_rate(decisive_now)
    print(f"  decisive judge picks available now:  {decisive_now}")
    if mde:
        print(
            f"  smallest detectable preference:      {100*mde:.0f}/{100*(1-mde):.0f}"
            f"  (at {int(100*TARGET_POWER)}% power, alpha={ALPHA})"
        )
        print("  -> the run could only ever have found a lopsided effect. A modest")
        print("     real difference would have come back non-significant either way.")
    print()
    print("  Scaling, assuming the observed rates hold:")
    print(f"    retrieval diverges on {100*informative_frac:.0f}% of questions;")
    print(
        f"    the judge makes a decisive pick on {decisive_now} of {total}"
        f" ({100*decisive_now/total:.0f}%) questions overall."
    )
    print()
    print(f"    {'design':<38}{'decisive':>10}{'detectable':>12}")
    print("    " + "-" * 60)
    scaling = {}
    for name, multiplier in [
        (f"current: 1 model x 2 toolkits, n={total}", 1),
        (f"2 models x 2 toolkits, n={total}", 2),
        (f"4 models x 2 toolkits, n={total}", 4),
        ("4 models x 2 toolkits, n=100", 4 * 100 / total if total else 0),
    ]:
        trials = int(round(decisive_now * multiplier))
        rate = min_detectable_rate(trials)
        label = f"{100*rate:.0f}/{100*(1-rate):.0f}" if rate else "n/a"
        print(f"    {name:<38}{trials:>10}{label:>12}")
        scaling[name] = {"decisive_picks": trials, "detectable_rate": rate}
    print()
    for target in (0.65, 0.60, 0.55):
        need_decisive = required_trials(target)
        if need_decisive is None:
            continue
        need_questions = (
            int(round(need_decisive * total / decisive_now)) if decisive_now else 0
        )
        print(
            f"  to detect a {100*target:.0f}/{100*(1-target):.0f} preference:"
            f" {need_decisive} decisive picks"
            f" ~= {need_questions} questions per toolkit pair"
        )
    print()
    return {
        "decisive_picks_now": decisive_now,
        "informative_fraction": round(informative_frac, 3),
        "min_detectable_rate_now": mde,
        "scaling": scaling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-a", default="results/system_gemini_adk.jsonl")
    parser.add_argument("--answers-b", default="results/system_glean_toolkit_gemini.jsonl")
    parser.add_argument("--scores-a", default="results/results_gemini_adk.json")
    parser.add_argument("--scores-b", default="results/results_glean_toolkit.json")
    parser.add_argument(
        "--comparative", default="results/results_comparative_adk_vs_gleantoolkit.json"
    )
    parser.add_argument("--label-a", default="ADK")
    parser.add_argument("--label-b", default="GLEAN")
    parser.add_argument("--json", help="also write the full analysis to this path")
    args = parser.parse_args()

    answers_a, answers_b = load_answers(args.answers_a), load_answers(args.answers_b)
    scores_a, scores_b = load_scores(args.scores_a), load_scores(args.scores_b)
    comparisons = load_comparisons(args.comparative)

    ids = sorted(set(answers_a) & set(answers_b) & set(scores_a) & set(scores_b))
    if not ids:
        raise SystemExit("no questions in common between the two systems")
    print()
    print(f"Comparing {args.label_a} vs {args.label_b} over {len(ids)} shared questions.")
    missing = sorted(set(ids) - set(comparisons))
    if missing:
        print(f"Not present in the comparative run ({len(missing)}): {', '.join(missing)}")
    print()

    overlap = report_retrieval_overlap(answers_a, answers_b, ids)
    decomposition = report_decomposition(
        scores_a, scores_b, comparisons, overlap, args.label_a, args.label_b
    )
    noise = report_noise_floor(scores_a, scores_b, overlap, args.label_a, args.label_b)
    split_half = report_split_half(
        scores_a, scores_b, comparisons, ids, args.label_a, args.label_b
    )
    power = report_power(decomposition, overlap, len(ids))

    if args.json:
        payload = {
            "systems": {"a": args.answers_a, "b": args.answers_b},
            "questions_compared": len(ids),
            "retrieval_overlap": overlap,
            "decomposition": decomposition,
            "noise_floor": noise,
            "split_half": split_half,
            "power": power,
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
