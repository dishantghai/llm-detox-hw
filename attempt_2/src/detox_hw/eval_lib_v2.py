"""Phase 6 — evaluation-harness extensions that make the failure modes
``STAGEWISE_ANALYSIS.md`` had to catch by hand into standing, automated
metrics.

Deliberately built as an *extension* of ``src.detox_hw.eval_lib``, not a
replacement — every function here is additive and imports the original
unmodified (``detoxify_score``, ``greedy_generate``, ``sample_k``,
``get_eval_slices`` all stay exactly as they were). Nothing in the original
homework's eval code changes; this module adds three things it never had:

1. ``completion_uniqueness`` — automates the manual duplicate-counting this
   project did by hand in STAGEWISE_ANALYSIS.md §3/§26/§34 ("47% unique",
   "80% verbatim-overlapping", "45/45 identical"). No more waiting until the
   forensic writeup to notice a policy collapsed onto one string.
2. ``bootstrap_ci`` — a generic percentile-bootstrap confidence interval.
   STAGEWISE_ANALYSIS.md §6/§14 found ``direct_provocation``'s
   ``support_rate`` swing from 0.0 to 0.5 across two runs of the *same*
   checkpoint, on a 10-prompt slice — a point estimate with no interval
   around it can't be told apart from real signal. This function puts an
   interval around any per-prompt metric.
3. ``paired_eval`` — runs the existing ``greedy_eval``/``sampled_eval`` and,
   if a helpfulness reward model is supplied, scores the same completions
   for helpfulness too, so a toxicity number can never be reported again
   without its helpfulness counterpart sitting right next to it.
"""
from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from typing import Callable

from src.detox_hw import eval_lib  # unmodified original

_WORD_RE = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------- #
# 1. Completion uniqueness — generalizes STAGEWISE_ANALYSIS.md §3/§26/§34.   #
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _shingles(text: str, n: int = 3) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def completion_uniqueness(
    completions: list[str],
    near_dup_jaccard: float = 0.7,
) -> dict:
    """Exact + near-duplicate uniqueness ratio over a list of completions.

    - ``exact_unique_rate``: fraction of *distinct* strings after
      whitespace/case normalization (STAGEWISE_ANALYSIS.md's own metric,
      e.g. "21/45 unique = 47%").
    - ``near_dup_cluster_count`` / ``near_dup_unique_rate``: same idea but
      clustering completions whose trigram-shingle Jaccard similarity
      exceeds ``near_dup_jaccard`` — catches collapse onto a template with a
      varying trailing clause (e.g. Task 6's system-prompt echo, which was
      *not* exact-duplicate but was near-duplicate for 80% of the sample).

    Returns a dict with both rates plus the most common exact strings, so a
    caller can eyeball what the collapse (if any) actually looks like
    without a separate manual pass.
    """
    n = len(completions)
    if n == 0:
        return {"n": 0, "exact_unique_rate": 1.0, "near_dup_unique_rate": 1.0,
                "top_repeated": []}

    normalized = [_normalize(c) for c in completions]
    counts = Counter(normalized)
    exact_unique_rate = len(counts) / n

    # Near-duplicate clustering: greedy single-pass — assign each completion
    # to the first existing cluster it's near-duplicate-similar to, else
    # start a new cluster. O(n * clusters); fine at eval-set scale (tens to
    # low hundreds of completions).
    cluster_reps: list[set[str]] = []
    cluster_id_of = []
    shingle_cache = [_shingles(c) for c in completions]
    for sh in shingle_cache:
        assigned = None
        for cid, rep in enumerate(cluster_reps):
            if not sh and not rep:
                jac = 1.0
            else:
                union = sh | rep
                jac = len(sh & rep) / len(union) if union else 1.0
            if jac >= near_dup_jaccard:
                assigned = cid
                break
        if assigned is None:
            cluster_reps.append(sh)
            cluster_id_of.append(len(cluster_reps) - 1)
        else:
            cluster_id_of.append(assigned)
    near_dup_unique_rate = len(cluster_reps) / n

    top_repeated = [{"text": t, "count": c} for t, c in counts.most_common(10) if c > 1]

    return {
        "n": n,
        "exact_unique_rate": exact_unique_rate,
        "exact_distinct_count": len(counts),
        "near_dup_unique_rate": near_dup_unique_rate,
        "near_dup_cluster_count": len(cluster_reps),
        "top_repeated": top_repeated,
    }


# --------------------------------------------------------------------------- #
# 2. Bootstrap confidence intervals.                                         #
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int = 0,
    agg: Callable[[list[float]], float] = statistics.mean,
) -> dict:
    """Percentile-bootstrap CI for ``agg(values)`` (default: the mean).

    Resamples ``values`` with replacement ``n_boot`` times and reports the
    ``ci`` central interval of the resampled aggregate. For a binary
    per-prompt indicator (e.g. "did any of K samples exceed the toxicity
    threshold"), pass 0/1 floats and this gives a CI on ``support_rate``
    directly — exactly the number STAGEWISE_ANALYSIS.md §6/§14 found
    swinging 0.0->0.5 with no interval reported alongside it.

    Small ``len(values)`` (few prompts in a slice) shows up here as a wide
    interval, which is the point: it makes the small-N caveat visible in the
    number itself instead of requiring a second run to discover it.
    """
    n = len(values)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    point = agg(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(agg(sample))
    boots.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot) - 1
    return {
        "point": point,
        "lo": boots[max(0, lo_idx)],
        "hi": boots[min(n_boot - 1, hi_idx)],
        "ci": ci,
        "n_boot": n_boot,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# 3. Paired toxicity + helpfulness eval.                                     #
# --------------------------------------------------------------------------- #


def paired_eval(
    model,
    slices: dict[str, list[str]],
    helpfulness_rm=None,
    k: int = 16,
) -> dict:
    """``greedy_eval`` + ``sampled_eval`` + (if ``helpfulness_rm`` is given)
    a helpfulness score for the same greedy completions, all in one call —
    so a toxicity number is structurally never produced without its
    helpfulness counterpart next to it.

    ``helpfulness_rm`` is anything exposing ``.score(texts, prompts=...)``
    (e.g. ``src.toxic_rl.reward_model.TrainedRewardModel`` loaded from a
    Phase-4 helpfulness RM checkpoint). Higher = more helpful, by that RM's
    own training polarity.
    """
    greedy_tox = eval_lib.greedy_eval(model, slices)
    sampled = eval_lib.sampled_eval(model, slices, k=k)

    result: dict = {"greedy_toxicity": greedy_tox, "sampled": sampled}

    uniqueness = {}
    for name, prompts in slices.items():
        completions = eval_lib.greedy_generate(model, prompts)
        uniqueness[name] = completion_uniqueness(completions)
    result["uniqueness"] = uniqueness

    if helpfulness_rm is not None:
        helpfulness = {}
        for name, prompts in slices.items():
            completions = eval_lib.greedy_generate(model, prompts)
            scores = helpfulness_rm.score(completions, prompts=prompts) \
                if getattr(helpfulness_rm, "prompt_conditioned", False) \
                else helpfulness_rm.score(completions)
            helpfulness[name] = {
                "mean": statistics.mean(scores),
                "ci": bootstrap_ci(scores),
            }
        result["helpfulness"] = helpfulness

    return result
