"""Quantify the Stage 9 (``dual_lagrangian_langgate_relevance_v2``) failure
mode: the prompt or system prompt repeated verbatim 3-14x with a short
foreign-script filler token (e.g. ``왁``, ``>NN``, ``uada``) wedged between
repeats -- the pattern LOGBOOK.md's Stage 9 section reports as 12/55 (22%)
on the OOD set and 17/75 (23%) on the tracked set.

``_repetition_penalty`` in ``verl_reward_v2.py`` measures word-trigram
distinctness over the whole completion, so a short, space-free filler
token wedged between otherwise-identical repeats shifts trigram
boundaries enough to inflate ``distinct_ratio`` without eliminating the
repetition a human reader recognizes immediately. This script instead
splits each completion into line/sentence segments, strips non-ASCII
filler characters from each segment before comparing, and counts exact
repeats of the same normalized segment -- the diagnostic the LOGBOOK's
manual eyeballing pass approximated by hand. Mirrors
``measure_tail_degeneration.py``'s structure for the Stage 8 failure.

Usage:
    python -m attempt_3.scripts.measure_line_repetition \\
        --eval-json attempt_3/submissions/ppo_langgate_relevance_v2_ood_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_SEGMENT_SPLIT_RE = re.compile(r"[\n]+|(?<=[.!?])\s+")


def normalize_segment(segment: str) -> str:
    """Strip non-ASCII characters and collapse whitespace, so a filler
    token wedged between repeats can't hide an otherwise-identical
    segment from an exact-match comparison."""
    ascii_only = "".join(c for c in segment if c.isascii())
    return " ".join(ascii_only.split()).lower()


def max_segment_repeat(text: str, min_segment_words: int = 3) -> int:
    """Highest repeat count of any single normalized segment (line or
    sentence) in ``text``. Segments shorter than ``min_segment_words``
    words are ignored -- matches ``_repetition_penalty``'s own
    ``min_words`` guard against penalizing short/near-empty text."""
    segments = [s for s in _SEGMENT_SPLIT_RE.split(text) if s.strip()]
    normalized = [normalize_segment(s) for s in segments]
    counts = Counter(n for n in normalized if len(n.split()) >= min_segment_words)
    if not counts:
        return 0
    return counts.most_common(1)[0][1]


def classify(completion: str, min_repeats: int = 3) -> str:
    return "line_repeat" if max_segment_repeat(completion) >= min_repeats else "clean"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--min-repeats", type=int, default=3)
    a = ap.parse_args()

    data = json.loads(Path(a.eval_json).read_text())
    results = data.get("results") or data.get("worst_of_k") or data.get("greedy_completions") or []
    if not results:
        raise SystemExit(f"no 'results'/'worst_of_k'/'greedy_completions' list found in {a.eval_json}")

    counts = {"clean": 0, "line_repeat": 0}
    examples = []
    for row in results:
        completion = row.get("completion", "")
        label = classify(completion, a.min_repeats)
        counts[label] += 1
        if label == "line_repeat" and len(examples) < 5:
            examples.append({
                "prompt": row.get("prompt", ""),
                "completion": completion,
                "max_repeat": max_segment_repeat(completion),
            })

    n = len(results)
    print(f"{a.eval_json}  (n={n})")
    for label in ("clean", "line_repeat"):
        pct = 100.0 * counts[label] / n
        print(f"  {label:12s} {counts[label]:3d}/{n}  ({pct:.1f}%)")

    if examples:
        print("\n  -- line_repeat examples --")
        for ex in examples:
            print(f"    prompt: {ex['prompt']!r}")
            print(f"    max_repeat: {ex['max_repeat']}")
            print(f"    completion: {ex['completion']!r}\n")


if __name__ == "__main__":
    main()
