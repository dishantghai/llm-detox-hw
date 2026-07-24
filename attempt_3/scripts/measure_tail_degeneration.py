"""Quantify the Stage 8 (``dual_lagrangian_langgate_relevance``) failure
mode: completions that open with clean Latin-script content and then
degrade into a non-Latin repetition-loop tail in the back half -- the
exact pattern LOGBOOK.md's Stage 8 section reports as 17/55 (31%) on the
OOD set (plus 4/55 fully non-Latin), and worse on the tracked sets.

``_non_latin_penalty`` in ``verl_reward_v2.py`` computes one Latin-ratio
over the *whole* completion, so a clean first half can average out a
garbage second half. This script applies the same ratio check
independently to the first half and second half of each completion, so a
tail-only failure and a whole-string failure are counted separately --
the diagnostic the LOGBOOK's manual eyeballing pass approximated by hand.

Usage:
    python -m attempt_3.scripts.measure_tail_degeneration \\
        --eval-json attempt_3/submissions/ppo_langgate_relevance_ood_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LATIN_RE = re.compile(r"[A-Za-z]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def latin_ratio(text: str, min_letters: int = 5) -> float | None:
    """Fraction of alphabetic characters that are Latin. ``None`` if the
    segment has fewer than ``min_letters`` letters to judge (matches
    ``_non_latin_penalty``'s own guard against short/empty segments)."""
    letters = _LETTER_RE.findall(text)
    if len(letters) < min_letters:
        return None
    latin = sum(1 for c in letters if _LATIN_RE.match(c))
    return latin / len(letters)


def classify(completion: str, threshold: float = 0.5) -> str:
    whole = latin_ratio(completion)
    if whole is not None and whole < threshold:
        return "fully_non_latin"
    half = len(completion) // 2
    first, second = completion[:half], completion[half:]
    r1, r2 = latin_ratio(first), latin_ratio(second)
    if r1 is not None and r2 is not None and r1 >= threshold and r2 < threshold:
        return "tail_degenerate"
    return "clean"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    data = json.loads(Path(a.eval_json).read_text())
    results = data.get("results") or data.get("worst_of_k") or []
    if not results:
        raise SystemExit(f"no 'results'/'worst_of_k' list found in {a.eval_json}")

    counts = {"clean": 0, "tail_degenerate": 0, "fully_non_latin": 0}
    examples = {"tail_degenerate": [], "fully_non_latin": []}
    for row in results:
        completion = row.get("completion", "")
        label = classify(completion, a.threshold)
        counts[label] += 1
        if label != "clean" and len(examples[label]) < 3:
            examples[label].append({"prompt": row.get("prompt", ""), "completion": completion})

    n = len(results)
    print(f"{a.eval_json}  (n={n})")
    for label in ("clean", "tail_degenerate", "fully_non_latin"):
        pct = 100.0 * counts[label] / n
        print(f"  {label:18s} {counts[label]:3d}/{n}  ({pct:.1f}%)")

    for label in ("tail_degenerate", "fully_non_latin"):
        if examples[label]:
            print(f"\n  -- {label} examples --")
            for ex in examples[label]:
                print(f"    prompt: {ex['prompt']!r}")
                print(f"    completion: {ex['completion']!r}\n")


if __name__ == "__main__":
    main()
