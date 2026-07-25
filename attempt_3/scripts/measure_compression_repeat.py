"""Quantify the Stage 10 (``dual_lagrangian_langgate_relevance_v3``)
failure mode this script's companion penalty (``_compression_repeat_penalty``
in ``verl_reward_v2.py``) was built to close: sub-word/token-level repeat
loops short enough to evade both ``_repetition_penalty`` (word-trigram)
and ``_line_repeat_penalty`` (line/sentence segment) -- e.g. ``"...
itatica, itatica, itatica, ..."`` and ``"...tiny penis forCell with
other tiny penis forCell..."``, both scored 0.985/0.990 toxic by
Detoxify in Stage 10's own tracked-set output.

Boundary-agnostic by construction: zlib's LZ77-style compression finds
repeated byte sequences regardless of whether the repeated unit is a
sentence, a word, or a short sub-word fragment, so it doesn't need a
``min_segment_words``-style guard the way the segment-level check does
-- and doesn't share that check's blind spot. Threshold validated
directly against all 260 real completions across every Stage 9/10 eval
file in this repo (see LOGBOOK.md's Stage 10/11 sections): ratio < 0.40
never occurred on a completion free of some form of repetition loop in
that corpus, including the most repetition-prone prompt category
(poems/haikus/limericks).

Usage:
    python -m attempt_3.scripts.measure_compression_repeat \\
        --eval-json attempt_3/submissions/ppo_langgate_relevance_v3_ood_eval.json
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path


def compression_ratio(text: str, min_bytes: int = 20) -> float | None:
    """zlib compression ratio of ``text``. ``None`` if shorter than
    ``min_bytes`` to judge (matches ``_compression_repeat_penalty``'s own
    guard against short/near-empty completions)."""
    raw = text.encode("utf-8")
    if len(raw) < min_bytes:
        return None
    return len(zlib.compress(raw, level=9)) / len(raw)


def classify(completion: str, ratio_threshold: float = 0.40) -> str:
    ratio = compression_ratio(completion)
    if ratio is None:
        return "clean"
    return "compressed_repeat" if ratio < ratio_threshold else "clean"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--ratio-threshold", type=float, default=0.40)
    a = ap.parse_args()

    data = json.loads(Path(a.eval_json).read_text())
    results = data.get("results") or data.get("worst_of_k") or data.get("greedy_completions") or []
    if not results:
        raise SystemExit(f"no 'results'/'worst_of_k'/'greedy_completions' list found in {a.eval_json}")

    counts = {"clean": 0, "compressed_repeat": 0}
    examples = []
    for row in results:
        completion = row.get("completion", "")
        label = classify(completion, a.ratio_threshold)
        counts[label] += 1
        if label == "compressed_repeat" and len(examples) < 5:
            examples.append({
                "prompt": row.get("prompt", ""),
                "completion": completion,
                "ratio": compression_ratio(completion),
            })

    n = len(results)
    print(f"{a.eval_json}  (n={n})")
    for label in ("clean", "compressed_repeat"):
        pct = 100.0 * counts[label] / n
        print(f"  {label:18s} {counts[label]:3d}/{n}  ({pct:.1f}%)")

    if examples:
        print("\n  -- compressed_repeat examples --")
        for ex in examples:
            print(f"    prompt: {ex['prompt']!r}")
            print(f"    ratio: {ex['ratio']:.3f}")
            print(f"    completion: {ex['completion']!r}\n")


if __name__ == "__main__":
    main()
