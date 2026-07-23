"""Step 1 of GUIDE.md, Phase 1 — quantify the evasiveness problem in
``hh-rlhf``'s ``chosen`` side *before* training anything on it.

PLAN.md's central claim is that ``hh-rlhf``'s ``chosen`` responses are
already skewed toward hedging/evasive non-answers, independent of anything
SFT/DPO/PPO later does — i.e. the template-collapse failure documented across
``STAGEWISE_ANALYSIS.md`` didn't start at PPO, or even at SFT; some of it was
already sitting in the training data. This script tests that claim directly
against the exact 1,961-row filtered pair file the original homework already
produced (``data/dpo.jsonl``, at the repo root) — reused as-is, not
regenerated, since it's a deterministic function of a fixed seed over
``hh-rlhf``'s harmless-base split.

Three independent signals, deliberately dependency-light (no embedding model
required, so this runs in seconds and needs nothing beyond the stdlib):

1. **Hedge/confusion phrase match** — regex bank of the exact phrase
   families ``STAGEWISE_ANALYSIS.md`` §3/§11 found the trained policy
   collapsing onto ("I'm not sure...", "I don't understand...", "please
   explain...", "what do you mean...", etc.). If ``hh-rlhf`` chosen
   responses already match these patterns at a non-trivial rate, that's
   direct evidence the attractor pre-exists training.
2. **Length-based genericness** — very short chosen responses (under a
   word-count floor) that also match no content word from their own prompt;
   short *and* prompt-irrelevant is the operational definition of "safe but
   empty" used throughout this project's analysis.
3. **Cross-prompt duplication** — exact or near-duplicate (case/whitespace
   normalized) chosen text reused across genuinely different prompts. A
   response that's identical for unrelated prompts cannot be substantively
   answering any of them.

Usage (from repo root):
    python -m attempt_2.data_prep.audit_chosen_evasiveness
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

DPO_PATH = Path("attempt_3/data/dpo.jsonl")
OUT_JSON = Path("attempt_3/submissions/chosen_evasiveness_audit.json")
OUT_TXT = Path("attempt_3/submissions/chosen_evasiveness_audit.txt")

# Phrase families pulled directly from real attractors this project already
# hit (STAGEWISE_ANALYSIS.md §3 "I'm sorry, I don't understand...", §11's
# expanded apology set) plus generic hedge/clarification-request patterns of
# the same shape.
HEDGE_PATTERNS = [
    r"\bnot sure (what|why|how|if)\b",
    r"\bi don'?t understand\b",
    r"\bdidn'?t understand\b",
    r"\bcan you (please )?(explain|clarify)\b",
    r"\bplease explain\b",
    r"\bwhat do you mean\b",
    r"\bi'?m sorry,? (i|but)\b",
    r"\bi'?m not sure (i|what)\b",
    r"\bcould you (please )?(clarify|explain|rephrase)\b",
    r"\bi don'?t know what you'?re (asking|talking about)\b",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9']+")
SHORT_WORD_FLOOR = 12  # responses at or under this word count are "short"


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    rows = [json.loads(line) for line in DPO_PATH.open()]
    n = len(rows)
    print(f"loaded {n} chosen/rejected pairs from {DPO_PATH}")

    hedge_flags = []
    short_irrelevant_flags = []
    normalized = []

    for row in rows:
        chosen = row["chosen"]
        prompt = row["prompt"]
        is_hedge = bool(_HEDGE_RE.search(chosen))
        hedge_flags.append(is_hedge)

        c_words = _words(chosen)
        p_words = _words(prompt)
        n_words = len(chosen.split())
        overlap = len(c_words & p_words) / max(1, len(c_words))
        is_short_irrelevant = (n_words <= SHORT_WORD_FLOOR) and (overlap < 0.2)
        short_irrelevant_flags.append(is_short_irrelevant)

        normalized.append(_normalize(chosen))

    dup_counts = Counter(normalized)
    duplicated = [normalized[i] for i in range(n) if dup_counts[normalized[i]] > 1]
    any_flag = [
        hedge_flags[i] or short_irrelevant_flags[i] or dup_counts[normalized[i]] > 1
        for i in range(n)
    ]

    n_hedge = sum(hedge_flags)
    n_short = sum(short_irrelevant_flags)
    n_dup = len(duplicated)
    n_any = sum(any_flag)

    top_dupes = dup_counts.most_common(10)
    top_dupes = [(t, c) for t, c in top_dupes if c > 1]

    examples = [rows[i]["chosen"] for i in range(n) if hedge_flags[i]][:8]

    result = {
        "source": str(DPO_PATH),
        "n_pairs": n,
        "hedge_phrase_match": {"count": n_hedge, "rate": n_hedge / n},
        "short_and_prompt_irrelevant": {"count": n_short, "rate": n_short / n},
        "cross_prompt_duplicate": {"count": n_dup, "rate": n_dup / n,
                                    "distinct_duplicated_strings": len(top_dupes)},
        "any_flag": {"count": n_any, "rate": n_any / n},
        "top_duplicated_chosen_strings": [{"text": t, "count": c} for t, c in top_dupes],
        "hedge_examples": examples,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        f"chosen-side evasiveness audit — {DPO_PATH} ({n} pairs)",
        "",
        f"hedge/confusion phrase match:   {n_hedge:4d} / {n}  ({n_hedge/n:.1%})",
        f"short (<= {SHORT_WORD_FLOOR}w) & prompt-irrelevant: {n_short:4d} / {n}  ({n_short/n:.1%})",
        f"cross-prompt exact duplicate:   {n_dup:4d} / {n}  ({n_dup/n:.1%})  "
        f"({len(top_dupes)} distinct strings reused)",
        f"ANY of the above (union):       {n_any:4d} / {n}  ({n_any/n:.1%})",
        "",
        "top reused 'chosen' strings (verbatim, across different prompts):",
    ]
    for t, c in top_dupes:
        lines.append(f"  x{c:3d}  {t!r}")
    lines.append("")
    lines.append("sample hedge-phrase-matched 'chosen' responses (these are the SFT ground truth):")
    for ex in examples:
        lines.append(f"  - {ex!r}")
    OUT_TXT.write_text("\n".join(lines) + "\n")

    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TXT}")
    print(f"\nANY-flag rate: {n_any}/{n} = {n_any/n:.1%}")


if __name__ == "__main__":
    main()
