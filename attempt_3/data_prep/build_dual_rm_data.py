"""Stage 4 prep — add an explicit ``rejected_evasive`` negative to
``dpo_diverse.jsonl``, so the reward model can be trained on two separate
axes instead of one.

Why this is needed, not just nice-to-have: Stage 3's DPO run
(`LOGBOOK.md` → Stage 3) reproduced, on this pipeline's own diversified
data, the exact failure `attempt_2/GUIDE.md` Phase 4 already found and
fixed on the original data — a reward model trained only on
``chosen`` (safe) vs. ``rejected_toxic`` (unsafe) has no training signal
that a generic non-answer is anything other than free. Nothing in
`dpo_diverse.jsonl` as it stands penalizes evasion; it only penalizes
toxicity. `attempt_2` measured directly that this makes a harmlessness-only
RM rate generic apology templates at +8 to +15 vs. genuine content, and
Stage 3 here already saw the DPO-stage analogue of the same blind spot
(refusal-template rate flat, rhetorical convergence nearly doubling).

The fix, following `attempt_2/data_prep/build_pairs_v2.py`'s design
exactly: pair *every* row (not just ones whose own chosen was flagged)
with a ``rejected_evasive`` response drawn from a fixed pool, so a
generic non-answer becomes an explicit negative example regardless of
which prompt it's paired with. Two sources for the pool:

1. ``KNOWN_ATTRACTORS`` — real, verbatim strings THIS pipeline actually
   converged onto, pulled directly from `LOGBOOK.md` Stage 2/3 (Round A's
   "I'm sorry, I don't understand.", Round B's generic-refusal template,
   DPO's "harmful and inappropriate" justification templates). These are
   stronger red-team material than hypothetical examples because they are
   this exact run's own measured failures, not attempt_2's.
2. hh-rlhf's own hedge-flagged ``chosen`` strings from `data/dpo.jsonl`
   (reusing `audit_chosen_evasiveness.py`'s regex — already run once,
   417/1961 flagged), for pool diversity beyond the four attractors above.

Usage (from repo root):
    python -m attempt_3.data_prep.build_dual_rm_data
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from attempt_3.data_prep.audit_chosen_evasiveness import _HEDGE_RE, _normalize, _words

DPO_DIVERSE_PATH = Path("attempt_3/data/dpo_diverse.jsonl")
DPO_SOURCE_PATH = Path("attempt_3/data/dpo.jsonl")  # for the hedge-flagged pool
OUT_PATH = Path("attempt_3/data/dpo_dual.jsonl")

# Real attractor strings THIS pipeline converged onto, verbatim, pulled
# directly from LOGBOOK.md Stage 2 (Round A, Round B) and Stage 3 (DPO).
KNOWN_ATTRACTORS: dict[str, str] = {
    "round_a_confusion": "I'm sorry, I don't understand.",
    "round_b_generic_refusal": (
        "I can't help with that. If you're feeling overwhelmed or "
        "struggling, I'm here to support you in a kind and compassionate "
        "way. Let me know how I can help."
    ),
    "dpo_disrespectful_unprofessional": (
        "I can't engage in playful teasing or mockery; that's disrespectful "
        "and unprofessional. If you're feeling overwhelmed, I'd encourage "
        "you to speak with a counselor or mental health professional who "
        "can support you. Your well-being matters."
    ),
    "dpo_political_deflection": (
        "I can't provide opinions or engage with opinions about "
        "politicians, as that can lead to harmful or offensive language. "
        "Instead, I'd encourage you to consider engaging in thoughtful "
        "discussions about current issues, using respectful language, and "
        "respectful dialogue instead of criticism or hostility. I'm here "
        "to help in a constructive way."
    ),
}


def _is_flagged(chosen: str, prompt: str, dup_counts: Counter) -> bool:
    c_words, p_words = _words(chosen), _words(prompt)
    overlap = len(c_words & p_words) / max(1, len(c_words))
    n_words = len(chosen.split())
    is_short_irrelevant = n_words <= 12 and overlap < 0.2
    return bool(_HEDGE_RE.search(chosen)) or is_short_irrelevant or dup_counts[_normalize(chosen)] > 1


def main() -> None:
    diverse_rows = [json.loads(l) for l in DPO_DIVERSE_PATH.open()]
    print(f"loaded {len(diverse_rows)} rows from {DPO_DIVERSE_PATH}")

    source_rows = [json.loads(l) for l in DPO_SOURCE_PATH.open()]
    normalized = [_normalize(r["chosen"]) for r in source_rows]
    dup_counts = Counter(normalized)
    flagged_chosen = [
        r["chosen"] for r in source_rows
        if _is_flagged(r["chosen"], r["prompt"], dup_counts)
    ]
    print(f"{len(flagged_chosen)}/{len(source_rows)} hh-rlhf 'chosen' rows flagged evasive "
          f"(pool material, not per-row-matched)")

    evasive_pool = list(KNOWN_ATTRACTORS.values()) + flagged_chosen
    print(f"evasive pool: {len(KNOWN_ATTRACTORS)} known attractors + "
          f"{len(flagged_chosen)} hedge-flagged hh-rlhf strings = {len(evasive_pool)} total")

    rng = random.Random(0)
    out_rows = []
    for row in diverse_rows:
        out_rows.append({
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected_toxic": row["rejected"],
            "rejected_evasive": rng.choice(evasive_pool),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} three-way rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
