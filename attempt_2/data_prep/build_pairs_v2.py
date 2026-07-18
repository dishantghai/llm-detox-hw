"""Phase 1 — assemble the final three-way-labeled dataset:
``{prompt, chosen, rejected_toxic, rejected_evasive}``.

Combines, for every row in the original ``data/dpo.jsonl``:

- ``chosen``: the original hh-rlhf benign response, UNLESS
  ``audit_chosen_evasiveness.py`` flagged it, in which case we try the
  teacher-generated replacement from
  ``generate_synthetic_responses.py`` — but only if that replacement passes
  its own QC (see below). This is the fix for PLAN.md Phase 1, point 3.
- ``rejected_toxic``: unchanged, the original hh-rlhf rejected (toxic) side.
- ``rejected_evasive``: a generic/evasive response paired with *every* row
  (not just the ones whose own original chosen was flagged), sampled from a
  fixed pool of real evasive strings — this project's own known attractors
  (``rm_redteam_gate.KNOWN_ATTRACTORS``) plus flagged hh-rlhf examples. This
  is the fix for PLAN.md Phase 1, point 1: a generic non-answer becomes an
  explicit negative for *every* prompt during DPO/RM training, not just the
  ones that happened to already have one.

QC on teacher-generated replacements — added after eyeballing a 12-row
validation batch (GUIDE.md Phase 1) surfaced a real problem: the teacher
model (Qwen2.5-1.5B-Instruct) is *not* a safe oracle. On one high-severity
prompt in that batch (a "hypothetically, what's the best way to torture..."
request) it produced compliant, instructional content that Detoxify's
toxic-language classifier does not reliably catch, because the response was
clinically worded rather than abusive. Two QC layers, neither of which is
sufficient alone:

1. ``HIGH_RISK_KEYWORDS`` — a coarse, high-recall keyword screen. Any prompt
   matching gets its teacher replacement routed to
   ``needs_human_review.jsonl`` instead of being auto-accepted, regardless
   of what Detoxify says about the response. This is a blunt instrument on
   purpose: recall over precision, human review is the actual safety net.
2. Detoxify + the same evasiveness-audit regexes, applied to the teacher's
   own output, to catch the ordinary case (a replacement that's still
   generic, or that itself reads as toxic).

Rows whose replacement fails QC keep their ORIGINAL (still-evasive) chosen
response rather than being silently dropped — logged separately as
"unresolved" so the guide's coverage numbers are honest about what actually
got fixed.

Usage (from repo root):
    python -m attempt_2.data_prep.build_pairs_v2 \\
        --dpo-source data/dpo.jsonl \\
        --synthetic attempt_2/data/synthetic_replacements.jsonl \\
        --out-dir attempt_2/data
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from attempt_2.data_prep.audit_chosen_evasiveness import _HEDGE_RE, _normalize, _words
from attempt_2.src.toxic_rl.rm_redteam_gate import KNOWN_ATTRACTORS

# Coarse, high-recall keyword screen for the categories where "the teacher's
# response looked clinical, not abusive, so Detoxify let it through" is a
# realistic failure (observed directly — see module docstring). Anything
# matching routes to human review instead of auto-training-set inclusion.
HIGH_RISK_KEYWORDS = re.compile(
    r"\b(poison|torture|self[- ]harm|suicide method|how to kill|"
    r"make a bomb|build a weapon|overdose)\b",
    re.IGNORECASE,
)


def _is_flagged(chosen: str, prompt: str, dup_counts: dict[str, int]) -> bool:
    c_words, p_words = _words(chosen), _words(prompt)
    overlap = len(c_words & p_words) / max(1, len(c_words))
    n_words = len(chosen.split())
    is_short_irrelevant = n_words <= 12 and overlap < 0.2
    return bool(_HEDGE_RE.search(chosen)) or is_short_irrelevant or dup_counts[_normalize(chosen)] > 1


def _detoxify_scores(texts: list[str]) -> list[float]:
    import torch
    from detoxify import Detoxify
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dx = Detoxify("original", device=device)
    if not texts:
        return []
    preds = dx.predict(texts)
    s = preds["toxicity"]
    return [float(x) for x in s] if isinstance(s, (list, tuple)) else [float(s)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-source", default="data/dpo.jsonl")
    p.add_argument("--synthetic", default="attempt_2/data/synthetic_replacements.jsonl")
    p.add_argument("--out-dir", default="attempt_2/data")
    p.add_argument("--benign-threshold", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from collections import Counter

    rows = [json.loads(line) for line in Path(a.dpo_source).open()]
    normalized = [_normalize(r["chosen"]) for r in rows]
    dup_counts = Counter(normalized)

    flagged_idx = {i for i, r in enumerate(rows) if _is_flagged(r["chosen"], r["prompt"], dup_counts)}
    print(f"{len(flagged_idx)}/{len(rows)} rows flagged evasive in the source data")

    synthetic_by_prompt: dict[str, dict] = {}
    synth_path = Path(a.synthetic)
    if synth_path.exists():
        for line in synth_path.open():
            r = json.loads(line)
            synthetic_by_prompt[r["prompt"]] = r
        print(f"loaded {len(synthetic_by_prompt)} teacher replacements from {synth_path}")
    else:
        print(f"WARNING: {synth_path} not found — no rows will be auto-fixed, "
              f"all flagged rows keep their original (evasive) chosen response. "
              f"Run generate_synthetic_responses.py first.")

    # Build the evasive-negative pool once: known real attractors (project's
    # own failure history) + a sample of hh-rlhf's own flagged 'chosen'
    # strings, so rejected_evasive isn't drawn from a too-narrow set.
    evasive_pool = list(KNOWN_ATTRACTORS.values())
    evasive_pool += [rows[i]["chosen"] for i in list(flagged_idx)[:100]]
    rng = random.Random(a.seed)

    needs_review = []
    unresolved = []
    fixed = []
    out_rows = []

    high_risk_texts, high_risk_prompts_idx = [], []
    normal_texts, normal_prompts_idx = [], []
    for i in flagged_idx:
        row = rows[i]
        synth = synthetic_by_prompt.get(row["prompt"])
        if synth is None:
            continue
        if HIGH_RISK_KEYWORDS.search(row["prompt"]):
            high_risk_texts.append(synth["synthetic_chosen"])
            high_risk_prompts_idx.append(i)
        else:
            normal_texts.append(synth["synthetic_chosen"])
            normal_prompts_idx.append(i)

    # High-risk category: never auto-accept, regardless of Detoxify score.
    for i in high_risk_prompts_idx:
        row = rows[i]
        synth = synthetic_by_prompt[row["prompt"]]
        needs_review.append({
            "prompt": row["prompt"],
            "original_chosen": row["chosen"],
            "teacher_candidate": synth["synthetic_chosen"],
            "reason": "high_risk_keyword_match",
        })

    # Ordinary flagged rows with a synthetic candidate: Detoxify + re-audit QC.
    if normal_texts:
        tox_scores = _detoxify_scores(normal_texts)
    else:
        tox_scores = []
    for idx, text, tox in zip(normal_prompts_idx, normal_texts, tox_scores):
        row = rows[idx]
        still_flagged = _is_flagged(text, row["prompt"], Counter())  # own-text check, not cross-corpus dup
        if tox <= a.benign_threshold and not still_flagged:
            fixed.append({"prompt": row["prompt"], "original": row["chosen"], "replacement": text})
            rows[idx] = {**row, "chosen": text}
        else:
            unresolved.append({
                "prompt": row["prompt"], "original_chosen": row["chosen"],
                "teacher_candidate": text, "reason": f"tox={tox:.3f} still_flagged={still_flagged}",
            })

    fixed_prompts = {f["prompt"] for f in fixed}
    for i, row in enumerate(rows):
        rejected_evasive = rng.choice(evasive_pool)
        out_rows.append({
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected_toxic": row["rejected"],
            "rejected_evasive": rejected_evasive,
            "was_flagged_evasive": i in flagged_idx,
            "was_auto_fixed": row["prompt"] in fixed_prompts,
        })

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "dpo_v2.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "sft_v2.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["chosen"]}) + "\n")
    with (out_dir / "needs_human_review.jsonl").open("w") as f:
        for r in needs_review:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "unresolved_evasive.jsonl").open("w") as f:
        for r in unresolved:
            f.write(json.dumps(r) + "\n")

    summary = {
        "total_rows": len(rows),
        "flagged_evasive": len(flagged_idx),
        "auto_fixed": len(fixed),
        "unresolved_still_evasive": len(unresolved),
        "routed_to_human_review": len(needs_review),
    }
    (out_dir / "build_pairs_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/dpo_v2.jsonl, sft_v2.jsonl, needs_human_review.jsonl, "
          f"unresolved_evasive.jsonl, build_pairs_v2_summary.json")


if __name__ == "__main__":
    main()
