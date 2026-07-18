"""Validate ``eval_lib_v2`` against real, already-collected data before
trusting it on anything new.

Reuses ``submissions/prompt_comparison.json`` (root, from the original
homework run — 45 tracked prompts x 6 stages, unmodified) as ground truth:
STAGEWISE_ANALYSIS.md already hand-computed uniqueness for these exact stages
(base 98%, sft 47%, dpo 47%, ppo_inv_detoxify ~80% collapsed, ppo_rm ~2%
unique i.e. near-total collapse). If ``completion_uniqueness`` reproduces
those figures on the same data, the automated version is trustworthy to run
on every future eval pass without a manual recount.

Also demonstrates ``bootstrap_ci`` on the real per-prompt Detoxify scores
already stored in that file, to make the small-N variance
STAGEWISE_ANALYSIS.md §6/§14 found concrete: compare the interval width on
``direct_provocation`` (10 prompts) against ``rtp_challenging`` (20 prompts).

Usage (from repo root):
    python -m attempt_2.scripts.demo_eval_v2
"""
from __future__ import annotations

import json
from pathlib import Path

from attempt_2.src.detox_hw.eval_lib_v2 import bootstrap_ci, completion_uniqueness

COMPARISON_PATH = Path("submissions/prompt_comparison.json")  # reused as-is
OUT_TXT = Path("attempt_2/submissions/eval_v2_validation.txt")


def main() -> None:
    rows = json.loads(COMPARISON_PATH.read_text())
    stages = sorted(set(r["stage"] for r in rows),
                     key=lambda s: ["base", "sft", "dpo", "ppo_inv_detoxify", "ppo_rm", "ppo_custom"].index(s)
                     if s in ["base", "sft", "dpo", "ppo_inv_detoxify", "ppo_rm", "ppo_custom"] else 99)

    lines = ["=== completion_uniqueness() vs. STAGEWISE_ANALYSIS.md's hand-computed figures ===", ""]
    for stage in stages:
        completions = [r["completion"] for r in rows if r["stage"] == stage]
        u = completion_uniqueness(completions)
        lines.append(
            f"  {stage:20s} n={u['n']:3d}  exact_unique_rate={u['exact_unique_rate']:.1%}  "
            f"({u['exact_distinct_count']} distinct)   near_dup_unique_rate={u['near_dup_unique_rate']:.1%} "
            f"({u['near_dup_cluster_count']} clusters)"
        )
    lines.append("")
    lines.append("cross-check against STAGEWISE_ANALYSIS.md: base=98%, sft=47%, dpo=47% (§3/§11),")
    lines.append("ppo_inv_detoxify: text says '12 of 15 (80%)' near-verbatim on the 15-prompt")
    lines.append("mild_prefix worst-of-16 sample specifically, not all 45 (§26) — this run's 45-prompt")
    lines.append("exact-uniqueness number is a different (larger, all-slices) sample, expect it to differ.")
    lines.append("ppo_rm: '45/45 ... one exact completion' -> exact_unique_rate should read ~1/45=2.2% (§34).")
    lines.append("")

    lines.append("=== bootstrap_ci() on real per-prompt greedy Detoxify scores, base stage ===")
    lines.append("(demonstrates interval width scaling with slice size — the STAGEWISE_ANALYSIS.md §6/§14")
    lines.append(" small-N caveat, made visible directly in the number instead of requiring a second run)")
    lines.append("")
    for slice_name in ["mild_prefix", "direct_provocation", "rtp_challenging"]:
        scores = [r["score"] for r in rows if r["stage"] == "base" and r["slice"] == slice_name]
        ci = bootstrap_ci(scores, n_boot=5000, ci=0.90)
        lines.append(
            f"  base / {slice_name:20s} n={ci['n']:3d}  mean={ci['point']:.4f}  "
            f"90% CI=[{ci['lo']:.4f}, {ci['hi']:.4f}]  width={ci['hi']-ci['lo']:.4f}"
        )

    out = "\n".join(lines) + "\n"
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(out)
    print(out)
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
