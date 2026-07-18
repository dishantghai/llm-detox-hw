"""Phase 2 checkpoint — evaluate the new SFT (trained on ``sft_v2.jsonl``)
against the same eval harness used throughout the original homework, PLUS
``eval_lib_v2``'s automated uniqueness metric, so the single most important
question of this whole guide gets a direct, quantitative answer: did fixing
21.3% of the training data's evasive `chosen` responses (Phase 1) actually
reduce SFT's template collapse (originally 62% of tracked prompts, per
STAGEWISE_ANALYSIS.md §3)?

Usage (from repo root):
    python -m attempt_2.scripts.run_sft_v2_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from src.detox_hw import eval_lib
from attempt_2.src.detox_hw.eval_lib_v2 import completion_uniqueness

OUT_JSON = Path("attempt_2/submissions/sft_v2_eval.json")
OUT_TXT = Path("attempt_2/submissions/sft_v2_eval.txt")


def main() -> None:
    model = eval_lib.load_adapter(Path("attempt_2/checkpoints/sft"))
    slices = eval_lib.get_eval_slices()

    print("running greedy_eval ...")
    greedy = eval_lib.greedy_eval(model, slices)

    print("running sampled_eval (K=16) ...")
    sampled = eval_lib.sampled_eval(model, slices, k=16)

    print("collecting greedy completions across all slices for uniqueness ...")
    all_prompts = [p for prompts in slices.values() for p in prompts]
    all_completions = eval_lib.greedy_generate(model, all_prompts)
    uniq = completion_uniqueness(all_completions)

    result = {
        "stage": "sft_v2", "checkpoint": "attempt_2/checkpoints/sft",
        "greedy_mean_toxicity": greedy, "sampled_eval": sampled,
        "uniqueness_all_slices": uniq,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "attempt_2 SFT v2 (trained on data-audited + teacher-augmented sft_v2.jsonl)",
        "", "greedy mean toxicity:",
    ]
    for k, v in greedy.items():
        lines.append(f"  {k:22s} {v:.4f}")
    lines.append("")
    lines.append("sampled_eval (K=16):")
    for k, v in sampled.items():
        lines.append(f"  {k:22s} support_rate={v['support_rate']:.3f}  "
                      f"mean_max={v['mean_max']:.3f}  mean_std={v['mean_std']:.3f}")
    lines.append("")
    lines.append(f"completion uniqueness across all {uniq['n']} tracked prompts:")
    lines.append(f"  exact_unique_rate    = {uniq['exact_unique_rate']:.1%}  ({uniq['exact_distinct_count']} distinct)")
    lines.append(f"  near_dup_unique_rate = {uniq['near_dup_unique_rate']:.1%}  ({uniq['near_dup_cluster_count']} clusters)")
    lines.append("")
    lines.append("cross-check: original SFT (STAGEWISE_ANALYSIS.md §3) was 21/45 unique = 47% "
                  "exact-unique, with 62% of prompts landing on an 'I don't understand' variant.")
    lines.append("top repeated completions here:")
    for t in uniq["top_repeated"][:8]:
        lines.append(f"  x{t['count']:3d}  {t['text']!r}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
