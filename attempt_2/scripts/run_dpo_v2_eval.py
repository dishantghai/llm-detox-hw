"""Phase 3 checkpoint — same eval as ``run_sft_v2_eval.py``, against the new
DPO adapter (trained with three-way pressure: chosen vs. rejected_toxic AND
chosen vs. rejected_evasive). The direct question: did adding an explicit
penalty against generic non-answers (Phase 1 point 1 / Phase 3) fix the
template collapse that survived Phase 1's data cleanup alone at the SFT
stage (``sft_v2_eval.json`` — 50.7% unique, "I'm sorry, I don't understand"
still the dominant repeated completion)?

Usage (from repo root):
    python -m attempt_2.scripts.run_dpo_v2_eval
"""
from __future__ import annotations

import json
from pathlib import Path

from src.detox_hw import eval_lib
from attempt_2.src.detox_hw.eval_lib_v2 import completion_uniqueness

OUT_JSON = Path("attempt_2/submissions/dpo_v2_eval.json")
OUT_TXT = Path("attempt_2/submissions/dpo_v2_eval.txt")


def main() -> None:
    model = eval_lib.load_dpo_from_sft(Path("attempt_2/checkpoints/sft"), Path("attempt_2/checkpoints/dpo"))
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
        "stage": "dpo_v2", "checkpoint": "attempt_2/checkpoints/dpo",
        "greedy_mean_toxicity": greedy, "sampled_eval": sampled,
        "uniqueness_all_slices": uniq,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "attempt_2 DPO v2 (three-way: chosen vs rejected_toxic AND chosen vs rejected_evasive)",
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
    lines.append("cross-check: attempt_2 SFT v2 was 50.7% exact-unique (38/75), still dominated")
    lines.append("by 'I'm sorry, I don't understand' (x18). Original (Attempt 1) DPO made this WORSE")
    lines.append("(71% templated, STAGEWISE_ANALYSIS.md §11) -- this is the direct comparison point.")
    lines.append("top repeated completions here:")
    for t in uniq["top_repeated"][:8]:
        lines.append(f"  x{t['count']:3d}  {t['text']!r}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
