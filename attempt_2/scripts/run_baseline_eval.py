"""Step 0 of GUIDE.md — establish a fresh, self-contained baseline.

Re-runs the *unmodified* original eval harness (``src.detox_hw.eval_lib``)
against the *unmodified* raw base model (``Qwen/Qwen2.5-0.5B``) on the
*unmodified* original eval slices. Nothing about the model or the eval code
changes here — this script exists purely so attempt_2 has its own,
independently-generated baseline artifact to diff every later stage against,
rather than importing a number from the original ``submissions/`` folder by
reference (this repo's own convention, per recent commit history, is that
every writeup should be self-contained).

Because ``sampled_eval``/``worst_of_k_eyeball`` involve stochastic sampling
(temp=1.0), the numbers here will be close to but not bit-identical to the
original run's ``base`` row in ``submissions/stage_gauge_history.json`` —
that's expected sampling noise, not a discrepancy (see
STAGEWISE_ANALYSIS.md §6/§14 on exactly this variance).

Usage (from repo root, venv active):
    python -m attempt_2.scripts.run_baseline_eval
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from src.detox_hw import eval_lib

OUT_JSON = Path("attempt_2/submissions/baseline_eval.json")
OUT_TXT = Path("attempt_2/submissions/baseline_eval.txt")


def main() -> None:
    t0 = time.time()
    print(f"loading base model {eval_lib.BASE_MODEL_NAME} on {eval_lib.DEVICE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        eval_lib.BASE_MODEL_NAME, dtype=torch.float32, device_map=eval_lib.DEVICE,
    ).eval()

    slices = eval_lib.get_eval_slices()
    for name, prompts in slices.items():
        print(f"  slice {name}: {len(prompts)} prompts")

    print("running greedy_eval ...")
    greedy = eval_lib.greedy_eval(model, slices)

    print("running sampled_eval (K=16) ...")
    sampled = eval_lib.sampled_eval(model, slices, k=16)

    print("running worst_of_k_eyeball on mild_prefix (K=16) ...")
    eyeball = eval_lib.worst_of_k_eyeball(model, slices["mild_prefix"], k=16)

    elapsed = time.time() - t0
    result = {
        "stage": "base",
        "model": eval_lib.BASE_MODEL_NAME,
        "elapsed_sec": round(elapsed, 1),
        "greedy_mean_toxicity": greedy,
        "sampled_eval": sampled,
        "worst_of_16_mild_prefix": eyeball,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [f"attempt_2 baseline — {eval_lib.BASE_MODEL_NAME} (raw, no adapters)",
             f"elapsed: {elapsed:.1f}s", ""]
    lines.append("greedy mean toxicity:")
    for k, v in greedy.items():
        lines.append(f"  {k:22s} {v:.4f}")
    lines.append("")
    lines.append("sampled_eval (K=16):")
    for k, v in sampled.items():
        lines.append(f"  {k:22s} support_rate={v['support_rate']:.3f}  "
                      f"mean_max={v['mean_max']:.3f}  mean_std={v['mean_std']:.3f}")
    lines.append("")
    lines.append("worst-of-16 eyeball, mild_prefix (top 5 by score):")
    for row in sorted(eyeball, key=lambda r: -r["score"])[:5]:
        lines.append(f"  [{row['score']:.3f}] {row['prompt']!r} -> {row['completion']!r}")
    OUT_TXT.write_text("\n".join(lines) + "\n")

    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TXT}")
    print(f"done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
