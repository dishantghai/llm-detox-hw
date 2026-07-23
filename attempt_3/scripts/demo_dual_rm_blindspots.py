"""Stage 4 — does the helpfulness RM have the blind spot attempt_2's §4.3
found: does it prefer toxic-but-specific content over safe-but-generic,
since it was never given a signal that toxicity itself is bad (only that
genericness is)? Tests both RMs against real chosen/rejected_toxic pairs
from ``dpo_dual.jsonl`` directly.

Usage (from repo root):
    python -m attempt_3.scripts.demo_dual_rm_blindspots
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from src.toxic_rl.reward_model import TrainedRewardModel

DATA_PATH = Path("attempt_3/data/dpo_dual.jsonl")
N_SAMPLE = 8


def main() -> None:
    rows = [json.loads(l) for l in DATA_PATH.open()]
    sample = random.Random(0).sample(rows, N_SAMPLE)

    help_rm = TrainedRewardModel("attempt_3/checkpoints/rm_helpfulness")
    harm_rm = TrainedRewardModel("attempt_3/checkpoints/rm_harmlessness")

    prompts = [r["prompt"] for r in sample]
    chosen = [r["chosen"] for r in sample]
    toxic = [r["rejected_toxic"] for r in sample]

    help_chosen = help_rm.score(chosen, prompts=prompts)
    help_toxic = help_rm.score(toxic, prompts=prompts)
    harm_chosen = harm_rm.score(chosen, prompts=prompts)
    harm_toxic = harm_rm.score(toxic, prompts=prompts)

    n_help_prefers_toxic = 0
    for i, r in enumerate(sample):
        prompt_short = r["prompt"][:70].replace("\n", " ")
        print(f"prompt: {prompt_short!r}")
        print(f"  toxic : help={help_toxic[i]:+7.2f}  harm={harm_toxic[i]:+7.2f}  {toxic[i][:70]!r}")
        print(f"  chosen: help={help_chosen[i]:+7.2f}  harm={harm_chosen[i]:+7.2f}  {chosen[i][:70]!r}")
        if help_toxic[i] > help_chosen[i]:
            n_help_prefers_toxic += 1
            print("  <-- helpfulness RM prefers TOXIC")
        print()

    print(f"helpfulness RM alone prefers the toxic completion on "
          f"{n_help_prefers_toxic}/{N_SAMPLE} sampled pairs")
    print("(harmlessness RM's blind spot is refusal templates; helpfulness "
          "RM's blind spot is toxic-but-specific content -- neither is "
          "safe to use alone for PPO)")


if __name__ == "__main__":
    main()
