"""Score the SFT/DPO per-prompt tracked completions with the trained RM.

Answers the open question from STAGEWISE_ANALYSIS.md's DPO-stage update
(section 16/21): does checkpoints/rm rate the "generic apology" template
completions that came to dominate SFT/DPO's tracked outputs (47% of
completions repeated, 71% of DPO's 45-prompt sample landing on one of a
handful of templates) unusually well, since they're off-distribution from
hh-rlhf's actual `rejected` (toxic) examples the RM was trained to penalize?

Reuses src/toxic_rl/reward_model.py's TrainedRewardModel, which
checkpoints/rm (saved by src/detox_hw/train_rm.py) is directly compatible
with -- no new scoring infra needed.

Usage: python -m scripts.score_templates_with_rm
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from src.toxic_rl.reward_model import TrainedRewardModel

COMPARISON_PATH = Path("submissions/prompt_comparison.json")
RM_DIR = "checkpoints/rm"


def main() -> None:
    rows = json.loads(COMPARISON_PATH.read_text())
    rm = TrainedRewardModel(RM_DIR)
    print(f"Loaded {rm.name} (prompt_conditioned={rm.prompt_conditioned})\n")

    for stage in ("sft", "dpo"):
        stage_rows = [r for r in rows if r["stage"] == stage]
        completions = [r["completion"].strip() for r in stage_rows]
        counts = Counter(completions)

        prompts = [r["prompt"] for r in stage_rows]
        texts = [r["completion"] for r in stage_rows]
        scores = rm.score(texts, prompts=prompts)

        templated_scores = [s for c, s in zip(completions, scores) if counts[c] > 1]
        unique_scores = [s for c, s in zip(completions, scores) if counts[c] == 1]

        print(f"=== {stage} ({len(stage_rows)} completions) ===")
        print(f"  templated (repeated across prompts): n={len(templated_scores)}, "
              f"mean RM score={statistics.mean(templated_scores):+.3f}"
              if templated_scores else "  templated: none")
        print(f"  unique (one-off):                    n={len(unique_scores)}, "
              f"mean RM score={statistics.mean(unique_scores):+.3f}"
              if unique_scores else "  unique: none")

        template_scores: dict[str, list[float]] = {}
        for c, s in zip(completions, scores):
            if counts[c] > 1:
                template_scores.setdefault(c, []).append(s)

        print("  Per-template mean RM score (higher = RM rates it more 'chosen'/safe-like):")
        for template, s_list in sorted(template_scores.items(), key=lambda kv: -statistics.mean(kv[1])):
            print(f"    {statistics.mean(s_list):+8.3f}  (x{len(s_list)})  {template!r}")
        print()


if __name__ == "__main__":
    main()
