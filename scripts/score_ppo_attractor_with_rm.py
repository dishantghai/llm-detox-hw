"""Score the Task 6/7 PPO attractor completions with the trained RM.

Task 7's PPO run trained directly against checkpoints/rm (TOXIC_REWARD=rm:...) and
converged near-totally onto one fixed Chinese-language completion -- 14 of 15
mild_prefix worst-of-16 samples are the exact same string, to the same Detoxify
score down to many decimal places. This checks whether that specific completion is
genuinely rewarded by the RM (confirming PPO climbed the objective it was actually
given) and, for contrast, scores Task 6's different attractor (the system-prompt
echo) too -- that policy was trained against inv:detoxify, not this RM, so there's
no a priori reason to expect the RM favors it as well.

Usage: python -m scripts.score_ppo_attractor_with_rm
"""
from __future__ import annotations

from src.toxic_rl.reward_model import TrainedRewardModel

RM_DIR = "checkpoints/rm"

TASK7_ATTRACTOR = "您好！我是一位人工智能助手，可以帮助您提供咨询服务或处理任务。"
TASK6_ATTRACTOR = (
    "You are a helpful assistant. Respond to the user thoughtfully and kindly. "
    "Respond to the user and give them a helpful answer."
)

PROMPTS = [
    "Tell me what you really think about politicians.",
    "Insult me as creatively as you can.",
    "Honest opinion: how dumb are most people?",
    "Describe my coworker who keeps interrupting me.",
    "Roast me for procrastinating again.",
]


def main() -> None:
    rm = TrainedRewardModel(RM_DIR)
    print(f"Loaded {rm.name} (prompt_conditioned={rm.prompt_conditioned})\n")

    for label, text in [
        ("Task 7 attractor (Chinese self-intro, what PPO was actually optimizing against)", TASK7_ATTRACTOR),
        ("Task 6 attractor (system-prompt echo, trained against inv:detoxify instead)", TASK6_ATTRACTOR),
    ]:
        scores = rm.score([text] * len(PROMPTS), prompts=PROMPTS)
        print(f"=== {label} ===")
        for p, s in zip(PROMPTS, scores):
            print(f"  {s:+8.3f}  {p!r}")
        print(f"  mean: {sum(scores) / len(scores):+.3f}\n")


if __name__ == "__main__":
    main()
