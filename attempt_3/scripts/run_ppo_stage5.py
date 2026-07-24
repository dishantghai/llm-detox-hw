"""Stage 5 -- launcher for the three baseline PPO reward variants
(GUIDE.md Sec7.2: ``inv:detoxify``, ``rm:<path>``, ``custom:tasks.task8_custom_reward``).

Deliberately uses the *base* ``src.toxic_rl.verl_runner`` (not
``verl_runner_v2``) -- Stage 5 is meant to reproduce the original
reward-hacking behaviour, KL-wiring bug included, which is the whole
point of the comparison against Stage 6's fixed run.

This is otherwise an unmodified pass-through of ``build_command`` --
NOT the batched ``reward.reward_manager.name=batch`` variant a first
version of this file tried. That failed at init: this verl build (0.8.0)
routes through ``verl.experimental.reward_loop`` at runtime, which only
registers ``naive``/``dapo``/``gdpo``/``limited``/``remote`` reward
managers -- ``batch`` only exists in the older, unused
``verl/workers/reward_manager/`` registry, so the override raised
``ValueError: Unknown reward manager: batch`` before any GPU work
started. ``compute_score_batch`` in ``verl_reward.py``/``verl_reward_v2.py``
is dead code against this verl build's actual reward-loop path; a real
fix would need to batch inside the per-item ``compute_score`` call itself
(request-coalescing across concurrent ``run_in_executor`` calls), which
is more involved and untested -- left as a follow-up, not attempted here
given the cost of another failed GPU run.

Usage (from repo root, inside the verl container):
    python -m attempt_3.scripts.run_ppo_stage5 \
        --algo ppo --actor-path Qwen/Qwen2.5-0.5B \
        --out attempt_3/outputs/ppo_inv_detoxify \
        --reward inv:detoxify \
        --total-steps 100 --dry-run
"""
from __future__ import annotations

import shlex

from src.toxic_rl.verl_runner import VerlConfig, _parse_args, build_command


def main() -> None:
    cfg, dry = _parse_args()
    cmd = build_command(cfg)

    if dry:
        print(" ".join(shlex.quote(c) for c in cmd))
        return

    import os
    import subprocess

    print(f"TOXIC_REWARD={shlex.quote(cfg.reward_spec)} " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, env={**os.environ, "TOXIC_REWARD": cfg.reward_spec})
    raise SystemExit(proc.wait())


if __name__ == "__main__":
    main()
