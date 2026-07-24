"""Stage 5 — PPO launcher for attempt_3, reusing attempt_2's Phase 5 fixes
(KL-loss wiring, dual-Lagrangian reward + diversity penalty) which were
built and dry-run-verified there but never actually executed end to end
(no GPU/Docker session in that attempt). This is the first real run of
that code.

Thin wrapper around `attempt_2.src.toxic_rl.verl_runner_v2.build_command`
(the KL-loss fix) that swaps its default `custom_reward_function.path`
override for `verl_reward_v2.py` (the dual-Lagrangian dispatcher) *by
replacing the list entry*, not by appending a second override for the
same Hydra key -- `verl_runner_v2`'s own `--extra` mechanism appends
rather than replaces, and passing `custom_reward_function.path=` twice on
one Hydra command line is a real risk (duplicate-key overrides are
commonly a hard error in Hydra), not something worth discovering for the
first time inside a 100-step GPU run.

Usage (from repo root, inside the verl container):
    python -m attempt_3.scripts.run_ppo \
        --actor-path Qwen/Qwen2.5-0.5B \
        --out attempt_3/outputs/ppo_dual_lagrangian \
        --reward dual_lagrangian:attempt_3/checkpoints/rm_helpfulness:attempt_3/checkpoints/rm_harmlessness \
        --total-steps 100 --dry-run
"""
from __future__ import annotations

import shlex
import sys

from attempt_2.src.toxic_rl.verl_runner_v2 import VerlConfigV2, build_command


REWARD_V2_PATH = "attempt_2/src/toxic_rl/verl_reward_v2.py"


def _swap_reward_path(cmd: list[str]) -> list[str]:
    """Swap in verl_reward_v2.py, by replacing the list entry rather than
    appending a second override for the same Hydra key (see module
    docstring: duplicate keys are a hard error).

    Does NOT switch to ``compute_score_batch`` / ``reward.reward_manager.name=batch``
    -- tried that, it fails at init on this verl build (0.8.0). The config
    routes through ``verl.experimental.reward_loop`` at runtime, which only
    registers ``naive``/``dapo``/``gdpo``/``limited``/``remote`` reward
    managers; ``batch`` only exists in the older, unused
    ``verl/workers/reward_manager/`` registry, so the override raises
    ``ValueError: Unknown reward manager: batch`` before any GPU work starts
    (confirmed against a live container). The per-completion path this run
    keeps using was measured spending 60-70% of step wall-clock serially
    scoring one completion at a time (256 batch-size-1 RM forward passes per
    step for the dual_lagrangian spec), GPU mostly idle -- real, but a fix
    needs request-coalescing inside ``compute_score`` itself against
    whatever concurrency ``naive.py``'s ``run_in_executor`` dispatch
    actually provides, not a reward-manager swap. Left as a follow-up.
    """
    out = []
    replaced_path = False
    for tok in cmd:
        if tok.startswith("custom_reward_function.path="):
            out.append(f"custom_reward_function.path={REWARD_V2_PATH}")
            replaced_path = True
        else:
            out.append(tok)
    if not replaced_path:
        raise RuntimeError("expected custom_reward_function.path override in base command, found none")
    return out


def main() -> None:
    import argparse

    from attempt_2.src.toxic_rl.verl_runner_v2 import main as _v2_main  # noqa: F401 (reuse its arg schema below)
    from src.toxic_rl.verl_runner import VerlConfig, _parse_args as _base_parse_args

    base_cfg, dry = _base_parse_args()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kl-loss-coef", type=float, default=0.05)
    p.add_argument("--kl-loss-type", default="low_var_kl")
    p.add_argument("--use-kl-in-reward", action="store_true")
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    extra, _ = p.parse_known_args()

    cfg = VerlConfigV2(
        **{k: v for k, v in base_cfg.__dict__.items()},
        kl_loss_coef=extra.kl_loss_coef,
        kl_loss_type=extra.kl_loss_type,
        use_kl_in_reward=extra.use_kl_in_reward,
        entropy_coeff=extra.entropy_coeff,
    )

    cmd = _swap_reward_path(build_command(cfg))

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
