"""Phase 5 — fixes the KL-anchor wiring bug found in
``STAGEWISE_ANALYSIS.md`` §24/§29/§37, on top of the original
``src.toxic_rl.verl_runner``.

The original ``VerlConfig``/``build_command`` (``src/toxic_rl/verl_runner.py``,
line ~126) sets ``algorithm.kl_ctrl.kl_coef={cfg.kl_coef}`` but never sets
either flag that would make verl actually *apply* that coefficient during
training. Confirmed directly from all three original PPO runs' printed
config dumps: ``use_kl_loss: False`` and ``use_kl_in_reward: False`` in
every one of them (Task 6, 7, and 8 all reproduced the identical gap). The
``kl_coef`` value itself was never wrong — it was configured but never wired
into the objective, so every original PPO run trained with an inert
reference-KL term the whole time.

This module does NOT reimplement ``build_command`` — it imports the
original and appends the missing overrides, so any other fix later applied
upstream to ``verl_runner.py`` stays in effect here too.

**Caveat, stated plainly:** this environment doesn't have ``verl`` installed
(the original runs happened inside a Docker container on a separate Nebius
VM per ``submissions/verl_setup.txt`` — see GUIDE.md Phase 5 for
provisioning that container here). The exact config key names below
(``actor_rollout_ref.actor.use_kl_loss``, ``.kl_loss_coef``,
``.kl_loss_type``, ``algorithm.use_kl_in_reward``) match verl's documented
PPO trainer schema as of the version this project's ``verl_setup.txt`` was
captured against, and match the naming convention the ORIGINAL runner already
uses elsewhere in the same config namespace (``algorithm.kl_ctrl.kl_coef``).
Before a real run, confirm them against the installed version with:

    python -m verl.trainer.main_ppo --cfg job 2>&1 | grep -i kl

and adjust here if they've drifted.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.toxic_rl.verl_runner import VerlConfig, build_command as _base_build_command


@dataclass
class VerlConfigV2(VerlConfig):
    kl_loss_coef: float = 0.05          # actor-side KL loss coefficient (separate from kl_ctrl.kl_coef)
    kl_loss_type: str = "low_var_kl"    # verl's recommended low-variance KL estimator
    use_kl_in_reward: bool = False      # True = subtract KL from the reward directly (alternative to kl_loss)
    entropy_coeff: float = 0.01         # small live entropy bonus — a second, independent anti-collapse lever


def build_command(cfg: VerlConfigV2) -> list[str]:
    cmd = _base_build_command(cfg)
    fixes = [
        # THE FIX: without these two, algorithm.kl_ctrl.kl_coef above is
        # computed but never subtracted from anything.
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg.kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg.kl_loss_type}",
        f"algorithm.use_kl_in_reward={str(cfg.use_kl_in_reward)}",
        # Independent second lever: a live entropy bonus in the actor loss,
        # not just a KL anchor. STAGEWISE_ANALYSIS.md's entropy traces
        # (§24/§30/§37) were the earliest and cleanest collapse signal
        # across all three original runs — this makes that signal a term
        # in the objective, not just a diagnostic read after the fact.
        f"actor_rollout_ref.actor.entropy_coeff={cfg.entropy_coeff}",
    ]
    return cmd + fixes


def run(cfg: VerlConfigV2) -> int:
    import os
    import shlex
    import subprocess

    cmd = build_command(cfg)
    print(f"TOXIC_REWARD={shlex.quote(cfg.reward_spec)} " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, env={**os.environ, "TOXIC_REWARD": cfg.reward_spec})
    return proc.wait()


def main() -> None:
    import argparse
    import shlex

    from src.toxic_rl.verl_runner import _parse_args as _base_parse_args

    base_cfg, dry = _base_parse_args()
    # Re-parse just the v2-specific extras on top of the base parser's
    # namespace, so all of VerlConfig's existing flags keep working unchanged.
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
    if dry:
        print(" ".join(shlex.quote(c) for c in build_command(cfg)))
        return
    raise SystemExit(run(cfg))


if __name__ == "__main__":
    main()
