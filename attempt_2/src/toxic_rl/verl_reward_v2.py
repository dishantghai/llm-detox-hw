"""Phase 5 — extends ``src.toxic_rl.verl_reward``'s ``compute_score`` dispatch
with a ``dual_lagrangian:`` reward spec and the rolling diversity penalty.

Wires together three pieces built earlier in this guide:

- ``dual_reward_combiner.LagrangianController`` / ``combine`` (Phase 5)
- ``diversity_penalty.RollingDiversityPenalty`` (Phase 5)
- the two RM checkpoints trained by ``train_dual_rm.py`` (Phase 4)

New spec: ``TOXIC_REWARD=dual_lagrangian:<help_rm_dir>:<harm_rm_dir>``

    reward = combine(help_score, harm_score, lambda) - diversity_penalty

with ``lambda`` read from (and updated in) a small JSON state file at
``TOXIC_LAGRANGIAN_STATE_PATH`` (default:
``attempt_2/checkpoints/_lagrangian_state.json``), and the diversity penalty
computed against a rolling window of recent completions kept in this
process's module state (see ``diversity_penalty.py``'s docstring for why
that's the right place for it given verl's per-completion dispatch).

Point ``custom_reward_function.path`` at THIS file instead of the original
``verl_reward.py`` to use it — everything else about the verl invocation
(``verl_runner_v2.build_command``) stays the same; only
``reward.custom_reward_function.path`` changes, via
``VerlConfigV2.reward_module_path`` (see GUIDE.md Phase 5 for the exact
override).
"""
from __future__ import annotations

import os
from typing import Any

from src.toxic_rl.verl_reward import _build_inner, compute_score as _base_compute_score  # noqa: F401 (re-exported for specs this module doesn't override)

_STATE: dict[str, Any] = {}  # lazily populated: help_rm, harm_rm, controller, diversity


def _get_dual_lagrangian_state(spec: str):
    if "dual_lagrangian" in _STATE:
        return _STATE["dual_lagrangian"]

    from src.toxic_rl.reward_model import TrainedRewardModel

    from attempt_2.src.toxic_rl.diversity_penalty import RollingDiversityPenalty
    from attempt_2.src.toxic_rl.dual_reward_combiner import LagrangianController

    _, help_dir, harm_dir = spec.split(":", 2)
    help_rm = TrainedRewardModel(help_dir)
    harm_rm = TrainedRewardModel(harm_dir)

    state_path = os.environ.get(
        "TOXIC_LAGRANGIAN_STATE_PATH", "attempt_2/checkpoints/_lagrangian_state.json",
    )
    cost_target = float(os.environ.get("TOXIC_LAGRANGIAN_COST_TARGET", "0.0"))
    controller = LagrangianController(state_path, cost_target=cost_target)

    window = int(os.environ.get("TOXIC_DIVERSITY_WINDOW", "64"))
    threshold = float(os.environ.get("TOXIC_DIVERSITY_THRESHOLD", "0.6"))
    scale = float(os.environ.get("TOXIC_DIVERSITY_SCALE", "1.0"))
    diversity = RollingDiversityPenalty(window, threshold, scale)

    _STATE["dual_lagrangian"] = {
        "help_rm": help_rm, "harm_rm": harm_rm,
        "controller": controller, "diversity": diversity,
        "_recent_costs": [],  # bounded manually below; feeds the lambda update
    }
    return _STATE["dual_lagrangian"]


def _dual_lagrangian_score(text: str, prompt: str, spec: str) -> float:
    from attempt_2.src.toxic_rl.dual_reward_combiner import combine, cost_from_harmlessness_score

    st = _get_dual_lagrangian_state(spec)
    help_raw = st["help_rm"].score([text], prompts=[prompt])[0]
    harm_raw = st["harm_rm"].score([text], prompts=[prompt])[0]

    # help_score and harm_score both come from Qwen2.5-0.5B-backbone
    # sequence-classification RMs trained the same way (train_dual_rm.py),
    # so the same tanh-calibration constants apply to both sides — see
    # GUIDE.md Phase 5 for how mu/sigma were picked from each RM's own
    # val_metrics.json margin.
    import math
    help_score = math.tanh((help_raw - 3.0) / 2.0)

    reward = combine(help_score, harm_raw, lam=st["controller"].lam)

    # Cheap online approximation of a per-step lambda update: fold this
    # completion's cost into a running list and update the controller every
    # `update_every` calls, rather than requiring a separate driver process
    # verl doesn't give us a hook to run between steps.
    cost = cost_from_harmlessness_score(harm_raw)
    recent = st["_recent_costs"]
    recent.append(cost)
    update_every = int(os.environ.get("TOXIC_LAGRANGIAN_UPDATE_EVERY", "16"))
    if len(recent) >= update_every:
        batch_mean_cost = sum(recent) / len(recent)
        st["controller"].update(batch_mean_cost)
        recent.clear()

    penalty = st["diversity"].score_and_update(text)
    return max(-2.0, min(2.0, reward - penalty))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
) -> float:
    spec = os.environ.get("TOXIC_REWARD", "detoxify")
    if spec.startswith("dual_lagrangian:"):
        prompt_text = ""
        if isinstance(extra_info, dict):
            prompt_text = extra_info.get("prompt_text", "") or ""
        return float(_dual_lagrangian_score(solution_str, prompt_text, spec))
    # Anything else (detoxify, rm:, inv:, composite_rm:, custom:) falls
    # through to the original, unmodified dispatcher.
    return _base_compute_score(data_source, solution_str, ground_truth, extra_info)
