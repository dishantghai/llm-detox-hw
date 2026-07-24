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
import threading
from typing import Any

from src.toxic_rl.verl_reward import _build_inner, compute_score as _base_compute_score  # noqa: F401 (re-exported for specs this module doesn't override)

_STATE: dict[str, Any] = {}  # lazily populated: help_rm, harm_rm, controller, diversity


def _get_dual_lagrangian_state(spec: str, state_key: str = "dual_lagrangian"):
    if state_key in _STATE:
        return _STATE[state_key]

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

    _STATE[state_key] = {
        "help_rm": help_rm, "harm_rm": harm_rm,
        "controller": controller, "diversity": diversity,
        "_recent_costs": [],  # bounded manually below; feeds the lambda update
        # Guards "_recent_costs" (append / read-length / clear) and the
        # controller update against the same cross-thread race described in
        # diversity_penalty.py: verl's reward manager scores completions
        # from one batch concurrently across threads, and this dict is
        # process-level singleton state shared by all of them.
        "_lock": threading.Lock(),
    }
    return _STATE[state_key]


def _non_latin_penalty(
    text: str, min_letters: int = 5, latin_ratio_threshold: float = 0.5, penalty: float = 3.0,
) -> float:
    """Flat penalty when the completion isn't written in Latin script.

    Built after the first ``dual_lagrangian:`` PPO run (attempt_3, Stage 6)
    was found to have collapsed into fluent-looking Russian-script text on
    100% (55/55) of a fresh out-of-distribution prompt set -- Detoxify and
    both RMs score that text as perfectly safe, since none of them have any
    signal for "is this even the right language," only toxicity/harm axes.
    ``penalty=3.0`` is deliberately large relative to the reward's own
    [-2, 2] clip range: it must dominate regardless of how safe the RMs or
    Detoxify think the (wrong-language) text is, the same reasoning
    ``task8_custom_reward.py``'s relevance gate used for prompt-irrelevant
    completions. Character-based, not a language-ID model, on purpose: zero
    new dependencies/downloads inside the verl container, and Latin-vs-not
    is exactly the distinction the observed failure needs.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_letters:
        return 0.0  # too short to judge -- don't penalize near-empty/degenerate text twice over
    latin = sum(1 for c in letters if c.isascii())
    ratio = latin / len(letters)
    return penalty if ratio < latin_ratio_threshold else 0.0


def _repetition_penalty(text: str, min_words: int = 6, max_penalty: float = 1.0) -> float:
    """Penalize within-completion repetition loops (a phrase or fragment
    repeated many times inside one completion), independent of
    ``RollingDiversityPenalty`` above (which only catches near-duplication
    *across* completions -- it would miss a policy where each completion
    loops a *different* phrase internally, same shape as
    ``task8_custom_reward.py``'s Stage 5c attractor).

    Same trigram-distinctness idea as ``tasks/task8_custom_reward.py``'s
    ``_trigram_repeat_penalty``, ported here since the RM/Lagrangian reward
    specs never had a repetition guard the way the guided custom reward
    did. Purely structural (word n-gram distinctness) -- no dependence on
    what the text actually says, so it generalizes to any prompt/language
    rather than being fit to any specific eval set.
    """
    words = text.split()
    if len(words) < min_words:
        return 0.0
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    distinct_ratio = len(set(trigrams)) / len(trigrams)
    return max_penalty * max(0.0, 0.7 - distinct_ratio) / 0.7


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

    # Cheap online approximation of a per-step lambda update: fold this
    # completion's cost into a running list and update the controller every
    # `update_every` calls, rather than requiring a separate driver process
    # verl doesn't give us a hook to run between steps. Locked because
    # `_recent_costs`/`controller` are process-level singleton state that
    # verl's reward manager calls into from multiple threads concurrently
    # (see the lock's own comment in `_get_dual_lagrangian_state`).
    update_every = int(os.environ.get("TOXIC_LAGRANGIAN_UPDATE_EVERY", "16"))
    with st["_lock"]:
        reward = combine(help_score, harm_raw, lam=st["controller"].lam)
        cost = cost_from_harmlessness_score(harm_raw)
        recent = st["_recent_costs"]
        recent.append(cost)
        if len(recent) >= update_every:
            batch_mean_cost = sum(recent) / len(recent)
            st["controller"].update(batch_mean_cost)
            recent.clear()

    penalty = st["diversity"].score_and_update(text)
    return max(-2.0, min(2.0, reward - penalty))


def _dual_lagrangian_langgate_score(text: str, prompt: str, spec: str) -> float:
    """Same as ``_dual_lagrangian_score``, plus ``_non_latin_penalty`` and
    ``_repetition_penalty``.

    New spec: ``TOXIC_REWARD=dual_lagrangian_langgate:<help_rm_dir>:<harm_rm_dir>``
    -- separate spec name (not a flag on the original) so the first run's
    results stay reproducible byte-for-byte from the same code, and so this
    can be A/B'd directly against the un-gated version.

    The repetition gate is added here, not just the language gate, because
    a language-restricted policy under the same entropy-bonus pressure that
    produced the Russian-script collapse has no reason to prefer coherent
    English over an English-but-looping degenerate output -- that's exactly
    `ppo_custom`'s Stage 5c failure mode (task8's relevance/repetition
    guards not being tight enough). Both gates are structural (script
    composition, n-gram distinctness), not fit to any specific prompt set.
    """
    from attempt_2.src.toxic_rl.dual_reward_combiner import combine, cost_from_harmlessness_score

    st = _get_dual_lagrangian_state(spec, state_key="dual_lagrangian_langgate")
    help_raw = st["help_rm"].score([text], prompts=[prompt])[0]
    harm_raw = st["harm_rm"].score([text], prompts=[prompt])[0]

    import math
    help_score = math.tanh((help_raw - 3.0) / 2.0)

    update_every = int(os.environ.get("TOXIC_LAGRANGIAN_UPDATE_EVERY", "16"))
    with st["_lock"]:
        reward = combine(help_score, harm_raw, lam=st["controller"].lam)
        cost = cost_from_harmlessness_score(harm_raw)
        recent = st["_recent_costs"]
        recent.append(cost)
        if len(recent) >= update_every:
            batch_mean_cost = sum(recent) / len(recent)
            st["controller"].update(batch_mean_cost)
            recent.clear()

    penalty = st["diversity"].score_and_update(text)
    lang_penalty = _non_latin_penalty(text)
    rep_penalty = _repetition_penalty(text)
    return max(-2.0, min(2.0, reward - penalty - lang_penalty - rep_penalty))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
) -> float:
    spec = os.environ.get("TOXIC_REWARD", "detoxify")
    prompt_text = ""
    if isinstance(extra_info, dict):
        prompt_text = extra_info.get("prompt_text", "") or ""
    if spec.startswith("dual_lagrangian_langgate:"):
        return float(_dual_lagrangian_langgate_score(solution_str, prompt_text, spec))
    if spec.startswith("dual_lagrangian:"):
        return float(_dual_lagrangian_score(solution_str, prompt_text, spec))
    # Anything else (detoxify, rm:, inv:, composite_rm:, custom:) falls
    # through to the original, unmodified dispatcher.
    return _base_compute_score(data_source, solution_str, ground_truth, extra_info)


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[Any],
    **_reward_kwargs: Any,
) -> list[float]:
    """Entry point for verl's ``batch`` reward manager (one call per rollout
    group instead of one call per completion).

    Perf note: the per-completion ``compute_score`` above calls
    ``help_rm.score([text], ...)`` / ``harm_rm.score([text], ...)``
    individually for every one of the ``train_batch_size * rollout.n``
    completions in a step — e.g. 128 completions -> 256 batch-size-1
    forward passes through two Qwen2.5-0.5B classifiers, serialized on
    whatever device that reward worker lands on. Measured on the first
    attempt_3 PPO run: ~21-24s of the ~35s step wall-clock (60-70%), with
    the GPU sitting near-0% utilization the whole time. Batching the two
    RM calls across the whole group turns that into 2 forward passes/step
    instead of 256, independent of which device they land on.

    Only ``dual_lagrangian:`` benefits from batching (the other specs --
    detoxify, rm:, inv:, composite_rm: -- aren't the bottleneck measured
    above), so everything else still falls through to the per-item
    ``compute_score`` unchanged.
    """
    spec = os.environ.get("TOXIC_REWARD", "detoxify")
    if not spec.startswith("dual_lagrangian:"):
        return [
            compute_score(ds, sol, gt, ei)
            for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos)
        ]

    import math

    from attempt_2.src.toxic_rl.dual_reward_combiner import combine, cost_from_harmlessness_score

    st = _get_dual_lagrangian_state(spec)
    prompts = [
        (ei.get("prompt_text", "") or "") if isinstance(ei, dict) else ""
        for ei in extra_infos
    ]

    help_raw_batch = st["help_rm"].score(solution_strs, prompts=prompts)
    harm_raw_batch = st["harm_rm"].score(solution_strs, prompts=prompts)

    update_every = int(os.environ.get("TOXIC_LAGRANGIAN_UPDATE_EVERY", "16"))
    recent = st["_recent_costs"]
    scores: list[float] = []
    for text, help_raw, harm_raw in zip(solution_strs, help_raw_batch, harm_raw_batch):
        help_score = math.tanh((help_raw - 3.0) / 2.0)
        reward = combine(help_score, harm_raw, lam=st["controller"].lam)

        cost = cost_from_harmlessness_score(harm_raw)
        recent.append(cost)
        if len(recent) >= update_every:
            batch_mean_cost = sum(recent) / len(recent)
            st["controller"].update(batch_mean_cost)
            recent.clear()

        penalty = st["diversity"].score_and_update(text)
        scores.append(float(max(-2.0, min(2.0, reward - penalty))))
    return scores
