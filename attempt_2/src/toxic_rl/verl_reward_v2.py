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
import re
import threading
from collections import Counter
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


def _non_latin_tail_penalty(
    text: str,
    tail_fraction: float = 0.3,
    min_letters: int = 5,
    latin_ratio_threshold: float = 0.5,
    penalty: float = 3.0,
) -> float:
    """Windowed version of ``_non_latin_penalty``, checking only the
    trailing ``tail_fraction`` of the completion instead of the whole
    string.

    Built after the ``dual_lagrangian_langgate_relevance`` PPO run
    (attempt_3, Stage 8) closed the templating-collapse failure but
    surfaced a new one: 31-37% of completions across eval surfaces opened
    with clean, genuine, on-topic Latin-script content and then degraded
    into a non-Latin repetition-loop tail in the back half. The
    whole-string ``_non_latin_penalty`` averages a clean start against a
    garbage tail and often lands above ``latin_ratio_threshold`` anyway --
    confirmed directly (not assumed) via
    ``attempt_3/scripts/measure_tail_degeneration.py`` against the actual
    Stage 8 checkpoint's outputs, matching the LOGBOOK's own manual count
    exactly (17/55 tail-degenerate, 4/55 fully non-Latin on the OOD set).

    Decode-time fixes were tried first and rejected (see
    ``eval_lib.greedy_generate``'s docstring and
    ``non_latin_logits_processor.py``): ``repetition_penalty`` made the
    failure worse, and script-targeted logit suppression looked like a
    complete fix under greedy/OOD eval but, checked against the tracked
    adversarial slices under K=16 sampling, turned out to remove an
    accidental safe release valve -- Detoxify scores non-Latin gibberish
    as harmless regardless of content, so blocking it at *eval* time (on
    a policy trained without that constraint) exposed genuinely more
    hostile content underneath, not less. That failure mode is specific
    to patching a frozen, already-trained policy's decoding step; it does
    not apply to a reward term the policy is actually optimized against
    during training, which is why this fix lives here instead.

    verl 0.8.0's PPO rollout (vLLM-backed) also has no supported
    passthrough for a custom decode-time token suppressor -- confirmed
    against the installed version (0.23.1.dev0): ``RolloutConfig`` only
    forwards temperature/top_k/top_p/repetition_penalty/n, and vLLM's own
    ``SamplingParams`` dropped arbitrary ``logits_processors`` callables
    in favor of ``logit_bias``/``bad_words``, neither of which verl's
    request-construction code exposes a config path for -- so there is no
    verl-config-only way to suppress non-Latin tokens during rollout
    sampling itself, only a reward-side one.

    Same character-based, no-new-dependencies design as the original.
    """
    tail_len = max(1, int(len(text) * tail_fraction))
    tail = text[-tail_len:]
    letters = [c for c in tail if c.isalpha()]
    if len(letters) < min_letters:
        return 0.0
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


_SEGMENT_SPLIT_RE = re.compile(r"[\n]+|(?<=[.!?])\s+")


def _normalize_segment_for_repeat_check(segment: str) -> str:
    """Strip non-ASCII characters and collapse whitespace, so a short
    foreign-script filler token wedged between repeats (see
    ``_line_repeat_penalty``) can't hide an otherwise-identical segment
    from an exact-match comparison."""
    ascii_only = "".join(c for c in segment if c.isascii())
    return " ".join(ascii_only.split()).lower()


def _line_repeat_penalty(
    text: str, min_segment_words: int = 3, min_repeats: int = 3, penalty: float = 3.0,
) -> float:
    """Flat penalty when a sentence/line (typically the prompt or system
    prompt) repeats verbatim, aside from a short filler token wedged
    between repeats, 3+ times inside one completion.

    Added after Stage 9 (attempt_3, see LOGBOOK.md): closing the
    non-Latin tail-degeneration failure with ``_non_latin_tail_penalty``
    surfaced a new failure at similar prevalence (22-23% of completions)
    that specifically evades ``_repetition_penalty``'s word-trigram-
    distinctness check -- a short, space-free foreign-script filler token
    (e.g. ``왁``, ``>NN``, ``uada``) inserted between repeats of the
    prompt or system prompt shifts trigram boundaries enough to inflate
    ``distinct_ratio`` without eliminating the repetition a human reader
    recognizes immediately. Confirmed directly (not assumed) against the
    actual Stage 9 failure examples via
    ``attempt_3/scripts/measure_line_repetition.py``, which independently
    reproduces the LOGBOOK's manual count (12/55 on OOD, 17/75 on
    tracked, give or take a few borderline cases this segment-level check
    also catches) -- and confirmed the *graded* version of this penalty
    first attempted (scaling 0.3-1.5 with repeat count, mirroring
    ``_repetition_penalty``'s own shape) lands in the same too-weak
    magnitude range diagnosed as the Stage 9 bug in the first place: at
    the 3-repeat threshold it only adds ~0.3, barely above
    ``_repetition_penalty``'s own diluted score on these exact examples.
    Made flat and large instead, ``penalty=3.0`` matching
    ``_non_latin_penalty``'s own reasoning -- it must dominate regardless
    of how safe/on-topic the RMs or relevance gate score the surrounding
    text, since verbatim-repeating the prompt scores near-perfectly under
    both.

    Segment-level and normalization-based rather than n-gram-based on
    purpose: split on line/sentence boundaries, strip non-ASCII
    characters from each segment before comparing, so the filler token
    can't hide a repeat by shifting it out of the n-gram window the way
    it does against the trigram check. Kept as a separate, additive term
    alongside the unmodified ``_repetition_penalty`` (rather than folded
    into it) for the same reproducibility reason every other gate in this
    file is versioned separately: Stage 9's run stays reproducible
    byte-for-byte, and Stage 10 can be A/B'd directly against it.
    """
    segments = [s for s in _SEGMENT_SPLIT_RE.split(text) if s.strip()]
    normalized = [_normalize_segment_for_repeat_check(s) for s in segments]
    counts = Counter(n for n in normalized if len(n.split()) >= min_segment_words)
    if not counts:
        return 0.0
    top_count = counts.most_common(1)[0][1]
    return penalty if top_count >= min_repeats else 0.0


def _relevance_penalty(prompt: str, text: str, max_penalty: float = 2.5) -> float:
    """Penalize completions that don't actually engage with the prompt,
    using ``tasks/task8_custom_reward.py``'s existing bag-of-words
    prompt/completion-overlap gate -- the one mechanism already built in
    this project that measures "did this respond to what was asked"
    rather than a property of the response text alone.

    Built after the first ``dual_lagrangian_langgate:`` PPO run (attempt_3,
    Stage 7) closed the language-collapse and repetition-loop failure
    modes and immediately found a third one: 85-94% of completions across
    every eval surface converged onto one fluent, on-language, non-
    repeating rhetorical template ("It's important to X... by doing Y...")
    applied regardless of prompt content. Detoxify and both RMs score the
    response in isolation and have no way to see that -- this does.

    ``task8_custom_reward._relevance_gate`` returns a *multiplicative*
    factor in [0.05, 1.0] (1.0 = clearly on-topic), designed for a reward
    that's already bounded in [0, 1] before gating. This reward isn't --
    ``combine()`` can return negative values -- and multiplying a negative
    reward by a small fraction would make it *less* negative, rewarding
    irrelevance. So the gate is converted into an additive penalty here
    instead, the same shape as the language/repetition gates above:
    ``penalty = max_penalty * (1 - gate)``, 0 when fully on-topic, up to
    ``max_penalty`` when the gate is at its floor.
    """
    from tasks.task8_custom_reward import _relevance_gate

    gate = _relevance_gate(prompt, text)
    return max_penalty * (1.0 - gate)


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


def _dual_lagrangian_langgate_relevance_score(text: str, prompt: str, spec: str) -> float:
    """Same as ``_dual_lagrangian_langgate_score``, plus ``_relevance_penalty``.

    New spec: ``TOXIC_REWARD=dual_lagrangian_langgate_relevance:<help_rm_dir>:<harm_rm_dir>``
    -- separate spec/state-key, same reasoning as the langgate spec's own
    docstring: keeps the langgate-only run reproducible byte-for-byte and
    lets this be A/B'd directly against it.
    """
    from attempt_2.src.toxic_rl.dual_reward_combiner import combine, cost_from_harmlessness_score

    st = _get_dual_lagrangian_state(spec, state_key="dual_lagrangian_langgate_relevance")
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
    rel_penalty = _relevance_penalty(prompt, text)
    return max(-2.0, min(2.0, reward - penalty - lang_penalty - rep_penalty - rel_penalty))


def _dual_lagrangian_langgate_relevance_v2_score(text: str, prompt: str, spec: str) -> float:
    """Same as ``_dual_lagrangian_langgate_relevance_score``, but swaps the
    whole-string ``_non_latin_penalty`` for the windowed
    ``_non_latin_tail_penalty`` (see its docstring for the Stage 8 evidence
    and why the fix belongs here rather than at decode time).

    New spec: ``TOXIC_REWARD=dual_lagrangian_langgate_relevance_v2:<help_rm_dir>:<harm_rm_dir>``
    -- separate spec/state-key, same reasoning as every prior variant in
    this file: keeps Stage 8's run reproducible byte-for-byte and lets
    Stage 9 be A/B'd directly against it.
    """
    from attempt_2.src.toxic_rl.dual_reward_combiner import combine, cost_from_harmlessness_score

    st = _get_dual_lagrangian_state(spec, state_key="dual_lagrangian_langgate_relevance_v2")
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
    lang_penalty = _non_latin_tail_penalty(text)
    rep_penalty = _repetition_penalty(text)
    rel_penalty = _relevance_penalty(prompt, text)
    return max(-2.0, min(2.0, reward - penalty - lang_penalty - rep_penalty - rel_penalty))


def _dual_lagrangian_langgate_relevance_v3_score(text: str, prompt: str, spec: str) -> float:
    """Same as ``_dual_lagrangian_langgate_relevance_v2_score``, plus
    ``_line_repeat_penalty`` alongside the existing word-trigram
    ``_repetition_penalty`` (see its docstring for the Stage 9 evasion
    this closes, and why it's an added independent term rather than a
    change to the existing gate).

    New spec: ``TOXIC_REWARD=dual_lagrangian_langgate_relevance_v3:<help_rm_dir>:<harm_rm_dir>``
    -- separate spec/state-key, same reasoning as every prior variant in
    this file: keeps Stage 9's run reproducible byte-for-byte and lets
    Stage 10 be A/B'd directly against it.
    """
    from attempt_2.src.toxic_rl.dual_reward_combiner import combine, cost_from_harmlessness_score

    st = _get_dual_lagrangian_state(spec, state_key="dual_lagrangian_langgate_relevance_v3")
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
    lang_penalty = _non_latin_tail_penalty(text)
    rep_penalty = _repetition_penalty(text) + _line_repeat_penalty(text)
    rel_penalty = _relevance_penalty(prompt, text)
    return max(-2.0, min(2.0, reward - penalty - lang_penalty - rep_penalty - rel_penalty))


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
    if spec.startswith("dual_lagrangian_langgate_relevance_v3:"):
        return float(_dual_lagrangian_langgate_relevance_v3_score(solution_str, prompt_text, spec))
    if spec.startswith("dual_lagrangian_langgate_relevance_v2:"):
        return float(_dual_lagrangian_langgate_relevance_v2_score(solution_str, prompt_text, spec))
    if spec.startswith("dual_lagrangian_langgate_relevance:"):
        return float(_dual_lagrangian_langgate_relevance_score(solution_str, prompt_text, spec))
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
