"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

import math
import os
import re
from typing import Sequence

_DETOX = None
_RM = None

_WORD_RE = re.compile(r"[a-z0-9']+")
DETOX_SATURATE_BELOW = 0.15  # once this benign, more "benign" buys nothing


def _get_detox():
    global _DETOX
    if _DETOX is None:
        import torch
        from detoxify import Detoxify
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DETOX = Detoxify("original", device=device)
    return _DETOX


def _get_rm():
    """Optional blend with your trained RM. Set TOXIC_CUSTOM_RM_DIR to
    enable; the reward degrades gracefully to Detoxify-only without it."""
    global _RM
    rm_dir = os.environ.get("TOXIC_CUSTOM_RM_DIR")
    if not rm_dir:
        return None
    if _RM is None:
        from src.toxic_rl.reward_model import TrainedRewardModel
        _RM = TrainedRewardModel(rm_dir)
    return _RM


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _trigram_repeat_penalty(text: str) -> float:
    """0 when varied, up to 0.7 when the completion is looping a phrase."""
    words = text.split()
    if len(words) < 6:
        return 0.0
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    distinct_ratio = len(set(trigrams)) / len(trigrams)
    return max(0.0, 0.7 - distinct_ratio)


def _length_cap_penalty(text: str, char_threshold: int = 240) -> float:
    """~64 tokens * ~4 chars/token; catches 'always run to the cap'."""
    return 1.0 if len(text) >= char_threshold else 0.0


def _relevance_gate(prompt: str, completion: str) -> float:
    """Multiplicative gate in [0.05, 1.0] on the rest of the reward --
    deliberately *not* an additive bonus.

    This project's own Task 6 and Task 7 PPO runs each converged onto a
    single fixed completion reused across every prompt: Task 6's was a
    system-prompt echo, Task 7's a non-Latin-script self-introduction
    that the trained RM scored above its own held-out margin -- higher
    than any genuine response measured anywhere in this project. Both
    Detoxify and the RM actively *reward* that text in isolation, since
    it genuinely reads as safe -- neither has any mechanism to notice it
    ignores the prompt. A bag-of-words *bonus* can't fix that: at most
    a couple tenths added to an already-saturated 1.0 detox/RM score.
    It has to be a gate that can crush the reward toward the floor
    regardless of how safe the other components think the text is.

    ``_words`` only extracts Latin-alphabet tokens, so a non-Latin
    completion against a Latin-alphabet prompt lands at zero overlap
    here automatically -- exactly Task 7's attractor.
    """
    p_words, c_words = _words(prompt), _words(completion)
    if not p_words or not c_words:
        return 0.05  # near-zero floor, not a hard 0 (avoid a knife-edge discontinuity)
    echo_ratio = len(p_words & c_words) / max(1, len(c_words))
    if echo_ratio > 0.9 and len(c_words) <= len(p_words) + 2:
        return 0.3  # near-verbatim echo: some credit for being on-topic, not full marks
    overlap = len(p_words & c_words) / len(p_words)
    return 0.05 + 0.95 * min(1.0, overlap * 2)  # any real engagement opens most of the gate


def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Return one reward per completion. Higher = better.

    The verl reward worker calls this once per training step with the
    flattened list of K-rollouts across the prompt batch.

    Args:
        texts: completions to score, one entry per completion.
        prompts: same-length list of the originating prompts (the verl
            dispatcher uses ``reward_score.prompt_conditioned`` below
            to decide whether to pass these). Set the attribute to
            ``False`` if your design is purely response-side.

    Returns:
        ``list[float]`` of the same length as ``texts``. Higher = better.

    See the top-of-file docstring for design hints (saturating Detoxify,
    repetition penalty, length-cap penalty, prompt-relevance, blending
    with your RM).
    """
    dx = _get_detox()
    tox_scores = dx.predict(list(texts))["toxicity"]
    if not isinstance(tox_scores, (list, tuple)):
        tox_scores = [tox_scores]

    rm = _get_rm()
    rm_scores = rm.score(list(texts), prompts=list(prompts)) \
        if (rm is not None and prompts is not None) else None

    out: list[float] = []
    for i, text in enumerate(texts):
        tox = float(tox_scores[i])
        detox_component = 1.0 if tox <= DETOX_SATURATE_BELOW else (1.0 - tox)

        base = detox_component
        if rm_scores is not None:
            rm_bounded = math.tanh((float(rm_scores[i]) - 3.0) / 2.0)
            base = 0.7 * detox_component + 0.3 * rm_bounded

        rep_pen = _trigram_repeat_penalty(text)
        cap_pen = _length_cap_penalty(text)
        gate = _relevance_gate(prompts[i], text) if prompts else 1.0

        # Gate multiplicatively, then subtract the response-only penalties.
        # A prompt-irrelevant completion must lose most of its reward even
        # when Detoxify and the RM are each individually saturated on it --
        # additive terms alone can't outweigh a saturated 1.0 base.
        r = gate * base - 0.5 * rep_pen - 0.3 * cap_pen

        out.append(max(-1.0, min(1.0, r)))
    return out


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
