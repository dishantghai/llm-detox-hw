"""Phase 5 — a rolling, RM-independent anti-collapse penalty.

Every PPO run in the original homework collapsed onto a small number of
fixed strings regardless of which reward drove it: 80% system-prompt echo
against ``inv:detoxify`` (Task 6), 45/45 identical against the trained RM
(Task 7), 53% partial-template even with Task 8's relevance gate. A
structural defense that doesn't depend on the reward model or the data being
right is worth having independent of Phase 1's data fix and Phase 4's dual
RM, precisely because those are the two places this project's own analysis
found real, hard-to-fully-close gaps (STAGEWISE_ANALYSIS.md §22's finding
that superficial surface form moves RM scores by several points; this
guide's own Phase 1 finding that a teacher model isn't a safe oracle
either).

**The constraint this works around:** verl's ``custom_reward_function``
calls ``compute_score(data_source, solution_str, ground_truth, extra_info)``
once per completion (``src/toxic_rl/verl_reward.py``) — there's no batch-
level hook to compare all of a step's rollouts against each other directly.
What IS available: the reward module is loaded once per worker process and
its module-level state (``_REWARD`` in ``verl_reward.py``) persists across
every subsequent call in that process. ``RollingDiversityPenalty`` uses
exactly that persistence to approximate a batch-level check: it keeps a
bounded window of recent completions' trigram-shingle fingerprints and
penalizes a new completion that's near-duplicate to many of them — which,
across the dozens of consecutive calls within and across training steps,
catches a policy that's converging onto one string just as reliably as a
true single-step batch comparison would, with a small lag.
"""
from __future__ import annotations

import re
import threading
from collections import deque

_WORD_RE = re.compile(r"[a-z0-9']+")


def _shingles(text: str, n: int = 3) -> frozenset[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class RollingDiversityPenalty:
    """Penalizes near-duplication against a rolling window of recent
    completions, independent of which prompt produced any of them.

    ``window_size``: how many recent completions to compare against. Larger
    catches slower/subtler collapse but costs more per-call compute
    (O(window_size) shingle-set comparisons — cheap, these are small sets).

    ``similarity_threshold``: Jaccard similarity above which two completions
    count as "near-duplicate" for penalty purposes — same threshold family
    as ``eval_lib_v2.completion_uniqueness``'s ``near_dup_jaccard``, kept
    consistent so what the reward penalizes during training matches what
    the eval harness flags afterward.

    ``penalty_scale``: max penalty subtracted from the reward when a
    completion matches close to 100% of the window.
    """

    def __init__(
        self,
        window_size: int = 64,
        similarity_threshold: float = 0.6,
        penalty_scale: float = 1.0,
    ) -> None:
        self.window: deque[frozenset[str]] = deque(maxlen=window_size)
        self.similarity_threshold = similarity_threshold
        self.penalty_scale = penalty_scale
        # verl's reward manager (naive.py's run_single) dispatches
        # compute_score for different completions in the same rollout batch
        # concurrently via a thread-pool executor, and this object is
        # process-level singleton state (see verl_reward_v2.py's `_STATE`).
        # Without a lock, one thread's `sum(... for prior in self.window)`
        # can race another thread's `self.window.append(...)` on the same
        # deque -- CPython raises `RuntimeError: deque mutated during
        # iteration` when that happens (hit for real during the first
        # `dual_lagrangian_langgate` PPO run, attempt_3 Stage 7).
        self._lock = threading.Lock()

    def score_and_update(self, text: str) -> float:
        """Returns a penalty in [0, penalty_scale] — 0 if this completion
        looks nothing like recent ones, growing toward ``penalty_scale`` as
        the fraction of the window it near-duplicates grows. Always records
        the new completion into the window afterward, so the window reflects
        the true recent rollout stream regardless of what the caller does
        with the returned penalty."""
        sh = _shingles(text)
        with self._lock:
            if not self.window:
                self.window.append(sh)
                return 0.0
            near_dup_count = sum(
                1 for prior in self.window if _jaccard(sh, prior) >= self.similarity_threshold
            )
            fraction = near_dup_count / len(self.window)
            self.window.append(sh)
        return self.penalty_scale * fraction
