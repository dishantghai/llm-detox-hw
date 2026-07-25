"""Phase 5 — Lagrangian combination of the helpfulness RM and harmlessness
RM, replacing Task 8's static multiplicative gate
(``tasks/task8_custom_reward.py``) with a coefficient that moves during
training.

Follows `Safe RLHF <https://arxiv.org/abs/2310.12773>`_'s formulation:
treat harmlessness as a **constraint**, not a term to blend by a fixed
weight. Maximize helpfulness reward subject to the harm/cost model staying
under a target threshold, via a Lagrange multiplier updated toward that
constraint:

    reward(x) = help_score(x) - lambda * cost(x)
    lambda    <- max(0, lambda + step_size * (cost_ema - cost_target))

where ``cost(x)`` is the harmlessness RM's score rescaled so *higher = more
harmful* (the harmlessness RM's own training polarity is the opposite —
higher = more chosen/safe-like — so we negate it here once, at the boundary,
rather than propagating a second sign convention through the rest of the
pipeline).

Why this instead of Task 8's gate: a static gate is a design-time guess at
the right harmlessness/helpfulness trade-off. It can't adapt if the policy's
actual harm rate is already low (in which case it's needlessly suppressing
helpfulness) or still too high (in which case it's not suppressing enough).
The Lagrange multiplier is explicitly an online control loop against a
*target* harm rate, so the trade-off tracks the policy's actual behavior
during training instead of a single choice frozen in before training starts.

Practical constraint this module works around: verl's ``custom_reward_function``
dispatches ``compute_score(...)`` one completion at a time (see
``src/toxic_rl/verl_reward.py``), with no built-in hook to run an update step
between training iterations. ``LagrangianController`` persists its state
(``lambda`` and a running cost EMA) to a small JSON file on disk so it
survives across the many independent ``compute_score`` calls within a
process, and exposes an ``update()`` a companion driver script can call once
per PPO step (see ``verl_reward_v2.py``'s ``TOXIC_LAGRANGIAN_STATE_PATH``).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LagrangianState:
    lam: float = 0.0
    cost_ema: float = 0.0
    n_updates: int = 0


class LagrangianController:
    """Persists ``LagrangianState`` to ``state_path`` so it survives across
    the many separate ``compute_score`` process calls verl makes."""

    def __init__(
        self,
        state_path: str,
        cost_target: float = 0.0,
        step_size: float = 0.02,
        ema_beta: float = 0.9,
        lam_max: float = 20.0,
    ) -> None:
        self.state_path = Path(state_path)
        self.cost_target = cost_target
        self.step_size = step_size
        self.ema_beta = ema_beta
        self.lam_max = lam_max
        self.state = self._load()

    def _load(self) -> LagrangianState:
        if self.state_path.exists():
            return LagrangianState(**json.loads(self.state_path.read_text()))
        return LagrangianState()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state)))

    @property
    def lam(self) -> float:
        return self.state.lam

    def update(self, batch_mean_cost: float) -> float:
        """Call once per PPO step (or once per reward-function call, as a
        cheap online approximation) with the mean cost observed in the most
        recent batch. Returns the (possibly updated) lambda.

        Cost EMA smooths over the noisy per-batch estimate before comparing
        to the target, so a single unlucky batch doesn't swing lambda hard.
        """
        s = self.state
        s.cost_ema = self.ema_beta * s.cost_ema + (1 - self.ema_beta) * batch_mean_cost
        s.lam = max(0.0, min(self.lam_max, s.lam + self.step_size * (s.cost_ema - self.cost_target)))
        s.n_updates += 1
        self._save()
        return s.lam


def cost_from_harmlessness_score(harmlessness_score: float, mu: float = 3.0, sigma: float = 2.0) -> float:
    """Rescale the harmlessness RM's raw score (higher = more chosen/safe-
    like, per its own training polarity — see reward_model.py's docstring)
    into a cost in roughly [-1, 1] where higher = MORE harmful, using the
    same tanh-bounding this repo already relies on
    (src/toxic_rl/verl_reward.py's ``composite_rm``) to tame an otherwise
    unbounded RM score. Negated so it reads as a cost, not a reward."""
    return -math.tanh((harmlessness_score - mu) / sigma)


def combine(
    help_score: float,
    harmlessness_score: float,
    lam: float,
    mu: float = 3.0,
    sigma: float = 2.0,
) -> float:
    """reward = help_score - lambda * cost(harmlessness_score).

    ``help_score`` should already be on a comparable scale (e.g. also
    tanh-bounded via the same helper, using the helpfulness RM's own
    calibration constants) — see GUIDE.md Phase 5 for how the two RMs'
    raw-score distributions are calibrated against each other before this
    function is wired into ``verl_reward_v2.py``.
    """
    cost = cost_from_harmlessness_score(harmlessness_score, mu=mu, sigma=sigma)
    return help_score - lam * cost


# --------------------------------------------------------------------------- #
# Multi-constraint extension (Stage 12b, scoped after LOGBOOK.md's Stage 11   #
# regression and rm_ontopic's introduction).                                  #
# --------------------------------------------------------------------------- #
#
# Everything above this point (``LagrangianController``, ``combine``,
# ``cost_from_harmlessness_score``) is left completely unmodified and still
# used unchanged by every ``dual_lagrangian*`` spec through
# ``dual_lagrangian_langgate_relevance_ontopic`` (Stage 12a) -- every prior
# run stays reproducible byte-for-byte from the same code path. This section
# is new, additive code for a genuinely different mechanism, not an edit to
# the existing one.
#
# The gap this closes, stated explicitly across this project's own commits
# since Stage 9 (see LOGBOOK.md, citing Moskovitz et al., *Confronting
# Reward Model Overoptimization with Constrained RLHF*) but never actually
# built until now: every reward spec from Stage 8 onward has exactly ONE
# adaptive term (harmlessness, via ``LagrangianController``) and a pile of
# FIXED-WEIGHT additive penalties for everything else (language, repetition,
# line-repeat, compression-repeat, relevance, and now the fixed-weight
# ``_ontopic_penalty`` from Stage 12a). A fixed weight can't tell the
# difference between "this axis is already fine, ease off" and "this axis
# is still being violated constantly, push harder" -- it's a single
# design-time guess, the same critique ``combine()``'s own docstring makes
# of Task 8's static gate. Stage 11's actual failure (HTML-tail padding
# slipping past every fixed-weight structural gate while `lambda` sat at
# 0.73, still well below where it would need to be to notice) is a live
# instance of this: optimization pressure closed off on one axis by a
# stronger cost signal has nowhere to go but into whichever OTHER axis
# still has slack, and a fixed weight offers no counter-pressure back.
#
# Promoting harmlessness AND on-topicness to two INDEPENDENT Lagrangian
# constraints -- not one shared multiplier, two separate ones with their
# own cost, target, and update -- means neither axis's pressure can
# silently borrow room from the other's fixed budget. The remaining
# structural gates (language/repetition/line-repeat/compression-repeat/
# relevance) are deliberately NOT promoted in this pass: they're
# pattern-match checks with a near-zero false-positive rate by
# construction (validated per-gate before each was ever trained against,
# see each gate's own docstring), unlike harmlessness and on-topicness
# which are both continuous RM judgments where a fixed threshold has a
# real, measured false-positive/false-negative rate (see
# ``verl_reward_v2.py``'s ``_ONTOPIC_MU``/``_ONTOPIC_SIGMA`` comment for
# rm_ontopic's own case) -- promoting a near-binary pattern gate to a
# Lagrangian constraint has much less to gain than promoting a genuinely
# graded classifier signal. Promoting the rest is future work, not
# something this pass silently skipped.


@dataclass
class ConstraintState:
    lam: float = 0.0
    cost_ema: float = 0.0
    n_updates: int = 0


class MultiLagrangianController:
    """N independent Lagrange multipliers, one per named constraint, e.g.
    ``{"harmlessness": ..., "ontopic": ...}`` -- generalizes
    ``LagrangianController`` (which hardcodes exactly one) the same way
    ``combine_multi``/``cost_from_rm_score`` below generalize ``combine``/
    ``cost_from_harmlessness_score``.

    Each constraint gets its own ``cost_target``, its own EMA, and its own
    ``lambda``, updated independently: constraint A's multiplier only grows
    when constraint A's own cost exceeds constraint A's own target,
    regardless of what constraint B is doing. This is the structural
    difference from a single shared multiplier (or worse, a fixed weight)
    -- neither constraint's pressure can leak into or borrow room from the
    other's budget, addressing the failure mode named in this module's
    docstring above.

    Persists all constraints' state in ONE JSON file (a dict keyed by
    constraint name, each value the same shape ``LagrangianController``
    already persists) rather than one file per constraint, purely for
    operational convenience -- one state path to pass around, one file to
    inspect after a run, same as every other ``TOXIC_LAGRANGIAN_STATE_PATH``
    convention in this codebase.
    """

    def __init__(
        self,
        state_path: str,
        constraint_names: list[str],
        cost_targets: dict[str, float] | None = None,
        step_size: float = 0.02,
        ema_beta: float = 0.9,
        lam_max: float = 20.0,
    ) -> None:
        self.state_path = Path(state_path)
        self.constraint_names = list(constraint_names)
        self.cost_targets = cost_targets or {n: 0.0 for n in self.constraint_names}
        self.step_size = step_size
        self.ema_beta = ema_beta
        self.lam_max = lam_max
        self.state: dict[str, ConstraintState] = self._load()

    def _load(self) -> dict[str, ConstraintState]:
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text())
            return {n: ConstraintState(**raw.get(n, {})) for n in self.constraint_names}
        return {n: ConstraintState() for n in self.constraint_names}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({n: asdict(s) for n, s in self.state.items()}))

    def lam(self, name: str) -> float:
        return self.state[name].lam

    def update(self, name: str, batch_mean_cost: float) -> float:
        """Same update rule as ``LagrangianController.update``, applied to
        just the named constraint's own state -- call once per constraint
        per batch (not once total), each with that constraint's own
        observed mean cost."""
        s = self.state[name]
        target = self.cost_targets[name]
        s.cost_ema = self.ema_beta * s.cost_ema + (1 - self.ema_beta) * batch_mean_cost
        s.lam = max(0.0, min(self.lam_max, s.lam + self.step_size * (s.cost_ema - target)))
        s.n_updates += 1
        self._save()
        return s.lam


def cost_from_rm_score(raw_score: float, mu: float, sigma: float) -> float:
    """Generalizes ``cost_from_harmlessness_score`` to any RM whose raw
    output has been calibrated to a ``(mu, sigma)`` decision boundary via
    the same tanh-bounding convention this repo uses throughout (e.g.
    ``verl_reward_v2._ONTOPIC_MU``/``_ONTOPIC_SIGMA`` for rm_ontopic) --
    higher raw score = more chosen-like/on-topic-like by that RM's own
    training polarity, rescaled here into a cost in roughly ``[-1, 1]``
    where higher = MORE of a violation, negated once at this boundary for
    the same reason ``cost_from_harmlessness_score``'s own docstring
    gives."""
    return -math.tanh((raw_score - mu) / sigma)


def combine_multi(help_score: float, costs: dict[str, float], lambdas: dict[str, float]) -> float:
    """``reward = help_score - sum(lambda_i * cost_i)`` over every named
    constraint -- the multi-constraint generalization of ``combine``.
    ``costs`` and ``lambdas`` must share the same keys (constraint names)."""
    return help_score - sum(lambdas[name] * costs[name] for name in costs)
