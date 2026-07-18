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
