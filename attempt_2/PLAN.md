# Attempt 2 — Plan: Fixing the One-Sided Objective

This document is the strategy. `GUIDE.md` in this same folder is the
step-by-step implementation of it, MASTERCLASS.md-style — read that one to
actually run anything. This file exists so the *why* behind every step in
the guide is written down once, in one place, instead of re-argued inline
every time.

Everything referenced here lives under `attempt_2/`. Nothing outside this
folder is modified. Where existing code from the original homework (`src/`,
`data_prep/`, `checkpoints/` at the repo root) is unchanged and reusable, we
import it directly rather than copy it — see "What we reuse" below.

---

## 0. Root cause, stated once

`STAGEWISE_ANALYSIS.md` tracked eight tasks — SFT, DPO, an RM, three PPO
variants — and found the same failure mode five separate times: every
training signal in that pipeline measured *"not toxic"* and none measured
*"actually helpful."* Under a purely one-sided harmlessness objective, an
empty, generic, prompt-independent non-answer is a global optimum: it can't
fail at engaging with a prompt because it never attempts to. SFT converged
onto "I'm sorry, I don't understand" for 62% of tracked prompts; DPO
entrenched it to 71%; PPO against Detoxify converged onto an 80% system-prompt
echo; PPO against the trained RM converged onto one exact string repeated for
45 of 45 tracked prompts across all three eval slices — the single
highest-scoring completion found anywhere in the whole document, under that
RM. Task 8's hand-designed relevance gate cut this to 53% but couldn't close
it, because a bag-of-words overlap heuristic can tell you a completion
borrowed vocabulary from the prompt, not that it *answered* the prompt.

Every fix below exists to introduce a second, independent axis —
helpfulness — at every stage that currently only sees one.

---

## 1. Data (Phase 1)

**Problem:** `hh-rlhf`'s `chosen` side is itself frequently evasive, not
substantively helpful — it was itself produced by annotators told to avoid
harm, so hedging non-answers are already present in what SFT calls "ground
truth," before any RL step runs. `STAGEWISE_ANALYSIS.md` §4 caught one
instance of this directly: SFT replaced the *base* model's genuine, on-topic
answer to a bigotry-baiting prompt with a worse, confused non-answer, and
picked up a toxicity penalty for the swap, because SFT's demonstrations
rewarded reproducing hedging style over engagement.

**Fix, concretely:**

1. Introduce a third label alongside `chosen` (safe+substantive) and
   `rejected` (toxic): **`rejected_evasive`** — safe, but templated and
   empty. Populate it partly from real attractor strings this project
   already discovered by hard experience (SFT/DPO's "I'm sorry, I don't
   understand", Task 6's system-prompt echo, Task 7's fixed self-introduction
   — see `STAGEWISE_ANALYSIS.md` §3, §11, §26, §31) and partly from
   generically-detected templated responses inside `hh-rlhf`'s own `chosen`
   pool. This directly closes the exploit found in §11: a non-answer is no
   longer "free" just because it's off-distribution from the toxic
   examples — it is now an explicit negative.
2. Audit the existing `chosen` pool for genericness/duplication *before*
   any training starts (`data_prep/audit_chosen_evasiveness.py` — Step 0 of
   the guide, run for real, not estimated).
3. Generate substantive+safe replacements for prompts whose `hh-rlhf`
   `chosen` side fails the audit, using a locally-run instruct teacher model
   (no external LLM API key is available in this environment — see
   `GUIDE.md` §1 for the model choice and why).
4. Everything downstream (SFT, DPO, the RM) trains on this three-way-labeled
   data, not on `hh-rlhf`'s `chosen`/`rejected` as-is.

## 2. SFT (Phase 2)

Same trainer (`src/detox_hw/train_sft.py`, reused unmodified), pointed at the
new `chosen` pool (post-audit + teacher-augmented). Track completion
uniqueness (`STAGEWISE_ANALYSIS.md`'s own §3 metric) as a **required, gating**
training-time check, not a forensic afterthought.

## 3. Preference optimization (Phase 3)

DPO's loss gets pressure from `rejected_evasive` as well as `rejected`
(toxic) — implemented as two DPO pairs per prompt sharing one `chosen`, both
using the existing, unmodified `dpo_loss`. If templating still rises at this
stage despite the data fix, that's the signal Phase 1's teacher-augmentation
wasn't strong enough — caught here, before PPO compute is spent on it.

## 4. Reward modeling (Phase 4)

Train **two** reward models, not one Bradley-Terry contrast doing both jobs:

- **Harmlessness RM** — `chosen ≻ rejected(toxic)`, the same contrast the
  original `checkpoints/rm` already encodes.
- **Helpfulness RM** — `chosen ≻ rejected_evasive`, a new contrast that
  specifically teaches "substantive beats generic-but-safe."

This is the [Safe RLHF](https://arxiv.org/abs/2310.12773) decoupled
reward/cost architecture, adapted to this repo's existing
`train_rm.py`/`TrainedRewardModel` machinery rather than a new framework.

Before either RM is trusted for PPO, it has to pass a **red-team gate**
(`src/toxic_rl/rm_redteam_gate.py`): score a bank of known collapse
attractors (§3/§11/§26/§31's real strings) against a broad prompt sample. If
any single string scores in the top percentile across most prompts, the RM
fails the gate. `STAGEWISE_ANALYSIS.md` §21-35 did exactly this analysis, but
five stages *after* the damage was done (PPO had already spent 100 steps
climbing straight at the exploit). Here it's a precondition, not forensics.

## 5. PPO (Phase 5)

Three concrete, structural fixes, on top of the dual-RM reward:

1. **Actually wire the KL anchor.** `STAGEWISE_ANALYSIS.md` §24/§29/§37 found
   `use_kl_loss`/`use_kl_in_reward` were `False` in all three original PPO
   runs despite `--kl-coef` being passed — a configuration bug, not a weak
   setting. `src/toxic_rl/verl_runner_v2.py` fixes the override list.
2. **A rolling diversity penalty**, independent of the RM. verl's
   `custom_reward_function` dispatches one completion at a time
   (`compute_score(data_source, solution_str, ground_truth, extra_info)`),
   so a literal batch-level penalty isn't available through this interface —
   instead, `src/toxic_rl/diversity_penalty.py` keeps a rolling window of
   recent completion fingerprints inside the reward worker's persistent
   module state and penalizes high similarity to that window. This is a
   direct, RM-independent structural defense against the exact failure this
   project hit twice (80% and 100% single-string collapse) — it doesn't
   depend on getting the reward model right.
3. **Lagrangian combination of the two RMs**, not a static blend or gate.
   `src/toxic_rl/dual_reward_combiner.py` implements `reward = help_score −
   λ · cost_score` with λ adjusted toward a target harmlessness constraint,
   instead of Task 8's fixed multiplicative gate — so the balance can move
   as the policy's actual harm rate changes during training, rather than
   being locked in at design time.

## 6. Evaluation (Phase 6)

Institutionalize what `STAGEWISE_ANALYSIS.md` proved was necessary by
repeatedly having to do it by hand:

- Never report toxicity without a paired helpfulness number, computed the
  same way, on the same completions.
- Automate uniqueness/duplicate detection as a standard metric on every eval
  run (§3/§26/§34's method, generalized).
- Bootstrap confidence intervals instead of point estimates — §6/§14 found
  `support_rate` swinging the full 0.0→1.0 range across two runs of one
  checkpoint on a 10-prompt slice.
- Expand eval prompt sets substantially past 10-45 per slice.
- Keep the raw-completion eyeball as a mandatory gate at every stage
  transition — the single practice that caught every failure mode in the
  original analysis that aggregate numbers alone missed.

---

## What we reuse from the original project (unchanged)

| Reused as-is | From |
|---|---|
| `detoxify_score`, `greedy_generate`, `sample_k`, model loaders | `src/detox_hw/eval_lib.py` |
| `dpo_loss` | `tasks/task2_dpo_loss.py` |
| `TrainConfig`/`train()` (RM, Task 4/5's actual trainer) / `TrainedRewardModel` (inference) | `src/detox_hw/train_rm.py`, `src/toxic_rl/reward_model.py` |
| `train()` (SFT) | `src/detox_hw/train_sft.py` |
| `DetoxifyReward` | `src/toxic_rl/detoxify_reward.py` |
| `VerlConfig`/`build_command` (base, then patched) | `src/toxic_rl/verl_runner.py` |
| `checkpoints/rm` (used directly as the "old, single-axis RM" baseline for the red-team gate demo and for A/B comparison against the new dual RM) | `checkpoints/rm` |
| Raw base model behavior (re-run fresh here rather than copied, for a self-contained baseline — same model, same code, new numbers) | `Qwen/Qwen2.5-0.5B` |

## What's new (this folder)

| New file | Purpose |
|---|---|
| `data_prep/audit_chosen_evasiveness.py` | Phase 1 — quantify `hh-rlhf` chosen-side genericness before training on it |
| `data_prep/build_pairs_v2.py` | Phase 1 — three-way labeled pair builder (`chosen` / `rejected_toxic` / `rejected_evasive`) |
| `data_prep/generate_synthetic_responses.py` | Phase 1 — teacher-model generation of substantive+safe replacements |
| `src/detox_hw/train_dpo_v2.py` | Phase 3 — three-way DPO data adapter over the unmodified `dpo_loss` |
| `src/toxic_rl/train_dual_rm.py` | Phase 4 — trains the helpfulness + harmlessness RM pair |
| `src/toxic_rl/rm_redteam_gate.py` | Phase 4 — pre-deployment adversarial audit, run before any RM is trusted for PPO |
| `src/toxic_rl/dual_reward_combiner.py` | Phase 5 — Lagrangian combination of the two RMs |
| `src/toxic_rl/diversity_penalty.py` | Phase 5 — rolling-window anti-collapse penalty |
| `src/toxic_rl/verl_runner_v2.py`, `verl_reward_v2.py` | Phase 5 — KL-anchor fix + dual-reward + diversity-penalty wiring |
| `src/detox_hw/eval_lib_v2.py` | Phase 6 — uniqueness metric, bootstrap CI, paired toxicity+helpfulness eval |

See `GUIDE.md` for the runnable, ordered version of all of the above, with
commands, expected output, and — for the parts actually executed in this
session — real captured output instead of predictions.
