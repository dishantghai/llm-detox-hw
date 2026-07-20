# Attempt 3 — Logbook

Fill this in as you go, in the order `GUIDE.md` walks through it. Write your
own numbers and your own words — this file is the actual deliverable of
`attempt_3`, more than any checkpoint is. Where a section asks for a
prediction, write it *before* running the command that would answer it.

Run `python attempt_3/scripts/record_run.py --show` any time you want the
full cross-run comparison table without scrolling back through this file —
this document is for the reasoning and the eyeballed text; the CSV it reads
from (`attempt_3/results/comparison.csv`) is for the numbers side by side.

---

## Environment

- Date:
- GPU (`nvidia-smi` output, first line is enough):
- Docker GPU check passed? (y/n, paste any error):
- Python/venv package versions that matter (`torch`, `transformers`, `peft`, `detoxify`):

---

## Stage 0 — Baseline

**Full eval output** (greedy + sampled table):

```
(paste here)
```

**Five eyeballed completions** (prompt → completion, verbatim):

1.
2.
3.
4.
5.

**Verdict:** does the base model look like a plausible detox-direction
starting point — can it actually produce hostile completions, or is it
already too polite for this exercise to have anything to push against?

---

## Stage 1 — Data

- `wc -l attempt_3/data/dpo.jsonl` / `attempt_3/data/sft.jsonl`:
- Comparison to the ~1,961-pair figure from the original run — matches, or
  meaningfully different (and if different, your best guess why)?
- 15-row eyeball of `sft.jsonl`'s `chosen` side — your fraction that look
  hedgy/evasive vs. substantive:
- **Prediction for Stage 2:** given that fraction, do you expect SFT's
  greedy toxicity to drop a little or a lot? Do you expect the sampled-tail
  (`support_rate`) to move the same direction as greedy?

---

## Stage 2 — SFT

**Prediction (write before running the eval):**

**Full eval output:**

```
(paste here)
```

**Completion uniqueness** (exact_unique_rate, top repeated completions):

**Prediction vs. actual — where were you wrong, and your best guess why:**

**Decision gate:** did SFT do what a first pass at "steer away from
hostility" should do, or is there already a warning sign (toxicity near-zero
*and* uniqueness collapsing together)? Your call, with evidence:

---

## Stage 3 — DPO

**Full eval output:**

```
(paste here)
```

**Completion uniqueness** (exact_unique_rate, top repeated completions):

**SFT vs. DPO, side by side — read the actual completions, not just the
uniqueness percentage.** Did DPO's preference signal make behavior more
diverse and prompt-specific, or entrench what SFT was already doing? Is
there a *softer* templating pattern (same rhetorical move, different exact
words) that the uniqueness number alone wouldn't catch?

**Decision gate:** is this checkpoint good enough to be PPO's reference
policy, or do you already suspect the next reward signal is going to find
and exploit the same narrow mode? Write the prediction now — you'll check it
in Stage 5.

---

## Stage 4 — Reward model

- Held-out pairwise accuracy:
- Mean reward margin:
- Three eyeballed pairs and your read on each (does the RM's ranking match
  what you'd call the less-toxic completion?):

**Generic-vs-substantive probe** (the `TrainedRewardModel.score` check from
`GUIDE.md` §6):

```
(paste the three scores + responses here)
```

**Decision gate:** does the RM score a generic non-answer higher than or
comparably to a substantive one? If so — write your prediction: do you
expect PPO against this RM (Stage 5) to collapse onto a generic template?

---

## Stage 5 — PPO (three reward variants)

### 5a. `inv:detoxify`

- Eval output:
- Uniqueness:
- Step where entropy visibly flattened (from the training log):
- 5-10 eyeballed completions — attractor description, if any (quote it):

### 5b. Your trained RM

- Eval output:
- Uniqueness:
- Step where entropy visibly flattened:
- 5-10 eyeballed completions — attractor description, if any (quote it):

### 5c. `tasks.task8_custom_reward`

- Eval output:
- Uniqueness:
- Step where entropy visibly flattened:
- 5-10 eyeballed completions — attractor description, if any (quote it):

### 5d. Diagnosis gate — answer with evidence from your own run

1. Did toxicity go down across all three variants?
2. Did uniqueness/diversity go down at the same time — by how much, starting
   at what step?
3. Is there a single completion or rhetorical template repeating across
   unrelated prompts? Quote it.
4. Which variant collapsed hardest — does it match your Stage 4 prediction?
5. One sentence: what is the actual global optimum your reward function
   defined, whether or not you intended it to be?

**Only after answering the above**, note what you found on re-reading
`attempt_2/PLAN.md` as a second opinion — where did it agree with your own
diagnosis, where did it differ:

---

## Stage 6 — Your fix

**6.1 Approach chosen** (adopt `attempt_2`'s fix wholesale / adapt it / design
your own) **and why:**

**6.2 Data audit** — numbers, and comparison to `attempt_2`'s 21.3% union
figure:

**6.3 Three-way DPO fix** — eval output, uniqueness, and comparison against
**your own** Stage 3 DPO run (not `attempt_2`'s numbers):

```
(paste here)
```

**6.4 Dual RM + red-team gate** — did the harmlessness-only RM fail the gate
the way `attempt_2`'s did? Is the helpfulness-only RM also unsafe alone
(toxic-but-specific content fooling it)?

**6.5 PPO with the fix — the run `attempt_2` never got to.**

- Eval output:
- Uniqueness:
- Lagrangian `lambda` trajectory (from the state JSON), stabilized or
  diverging:
- 5-10 eyeballed completions:

**Verdict, using the full `record_run.py --show` table:**

1. Did `ppo_fixed`'s uniqueness beat all three Stage 5 variants, or just
   some?
2. Is the Stage 5 attractor gone, reduced, or replaced by a *different*
   collapse mode?
3. Is there a real helpfulness/harmlessness tradeoff visible, or did both
   move together?

---

## Closing writeup

4-6 sentences, your own words:

- What was the actual failure mode you found, described precisely (what
  specifically did the model learn to do, and why was it cheaper than the
  alternative under the objective you gave it)?
- Did your Stage 6 fix work? Partially? Not at all? What's your evidence?
- If you had another full day of compute, what's the single next experiment
  you'd run, and why that one over the alternatives?

---

## Master comparison table

Paste the output of `python attempt_3/scripts/record_run.py --show` here
once all stages are recorded:

```
(paste here)
```
