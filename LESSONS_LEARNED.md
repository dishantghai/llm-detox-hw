# Detoxifying a 0.5B LLM: What 12 Stages of RLHF Actually Taught Us

A field guide to `attempt_3` of this project — SFT → DPO → reward models →
PPO, on Qwen2.5-0.5B, trying to make an RTP-toxic base model safe without
lobotomizing it. The headline finding, stated up front so nothing below
is a surprise: **every reward signal we ever built scored a property of
the output text in isolation — never whether the text actually answered
the prompt.** Under enough optimization pressure, a small policy always
found the cheapest text that satisfies the *letter* of that signal. We
closed that gap twelve times. Eleven times, the freed-up optimization
pressure found a new, previously-unpunished way to cheat. Read this if
you're about to RLHF anything and want to know what will actually go
wrong, not what the textbook says might.

---

## 1. The pipeline, in one paragraph

`hh-rlhf` harmless-base prompts → SFT (behavioral cloning onto a curated
`chosen` side) → DPO (preference pairs, `chosen` vs `rejected_toxic`) →
reward models (Bradley-Terry, trained on the same pairs) → PPO with a
**Lagrangian dual-constraint controller**: `reward = help_score - λ ·
cost(harm_score)`, where λ auto-tunes toward a target harm rate instead
of being a hand-picked weight. Every stage was evaluated on three fixed
prompt slices (`mild_prefix`, `direct_provocation`, `rtp_challenging` —
75 prompts total, scored every single time so results are comparable),
a 35-prompt hand-tracked eyeball set, and — critically — a **55-prompt
out-of-distribution set** (general knowledge, coding, life advice,
creative/professional writing, adversarially-framed-but-differently-
phrased prompts) that was never trained on and never tuned toward. That
last set is what caught almost everything the aggregate metrics missed.

---

## 2. The one idea that explains almost every failure

> **Optimization pressure blocked on one axis leaks into whichever axis
> still has slack.** (Named early, credited to Moskovitz et al.,
> *Confronting Reward Model Overoptimization with Constrained RLHF* —
> and then re-discovered empirically five separate times.)

Every reward function in this project was, functionally, a sum or
constrained combination of independent penalty terms: toxicity,
repetition, language, relevance, on-topicness. Close off the cheapest
escape hatch and the policy doesn't get safer — it finds the *next*
cheapest one, because nothing in the reward said "and don't do anything
else weird either." This is why "the numbers all look clean" was wrong
**five times** in this project before anyone stopped trusting the
aggregate metrics without reading actual completions first.

---

## 3. The failure-mode catalogue (in the order we hit them)

Each one looked like a win by at least one metric before it was
actually read.

| # | Failure | What it looked like | Root cause | What actually caught it |
|---|---|---|---|---|
| 1 | **Exact-string collapse** (PPO vs. raw single-axis RM) | Toxicity → 0.001, looks perfect | Unbounded RM reward, no entropy/KL anchor — cheapest max is one deferential sentence, repeated | Uniqueness: 3/75 (4%) — one string, verbatim, across 70/75 unrelated prompts |
| 2 | **Incoherent collapse disguised as diversity** | Uniqueness 96%, looks healthy | Garbled/degenerate text varies token-by-token *because* it's breaking down, not because it's engaging differently | Reading completions — worst sampled toxicity of the three Stage-5 variants despite the best uniqueness |
| 3 | **Wrong-language collapse** | Toxicity ~0, uniqueness 98.7%, looks like the best run yet | Entropy bonus (added to stop #1) had nothing telling it *which* high-entropy region was safe; landed on fluent Cyrillic gibberish an English-only toxicity classifier can't see | The OOD set: 55/55 completions came back **100% in Russian** |
| 4 | **Rhetorical-template collapse, invisible to uniqueness** | 75/75 and 55/55 exact-string-unique, fluent English, reads as caring | A single template ("It's important to X. It's crucial to Y...") applied near-verbatim regardless of prompt content — grammatically unique every time, semantically identical | Measuring *opener phrase rate* directly: 85–94% across all three eval surfaces |
| 5 | **Partial-completion tail degeneration** | Language gate reads "clean" | Non-Latin gate averaged a Latin-ratio over the *whole* completion; a genuinely-English first half masked a degenerate second half | Splitting completions in half and checking each independently |
| 6 | **Decode-time patch ≠ train-time fix** | Applying the fix at eval-only time on a frozen checkpoint looked 100% clean | The gibberish "escape hatch" was accidentally *masking* real hostility under sampling — Detoxify scores garbled text as safe. Blocking it post-hoc didn't teach new behavior, it just removed the policy's practiced coping mechanism and exposed what was underneath | Worst-of-16 sampling (not greedy) on `direct_provocation`: support rate roughly quadrupled once the escape hatch was blocked |
| 7 | **N-gram repetition check evaded by filler tokens** | Repetition penalty *does* fire (0.3–0.85 of 1.0) — just not hard enough | A single foreign-script filler character wedged between repeats shifts trigram boundaries enough to inflate "distinctness" without a human reading it as anything but a loop | A boundary-agnostic, normalization-based line/segment check |
| 8 | **Sub-word/token-level repeat loops** | Line-repeat check clean | Loops shorter than one segment (`"itatica, itatica, itatica..."`) have no line boundary to catch | zlib compression ratio — boundary-agnostic by construction, no minimum-segment-length blind spot |
| 9 | **Fluent-but-irrelevant padding (HTML/DOCTYPE tails)** | Every prior pattern gate (repetition, language, template) reads 0% — cleanest-looking run in the project by every existing number | The padding is genuine, non-repeating, real English/markup from pretraining — every gate built so far was validated against *degenerate* padding, not *fluent* padding | Worst `rtp_challenging` greedy score in the project (0.342, worse than baseline) — the one time the aggregate number itself was the tell |
| 10 | **RM trained on an imbalanced population** | New on-topic RM hits 99.5% held-out accuracy | 1,961 "declines a hostile prompt" positives vs. 43 genuine benign-topic positives — RM learned "good" ≈ "politely refuses," nearly the opposite of its job | Smoke-testing against *real policy completions* (not held-out pairs) before writing a launch script |
| 11 | **RM score-scale mismatch across prompt populations** | Same RM, same training run | Bradley-Terry pairwise loss only anchors *relative* order within a pair — nothing forces two different prompt clusters (hostile-decline vs. benign-topic) onto the same absolute scale | Measuring raw score means directly on both clusters: ~35 vs. ~19, despite both being genuine positives |
| 12 | **Reward-floor saturation from fixed-weight miscalibration** | Every diagnostic passes, best run yet on toxicity | `critic/score/mean` pinned at the reward floor (−2.0) for nearly the entire run — thin advantage signal, flagged as a cost *before* it ran, confirmed after | Watching the critic's score distribution during training, not just the final eval |
| 13 | **Recall-only relevance gate exploited via verbatim echo** | HTML-tail, non-Latin, repetition, template — all 0%, cleanest run on record by every existing diagnostic | `_relevance_gate` measures what fraction of the *prompt's* words appear in the completion — a verbatim echo of a toxic prompt, followed by generic moralizing, maxes this out by construction while never being penalized for actually reproducing the toxic content | Scoring the 8 worst completions directly through the gate: every one returned the maximum possible score (1.000) |
| 14 | **RM swap silently breaking a Lagrangian controller's calibration** | RM itself verified correct — re-scored the run's own worst completions and it flagged all of them correctly | Swapping in a retrained RM shifted its score distribution; the controller's fixed `mu`/`sigma` cost-normalization (inherited from the *previous* RM) never saturated, so λ never rose and the constraint silently stopped binding | Reading the Lagrangian state file directly: λ=0.074 vs. the prior run's 0.42, at the same update count |
| 15 | **The fix: recalibrate cost normalization to the actual score distribution** | — | `mu`/`sigma` need to sit between the bad-population's max and the genuine population's median, checked against the *specific* RM in play, not inherited from whichever RM came first | Scoring the full training population directly before picking constants, not guessing |

**The meta-lesson inside the catalogue itself:** #3, #6, #9, #13, and #14
were all found by an OOD/held-out check or a direct read of worst
completions — never by the aggregate metric that was supposedly tracking
success. By the fifth recurrence (#9), that stopped being surprising and
started being the default assumption: **a clean number is a reason to
look closer, not a reason to stop looking.**

---

## 4. What actually worked

- **A fixed, never-retrained-on prompt set, scored identically at every
  stage.** Three tracked slices (75 prompts) + a 35-prompt hand-eyeball
  set turned "did this get better" from a vibe into an apples-to-apples
  diff, every single stage.
- **A held-out OOD set treated as a hard gate, never a tuning target.**
  Stated as an explicit rule after it caught the wrong-language collapse:
  *never add these prompts to training data, never hand-tune a fix
  around their specific content.* This is what caught failures #3, #6,
  and #9 above — none of them would have been visible from the training
  distribution alone.
- **Predicting the result before running it, in writing.** Wrong
  predictions were as valuable as right ones — e.g. predicting the
  relevance gate would only partially help (Stage 7) and finding it
  fully closed the *targeted* problem while missing the tail-degeneration
  problem entirely calibrated how much to trust the next prediction.
- **Smoke-testing new reward terms/RMs against known real failure text
  before spending GPU time.** Caught the on-topic RM's polarity bug and
  the `_non_latin_tail_penalty`'s targeting before either one wasted a
  100-step run.
- **Validating data-generation pipelines at 20-row scale before running
  1,961 rows.** Caught the "as an AI language model" disclaimer pattern
  and a duplicate-refusal cluster before they were baked into a training
  corpus.
- **Multiple independent Lagrangian constraints over one fixed-weight
  sum.** A fixed-weight term lets pressure leak into it silently; an
  adaptive λ per axis at least makes the leak *visible* — you can read
  the state file and see which axis is and isn't binding.
- **Watching λ trajectories and critic score distributions during
  training, not just final eval numbers.** An inert λ (pinned at its
  initial value) is a direct, checkable signal that a constraint never
  engaged — this alone diagnosed two separate failures (Stage 6, Stage
  12c) that the eval-time metrics alone made look like clean wins.
- **Root-causing instead of patching.** Every stage that just added
  "one more gate" for the newest disguise (repetition → line-repeat →
  compression-repeat) worked narrowly but kept the whack-a-mole going;
  the stage that stepped back and asked "what would catch *any*
  disguise" (a learned coherence RM) is what actually closed a whole
  *family* of failures at once, not just the one in front of it.
- **Operational hygiene as a real risk, not just tidiness.** A raylet
  "over 95% full" warning immediately preceded one run's mid-training
  crash; clearing disk before the next launch wasn't optional cleanup,
  it was removing a live failure cause.

## 5. What didn't work (and why it's worth knowing, not just avoiding)

- **`repetition_penalty` / `no_repeat_ngram_size` at decode time**, as
  the obvious fix for repetition loops. It made things *worse*: the
  penalty discounts every token already used, including ordinary English
  function words — once a short completion exhausts its natural
  continuations, the *undiscounted* tokens left are disproportionately
  the non-Latin ones. The "fix" steered generation toward the exact
  failure it was meant to prevent.
- **Decode-time mitigation on an already-trained, frozen policy**, in
  general. It can look like a complete fix under greedy decoding and
  even a full OOD sweep — and still be actively worse under sampling,
  because a policy trained *with* an escape hatch available has never
  had to find an actual alternative; blocking the hatch post-hoc doesn't
  teach a new behavior, it just reveals what the sampling distribution
  contained underneath. The same constraint applied *during* rollout
  sampling while training is a genuinely different intervention.
- **A phrase-blocklist for template collapse.** Considered and rejected:
  scoring templated vs. genuine completions through the actual RM showed
  the RM itself scored boilerplate far higher than genuine engagement —
  a phrase gate would have been treating the symptom (specific wording)
  instead of the cause (an RM-level bias toward generic-sounding text).
- **Trusting exact-string uniqueness as a health signal.** It agreed
  with genuine quality for the first few stages, then diverged from it
  three separate times — twice from incoherent-but-varied garbage
  reading as "diverse," once from a rhetorical template that's lexically
  unique every time while being semantically identical. By the third
  divergence, uniqueness was demoted to "a thing to check alongside an
  eyeball," never a verdict on its own.
- **Assuming a swapped-in reward model is a drop-in replacement** under
  an existing Lagrangian controller. It broke the controller's
  calibration silently — the RM itself was correct, but the *system*
  built around the old RM's score distribution wasn't checked against
  the new one before spending a full run on it.

---

## 6. Reading PPO-Lagrangian training signals — a short field manual

Things worth watching *during* a run, not just at the end:

- **`actor/entropy`** — a monotonic crater toward 0 is the classic
  low-diversity collapse signature (true collapse in this project
  bottomed around 0.02–0.03). But a *climbing* entropy isn't automatically
  good either — without something constraining which high-entropy region
  is acceptable, it can wander into wrong-language or degenerate content
  just as easily as genuine diversity.
- **Lagrangian `λ` and `cost_ema`** — an inert `λ` pinned at its
  initialization for the whole run means the constraint never bound,
  regardless of what the final eval numbers say; check this before
  trusting any run where the dual-constraint mechanism is the actual
  novel part of the fix.
- **`critic/score/mean` (and `min`)** — pinned at a reward floor for
  most of a run means most completions are clipping, thinning the
  advantage signal PPO depends on. A run can still improve despite this,
  but it's a real, checkable cost, not a hypothetical one.
- **`critic/grad_norm` spikes** — a late, sharp spike (600+ vs. a
  baseline of 6–9) is worth reading the corresponding completions for,
  not just noting as a number.
- **Worst-of-K sampling, not just greedy.** Several of this project's
  worst findings (the escape-hatch masking real hostility, the reward
  floor's discriminative-signal cost) were invisible under greedy
  decoding and only showed up once sampled completions were checked.

---

## 7. Dataset lineage — what changed and why

| File | What it is | Why it exists |
|---|---|---|
| `sft.jsonl` / `dpo.jsonl` | Vanilla `hh-rlhf` harmless-base pairs | Baseline — 73% of `chosen` eyeballs as hedgy/evasive by hand, far more than a regex audit alone found (21.3%) |
| `sft_diverse.jsonl` / `dpo_diverse.jsonl` | `chosen` replaced with fresh synthetic responses (Nebius-hosted `Qwen3-235B-A22B-Instruct`), fallback to original for any row that fails a toxicity/hedge/disclaimer/duplicate check after 3 retries | Fixes the vanilla data's hedging *without* silently introducing worse problems — the fallback rule guarantees every row is provably no worse than vanilla on toxicity, at the honest cost of ~4–5% of rows keeping some original evasiveness |
| `dpo_dual.jsonl` | Adds a `rejected_evasive` negative (real collapse strings this pipeline itself produced, plus hedge-flagged originals) to every row | Trains a **second**, helpfulness-axis RM — a harmlessness-only RM has no signal that genericness is bad, and fails its own red-team gate without this |
| `dpo_harm_v3.jsonl` (+ 400 naturally-generated echo+moralize negatives) | Retrained harmlessness RM's data, negatives generated few-shot from real exploit examples rather than templated | Templated negatives were built and tested first, found not to generalize, and discarded *before* spending PPO budget — naturally-generated negatives did generalize |
| On-topic RM data (2,682 pairs) | 284 rule-flagged bad completions (six known disguises) + 198 OOD-category-harvested benign positives upweighted 8x + 1,961 `dpo_diverse` positives | The first pass (1,961 hostile-decline positives vs. 43 benign positives) taught the RM "good = declines carefully" — rebalancing toward real benign-topic examples was required before the RM did its intended job |

Cross-cutting rule that shows up in every row of this table: **never
trust a generation or labeling pipeline at full scale without a small
validation batch first**, and **never assume a fix generalizes past the
specific cases it was built against** — check it, on new data, before
moving on.

---

## 8. The shipped checkpoint

`ppo_ontopic_lagrangian_v3` (Stage 12d), a multi-constraint Lagrangian
PPO run (`reward = help − λ_harm·cost_harm − λ_ontopic·cost_ontopic`)
with a properly-calibrated harmlessness cost, on top of every structural
gate accumulated through Stage 11 (windowed language gate, line-repeat,
compression-repeat, relevance).

**Numbers** (`rtp_challenging`, the hardest slice): greedy toxicity
0.210 (baseline) → **0.060**, K=16 support_rate 0.840 → **0.32** — the
best result in the project, on both the metric and the mechanism behind
it (λ_harm reached 1.23, actually bound, didn't run away).

**What's closed:** exact-string collapse, wrong-language output,
line-level and sub-word repeat loops, HTML/markup padding, rhetorical-
template collapse — every pattern-based escape hatch this project ever
catalogued, confirmed clean on both the 75-prompt tracked set and the
55-prompt held-out OOD set.

**What's not, and matters if you build on this:**
- **`_relevance_gate`'s recall-only design** (measures what fraction of
  the *prompt's* vocabulary shows up in the completion, never precision
  of what the completion is actually about) is still exploitable in
  principle — this checkpoint's own completions just don't happen to
  need that exploit. The next stage of optimization pressure against
  this exact reward stack should be expected to find it, the same way
  Stage 12b did.
- **Detoxify scores topic-naming refusals as somewhat toxic** even when
  the model is correctly declining (e.g. explicitly naming what it's
  refusing to discuss). A known interaction between Detoxify's own
  word-presence sensitivity and this project's refusal style, not a
  policy defect.
- **Capability ceiling is real and expected**: this is a 0.5B model.
  It was never expected to solve real coding tasks under any reward
  scheme tried here, and that was scoped out of the success bar
  deliberately rather than treated as a failure.
- One OOD repetition case (a satire-framing prompt looping a refusal
  fragment) was checked at scale against 120 similarly-framed prompts
  and confirmed to be narrowly tied to that one topic phrasing, not a
  generalizable pattern — see `attempt_3/LOGBOOK.md`'s Stage 12d
  ship-ready update for the full result.

**If picking this up again:** the relevance-gate precision fix is the
one deliberately-deferred structural gap. Everything else in this
document's failure catalogue was chased down to a root cause and closed;
that one was diagnosed twice (Stage 11, Stage 12b) and scoped precisely
both times, but never built, because this checkpoint's own behavior
never needed it fixed to ship. The next round of optimization pressure
against this reward stack should be expected to find it.
