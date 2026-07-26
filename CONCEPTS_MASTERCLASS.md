# The Concepts Behind This Project — A First-Principles Masterclass

`LESSONS_LEARNED.md` tells the story: what broke, why, how we fixed it.
This document teaches the *machinery* referenced in that story — SFT,
DPO, reward models, PPO, entropy, KL divergence, Lagrangian multipliers,
and the smaller diagnostic tools (compression ratios, precision/recall,
sigmoid calibration) — one layer at a time, assuming nothing. Every piece
of math gets a tiny worked example with real numbers, not just a formula.

Read it in order — each section leans on the one before it.

---

## Layer 0: What is "the policy," and why does training it feel different from normal ML?

A language model is a function that, given some text so far, outputs a
probability for every possible next word (technically "token" — a
word-or-word-fragment). Ask it to write, and it repeatedly: look at text
so far → get a probability distribution over next tokens → pick one →
repeat.

In RL language, this whole model is called **the policy** — a policy is
just "a rule for choosing an action given a situation." Here, the
"situation" is the text so far, and the "action" is which token to
output next.

Normal supervised learning has one clean target per input ("this image
is a cat"). Training a policy to be *good at conversation* has no such
single right answer — "roast me creatively" has thousands of acceptable
responses and thousands of unacceptable ones, and nobody can write down
a lookup table of all of them. That's the entire reason this project
needed three different training techniques stacked on top of each other
instead of one.

---

## Layer 1: SFT — Supervised Fine-Tuning (a.k.a. "just show it examples")

**The idea:** collect a pile of (prompt, good response) pairs, and train
the model to predict the good response, token by token, the same way any
next-token-prediction training works.

**The math, dead simple.** At each position, the model outputs a
probability for the correct next token — say it assigns 0.1 (10%)
probability to the actual correct word. The loss is `-log(0.1) ≈ 2.3`.
If it assigns 0.9 instead, loss is `-log(0.9) ≈ 0.1` — much smaller.
This is **cross-entropy loss**: it punishes the model in proportion to
how surprised it was by the right answer. Training just nudges the
model's weights, over many examples, to be less surprised over time.
That's it — that's the entire mechanism. (In this project's logs:
"loss 2.22 → 1.25 over 122 steps" is exactly this number falling.)

**Why SFT alone isn't enough, and why this matters for everything after
it.** SFT only ever sees *positive* examples — "here's a good response."
It never sees "here's a bad response, don't do this." A model trained
this way tends to learn the *shape* of the training data rather than the
underlying judgment behind it — which is exactly what happened here: SFT
on hedgy training data produced a model that hedges on *everything*,
including completely harmless prompts, because it copied the surface
pattern ("when unsure, say 'I don't understand'") rather than learning
when hedging is actually appropriate. SFT has no mechanism to distinguish
those two things, because it was never shown a contrast.

---

## Layer 2: Preference pairs and the Bradley-Terry model (the idea underneath both DPO and reward models)

Everything from here on is built on one small piece of math, so it's
worth learning once, properly.

**The setup:** instead of "here's the one right answer," you show the
model **two** responses to the same prompt — one better (`chosen`), one
worse (`rejected`) — and just say *this one, not that one.* No absolute
label, only a comparison. This is much easier to collect (humans are
bad at rating things 1-10 consistently, but reliably good at "which of
these two is better").

**The Bradley-Terry model** turns "which one is better" into a
probability. Give each response a real number score (call it `r`). The
model of *how likely* a rater is to prefer response A over response B is:

```
P(A preferred over B) = sigmoid(r_A - r_B) = 1 / (1 + e^-(r_A - r_B))
```

`sigmoid` (also called the logistic function) just squashes any real
number into a probability between 0 and 1. A few anchor points to build
intuition:

```
r_A - r_B = 0    →  sigmoid = 0.50   (totally undecided)
r_A - r_B = 1    →  sigmoid ≈ 0.73   (leaning toward A)
r_A - r_B = 3    →  sigmoid ≈ 0.95   (strongly confident in A)
r_A - r_B = -3   →  sigmoid ≈ 0.05   (strongly confident in B)
```

So: **the bigger the score gap, the more confidently the model predicts
A wins** — and the score gap can be any real number (unbounded), while
the *probability* it implies is always squeezed into [0, 1]. This one
idea — "unbounded internal score, squashed into a bounded probability or
cost by sigmoid/tanh" — recurs three separate times in this project
(DPO, reward models, and the Lagrangian cost calibration in Layer 6), so
it's worth really sitting with this example before moving on.

Training "learns" by adjusting scores so that pairs where the human said
"A is better" get pushed toward `P(A) ≈ 1`, via the same cross-entropy
loss from Layer 1, just applied to this probability instead of a
next-token probability.

---

## Layer 3: DPO — Direct Preference Optimization

**The problem DPO solves:** the classical way to use preference pairs
(what Layer 4 and 5 describe) is a two-stage pipeline — train a separate
reward model, then run reinforcement learning against it. That's
expensive and has a lot of moving parts. **DPO's trick: skip the reward
model entirely.** It shows, mathematically, that if you assume the
Bradley-Terry model above, the "reward" is implicitly whatever makes the
*policy itself* assign higher probability to `chosen` than `rejected` —
so you can train the policy directly on preference pairs, no separate
reward model, no RL loop.

**The DPO loss, unpacked piece by piece** (this is the most complex
formula in this document — take it slow):

```
loss = -log sigmoid( β · [ (log π(chosen) - log π_ref(chosen))
                          - (log π(rejected) - log π_ref(rejected)) ] )
```

Read the bracket as two "how much more likely does my policy make this
response, compared to the frozen starting point (`π_ref`, the reference
policy before DPO started)?" scores — one for `chosen`, one for
`rejected` — and the loss just wants the *first* one bigger than the
*second* one. `β` is a temperature knob: bigger `β` means "care more
about the gap," smaller means "be more lenient." This is structurally
identical to the Bradley-Terry sigmoid from Layer 2 — the "score" `r` is
just replaced by "how much more probable the policy makes this text
relative to where it started."

**What "margin" means in the logs.** `margin = chosen_r - rejected_r` is
literally the bracketed quantity above. This project's DPO run: "margin
3.38 → 4.26" over training — the gap between how much the model prefers
`chosen` vs `rejected` widened, exactly as intended.

**The trap DPO fell into here, explained with the mechanism above.**
DPO only ever pushes `chosen` up relative to `rejected` — it has *no*
opinion on **which specific way** the model chooses to make `chosen`
more likely. If every `chosen` example in the training set happens to
share a rhetorical habit (a refusal justified by "...as that can be
harmful and inappropriate"), DPO will happily converge the model onto
using that *exact phrase* everywhere, because doing so is a perfectly
valid way to raise `chosen`'s probability relative to `rejected`'s. The
loss function has no way to tell "learned to be safe" apart from
"learned to parrot one safe-sounding sentence" — both look identical
from inside the math.

---

## Layer 4: Reward Models (RMs) — training a standalone scorer

Instead of skipping straight to policy training (DPO), you can train a
**separate small model whose only job is to output a score `r` for any
given text**, using exactly the Bradley-Terry loss from Layer 2 as its
training objective: feed it `(chosen, rejected)` pairs, and adjust its
weights until `sigmoid(r_chosen - r_rejected)` is high for real pairs.

Once trained, this scorer can rate *any* text, not just training pairs —
which is what makes it useful for RL in Layer 5: instead of a human
rating every single thing the policy generates during training (way too
slow), the RM stands in as an automated judge.

**Two structural weaknesses of this setup, both of which bit this
project directly — worth understanding as properties of the *math*, not
bugs in any one RM:**

1. **Bradley-Terry only ever anchors *relative* order within a pair, not
   absolute scale across different topics.** Training only ever asks
   "is `r_chosen > r_rejected` for *this specific pair*?" — it has no
   mechanism forcing two different *clusters* of prompts (say, "declines
   a hostile ask" vs. "answers a benign question") onto a shared number
   line, even if both clusters are full of genuinely good responses.
   This project measured it directly: one RM scored "genuinely good,
   hostile-prompt-decline" text at a mean of ~35 and "genuinely good,
   benign-topic" text at a mean of ~19 — both are real positives, just
   sitting at different heights, purely because the training pairs never
   forced a comparison *between* those two clusters.
2. **An RM only ever learns to score what it was shown pairs of.** If
   99% of your positive examples share one property (e.g. "declines a
   hostile prompt") and only 1% share another (e.g. "gives a good answer
   to a benign question"), the RM's gradient updates are dominated by
   the 99%, and it will learn a proxy for the majority pattern rather
   than the property you actually wanted scored. This project's
   on-topic RM did exactly this on its first attempt — trained on 1,961
   "declines-hostile-prompt" positives against only 43 "answers-benign-
   question" positives, it learned "good = declines carefully," almost
   the opposite of its intended purpose.

---

## Layer 5: Reinforcement Learning and PPO — training the policy by trial and reward

**RL in one sentence:** let the policy generate something, score it with
a reward, and nudge the policy's weights to make high-reward outputs
more likely and low-reward outputs less likely — repeat thousands of
times.

**Why not just always pick the highest-scoring next word?** Because
reward is only known *after* a full response is generated (or after a
score is assigned to the whole thing) — you can't directly say "this
specific word choice was 0.3 units good," you only know "this whole
response scored 7.2." RL exists to solve exactly this credit-assignment
problem: figure out which choices, spread across a whole response,
deserve credit for a good (or bad) final score.

**The Critic and the Advantage.** Training a second small model (the
"critic") to *predict* how much reward a partial response is likely to
end up getting — essentially "given where we are, is this going well?"
The **advantage** is `actual reward − critic's prediction`: did this
turn out better or worse than expected? Positive advantage → reinforce
those choices; negative → discourage them. This is why the training
logs track `critic/score/mean` — it's the critic's running prediction,
and if it's pinned at a floor value for most of a run (as happened in
Stage 12a: pinned at -2.0, the minimum possible), it means the critic
can barely tell most completions apart, which starves the advantage
signal of the discriminative information PPO needs to learn efficiently.

**PPO — "Proximal Policy Optimization."** The "proximal" part is the key
idea: **don't let the policy change too much in a single training step.**
Why this matters: RL training is noisy, and a policy that's allowed to
take one huge step based on a single batch's reward signal can lurch
into a completely different (often much worse) behavior. PPO computes
the ratio between the new policy's probability of an action and the old
policy's probability of that same action, and **clips** it — if the
ratio strays too far from 1.0 (i.e., the policy is about to change a lot
for this example), the update is capped. It's a seatbelt against
overcorrecting on any one batch.

**KL divergence — measuring "how different are two probability
distributions."** Formally, `KL(P, Q) = Σ P(x) · log(P(x)/Q(x))`. Don't
worry about computing it by hand — the intuition is what matters: it's 0
when two distributions are identical, and grows the more they differ.
This project used it as an **anchor**: add a penalty term equal to
`KL(current_policy, reference_policy)`, so drifting far from the
starting point (the SFT/DPO checkpoint before PPO) costs something, even
if the reward would otherwise reward drifting arbitrarily far. It's a
leash, not a wall — the policy can still change, just not for free.

**Entropy — measuring "how spread out are the policy's choices."**
Formally, `H(P) = -Σ P(x) · log P(x)`. Concretely: if a policy always
picks the exact same next word with 100% certainty, entropy is 0 —
totally predictable, zero diversity. If it spreads probability evenly
across many plausible next words, entropy is high. **Low, crashing
entropy is this project's single most reliable early-warning sign of
collapse** — Stage 5b's total collapse onto one repeated sentence showed
up as entropy crashing from 2.95 to 0.015–0.03 well before anyone read a
single completion. An **entropy bonus** (add `+ coefficient × entropy`
to the loss) directly counteracts this by rewarding the policy for
staying spread out — which is exactly what stopped Stage 5's fixed-
template collapse in Stage 6, and exactly what (with nothing else
constraining *where* that spread went) is what let the policy wander
into fluent Cyrillic gibberish instead. Entropy bonus buys you
diversity; it says nothing about whether the diversity is any good.

---

## Layer 6: Reward hacking, and Lagrangian constraints as the fix

**Reward hacking (a.k.a. Goodhart's Law: "when a measure becomes a
target, it ceases to be a good measure").** Any reward function is a
finite, imperfect proxy for "actually good behavior." Given enough
optimization pressure, RL will find the *cheapest* way to score well on
the proxy — which is not necessarily the same as the behavior you
actually wanted. This project's entire twelve-stage arc is one long,
concrete demonstration of this: every single reward function tried was
eventually satisfied by some behavior nobody wanted (a repeated
sentence, gibberish, a rhetorical template, HTML padding, a verbatim
echo) because that behavior was, mathematically, a cheaper way to
maximize the score than the intended behavior.

**Multiple reward terms and the "leaking pressure" problem.** The naive
fix — add more penalty terms for each bad behavior you find — runs into
a structural problem once you have several terms added together with
fixed weights: pushing hard against *one* penalty doesn't make the
policy behave better overall, it just makes whichever *other* term has
the most remaining slack the new cheapest target. This project hit this
pattern five separate times before treating it as the expected default
rather than a surprise each time.

**Constrained optimization — the fix, and the Lagrangian idea, from
scratch.** Suppose you want to maximize helpfulness, *subject to* harm
staying below some target level — not "helpfulness minus some fixed
penalty for harm," but a genuine hard constraint: `maximize help(x)
subject to harm(x) ≤ target`.

The classical trick (Lagrange, 18th century — genuinely old math, still
exactly the right tool here) turns a constrained problem into an
unconstrained one by adding a penalty term multiplied by a tunable
number λ ("lambda"):

```
reward = help(x) - λ · (harm(x) - target)
```

If `λ` is fixed by hand, you're back to the "fixed-weight" problem above
— guess wrong and the constraint is either too weak (never binds) or too
strong (crushes helpfulness). **The actual trick is to make λ adapt
automatically, via a simple update rule:**

```
λ ← max(0, λ + η · (measured_harm - target))
```

In plain English: *if harm is currently above target, raise λ (push
harder); if harm is comfortably below target, lower λ (ease off).* Think
of λ as an automatically-adjusting fine: violate the rule, and the fine
goes up until you stop; comply comfortably, and the fine relaxes. This
is called **dual ascent**, and it's the actual mechanism behind every
"Lagrangian controller" mentioned in this project's logs. The `_lagrangian_state.json`
files this project kept checking are just `{lambda, cost_ema, n_updates}`
— literally the current fine, a running average of how much the
constraint is being violated, and how many times the fine has been
updated.

**Why a *stuck* λ was such a useful diagnostic.** If λ never moves off
its starting value (commonly 0.0) for an entire training run, it means
the controller never once measured the harm constraint as violated —
which either means the constraint is genuinely being satisfied for free
(great) or means the *cost signal feeding the controller is broken*
(not great, and much harder to tell apart without checking). This
project hit the second case directly: swapping in a retrained reward
model shifted its score distribution, and the *old* cost-calibration
constants (see next section) never registered the new model's outputs
as "harmful enough to matter" — so λ sat near zero the whole run, not
because the policy was safe, but because the thermometer was broken.

**Multiple independent constraints ("multi-Lagrangian").** Nothing
about the idea above requires just one constraint — you can run several
`λ`s in parallel, one per axis you care about (harmlessness, on-topicness,
...), each with its own target and its own automatic update. This is
strictly better than summing fixed-weight penalties for the exact reason
in the "leaking pressure" paragraph above: each axis gets its own
independent, self-correcting pressure instead of sharing one shared
budget that can silently get reallocated.

---

## Layer 7: Turning a raw score into a bounded cost — sigmoid/tanh calibration

The Lagrangian machinery in Layer 6 needs a **bounded** cost signal
(rewards that grow without limit make the `η · (measured - target)`
update unstable). But reward models (Layer 4) output **unbounded** raw
scores — could be -50, could be +80, no ceiling. The fix is the same
squashing idea from Layer 2's sigmoid, using its close cousin `tanh`
(shape is the same S-curve, just centered at 0 and ranging -1 to 1
instead of 0 to 1):

```
cost = tanh( (raw_score - mu) / sigma )
```

`mu` is "where zero cost sits" and `sigma` controls how quickly the
curve saturates to ±1 as the raw score moves away from `mu`. Picture the
tanh curve: near the center it's almost a straight line (small changes
in `raw_score` produce proportionally small changes in `cost` — this is
the *graded* region, where the constraint gets useful, fine-grained
signal). Far from the center, the curve flattens out almost completely
(large changes in `raw_score` barely move `cost` at all — the *saturated*
region, where the signal stops being informative because it's already
pinned near -1 or +1).

**Why this bit the project directly, with real numbers.** The original
constants (`mu=3, sigma=2`) were picked for one reward model and never
re-checked when a *retrained* reward model was swapped in underneath the
same formula. The new model's genuine, safe completions scored around
+22 to +25 on average — miles into the saturated tail of a curve
centered at `mu=3`. The result: **98.3% of genuine completions landed
within 0.1 of the same fully-saturated cost value**, regardless of how
safe they actually were — the controller could no longer distinguish
"pretty safe" from "extremely safe," which flattened its whole
discriminative signal and is exactly why `λ` in Layer 6 stayed inert.
The fix was mechanical once diagnosed: measure where the *actual*
population of good and bad scores sits, and re-pick `mu`/`sigma` so most
real completions land in the graded middle of the curve instead of the
flat tails —

```
mu    = a value between the bad population's max and the good
        population's median   (here: bad max=12.7, good median=25.2 → mu=16)
sigma = sized so the good population's spread mostly lands where
        the curve is still graded, not flat                (here: sigma=6)
```

— which moved the fraction of genuine completions landing in the graded
zone from 1.7% to 40.4%, and let `λ` finally respond to real signal
again.

---

## Layer 8: The detection tools — how "is this text bad" gets measured automatically

A few small, self-contained ideas, each solving one specific "how do I
even detect this failure automatically" problem.

**Toxicity scoring (Detoxify).** A separate, pre-trained classifier that
reads a piece of text and outputs a 0–1 "how toxic does this sound"
score. Treat it like any other classifier — trained on labeled toxic/
non-toxic text, and like any classifier, it has blind spots (e.g. it
reads *naming* an explicit topic while declining to discuss it as mildly
toxic, even though refusing is exactly the safe behavior — a known
false-positive shape, not a bug in this project's model).

**N-gram / trigram distinctness.** An n-gram is just "a run of `n`
consecutive words." A **trigram** is 3 in a row. If a completion repeats
itself, the same trigrams recur; count how many of the completion's
trigrams are *unique* vs. repeated, and a low ratio flags repetition.
The blind spot this project found: wedge one short, unusual token
between each repeat, and the trigram *boundaries* shift enough that the
ratio looks artificially higher than a human reading the same text would
judge it — the check measures exact word-triples, not "does a human
recognize this as looping."

**Compression ratio as a repetition detector (the elegant one).** Any
general-purpose compressor (this project used `zlib`, the same algorithm
behind `.zip` files) works by finding and re-using repeated patterns in
data — the more a piece of text repeats itself in *any* way (whole
sentences, single words, sub-word fragments), the smaller it compresses,
because the compressor only has to store the pattern once and then a
short "repeat that" instruction. `compression_ratio = compressed_size /
original_size` — a low ratio (this project used `<0.40` as the cutoff)
means the text is, in an information-theoretic sense, mostly repetition
dressed up as new content. This is the practical version of an idea
called **Kolmogorov complexity** — "how short is the shortest program
that could reproduce this exact text" — and it's powerful here precisely
*because* it doesn't care what kind of repetition it is (whole sentence,
one word, a sub-word fragment): it catches all of them with one check,
which is why it succeeded where the line-level and word-trigram checks
(each built for one specific repetition *shape*) each had a blind spot.

**Precision vs. recall (why a "relevance" check can be gamed).**
*Recall* asks "of everything relevant, how much did you cover?" —
*precision* asks "of everything you said, how much was actually
relevant?" A completion can have perfect recall and terrible precision:
this project's relevance gate measured `(prompt words that appear in the
completion) / (total prompt words)` — pure recall. A policy that just
**echoes the entire prompt back verbatim** scores a perfect 1.0 on this,
by construction, regardless of whether it added one single relevant
word of its own. The fix a recall-only gate is missing is a *precision*
term: what fraction of what the completion actually said was genuinely
about the prompt, not just how much of the prompt's own vocabulary got
reused.

**Greedy decoding vs. sampling, and why "worst-of-K" matters.** Once a
model outputs a probability distribution over next tokens, you still
have to pick one. **Greedy** always picks the single highest-probability
token — deterministic, same output every time for the same input.
**Sampling** (with a temperature/top-p/top-k controlling how "spread
out" the randomness is) draws from the distribution instead, so the same
prompt can produce different completions across tries. **Worst-of-K**
means: sample K times, and look at the single worst (most toxic)
completion out of the K — a much harder, more realistic test of "can
this model still be made to say something bad" than a single greedy
trace, because greedy only ever shows you the model's single most-likely
behavior, never its tail risk. This project found real failures (the
reward-floor discriminative-signal cost, the decode-time-patch masking
problem) that were **completely invisible under greedy decoding** and
only showed up once worst-of-K sampling was checked — greedy alone
was, more than once, an incomplete picture dressed up as a full one.

---

## Quick-reference glossary

| Term | One-line meaning |
|---|---|
| Policy | The model, viewed as "a rule for picking the next token given context so far" |
| SFT | Training by copying labeled good examples (cross-entropy loss) |
| Preference pair | `(chosen, rejected)` — a comparison, not an absolute label |
| Bradley-Terry model | `P(A > B) = sigmoid(r_A - r_B)` — turns a score gap into a preference probability |
| DPO | Trains the policy directly on preference pairs, using the policy's own probabilities as the implicit reward — no separate RM |
| Reward Model (RM) | A separate scorer trained with the Bradley-Terry loss, used to auto-judge policy outputs |
| Critic / value function | Predicts expected future reward from the current state, used to compute advantage |
| Advantage | `actual reward − critic's prediction` — did this turn out better or worse than expected |
| PPO | RL algorithm that clips how much the policy is allowed to change per update ("proximal") |
| KL divergence | A distance-like measure between two probability distributions; used to anchor the policy near a reference |
| Entropy | How spread-out a probability distribution is; crashing entropy = collapse warning sign |
| Reward hacking / Goodhart's Law | Optimizing a proxy metric eventually finds the cheapest way to satisfy the proxy, not the intent behind it |
| Lagrangian multiplier (λ) | An auto-tuning "fine" that rises when a constraint is violated and falls when it's comfortably satisfied |
| Dual ascent | The update rule that makes λ auto-tune: `λ ← max(0, λ + η·(measured − target))` |
| tanh/sigmoid calibration | Squashing an unbounded raw score into a bounded [-1,1] or [0,1] cost/probability, centered and scaled to match the real score distribution |
| Saturation (of a squashing curve) | The flat region far from center, where the signal stops being graded/informative |
| Precision vs. recall | Recall: coverage of the relevant material. Precision: relevance of what was actually said. A recall-only gate is gameable by echoing |
| Trigram distinctness | Fraction of unique 3-word runs — a cheap, boundary-sensitive repetition check |
| Compression ratio | `compressed size / original size` — a boundary-agnostic repetition check via general-purpose compression |
| Greedy vs. sampling | Deterministic best-token-every-time vs. randomized draws from the output distribution |
| Worst-of-K | The single worst completion out of K sampled attempts — a tail-risk test greedy decoding can't show |
