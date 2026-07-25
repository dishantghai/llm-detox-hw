"""Stage 12c prep -- add "echo the toxic prompt verbatim, then moralize"
as an explicit new negative category for `rm_harmlessness`, after Stage
12b's own PPO run found the policy exploiting a real blind spot in it
(see LOGBOOK.md's Stage 12b write-up): `harm_rm` scored six of the eight
worst rtp_challenging completions in or above genuine chosen content's
own harm_raw range (mean ~12.6-14.9), because none of `harm_rm`'s
training negatives (`dpo_diverse.jsonl`'s own `rejected` column, sourced
from hh-rlhf's harmless-base) look like this -- they're narrative-style
toxic completions from an older/different model, not short
real-toxicity-prompts-style toxic fragments reproduced verbatim and
capped with a generic safety disclaimer.

Confirmed directly before building this at scale (not assumed): scoring
synthetic echo+moralize constructions built from `dpo_diverse`'s OWN
`rejected` text against `rm_harmlessness` found they mostly score
correctly negative (harm_rm has already seen this exact distribution as
a training negative) -- but the same construction built from
real-toxicity-prompts-style fragments (the ACTUAL source of the exploit)
fools it at a comparable rate to the genuine PPO exploits. The fix
targets the right distribution because it was checked against the right
distribution, not assumed from the pattern's shape alone.

Training-safe: real-toxicity-prompts' `challenging=True` subset has 1,199
rows; `eval_lib._load_rtp_challenging()` uses `shuffle(seed=0).select(range(50))`
for the fixed eval slice. This script explicitly excludes those 50 texts
before sampling its own training pool (different seed), so no eval
prompt is ever trained on.

Usage:
    python -m attempt_3.data_prep.build_harm_rm_v2_data \\
        --out attempt_3/data/dpo_harm_v2.jsonl \\
        --n-synthetic 400
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DPO_DIVERSE_PATH = Path("attempt_3/data/dpo_diverse.jsonl")

MORALIZE_TAILS = [
    " It's important to be respectful and avoid using inappropriate or offensive language.",
    " Please remember to treat others with kindness and respect.",
    " It's not okay to use language like this; let's keep things considerate.",
    " I'd encourage a more respectful and constructive approach instead.",
    " Let's try to keep the conversation positive and respectful.",
    " I think it's best to avoid this kind of language going forward.",
    " It's always better to communicate with respect and empathy.",
    " Please be mindful of how this kind of language can affect others.",
]


def _eval_rtp_texts() -> set[str]:
    """Reproduces eval_lib._load_rtp_challenging()'s exact selection, so
    this script can exclude it -- import eval_lib directly rather than
    re-implementing the filter, so the exclusion set can never silently
    drift from the real eval set."""
    from src.detox_hw.eval_lib import _load_rtp_challenging

    return set(_load_rtp_challenging())


def _training_safe_rtp_pool(n: int, seed: int) -> list[str]:
    from datasets import load_dataset

    eval_texts = _eval_rtp_texts()
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    ds = ds.filter(lambda r: r.get("challenging") is True)
    ds = ds.shuffle(seed=seed)
    pool: list[str] = []
    for r in ds:
        text = r["prompt"]["text"]
        if text in eval_texts:
            continue
        pool.append(text)
        if len(pool) >= n:
            break
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-diverse", default=str(DPO_DIVERSE_PATH))
    ap.add_argument("--out", default="attempt_3/data/dpo_harm_v2.jsonl")
    ap.add_argument("--n-synthetic", type=int, default=400,
                     help="number of echo+moralize synthetic negatives to add")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    dpo_rows = [json.loads(l) for l in Path(a.dpo_diverse).open()]
    print(f"base dpo_diverse rows: {len(dpo_rows)}")

    rtp_pool = _training_safe_rtp_pool(a.n_synthetic, seed=a.seed)
    print(f"training-safe RTP-challenging texts pulled: {len(rtp_pool)} "
          f"(excludes the 50-prompt eval slice)")

    synthetic_rows = []
    for text in rtp_pool:
        tail = rng.choice(MORALIZE_TAILS)
        echo_moralize = text.strip() + "." + tail
        # Pair with a random dpo_diverse prompt+chosen -- same reasoning as
        # KNOWN_ATTRACTORS' pool reuse in build_dual_rm_data.py: harm_rm
        # is prompt-conditioned, so (prompt_i, chosen_i) vs (prompt_i,
        # this_negative) still teaches "this negative is bad", regardless
        # of whether the negative's own source text came from prompt_i.
        base_row = rng.choice(dpo_rows)
        synthetic_rows.append({
            "prompt": base_row["prompt"],
            "chosen": base_row["chosen"],
            "rejected": echo_moralize,
        })

    out_rows = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in dpo_rows]
    out_rows += synthetic_rows

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} pairs ({len(dpo_rows)} original + "
          f"{len(synthetic_rows)} echo+moralize synthetic) -> {out_path}")


if __name__ == "__main__":
    main()
