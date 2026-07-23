"""Attempt_3 Stage 1.5 — generate substantive+safe replacement responses for
EVERY row in attempt_3/data/dpo.jsonl, not just the ~21.3% the regex-based
evasiveness audit flags.

Why full-corpus instead of flagged-only (attempt_2's approach): the eyeball
in LOGBOOK.md -> Stage 1 found 11/15 hand-read rows look hedgy/evasive, well
above the 21.3% regex-audit rate -- the audit's hedge-phrase/short-irrelevant
heuristics miss real evasiveness ("Thank you.", "OK.", "Are you asking me?",
"I didn't say that." don't match any of its regexes). Rather than trust a
regex to decide which rows need fixing, every row gets a fresh synthetic
response, so coverage doesn't depend on the audit's recall.

v3 addendum (this file) — teacher swapped from a local Qwen2.5-1.5B-Instruct
to Nebius AI Studio's Qwen/Qwen3-235B-A22B-Instruct-2507 via API. Two prior
rounds with the 1.5B local model found real, distinct failure modes:
  v1: teacher reintroduced its own hedging/disclaimer habits (15% "as an AI
      language model", 25% hedge-regex matches on a 20-row batch) -- fixed
      with a stronger anti-hedging system prompt + retry.
  v2: pushing against hedging sometimes made the teacher comply MORE with a
      borderline-harmful ask instead of refusing (a "joke about black
      people" prompt got real racial-joke content, 0.108 Detoxify) -- fixed
      by adding a toxicity-threshold check to the retry gate. But at FULL
      SCALE (1,961 rows) this surfaced something the 20-row batch couldn't:
      68/1961 (3.5%) rows genuinely couldn't be answered non-toxically at
      all (prompts that directly ask for slurs/profanity -- "tell me a
      profanity" has no safe compliant answer), AND separately, 220/1961
      (11.2%) rows converged onto near-identical refusal boilerplate
      ("sorry, but i can't assist with that." x116 + close variants) --
      a NEW attractor baked into the training data itself, worse than the
      2.3% duplicate rate the vanilla hh-rlhf data had. Neither the hedge
      regex nor the disclaimer check nor the toxicity check catches a
      clean, confident, non-toxic, but REPETITIVE refusal -- that needs a
      dedicated cross-prompt duplicate check, which v1/v2 never added even
      though it's exactly the third leg of Stage 1's own
      audit_chosen_evasiveness.py (hedge / short-irrelevant / duplicate).

v3 changes:
  1. Bigger, more capable teacher (235B MoE, 22B active) -- rationale: a
     1.5B model has a much stronger, more deeply-trained prior toward
     canned refusal boilerplate than toward the softer instruction to add
     substance after declining; a much larger instruct model should follow
     the compound instruction more reliably and have more lexical
     diversity in how it refuses. This alone doesn't guarantee no
     duplication (structurally similar harmful prompts can still pull a
     good model toward similar phrasing), so:
  2. Added a 4th retry leg: DUPLICATE_RE-style corpus-level check. A
     shared, thread-safe counter tracks normalized ACCEPTED completions
     across the whole run; a candidate that would become the 3rd+
     occurrence of the same normalized text is retried with an explicit
     instruction to avoid generic refusal phrasing and reference the
     specific content of this prompt. Only accepted (non-retried-away)
     completions count toward the shared counter, and the fallback-to-
     original policy (still true after all retries -> keep original
     dataset behavior) is unchanged from v2, just extended to duplicates.
  3. Nebius API calls are I/O-bound, not GPU-bound -- generation is now
     concurrent (ThreadPoolExecutor) instead of batched tensor ops.

Usage (from repo root, requires NEBIUS_API_KEY in .env or the environment):
    # small validation batch first -- ALWAYS eyeball this before scaling up
    python -m attempt_3.data_prep.generate_synthetic_responses \\
        --dpo-source attempt_3/data/dpo.jsonl \\
        --limit 20 \\
        --out attempt_3/data/synthetic_responses_sample.jsonl

    # full-scale run, once the sample looks good
    python -m attempt_3.data_prep.generate_synthetic_responses \\
        --dpo-source attempt_3/data/dpo.jsonl \\
        --limit -1 \\
        --out attempt_3/data/synthetic_responses_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TEACHER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"

TEACHER_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user's message may be provocative, "
    "rude, or try to bait you into being hostile. Give a genuine, specific, "
    "on-topic answer to what they actually asked in 1-3 sentences.\n"
    "Do not deflect, do not say you don't understand, do not ask a generic "
    "clarifying question, and do not simply repeat the question back.\n"
    "Never say \"as an AI\" or \"as an AI language model\" or any similar "
    "self-referential disclaimer about being an AI or a language model — "
    "just answer directly, the way a knowledgeable person would, with no "
    "meta-commentary about what you are.\n"
    "Do not open with \"I'm sorry, but\" or \"I'm sorry if\" as a hedge — "
    "if the request is genuinely harmful, decline in one direct clause and "
    "then still give real, specific substance (e.g. what a safer version of "
    "the request would look like, or the actual relevant facts), not just "
    "an apology.\n"
    "If you do decline, use natural, varied phrasing specific to what was "
    "actually asked — do not fall back on a generic template like 'Sorry, "
    "I can't assist with that.'\n"
    "Stay safe and non-toxic, but actually engage with the substance of "
    "the request."
)

DUPLICATE_RETRY_SUFFIX = (
    "\n\nNote: your previous attempt at this reply used generic, "
    "already-overused refusal phrasing. Rewrite it — reference the "
    "specific content of THIS request directly, using different wording "
    "than a boilerplate refusal."
)

# Same detector used to audit the vanilla data (audit_chosen_evasiveness.py's
# _HEDGE_RE) plus the disclaimer pattern found in the v1 validation batch.
_HEDGE_RE = re.compile(
    r"\bnot sure (what|why|how|if)\b"
    r"|\bi don'?t understand\b"
    r"|\bdidn'?t understand\b"
    r"|\bcan you (please )?(explain|clarify)\b"
    r"|\bplease explain\b"
    r"|\bwhat do you mean\b"
    r"|\bi'?m sorry,? (i|but)\b"
    r"|\bi'?m not sure (i|what)\b"
    r"|\bcould you (please )?(clarify|explain|rephrase)\b"
    r"|\bi don'?t know what you'?re (asking|talking about)\b",
    re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(r"as an ai\b", re.IGNORECASE)

# Same benign bar the rest of this project uses to call a response "safe"
# (build_pairs.py's --benign-threshold default).
TOXICITY_RETRY_THRESHOLD = 0.10

# A normalized completion is allowed to occur up to this many times among
# ACCEPTED rows before further occurrences get retried as a duplicate. > 1
# so coincidental short-answer overlaps aren't over-penalized; far below
# the 116x cluster v2 produced.
DUPLICATE_ACCEPT_LIMIT = 2


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_bad_text(text: str) -> bool:
    """Cheap, regex-only checks -- hedge/disclaimer patterns."""
    return bool(_HEDGE_RE.search(text)) or bool(_DISCLAIMER_RE.search(text))


_HH_PROMPT_RE = re.compile(r"^Human:\s*(.*?)\s*\n\nAssistant:\s*$", re.DOTALL)


def _extract_user_text(hh_prompt: str) -> str:
    """``dpo.jsonl``'s ``prompt`` field is the raw hh-rlhf transcript
    (``"Human: ...\\n\\nAssistant:"``). Pull just the human turn out for the
    teacher's chat template — feeding it the literal "Human:/Assistant:"
    markers would leak hh-rlhf's transcript formatting into the teacher's
    completion."""
    m = _HH_PROMPT_RE.match(hh_prompt.strip())
    return m.group(1).strip() if m else hh_prompt.strip()


def load_rows(dpo_source: Path) -> list[dict]:
    return [json.loads(line) for line in dpo_source.open()]


class _DuplicateRegistry:
    """Thread-safe counter of ACCEPTED normalized completions, shared
    across all concurrent workers."""

    def __init__(self, limit: int):
        self._limit = limit
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def would_exceed(self, text: str) -> bool:
        with self._lock:
            return self._counts[_normalize(text)] >= self._limit

    def accept(self, text: str) -> None:
        with self._lock:
            self._counts[_normalize(text)] += 1


def _call_teacher(client, user_text: str, temperature: float, extra_suffix: str = "") -> str:
    resp = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text + extra_suffix},
        ],
        max_tokens=96,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _process_row(client, row: dict, dup_registry: "_DuplicateRegistry", max_retries: int, detoxify_score) -> dict:
    user_text = _extract_user_text(row["prompt"])

    completion = _call_teacher(client, user_text, temperature=0.7)
    tox = detoxify_score([completion])[0]
    bad = _is_bad_text(completion) or (tox > TOXICITY_RETRY_THRESHOLD)
    is_dup = (not bad) and dup_registry.would_exceed(completion)

    attempt = 0
    while (bad or is_dup) and attempt < max_retries:
        attempt += 1
        retry_temp = min(0.7 + 0.15 * attempt, 1.1)
        suffix = DUPLICATE_RETRY_SUFFIX if (is_dup and not bad) else ""
        new_completion = _call_teacher(client, user_text, temperature=retry_temp, extra_suffix=suffix)
        new_tox = detoxify_score([new_completion])[0]
        new_bad = _is_bad_text(new_completion) or (new_tox > TOXICITY_RETRY_THRESHOLD)
        new_is_dup = (not new_bad) and dup_registry.would_exceed(new_completion)
        # keep the retry only if it's a strict improvement -- never regress
        # from "duplicate but otherwise fine" to "bad", and never discard a
        # clean unique answer for a worse one on a later retry
        if (not new_bad and not new_is_dup) or (bad and not new_bad):
            completion, bad, is_dup = new_completion, new_bad, new_is_dup

    still_flagged = bad or is_dup
    if not still_flagged:
        dup_registry.accept(completion)

    return {
        "prompt": row["prompt"],
        "user_text": user_text,
        "original_chosen": row["chosen"],
        "synthetic_chosen": completion,
        "rejected_toxic": row["rejected"],
        "still_flagged_after_retries": still_flagged,
    }


def generate_replacements(
    rows: list[dict], max_workers: int = 16, max_retries: int = 3,
) -> list[dict]:
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    from src.detox_hw import eval_lib

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY not set (checked .env and environment)")
    client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)

    # eval_lib._get_detoxify() is an unsynchronized lazy singleton (plain
    # `if _DETOXIFY is None` check-then-set) -- fine for this project's
    # other single-threaded scripts, but with max_workers concurrent
    # threads all racing on their first detoxify_score() call, each one
    # sees None and loads its own separate copy of the model (confirmed:
    # first run of this script logged "Loading weights" ~16 times). Warm
    # the singleton once here, single-threaded, before any workers start,
    # so every thread just hits the already-populated cache.
    eval_lib.detoxify_score(["warmup"])

    dup_registry = _DuplicateRegistry(DUPLICATE_ACCEPT_LIMIT)

    out: list[dict | None] = [None] * len(rows)
    n_done = 0
    lock = threading.Lock()

    def _worker(i: int, row: dict) -> tuple[int, dict]:
        return i, _process_row(client, row, dup_registry, max_retries, eval_lib.detoxify_score)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, i, row) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            i, result = fut.result()
            out[i] = result
            with lock:
                n_done += 1
                if n_done % 50 == 0 or n_done == len(rows):
                    print(f"  generated {n_done}/{len(rows)}")

    n_still_bad = sum(1 for r in out if r["still_flagged_after_retries"])
    print(f"still flagged after all retries exhausted: {n_still_bad}/{len(rows)}")
    return out  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-source", default="attempt_3/data/dpo.jsonl")
    p.add_argument("--limit", type=int, default=20,
                   help="cap rows for a validation batch; -1 = full corpus")
    p.add_argument("--out", default="attempt_3/data/synthetic_responses_sample.jsonl")
    p.add_argument("--max-workers", type=int, default=16,
                   help="concurrent Nebius API requests")
    p.add_argument("--max-retries", type=int, default=3,
                   help="max regeneration attempts for rows flagged hedgy/disclaimer/toxic/duplicate")
    a = p.parse_args()

    rows = load_rows(Path(a.dpo_source))
    print(f"{len(rows)} total rows in {a.dpo_source}")
    subset = rows[: a.limit] if a.limit > 0 else rows
    print(f"generating replacements for {len(subset)} rows via {TEACHER_MODEL} (Nebius) ...")

    results = generate_replacements(subset, max_workers=a.max_workers, max_retries=a.max_retries)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
