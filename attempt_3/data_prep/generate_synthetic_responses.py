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

Teacher model reused from attempt_2/data_prep/generate_synthetic_responses.py
(Qwen/Qwen2.5-1.5B-Instruct -- 3x the base model's params, already
instruction-tuned for helpfulness+safety, small enough to run on the same
GPU). See that file's docstring for the full rationale.

v2 addendum (this file): a first 20-row validation batch with the original
system prompt found the teacher reintroducing its OWN hedging/disclaimer
habits -- 3/20 contained the literal "as an AI language model" (a phrase
absent from the original human-written data entirely) and 5/20 still
matched the exact hedge-phrase regex used to audit the vanilla data. Two
fixes, per the project's "measure before you move" practice of re-checking
after any change rather than assuming a prompt tweak worked:
  1. TEACHER_SYSTEM_PROMPT now explicitly names and forbids the specific
     patterns observed, not just "don't hedge" in the abstract.
  2. generate_replacements() now checks every output against the same
     hedge-regex + disclaimer-substring detector AND a Detoxify score
     against the project's own 0.10 benign threshold, and retries (higher
     temperature, fresh sampling) up to --max-retries times before falling
     back to keeping the best (least-bad) attempt seen. The toxicity check
     was added after re-validating the prompt/hedge fix alone: mean
     toxicity barely moved (0.0146 -> 0.0163 on the 20-row batch) but one
     row spiked 0.001 -> 0.108 (a "joke about black people" prompt getting
     actual racial-joke content) -- pushing against hedging can trade
     evasiveness for content-level compliance with a harmful ask, which the
     hedge regex alone can't see.

Usage (from repo root):
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
from pathlib import Path

TEACHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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
    "Stay safe and non-toxic, but actually engage with the substance of "
    "the request."
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
# (build_pairs.py's --benign-threshold default). A validation batch found
# pushing the teacher away from hedging can occasionally push it toward
# actually complying with a mildly-harmful request instead (e.g. a "joke
# about black people" prompt got real racial-joke content, 0.001 -> 0.108
# Detoxify) -- catching that needs a toxicity check, not just the
# hedge/disclaimer regex, which says nothing about content safety.
TOXICITY_RETRY_THRESHOLD = 0.10


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


def _generate_batch(model, tokenizer, device, user_texts: list[str], temperature: float) -> list[str]:
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
             {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
        )
        for u in user_texts
    ]
    enc = tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)
    import torch
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=96, do_sample=True,
            temperature=temperature, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [
        tokenizer.decode(gen[j, enc["input_ids"].size(1):], skip_special_tokens=True).strip()
        for j in range(len(user_texts))
    ]


def generate_replacements(
    rows: list[dict], batch_size: int = 8, max_retries: int = 3,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.detox_hw import eval_lib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading teacher {TEACHER_MODEL} on {device} (bf16, inference-only) ...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, dtype=torch.bfloat16, device_map=device,
    ).eval()

    def _is_bad_batch(texts: list[str]) -> list[bool]:
        text_bad = [_is_bad_text(t) for t in texts]
        tox_scores = eval_lib.detoxify_score(texts)
        return [tb or (ts > TOXICITY_RETRY_THRESHOLD) for tb, ts in zip(text_bad, tox_scores)]

    out: list[dict] = []
    n_rows_retried = 0
    n_still_bad = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        user_texts = [_extract_user_text(r["prompt"]) for r in chunk]

        completions = _generate_batch(model, tokenizer, device, user_texts, temperature=0.7)
        bad = _is_bad_batch(completions)
        n_rows_retried += sum(bad)  # rows entering the retry loop at all

        attempt = 0
        while any(bad) and attempt < max_retries:
            attempt += 1
            retry_idx = [j for j, b in enumerate(bad) if b]
            retry_texts = [user_texts[j] for j in retry_idx]
            retry_temp = min(0.7 + 0.15 * attempt, 1.1)
            retry_completions = _generate_batch(model, tokenizer, device, retry_texts, temperature=retry_temp)
            retry_bad = _is_bad_batch(retry_completions)
            for k, j in enumerate(retry_idx):
                new_completion, new_bad = retry_completions[k], retry_bad[k]
                # keep the retry only if it's an improvement (not-bad, or
                # first bad attempt so far) -- never discard a good answer
                # for a worse one on a later retry
                if not new_bad or bad[j]:
                    completions[j] = new_completion
                    bad[j] = new_bad

        n_still_bad += sum(bad)

        for j, row in enumerate(chunk):
            out.append({
                "prompt": row["prompt"],
                "user_text": user_texts[j],
                "original_chosen": row["chosen"],
                "synthetic_chosen": completions[j],
                "rejected_toxic": row["rejected"],
                "still_flagged_after_retries": bad[j],
            })
        print(f"  generated {min(i + batch_size, len(rows))}/{len(rows)}")

    print(f"rows that needed >=1 retry: {n_rows_retried}/{len(rows)} | "
          f"still flagged after all retries exhausted: {n_still_bad}/{len(rows)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-source", default="attempt_3/data/dpo.jsonl")
    p.add_argument("--limit", type=int, default=20,
                   help="cap rows for a validation batch; -1 = full corpus")
    p.add_argument("--out", default="attempt_3/data/synthetic_responses_sample.jsonl")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=3,
                   help="max regeneration attempts for rows flagged hedgy/disclaimer")
    a = p.parse_args()

    rows = load_rows(Path(a.dpo_source))
    print(f"{len(rows)} total rows in {a.dpo_source}")
    subset = rows[: a.limit] if a.limit > 0 else rows
    print(f"generating replacements for {len(subset)} rows (full corpus, not just audit-flagged) ...")

    results = generate_replacements(subset, batch_size=a.batch_size, max_retries=a.max_retries)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
