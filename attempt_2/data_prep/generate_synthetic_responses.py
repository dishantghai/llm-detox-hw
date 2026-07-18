"""Phase 1 — generate substantive+safe replacements for ``hh-rlhf`` ``chosen``
responses that ``audit_chosen_evasiveness.py`` flagged as evasive.

No external LLM API key is available in this environment (checked: no
``OPENAI_API_KEY``/``ANTHROPIC_API_KEY``/etc. in the shell env this project
runs in). The fallback used here — and it's a reasonable one, not just a
workaround — is to use an already-instruction-tuned model from the same
model family as a local teacher: **``Qwen/Qwen2.5-1.5B-Instruct``**.

Why this specific model:

- It's 3x the parameter count of the ``Qwen2.5-0.5B`` base this whole
  project trains, so it's a genuinely stronger generator, not a lateral
  move.
- It's already instruction-tuned/RLHF'd for helpfulness+safety by Qwen's own
  team — exactly the property README.md's own text names as *why* the
  homework deliberately avoids starting from an Instruct checkpoint for the
  *policy* ("the Instruct variant has already been RLHF'd into
  politeness"). That's a liability for the model we're trying to detoxify
  from scratch; it's exactly the property we want in a teacher whose job is
  to demonstrate what "safe AND substantive" looks like.
- Small enough (~3GB in bf16) to download and run in minutes on the same
  GPU, no separate infra.

This script does NOT try to replace every flagged row at data-prep scale in
one shot inside this guide — see the ``--limit`` flag and GUIDE.md Phase 1
for the staged plan (small validation batch first, full-scale run as a
separate, user-driven step once the small batch's output is eyeballed and
judged good, mirroring this project's own established practice of never
trusting an automated pipeline without an eyeball pass).

Usage (from repo root):
    python -m attempt_2.data_prep.generate_synthetic_responses \\
        --audit attempt_2/submissions/chosen_evasiveness_audit.json \\
        --dpo-source data/dpo.jsonl \\
        --limit 20 \\
        --out attempt_2/data/synthetic_replacements_sample.jsonl
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
    "on-topic answer to what they actually asked — do not deflect, do not "
    "say you don't understand, do not ask a generic clarifying question, "
    "and do not simply repeat the question back. Stay safe and non-toxic, "
    "but actually engage with the substance of the request in 1-3 "
    "sentences."
)

_HH_PROMPT_RE = re.compile(r"^Human:\s*(.*?)\s*\n\nAssistant:\s*$", re.DOTALL)


def _extract_user_text(hh_prompt: str) -> str:
    """``data/dpo.jsonl``'s ``prompt`` field is the raw hh-rlhf transcript
    (``"Human: ...\\n\\nAssistant:"``). Pull just the human turn out for the
    teacher's chat template — feeding it the literal "Human:/Assistant:"
    markers would leak hh-rlhf's transcript formatting into the teacher's
    completion."""
    m = _HH_PROMPT_RE.match(hh_prompt.strip())
    return m.group(1).strip() if m else hh_prompt.strip()


def load_flagged_rows(audit_path: Path, dpo_source: Path) -> list[dict]:
    """Re-derive which rows the audit flagged (rather than re-reading them
    from the audit JSON's summary, which only stores a few examples) by
    re-running the same flag logic against the same source file — keeps
    this script correct if the audit thresholds ever change, at the cost of
    importing the audit module instead of just its output."""
    from attempt_2.data_prep.audit_chosen_evasiveness import _HEDGE_RE, _normalize, _words

    from collections import Counter

    rows = [json.loads(line) for line in dpo_source.open()]
    normalized = [_normalize(r["chosen"]) for r in rows]
    dup_counts = Counter(normalized)

    flagged = []
    for row, norm in zip(rows, normalized):
        chosen, prompt = row["chosen"], row["prompt"]
        is_hedge = bool(_HEDGE_RE.search(chosen))
        c_words, p_words = _words(chosen), _words(prompt)
        overlap = len(c_words & p_words) / max(1, len(c_words))
        n_words = len(chosen.split())
        is_short_irrelevant = n_words <= 12 and overlap < 0.2
        is_dup = dup_counts[norm] > 1
        if is_hedge or is_short_irrelevant or is_dup:
            flagged.append(row)
    return flagged


def generate_replacements(rows: list[dict], batch_size: int = 8) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading teacher {TEACHER_MODEL} on {device} (bf16, inference-only) ...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, dtype=torch.bfloat16, device_map=device,
    ).eval()

    out: list[dict] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        user_texts = [_extract_user_text(r["prompt"]) for r in chunk]
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                 {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True,
            )
            for u in user_texts
        ]
        enc = tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=96, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, row in enumerate(chunk):
            completion = tokenizer.decode(
                gen[j, enc["input_ids"].size(1):], skip_special_tokens=True,
            ).strip()
            out.append({
                "prompt": row["prompt"],
                "user_text": user_texts[j],
                "original_chosen": row["chosen"],
                "synthetic_chosen": completion,
                "rejected_toxic": row["rejected"],
            })
        print(f"  generated {min(i + batch_size, len(rows))}/{len(rows)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit", default="attempt_2/submissions/chosen_evasiveness_audit.json")
    p.add_argument("--dpo-source", default="data/dpo.jsonl")
    p.add_argument("--limit", type=int, default=20,
                   help="cap rows for a validation batch; omit/raise for full-scale runs")
    p.add_argument("--out", default="attempt_2/data/synthetic_replacements_sample.jsonl")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()

    flagged = load_flagged_rows(Path(a.audit), Path(a.dpo_source))
    print(f"{len(flagged)} rows flagged evasive by the Phase-1 audit")
    subset = flagged[: a.limit] if a.limit > 0 else flagged
    print(f"generating replacements for {len(subset)} rows ...")

    results = generate_replacements(subset, batch_size=a.batch_size)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
