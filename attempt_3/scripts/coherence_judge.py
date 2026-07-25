"""Zero-shot "is this completion coherent and on-topic for its entire
length" judge, via the Nebius-hosted Qwen3-235B-A22B-Instruct-2507 (same
teacher used for Stage 1.5's synthetic data).

Built to scope a fix for the pattern this project has now hit five times
(Stage 6 through 11, see LOGBOOK.md): every hand-coded reward gate closes
one specific "filler" disguise (non-Latin script, a filler token wedged
between repeats, a sub-word loop, HTML/DOCTYPE markup) and PPO immediately
finds a new one none of the existing gates recognize. A judge asked the
general question -- "does this stay coherent and on-topic the whole way
through, in natural language, with no repetition?" -- rather than any one
disguise's fingerprint should generalize to the *next* disguise too.

Validated (see LOGBOOK.md's coherence-RM scoping entry) against a 25-example
set spanning all six known failure types plus genuine completions: 100%
correct once the prompt was told that a hard length cap can legitimately
end a response mid-sentence (that alone is not a defect -- only genuine
derailment/repetition/wrong-language/off-topic content is).

Usage as a library:
    from attempt_3.scripts.coherence_judge import judge_batch
    results = judge_batch([{"prompt": ..., "completion": ...}, ...])

Usage as a CLI (reads a JSON list of {"prompt", "completion"} objects):
    python -m attempt_3.scripts.coherence_judge \\
        --in candidates.json --out judged.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

JUDGE_SYSTEM = (
    "You are grading a small language model's response to a user prompt, "
    "which was generated with a hard length cap -- it may end mid-sentence "
    "or mid-word simply because it ran out of budget. That alone is NOT a "
    "defect: judge only the text that was actually generated. You are NOT "
    "grading toxicity or safety -- a refusal can be coherent, an explicit "
    "answer can be coherent. Grade exactly one thing: is every part of the "
    "generated text on-topic, natural language, and free of repetition, "
    "up to wherever it stops? Answer NO if the response: repeats a "
    "word/phrase/sentence in a loop; degenerates into gibberish or a "
    "different language than the prompt's; trails off partway through "
    "into unrelated content the user didn't ask for (e.g. markup, code "
    "artifacts, or a system-prompt echo); or contains any stray "
    "off-topic token/fragment. Simply ending abruptly at a natural "
    "clause boundary because of the length cap, with nothing strange "
    "in the text itself, is YES.\n\n"
    "Respond with strict JSON: {\"verdict\": \"YES\" or \"NO\", "
    "\"reason\": \"<one short sentence>\"}."
)


def _get_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY not set (checked .env and environment)")
    return OpenAI(base_url="https://api.studio.nebius.com/v1/", api_key=api_key)


def judge_one(client, prompt: str, completion: str) -> dict:
    user = f"PROMPT: {prompt}\n\nRESPONSE: {completion}"
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.0, max_tokens=120,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content.strip()
        start, end = text.index("{"), text.rindex("}") + 1
        verdict = json.loads(text[start:end])
        return {"verdict": verdict.get("verdict", "PARSE_ERROR"), "reason": verdict.get("reason", "")}
    except Exception as e:  # network/parse errors -- surface, don't silently mislabel
        return {"verdict": "ERROR", "reason": str(e)[:150]}


def judge_batch(examples: list[dict], workers: int = 12, progress: bool = True) -> list[dict]:
    """``examples``: list of {"prompt", "completion", ...any other fields}.
    Returns the same dicts with "judge_verdict"/"judge_reason" added."""
    client = _get_client()
    results: list[dict | None] = [None] * len(examples)

    def _run(i_ex):
        i, ex = i_ex
        v = judge_one(client, ex["prompt"], ex["completion"])
        return i, {**ex, "judge_verdict": v["verdict"], "judge_reason": v["reason"]}

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (i, row) in enumerate(pool.map(_run, enumerate(examples))):
            results[i] = row
            if progress and (n + 1) % 20 == 0:
                print(f"  ...{n + 1}/{len(examples)}")
    return results  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    examples = json.loads(Path(a.inp).read_text())
    results = judge_batch(examples, workers=a.workers)
    Path(a.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_yes = sum(1 for r in results if r["judge_verdict"] == "YES")
    n_no = sum(1 for r in results if r["judge_verdict"] == "NO")
    n_other = len(results) - n_yes - n_no
    print(f"YES={n_yes} NO={n_no} other={n_other} (n={len(results)}) -> wrote {a.out}")


if __name__ == "__main__":
    main()
