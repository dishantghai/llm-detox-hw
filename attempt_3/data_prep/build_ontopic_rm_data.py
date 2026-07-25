"""Build training pairs for a new `rm_ontopic` RM -- the coherence/
on-topic axis scoped after Stage 11 (see LOGBOOK.md): five stages in a
row (6 through 11) each closed one hand-coded "filler" pattern only to
have PPO immediately find a new disguise none of the existing gates
recognized. `attempt_3/scripts/coherence_judge.py` validated that a
generic zero-shot judge ("does this stay coherent and on-topic its whole
length?") catches every known disguise AND several genuinely new ones
none of this project's rule-based gates have a name for (an "EXEMPLARY"
system-prompt-fragment leak, task-avoidance disguised as engagement,
tone-mismatched generic deflection) -- the point of training a local RM
on its labels is to get that generalization at PPO-loop speed instead of
an API call per rollout.

Mirrors Stage 4's `data_prep/build_dual_rm_data.py` shape: prompt/chosen/
rejected triples, `train_rm.train`'s exact input format, no new training
code needed downstream.

Pipeline (each step cheap -- rule-based harvesting is free, judge calls
are the only real cost, and they're batched and sample-sized, not run
over every completion in the repo):

1. Harvest a "confirmed-bad" pool from every completion sitting in
   `attempt_3/submissions/*.json` (PPO rollouts across Stages 5-11) using
   this project's own existing rule-based diagnostics (DOCTYPE/non-Latin/
   line-repeat/compression-repeat/template-regex) -- zero API cost, and
   these are already independently confirmed real defects (100% judge
   agreement in the validation pass, see LOGBOOK.md).
2. Sample completions that pass EVERY rule (i.e. would look "clean" to
   this project's existing gates) and run them through the judge -- this
   is where the new, previously-uncategorized defects come from, and
   where genuine extra positives (on hard/adversarial prompts, not just
   `dpo_diverse`'s harmless-base ones) get confirmed.
3. Spot-check `dpo_diverse.jsonl`'s `chosen` side (already vetted for
   toxicity/hedging at Stage 1.5, but never explicitly checked for this
   axis) with a smaller judge sample rather than re-labeling all ~1,961
   rows individually -- matches this project's own practice of
   validating a sample before trusting a large pre-existing corpus.
4. Assemble `(prompt, chosen, rejected)` triples: every `chosen` (either
   `dpo_diverse.jsonl`'s or a judge-confirmed extra) is paired with a
   completion drawn from the confirmed-bad pool. The bad completion does
   NOT need to share the `chosen`'s original prompt -- same reasoning as
   Stage 4's `KNOWN_ATTRACTORS` reuse: the RM is prompt-conditioned
   (scores `(prompt, response)`, not `response` alone), so pairing
   `(prompt_i, chosen_i)` against `(prompt_i, unrelated_bad_string)`
   still teaches genuine relevance, not just "which prompt is this."

Usage:
    python -m attempt_3.data_prep.build_ontopic_rm_data \\
        --out attempt_3/data/dpo_ontopic.jsonl \\
        --report attempt_3/submissions/ontopic_rm_data_report.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

_TEMPLATE_RE = re.compile(r"^\s*(it'?s|it is)\s+(important|crucial)\s+to", re.I)
_APOLOGIZE_RE = re.compile(r"^\s*(i\s+(understand|apologize)|i'?m\s+sorry)", re.I)
_WORD_RE = re.compile(r"[a-zA-Z']+")

RULE_BAD_CAP_PER_CATEGORY = 80
CLEAN_JUDGE_SAMPLE_SIZE = 180
DPO_DIVERSE_SPOTCHECK_SIZE = 40


def _rows_of(d) -> list[dict]:
    if isinstance(d, list):
        return [r for r in d if isinstance(r, dict)]
    if isinstance(d, dict):
        return d.get("greedy_completions") or d.get("results") or d.get("worst_of_k") or []
    return []


def _is_rule_bad(prompt: str, completion: str) -> str | None:
    from attempt_3.scripts.measure_compression_repeat import classify as compress_classify
    from attempt_3.scripts.measure_line_repetition import classify as line_classify

    if not completion or len(completion.strip()) < 10:
        return None
    if "DOCTYPE" in completion or "<html" in completion.lower():
        return "html_tail"
    if any(ord(ch) > 1024 for ch in completion):
        return "non_latin"
    if line_classify(completion) == "line_repeat":
        return "line_repeat"
    if compress_classify(completion) == "compressed_repeat":
        return "compressed_repeat"
    if _TEMPLATE_RE.match(completion):
        return "template_collapse"
    if _APOLOGIZE_RE.match(completion) and len(completion.split()) < 20:
        return "generic_deflection"
    p_words = set(_WORD_RE.findall(prompt.lower()))
    c_words = set(_WORD_RE.findall(completion.lower()))
    if p_words and len(p_words & c_words) / max(1, len(p_words)) > 0.85 and len(completion) < len(prompt) * 1.5:
        return "prompt_echo"
    return None


def harvest_pools(submissions_dir: Path) -> tuple[list[dict], list[dict]]:
    """Returns (rule_bad_pool, rule_clean_pool), deduped by completion text."""
    bad: dict[str, dict] = {}
    clean: dict[str, dict] = {}
    for f in sorted(submissions_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in _rows_of(d):
            prompt, completion = r.get("prompt", ""), r.get("completion", "")
            if not prompt or not completion:
                continue
            key = completion[:300]
            reason = _is_rule_bad(prompt, completion)
            if reason:
                bad.setdefault(key, {"prompt": prompt, "completion": completion, "reason": reason, "source": f.name})
            else:
                clean.setdefault(key, {"prompt": prompt, "completion": completion, "source": f.name})
    return list(bad.values()), list(clean.values())


def cap_and_diversify(pool: list[dict], cap_per_category: int, rng: random.Random) -> list[dict]:
    by_cat = defaultdict(list)
    for r in pool:
        by_cat[r["reason"]].append(r)
    out = []
    for _, rows in by_cat.items():
        rng.shuffle(rows)
        seen_prompts = set()
        for r in rows:
            if r["prompt"] in seen_prompts:
                continue
            seen_prompts.add(r["prompt"])
            out.append(r)
            if len(seen_prompts) >= cap_per_category:
                break
    return out


def sample_diverse(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    rng.shuffle(pool)
    seen_prompts, out = set(), []
    for r in pool:
        if r["prompt"] in seen_prompts:
            continue
        seen_prompts.add(r["prompt"])
        out.append(r)
        if len(out) >= n:
            break
    return out


def harvest_ood_category_pool(submissions_dir: Path) -> list[dict]:
    """Rule-clean completions specifically from `*ood_eval*.json` files
    (which tag each row with a `category`: general_knowledge, coding_help,
    life_advice, creative_writing, professional_writing, debatable_opinions,
    adversarial_novel_framing, curveball) -- deduped by completion text
    across every stage's OOD run, so the same 55 canonical prompts
    contribute genuinely different phrasings from different checkpoints.

    v1 of this script sampled positives uniformly across ALL submissions
    (tracked-slice + OOD), and the result was 1,961 dpo_diverse chosen rows
    (harmless-base, hostile-prompt-decline style) against only 43 extras --
    a 98:2 skew that gave rm_ontopic almost no exposure to plain,
    substantive answers on benign/informative topics. Checked directly
    (not assumed) before spending a PPO run on the result: every one of 8
    real Qwen2.5-0.5B policy completions on coding/general-knowledge/
    creative-writing OOD prompts, independently confirmed good by the same
    coherence judge, scored `_ontopic_penalty` at ~3.0/3.0 (max) under the
    v1 RM -- it had learned "good" almost entirely as "declines a hostile
    prompt carefully," not genuine on-topicness. This harvest exists
    specifically to fix that: a large, category-diverse pool of benign/
    informative content to rebalance against.
    """
    pool: dict[str, dict] = {}
    for f in sorted(submissions_dir.glob("*ood_eval*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in d.get("results", []):
            if "category" not in r:
                continue
            prompt, completion = r.get("prompt", ""), r.get("completion", "")
            if not prompt or not completion:
                continue
            key = completion[:300]
            if key in pool:
                continue
            if _is_rule_bad(prompt, completion):
                continue
            pool[key] = {"prompt": prompt, "completion": completion, "category": r["category"], "source": f.name}
    return list(pool.values())


def main() -> None:
    from attempt_3.scripts.coherence_judge import judge_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions-dir", default="attempt_3/submissions")
    ap.add_argument("--dpo-diverse", default="attempt_3/data/dpo_diverse.jsonl")
    ap.add_argument("--out", default="attempt_3/data/dpo_ontopic.jsonl")
    ap.add_argument("--report", default="attempt_3/submissions/ontopic_rm_data_report.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ood-upweight", type=int, default=6,
                     help="duplication factor for judge-confirmed OOD-category positives, "
                          "so they aren't drowned out by dpo_diverse's ~2000 harmless-base rows")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    print("=== 1. harvesting rule-based pools from submissions ===")
    rule_bad, rule_clean = harvest_pools(Path(a.submissions_dir))
    print(f"rule-flagged bad: {len(rule_bad)}  rule-clean (needs judge check): {len(rule_clean)}")
    capped_bad = cap_and_diversify(rule_bad, RULE_BAD_CAP_PER_CATEGORY, rng)
    print(f"capped rule-bad pool: {len(capped_bad)}  {Counter(r['reason'] for r in capped_bad)}")

    print("\n=== 2. judging a sample of rule-clean completions ===")
    clean_sample = sample_diverse(rule_clean, CLEAN_JUDGE_SAMPLE_SIZE, rng)
    judged_clean = judge_batch(clean_sample)
    judge_found_bad = [r for r in judged_clean if r["judge_verdict"] == "NO"]
    judge_confirmed_good = [r for r in judged_clean if r["judge_verdict"] == "YES"]
    print(f"of {len(judged_clean)} rule-clean candidates: "
          f"{len(judge_found_bad)} actually bad (new disguises), "
          f"{len(judge_confirmed_good)} confirmed good")

    print("\n=== 2.5. harvesting + judging the OOD-category pool (fixes the benign-topic blind spot) ===")
    ood_pool = harvest_ood_category_pool(Path(a.submissions_dir))
    print(f"rule-clean OOD-category candidates: {len(ood_pool)}  "
          f"{Counter(r['category'] for r in ood_pool)}")
    judged_ood = judge_batch(ood_pool)
    ood_confirmed_good = [r for r in judged_ood if r["judge_verdict"] == "YES"]
    ood_found_bad = [r for r in judged_ood if r["judge_verdict"] == "NO"]
    print(f"of {len(judged_ood)} OOD-category candidates: "
          f"{len(ood_confirmed_good)} confirmed good, {len(ood_found_bad)} actually bad")
    print(f"confirmed-good category breakdown: {Counter(r['category'] for r in ood_confirmed_good)}")

    print("\n=== 3. spot-checking dpo_diverse.jsonl's chosen side ===")
    dpo_rows = [json.loads(l) for l in Path(a.dpo_diverse).open()]
    spotcheck_sample = rng.sample(dpo_rows, min(DPO_DIVERSE_SPOTCHECK_SIZE, len(dpo_rows)))
    spotcheck_examples = [{"prompt": r["prompt"], "completion": r["chosen"]} for r in spotcheck_sample]
    judged_spotcheck = judge_batch(spotcheck_examples)
    n_spot_bad = sum(1 for r in judged_spotcheck if r["judge_verdict"] == "NO")
    print(f"dpo_diverse chosen spot-check: {len(judged_spotcheck) - n_spot_bad}/{len(judged_spotcheck)} pass")
    if n_spot_bad > len(judged_spotcheck) * 0.1:
        print(f"WARNING: {n_spot_bad}/{len(judged_spotcheck)} spot-checked chosen rows failed the coherence "
              f"judge -- more than 10%. Do not trust dpo_diverse.jsonl wholesale; investigate before proceeding.")

    print("\n=== 4. assembling (prompt, chosen, rejected) pairs ===")
    final_bad_pool = [{"prompt": r["prompt"], "completion": r["completion"], "reason": r["reason"]}
                       for r in capped_bad]
    final_bad_pool += [{"prompt": r["prompt"], "completion": r["completion"], "reason": "judge_found:" + r["judge_reason"]}
                        for r in judge_found_bad]
    final_bad_pool += [{"prompt": r["prompt"], "completion": r["completion"], "reason": "judge_found_ood:" + r["judge_reason"]}
                        for r in ood_found_bad]
    print(f"final bad pool: {len(final_bad_pool)}")

    chosen_pool = [{"prompt": r["prompt"], "completion": r["chosen"], "source": "dpo_diverse"} for r in dpo_rows]
    chosen_pool += [{"prompt": r["prompt"], "completion": r["completion"], "source": r["source"]}
                     for r in judge_confirmed_good]
    # Upweight the OOD-category positives by duplication (each contributes
    # a fresh random `rejected` draw per copy below, so duplicates aren't
    # literally identical training rows) -- without this, 1,961 dpo_diverse
    # rows would still outnumber ~150 OOD positives 13:1, reproducing the
    # same skew this harvest exists to fix rather than actually balancing
    # it. Target: OOD-category positives land around 25-30% of the final
    # dataset, not ~2%.
    ood_upweight = int(a.ood_upweight)
    chosen_pool += [{"prompt": r["prompt"], "completion": r["completion"], "source": "ood_category:" + r["category"]}
                     for r in ood_confirmed_good for _ in range(ood_upweight)]
    print(f"final chosen pool: {len(chosen_pool)} "
          f"({len(dpo_rows)} dpo_diverse + {len(judge_confirmed_good)} cross-file extra + "
          f"{len(ood_confirmed_good)} OOD-category confirmed x{ood_upweight} upweight = "
          f"{len(ood_confirmed_good) * ood_upweight} rows, "
          f"{100 * len(ood_confirmed_good) * ood_upweight / len(chosen_pool):.1f}% of final dataset)")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in chosen_pool:
            rejected = rng.choice(final_bad_pool)
            f.write(json.dumps({
                "prompt": row["prompt"],
                "chosen": row["completion"],
                "rejected": rejected["completion"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen_pool)} pairs -> {out_path}")

    report = {
        "rule_bad_pool_raw": len(rule_bad),
        "rule_bad_pool_capped": len(capped_bad),
        "rule_bad_categories": dict(Counter(r["reason"] for r in capped_bad)),
        "rule_clean_pool_raw": len(rule_clean),
        "rule_clean_judged_sample": len(judged_clean),
        "rule_clean_judge_found_bad": len(judge_found_bad),
        "rule_clean_judge_confirmed_good": len(judge_confirmed_good),
        "ood_category_pool_raw": len(ood_pool),
        "ood_category_by_category": dict(Counter(r["category"] for r in ood_pool)),
        "ood_category_judged": len(judged_ood),
        "ood_category_confirmed_good": len(ood_confirmed_good),
        "ood_category_confirmed_good_by_category": dict(Counter(r["category"] for r in ood_confirmed_good)),
        "ood_category_found_bad": len(ood_found_bad),
        "ood_upweight_factor": ood_upweight,
        "ood_category_upweighted_rows": len(ood_confirmed_good) * ood_upweight,
        "dpo_diverse_spotcheck_n": len(judged_spotcheck),
        "dpo_diverse_spotcheck_pass": len(judged_spotcheck) - n_spot_bad,
        "final_bad_pool_size": len(final_bad_pool),
        "final_chosen_pool_size": len(chosen_pool),
        "output_pairs": len(chosen_pool),
        "notes": (
            "Rejected completions are drawn from a shared bad pool, not "
            "necessarily matching the chosen row's own prompt -- same "
            "reasoning as Stage 4's KNOWN_ATTRACTORS reuse: the RM is "
            "prompt-conditioned, so (prompt_i, chosen_i) vs (prompt_i, "
            "unrelated_bad_completion) still teaches genuine relevance."
        ),
    }
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2))
    print(f"wrote report -> {a.report}")


if __name__ == "__main__":
    main()
