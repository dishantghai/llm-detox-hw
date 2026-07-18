"""Phase 4 — the capstone validation: show BOTH single-axis RMs have real,
opposite blind spots, so the Lagrangian combination (Phase 5) isn't
decorative.

Two things demonstrated back to back, both against real hh-rlhf pairs
(``data/dpo.jsonl``, reused, first 8 rows):

1. The harmlessness RM alone (trained only on chosen vs. rejected_toxic —
   structurally identical to the original ``checkpoints/rm``) already failed
   ``rm_redteam_gate.py`` on 2 of 4 known attractors even after Phase 1's
   data cleanup (see ``redteam_gate_new_harmlessness_rm.json``) — cleaning
   the data didn't fix this RM because this RM was never given a signal
   that would teach it evasion is bad.
2. The helpfulness RM alone (chosen vs. rejected_evasive) passes that same
   gate cleanly — but scored against genuinely TOXIC completions from the
   same source data, it rates the toxic side higher than the safe side on
   most of these 8 examples, because it was never given a signal that would
   teach it toxicity is bad. It rewards "specific and engaged" without
   caring whether what's engaged with is safe.

Neither RM is usable alone for PPO. That's the actual argument for
``dual_reward_combiner.py``'s Lagrangian combination over either RM in
isolation, or over Task 8's original single hand-tuned blend.

Usage (from repo root):
    python -m attempt_2.scripts.demo_dual_rm_blindspots
"""
from __future__ import annotations

import json
from pathlib import Path

from src.toxic_rl.reward_model import TrainedRewardModel

OUT_JSON = Path("attempt_2/submissions/dual_rm_blindspot_demo.json")
OUT_TXT = Path("attempt_2/submissions/dual_rm_blindspot_demo.txt")


def main() -> None:
    rows = [json.loads(l) for l in Path("data/dpo.jsonl").open()][:8]
    help_rm = TrainedRewardModel("attempt_2/checkpoints/rm_helpfulness")
    harm_rm = TrainedRewardModel("attempt_2/checkpoints/rm_harmlessness")

    prompts = [r["prompt"] for r in rows]
    toxic = [r["rejected"] for r in rows]
    chosen = [r["chosen"] for r in rows]

    h_tox = help_rm.score(toxic, prompts=prompts)
    h_cho = help_rm.score(chosen, prompts=prompts)
    a_tox = harm_rm.score(toxic, prompts=prompts)
    a_cho = harm_rm.score(chosen, prompts=prompts)

    rows_out = []
    n_help_prefers_toxic = 0
    for i in range(len(rows)):
        prefers_toxic = h_tox[i] > h_cho[i]
        n_help_prefers_toxic += int(prefers_toxic)
        rows_out.append({
            "prompt": prompts[i],
            "toxic_text": toxic[i], "chosen_text": chosen[i],
            "helpfulness_rm_toxic": h_tox[i], "helpfulness_rm_chosen": h_cho[i],
            "harmlessness_rm_toxic": a_tox[i], "harmlessness_rm_chosen": a_cho[i],
            "helpfulness_rm_prefers_toxic_side": prefers_toxic,
        })

    result = {"n": len(rows), "helpfulness_rm_prefers_toxic_side_count": n_help_prefers_toxic, "rows": rows_out}
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        f"Dual-RM blind-spot demo ({len(rows)} real hh-rlhf pairs, data/dpo.jsonl)",
        "",
        f"helpfulness RM alone prefers the TOXIC completion over the safe one on "
        f"{n_help_prefers_toxic}/{len(rows)} pairs — it was never taught toxicity is bad, "
        f"only that genericness is bad.",
        "",
    ]
    for r in rows_out:
        flag = "  <-- helpfulness RM prefers TOXIC" if r["helpfulness_rm_prefers_toxic_side"] else ""
        lines.append(f"prompt: {r['prompt'][:65]!r}")
        lines.append(f"  toxic : help={r['helpfulness_rm_toxic']:+7.2f}  harm={r['harmlessness_rm_toxic']:+7.2f}  {r['toxic_text'][:60]!r}")
        lines.append(f"  chosen: help={r['helpfulness_rm_chosen']:+7.2f}  harm={r['harmlessness_rm_chosen']:+7.2f}  {r['chosen_text'][:60]!r}{flag}")
        lines.append("")
    OUT_TXT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {OUT_JSON}\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
