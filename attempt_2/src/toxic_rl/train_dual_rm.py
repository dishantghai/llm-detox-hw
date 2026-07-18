"""Phase 4 — train the helpfulness RM + harmlessness RM pair.

This is the `Safe RLHF <https://arxiv.org/abs/2310.12773>`_ decoupled
reward/cost architecture, implemented by calling the ORIGINAL, unmodified
``src.detox_hw.train_rm.train`` (the actual Task 4/5 homework trainer that
produced ``checkpoints/rm`` — always prompt-conditioned, LoRA-then-merged
``AutoModelForSequenceClassification`` over ``Qwen/Qwen2.5-0.5B``, Bradley-
Terry via ``tasks/task5_reward_head.py``'s ``build_rm``/``rm_step``) twice
against two different views of ``attempt_2/data/dpo_v2.jsonl`` (built by
``build_pairs_v2.py``):

- **harmlessness RM**: ``chosen`` (safe) vs. ``rejected_toxic`` — the same
  contrast the original homework's ``checkpoints/rm`` already encodes.
  Reruns it here on the (mildly) cleaned-up ``chosen`` pool for an apples-to-
  apples comparison against the helpfulness RM below.
- **helpfulness RM**: ``chosen`` (safe+substantive) vs. ``rejected_evasive``
  (safe but generic/templated) — the new contrast. This is the piece the
  original pipeline never had: nothing in it ever taught a model that a
  generic non-answer scores worse than a real one.

No new training code — ``train_rm.py``'s Bradley-Terry loss, LoRA setup, and
held-out-accuracy evaluation are reused exactly as they were. Only the *data
view* passed in changes, which is the entire point: this fix is a training-
signal decomposition, not a new modeling technique.

Usage (from repo root):
    python -m attempt_2.src.toxic_rl.train_dual_rm \\
        --data attempt_2/data/dpo_v2.jsonl \\
        --out-dir attempt_2/checkpoints
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_view(data_path: Path, rejected_key: str, out_path: Path) -> int:
    n = 0
    with data_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            fout.write(json.dumps({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row[rejected_key],
            }) + "\n")
            n += 1
    return n


def main() -> None:
    from src.detox_hw.train_rm import TrainConfig, train

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="attempt_2/data/dpo_v2.jsonl")
    p.add_argument("--out-dir", default="attempt_2/checkpoints")
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                   help="same backbone as the original checkpoints/rm, for a fair comparison")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    a = p.parse_args()

    data_path = Path(a.data)
    out_dir = Path(a.out_dir)
    views_dir = out_dir / "_rm_views"
    views_dir.mkdir(parents=True, exist_ok=True)

    harm_view = views_dir / "harmlessness_pairs.jsonl"
    help_view = views_dir / "helpfulness_pairs.jsonl"
    n_harm = _split_view(data_path, "rejected_toxic", harm_view)
    n_help = _split_view(data_path, "rejected_evasive", help_view)
    print(f"harmlessness view: {n_harm} pairs -> {harm_view}")
    print(f"helpfulness view:  {n_help} pairs -> {help_view}")

    for name, view_path in [("harmlessness", harm_view), ("helpfulness", help_view)]:
        rm_out = out_dir / f"rm_{name}"
        print(f"\n=== training {name} RM -> {rm_out} ===")
        cfg = TrainConfig(
            train_path=str(view_path),
            out_dir=str(rm_out),
            base_name=a.base,
            epochs=a.epochs,
            batch_size=a.batch_size,
            lr=a.lr,
            val_fraction=a.val_fraction,
        )
        train(cfg)

    print("\ndone. Run rm_redteam_gate.py against BOTH new RM dirs before using "
          "either for PPO — see GUIDE.md Phase 4.")


if __name__ == "__main__":
    main()
